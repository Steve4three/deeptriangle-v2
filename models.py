"""
PyTorch neural architectures for DeepTriangle v2.

All architectures share:
  - Inputs:
      * ay_seq_input    : (batch, 9, 2)  — [paid_lags, case_lags] stacked
      * group_code_input: (batch, 1)     — integer company id
  - group_code embedding: Embedding(vocab_size, min(50, vocab_size-1)) -> Repeat
  - Mask value: -99.0 in input means padded
  - Dual output heads with ReLU final activation (matching Kuo 2019)

Architectures
-------------
1. GRU_BASELINE
   Encoder GRU (masked) -> RepeatVector -> Decoder GRU -> concat(company_embed) -> heads

2. GRU_ATTENTION
   Encoder GRU (masked) -> MHA(1 head, with padding mask) -> residual+LN -> Decoder GRU
   The encoder GRU skips padded timesteps (matching Baseline), and the attention layer
   excludes padded positions from softmax via key_padding_mask.

3. GRU_ATTENTION_UNMASKED
   Ablation variant: encoder GRU does NOT skip padded timesteps, attention has no padding mask.
   Included only as an ablation to demonstrate that the "better" convergence of unmasked
   attention is an artifact of attending to zero-padded noise, not a methodological advantage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Supported architecture names
ARCH_NAMES = (
    "gru_baseline",
    "gru_attention",
    "gru_attention_unmasked",
)


# ---------------------------------------------------------------------------
# Keras-matching weight initialization
# ---------------------------------------------------------------------------
#
# ⚠️  LOAD-BEARING FOR KUO 2019 REPLICATION — DO NOT REMOVE
#
# These helpers exist so the PyTorch rewrite reproduces the initialization
# distributions of Kuo 2019's original Keras/TensorFlow implementation.
# Dropping them and falling back to PyTorch defaults changes the starting
# weights materially:
#   - nn.Embedding default N(0, 1)   std ≈ 1.0    vs Keras U(-0.05, 0.05) std ≈ 0.029  (35x larger)
#   - nn.Linear   default kaiming(a=√5)          vs Keras xavier_uniform                (~45% off)
#   - nn.GRUCell  default uniform(-1/√H, 1/√H)   vs Keras orthogonal per-gate            (different scheme)
#
# Changing these will break paper replication. If you need to tweak anything
# in this section, run scripts/ab_mha_init_experiment.py first and compare
# attention entropy + final MAPE/RMSPE before committing.
#
# NOTE: init_model_keras_style deliberately SKIPS nn.MultiheadAttention.out_proj.
# PyTorch's conservative kaiming_uniform(a=√5) default for that layer is better
# tuned for residual+LayerNorm paths than Keras's glorot_uniform, which inflates
# out_proj weight std by ~74% and plausibly contributes to attention collapse.
# See init_model_keras_style docstring for the full rationale.

def _init_keras_gru_cell(cell: nn.GRUCell) -> None:
    """Initialize a PyTorch GRUCell to match Keras GRU defaults.

    Keras GRU uses:
      - kernel_initializer='glorot_uniform'     (input→hidden weights)
      - recurrent_initializer='orthogonal'      (hidden→hidden weights)
      - bias_initializer='zeros'
    PyTorch GRUCell packs weights as:
      - weight_ih: (3*hidden, input)  — input→hidden for [r, z, n] gates
      - weight_hh: (3*hidden, hidden) — hidden→hidden for [r, z, n] gates
      - bias_ih, bias_hh: (3*hidden,)
    """
    hidden_size = cell.hidden_size
    # Input weights: glorot_uniform
    nn.init.xavier_uniform_(cell.weight_ih)
    # Recurrent weights: orthogonal (applied per-gate)
    for i in range(3):
        nn.init.orthogonal_(cell.weight_hh[i * hidden_size:(i + 1) * hidden_size])
    # Biases: zeros
    nn.init.zeros_(cell.bias_ih)
    nn.init.zeros_(cell.bias_hh)


def _init_keras_linear(linear: nn.Linear) -> None:
    """Initialize nn.Linear to match Keras Dense(kernel_initializer='glorot_uniform', bias='zeros')."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_keras_embedding(emb: nn.Embedding) -> None:
    """Initialize nn.Embedding to match Keras Embedding (uniform, not N(0,1))."""
    nn.init.uniform_(emb.weight, -0.05, 0.05)


def init_model_keras_style(model: nn.Module) -> None:
    """Walk a model and apply Keras-matching initialization to all layers.

    LOAD-BEARING for Kuo 2019 replication. See section-level comment above for
    the layer-by-layer rationale.

    Exception — nn.MultiheadAttention.out_proj is deliberately LEFT at PyTorch's
    default (kaiming_uniform with a=√5). If we promote it to xavier_uniform (as
    a naive "match Keras" sweep would do), its weight std inflates by ~74% for
    typical gru_units. That over-initialization makes the attention output
    dominate the residual stream at step 0, drowning out the encoder signal
    through LayerNorm and plausibly contributing to attention collapse in the
    masked gru_attention variant. Verified empirically — see
    scripts/ab_mha_init_experiment.py for the A/B comparison.
    """
    for name, module in model.named_modules():
        # Skip MHA output projection — PyTorch's conservative default is
        # better tuned for residual+LayerNorm paths than Keras's glorot_uniform.
        if name.endswith("mha.out_proj"):
            continue
        if isinstance(module, nn.GRUCell):
            _init_keras_gru_cell(module)
        elif isinstance(module, nn.Linear):
            _init_keras_linear(module)
        elif isinstance(module, nn.Embedding):
            _init_keras_embedding(module)
        # LayerNorm already defaults to weight=1, bias=0 (same as Keras)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embedding_dim(vocab_size: int) -> int:
    return max(1, min(50, vocab_size - 1))


def _zero_mask(x: torch.Tensor, mask_value: float = -99.0) -> torch.Tensor:
    return x.masked_fill(x == mask_value, 0.0)


def _build_attention_mask(ay_seq_input: torch.Tensor, mask_value: float = -99.0) -> torch.Tensor:
    """Return valid positions mask: True where at least one feature != mask_value."""
    return (ay_seq_input != mask_value).any(dim=-1)  # (batch, T)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class RecurrentDropoutGRUCell(nn.Module):
    """GRU cell with recurrent dropout (matching Keras GRU recurrent_dropout).

    Keras applies the SAME dropout mask to recurrent weights at every timestep
    within a sequence (variational/locked dropout on h_{t-1}).
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0, recurrent_dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.gru_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None,
                return_sequences: bool = True,
                mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, T, input_size)
        h_0: (batch, hidden_size) or None
        mask: (batch, T) bool — True where valid, False where padded.
              When a timestep is masked, the hidden state is NOT updated
              (matching Keras Masking layer behavior).
        Returns: (output_seq, h_n)
        """
        batch, T, _ = x.shape
        device = x.device

        if h_0 is None:
            h = torch.zeros(batch, self.hidden_size, device=device)
        else:
            h = h_0

        # Generate locked dropout masks (same mask for all timesteps)
        if self.training and self.dropout > 0:
            input_mask = torch.bernoulli(torch.full((batch, x.shape[2]), 1.0 - self.dropout, device=device))
            input_mask = input_mask / (1.0 - self.dropout)  # scale
        else:
            input_mask = None

        if self.training and self.recurrent_dropout > 0:
            rec_mask = torch.bernoulli(torch.full((batch, self.hidden_size), 1.0 - self.recurrent_dropout, device=device))
            rec_mask = rec_mask / (1.0 - self.recurrent_dropout)
        else:
            rec_mask = None

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            if input_mask is not None:
                x_t = x_t * input_mask
            h_prev = h
            if rec_mask is not None:
                h_for_cell = h * rec_mask
            else:
                h_for_cell = h
            h_new = self.gru_cell(x_t, h_for_cell)

            # If mask provided, skip update for padded timesteps (Keras Masking behavior)
            if mask is not None:
                valid = mask[:, t].unsqueeze(1)  # (batch, 1)
                h = torch.where(valid, h_new, h_prev)
            else:
                h = h_new
            outputs.append(h)

        h_n = h  # (batch, hidden)
        if return_sequences:
            output_seq = torch.stack(outputs, dim=1)  # (batch, T, hidden)
        else:
            output_seq = h_n.unsqueeze(1)  # (batch, 1, hidden)
        return output_seq, h_n


class DualOutputHead(nn.Module):
    def __init__(self, input_dim: int, dense_units: int, dropout_rate: float):
        super().__init__()
        self.paid_dense = nn.Linear(input_dim, dense_units)
        self.case_dense = nn.Linear(input_dim, dense_units)
        self.paid_out = nn.Linear(dense_units, 1)
        self.case_out = nn.Linear(dense_units, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def _branch(self, x: torch.Tensor, dense: nn.Linear, out: nn.Linear) -> torch.Tensor:
        x = F.relu(dense(x))
        x = self.dropout(x)
        x = F.relu(out(x))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        paid = self._branch(x, self.paid_dense, self.paid_out)
        case = self._branch(x, self.case_dense, self.case_out)
        return paid, case


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, timesteps: int, gru_units: int, dropout_rate: float, dense_units: int, emb_dim: Optional[int] = None):
        super().__init__()
        self.timesteps = timesteps
        emb_dim = emb_dim if emb_dim is not None else _embedding_dim(vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Match TF: GRU with dropout + recurrent_dropout on both encoder and decoder
        self.enc_gru = RecurrentDropoutGRUCell(
            input_size=2, hidden_size=gru_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
        )
        self.dec_gru = RecurrentDropoutGRUCell(
            input_size=gru_units, hidden_size=gru_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
        )
        self.heads = DualOutputHead(gru_units + emb_dim, dense_units, dropout_rate)

    def forward(self, ay_seq_input: torch.Tensor, group_code_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ay_seq_input: (batch, T, 2)
        # TF uses Masking(-99.0): GRU skips masked timesteps entirely
        valid_mask = _build_attention_mask(ay_seq_input)  # True where valid
        zeroed = _zero_mask(ay_seq_input)

        # Encoder: pass mask so GRU skips padded timesteps (matching Keras Masking)
        _, encoded = self.enc_gru(zeroed, return_sequences=False, mask=valid_mask)

        # RepeatVector → decoder (decoder input is all-valid, no mask needed)
        encoded_seq = encoded.unsqueeze(1).repeat(1, self.timesteps, 1)
        decoded_seq, _ = self.dec_gru(encoded_seq, return_sequences=True)

        gc = group_code_input.squeeze(1)
        emb = self.embedding(gc)
        emb_rep = emb.unsqueeze(1).repeat(1, self.timesteps, 1)
        merged = torch.cat([decoded_seq, emb_rep], dim=-1)

        return self.heads(merged)


class GRUAttention(nn.Module):
    """GRU + Attention (masked) — the methodologically correct variant.

    Both the encoder GRU and the attention layer respect the padding mask:
      - Encoder GRU skips padded timesteps (matching GRU Baseline behavior)
      - MHA excludes padded positions from softmax via key_padding_mask
    This ensures a fair apples-to-apples comparison with the Baseline.
    """
    def __init__(self, vocab_size: int, timesteps: int, gru_units: int, dropout_rate: float, dense_units: int):
        super().__init__()
        self.timesteps = timesteps
        emb_dim = _embedding_dim(vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.enc_gru = RecurrentDropoutGRUCell(
            input_size=2, hidden_size=gru_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
        )
        self.mha = nn.MultiheadAttention(embed_dim=gru_units, num_heads=1, dropout=dropout_rate, batch_first=True)
        self.ln = nn.LayerNorm(gru_units)
        self.dec_gru = RecurrentDropoutGRUCell(
            input_size=gru_units, hidden_size=gru_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
        )
        self.heads = DualOutputHead(gru_units + emb_dim, dense_units, dropout_rate)

    def forward(self, ay_seq_input: torch.Tensor, group_code_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = _build_attention_mask(ay_seq_input)  # True where valid
        zeroed = _zero_mask(ay_seq_input)

        # Encoder: masked — skips padded timesteps (matching Baseline)
        enc_seq, _ = self.enc_gru(zeroed, return_sequences=True, mask=valid_mask)

        # Self-attention with padding mask + residual + LayerNorm
        padding_mask = ~valid_mask  # True where padded (PyTorch convention)
        attn_out, _ = self.mha(enc_seq, enc_seq, enc_seq, key_padding_mask=padding_mask)
        attn_out = self.ln(enc_seq + attn_out)

        # Decoder
        decoded_seq, _ = self.dec_gru(attn_out, return_sequences=True)

        gc = group_code_input.squeeze(1)
        emb = self.embedding(gc)
        emb_rep = emb.unsqueeze(1).repeat(1, self.timesteps, 1)
        merged = torch.cat([decoded_seq, emb_rep], dim=-1)
        return self.heads(merged)


class GRUAttentionUnmasked(nn.Module):
    """GRU + Attention (unmasked) — ablation variant.

    Neither the encoder GRU nor the attention layer uses a padding mask.
    Encoder processes zero-padded positions normally; attention attends to all 9 positions.
    Included only to demonstrate that "better" convergence of unmasked attention is an artifact
    of spreading softmax over zero-noise positions, not a real improvement.
    """
    def __init__(self, vocab_size: int, timesteps: int, gru_units: int, dropout_rate: float, dense_units: int):
        super().__init__()
        self.timesteps = timesteps
        emb_dim = _embedding_dim(vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.enc_gru = RecurrentDropoutGRUCell(
            input_size=2, hidden_size=gru_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
        )
        self.mha = nn.MultiheadAttention(embed_dim=gru_units, num_heads=1, dropout=dropout_rate, batch_first=True)
        self.ln = nn.LayerNorm(gru_units)
        self.dec_gru = RecurrentDropoutGRUCell(
            input_size=gru_units, hidden_size=gru_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
        )
        self.heads = DualOutputHead(gru_units + emb_dim, dense_units, dropout_rate)

    def forward(self, ay_seq_input: torch.Tensor, group_code_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zeroed = _zero_mask(ay_seq_input)

        # Encoder: NO mask — processes zero-padded positions (intentional ablation)
        enc_seq, _ = self.enc_gru(zeroed, return_sequences=True)

        # Self-attention: NO padding mask (intentional ablation)
        attn_out, _ = self.mha(enc_seq, enc_seq, enc_seq, key_padding_mask=None)
        attn_out = self.ln(enc_seq + attn_out)

        decoded_seq, _ = self.dec_gru(attn_out, return_sequences=True)

        gc = group_code_input.squeeze(1)
        emb = self.embedding(gc)
        emb_rep = emb.unsqueeze(1).repeat(1, self.timesteps, 1)
        merged = torch.cat([decoded_seq, emb_rep], dim=-1)
        return self.heads(merged)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    arch_name: str,
    vocab_size: int,
    timesteps: int = 9,
    gru_units: int = 128,
    dropout_rate: float = 0.10,
    dense_units: int = 64,
    emb_dim: Optional[int] = None,
) -> nn.Module:
    builders = {
        "gru_baseline": GRUBaseline,
        "gru_attention": GRUAttention,
        "gru_attention_unmasked": GRUAttentionUnmasked,
    }
    if arch_name not in builders:
        raise ValueError(f"Unknown architecture '{arch_name}'. Choose from {ARCH_NAMES}")

    kwargs = dict(
        vocab_size=vocab_size,
        timesteps=timesteps,
        gru_units=gru_units,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
    )
    if emb_dim is not None:
        kwargs["emb_dim"] = emb_dim

    model = builders[arch_name](**kwargs)
    # Apply Keras-matching weight initialization
    init_model_keras_style(model)
    return model


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(0)

    VOCAB = 350
    BATCH = 8
    T = 9

    # Create input with some masked positions (realistic triangle structure)
    dummy_seq = torch.from_numpy(np.random.randn(BATCH, T, 2).astype("float32"))
    # Mask later timesteps for some samples (simulating immature AYs)
    for i in range(BATCH):
        mask_from = max(2, T - i)  # progressively mask more timesteps
        if mask_from < T:
            dummy_seq[i, mask_from:, :] = -99.0

    dummy_gc = torch.from_numpy(np.random.randint(0, VOCAB, (BATCH, 1)).astype("int64"))

    for arch in ARCH_NAMES:
        model = build_model(arch, vocab_size=VOCAB)
        model.eval()
        with torch.no_grad():
            paid_out, case_out = model(dummy_seq, dummy_gc)
        print(
            f"  {arch:25s}  params={sum(p.numel() for p in model.parameters()):8d}  "
            f"paid_out={tuple(paid_out.shape)}  case_out={tuple(case_out.shape)}"
        )
        assert paid_out.min().item() >= 0.0, f"{arch}: paid output has negatives!"
        assert case_out.min().item() >= 0.0, f"{arch}: case output has negatives!"

    print("\nmodels.py OK — all architectures built and sanity-checked")
