"""
Training loop for DeepTriangle v2 (PyTorch).

Matches Kuo (2019) training configuration:
  - Optimizer  : Adam + AMSGrad, lr = 5e-4
  - Batch size : 512
  - Max epochs : 1000
  - Early stop : patience=200, min_delta=0.001, restore_best_weights=True
  - LR decay   : ReduceLROnPlateau factor=0.5, patience=50, min_lr=1e-6
  - Loss       : masked MSE (see loss.py), loss_weights = {paid: 0.5, case: 0.5}
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from loss import create_masked_mse_loss


# ---------------------------------------------------------------------------
# Training configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    learning_rate: float = 5e-4
    batch_size: int = 512
    epochs: int = 1000
    es_patience: int = 200
    min_delta: float = 0.001
    lr_patience: int = 50
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    verbose: int = 0
    mask_value: float = -99.0
    device: str | None = None


@dataclass
class SimpleHistory:
    history: Dict[str, list]
    params: Dict[str, Any]


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    config: TrainConfig,
    epoch_callback=None,
) -> Tuple[SimpleHistory, float]:
    """Train a PyTorch model and return history + wall time."""
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    masked_mse = create_masked_mse_loss(mask_value=config.mask_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
    )

    x_train = train_data["x"]["ay_seq_input"].astype(np.float32)
    gc_train = train_data["x"]["group_code_input"].astype(np.int64)
    y_paid = train_data["y"]["paid_output"].astype(np.float32)
    y_case = train_data["y"]["case_reserves_output"].astype(np.float32)

    x_val = val_data["x"]["ay_seq_input"].astype(np.float32)
    gc_val = val_data["x"]["group_code_input"].astype(np.int64)
    y_paid_val = val_data["y"]["paid_output"].astype(np.float32)
    y_case_val = val_data["y"]["case_reserves_output"].astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(gc_train),
        torch.from_numpy(y_paid),
        torch.from_numpy(y_case),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(gc_val),
        torch.from_numpy(y_paid_val),
        torch.from_numpy(y_case_val),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    history = {"loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience = 0

    t0 = time.perf_counter()

    for epoch in range(config.epochs):
        if epoch_callback is not None:
            epoch_callback(epoch)
        model.train()
        train_losses = []

        for xb, gcb, yb_paid, yb_case in train_loader:
            xb = xb.to(device)
            gcb = gcb.to(device)
            yb_paid = yb_paid.to(device)
            yb_case = yb_case.to(device)

            optimizer.zero_grad()
            paid_pred, case_pred = model(xb, gcb)
            loss_paid = masked_mse(yb_paid, paid_pred).mean()
            loss_case = masked_mse(yb_case, case_pred).mean()
            loss = 0.5 * loss_paid + 0.5 * loss_case
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, gcb, yb_paid, yb_case in val_loader:
                xb = xb.to(device)
                gcb = gcb.to(device)
                yb_paid = yb_paid.to(device)
                yb_case = yb_case.to(device)

                paid_pred, case_pred = model(xb, gcb)
                loss_paid = masked_mse(yb_paid, paid_pred).mean()
                loss_case = masked_mse(yb_case, case_pred).mean()
                loss = 0.5 * loss_paid + 0.5 * loss_case
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < (best_val - config.min_delta):
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if config.verbose == 2:
            print(f"Epoch {epoch+1:04d}  loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if patience >= config.es_patience:
            if config.verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.perf_counter() - t0
    history_obj = SimpleHistory(history=history, params={"epochs": config.epochs})
    return history_obj, training_time


# ---------------------------------------------------------------------------
# Convenience: extract summary statistics from a History object
# ---------------------------------------------------------------------------

def history_summary(history: SimpleHistory) -> Dict[str, Any]:
    val_losses = history.history.get("val_loss", [])
    train_losses = history.history.get("loss", [])
    epochs_trained = len(train_losses)

    return {
        "epochs_trained": epochs_trained,
        "best_val_loss": float(min(val_losses)) if val_losses else float("nan"),
        "final_train_loss": float(train_losses[-1]) if train_losses else float("nan"),
        "stopped_early": epochs_trained < history.params.get("epochs", epochs_trained),
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from models import build_model

    VOCAB = 50
    N_TRAIN = 200
    N_VAL = 40
    T = 9

    rng = np.random.default_rng(42)

    def _dummy_data(n: int) -> Dict[str, Any]:
        seq = rng.random((n, T, 2)).astype("float32") * 0.1
        gc = rng.integers(0, VOCAB, (n, 1)).astype("int32")
        paid_t = rng.random((n, T, 1)).astype("float32") * 0.1
        case_t = rng.random((n, T, 1)).astype("float32") * 0.05
        return {
            "x": {"ay_seq_input": seq, "group_code_input": gc},
            "y": {"paid_output": paid_t, "case_reserves_output": case_t},
        }

    train_d = _dummy_data(N_TRAIN)
    val_d = _dummy_data(N_VAL)

    cfg = TrainConfig(
        learning_rate=1e-3,
        batch_size=64,
        epochs=10,
        es_patience=5,
        min_delta=1e-4,
        lr_patience=3,
        verbose=0,
    )

    for arch in ("gru_baseline", "gru_attention", "gru_attention_unmasked"):
        torch.manual_seed(0)
        model = build_model(arch, vocab_size=VOCAB)
        hist, t_sec = train_model(model, train_d, val_d, cfg)
        summ = history_summary(hist)
        print(
            f"  {arch:20s}  "
            f"epochs={summ['epochs_trained']:3d}  "
            f"best_val={summ['best_val_loss']:.6f}  "
            f"time={t_sec:.2f}s"
        )

    print("\ntrain.py OK")
