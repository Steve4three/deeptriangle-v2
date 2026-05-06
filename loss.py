"""
Loss functions for DeepTriangle v2 (PyTorch).

Masked MSE:
    loss_i = sum_{t: y_true_{i,t} != mask} (y_true_{i,t} - y_pred_{i,t})^2
             / max(#{t: y_true_{i,t} != mask}, 1)
"""

from __future__ import annotations

import torch


def create_masked_mse_loss(mask_value: float = -99.0):
    """Factory returning a masked MSE loss function (per-sample)."""

    def masked_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        mask = (y_true != mask_value).float()
        squared_error = ((y_true - y_pred) * mask) ** 2
        sum_sq = squared_error.sum(dim=(1, 2))
        num_valid = mask.sum(dim=(1, 2)).clamp(min=1.0)
        return sum_sq / num_valid

    masked_mse.__name__ = "masked_mse"
    return masked_mse


masked_mse_loss = create_masked_mse_loss(mask_value=-99.0)


if __name__ == "__main__":
    import numpy as np

    loss_fn = create_masked_mse_loss(mask_value=-99.0)

    y = torch.tensor([[[0.1], [0.2], [0.3]]], dtype=torch.float32)
    loss = loss_fn(y, y)
    assert float(loss.numpy()[0]) == 0.0
    print(f"Test 1 passed  (loss={float(loss.numpy()[0]):.6f})")

    y_true = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
    y_pred = torch.tensor([[[2.0], [2.0], [3.0]]], dtype=torch.float32)
    loss = loss_fn(y_true, y_pred)
    expected = (1.0 ** 2 + 0.0 + 0.0) / 3.0
    assert abs(float(loss.numpy()[0]) - expected) < 1e-6
    print(f"Test 2 passed  (loss={float(loss.numpy()[0]):.6f}, expected={expected:.6f})")

    y_true_m = torch.tensor([[[1.0], [-99.0], [-99.0]]], dtype=torch.float32)
    y_pred_m = torch.tensor([[[2.0], [5.0], [5.0]]], dtype=torch.float32)
    loss = loss_fn(y_true_m, y_pred_m)
    expected_m = 1.0 ** 2 / 1.0
    assert abs(float(loss.numpy()[0]) - expected_m) < 1e-6
    print(f"Test 3 passed  (loss={float(loss.numpy()[0]):.6f}, expected={expected_m:.6f})")

    y_true_b = torch.tensor(
        [[[1.0], [2.0], [-99.0]],
         [[3.0], [-99.0], [-99.0]]],
        dtype=torch.float32,
    )
    y_pred_b = torch.tensor(
        [[[1.5], [2.5], [9.9]],
         [[4.0], [9.9], [9.9]]],
        dtype=torch.float32,
    )
    loss_b = loss_fn(y_true_b, y_pred_b)
    e0 = (0.5 ** 2 + 0.5 ** 2) / 2.0
    e1 = 1.0 ** 2 / 1.0
    assert abs(float(loss_b.numpy()[0]) - e0) < 1e-6
    assert abs(float(loss_b.numpy()[1]) - e1) < 1e-6
    print(f"Test 4 passed  (loss=[{float(loss_b.numpy()[0]):.6f}, {float(loss_b.numpy()[1]):.6f}])")

    print("\nloss.py OK — all tests passed")
