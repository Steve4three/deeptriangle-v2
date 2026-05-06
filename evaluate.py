"""
Evaluation utilities for DeepTriangle v2 (PyTorch).

Computes MAPE and RMSPE on the test split.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

MASK_VALUE = -99.0
EPSILON = 1e-8

# Env-var-aware defaults for temporal splits
_DEFAULT_AY_MIN = int(os.environ.get("DEEPTRIANGLE_AY_MIN", 2002))
_DEFAULT_AY_MAX = int(os.environ.get("DEEPTRIANGLE_TEST_MAX_AY", 2010))
_DEFAULT_TEST_CAL = int(os.environ.get("DEEPTRIANGLE_TEST_CAL", 2011))
DEFAULT_AY_RANGE = (_DEFAULT_AY_MIN, _DEFAULT_AY_MAX)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_paid_output(
    model,
    inputs: Dict[str, Any],
    batch_size: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    """
    Return paid_output predictions as numpy array of shape (N, 9, 1).
    """
    if isinstance(model, torch.nn.Module):
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(dev)
        model.eval()

        ay = torch.from_numpy(inputs["ay_seq_input"]).float()
        gc = torch.from_numpy(inputs["group_code_input"]).long()
        ds = TensorDataset(ay, gc)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        outputs = []
        with torch.no_grad():
            for ay_b, gc_b in dl:
                ay_b = ay_b.to(dev)
                gc_b = gc_b.to(dev)
                paid_pred, _ = model(ay_b, gc_b)
                outputs.append(paid_pred.detach().cpu().numpy())

        return np.concatenate(outputs, axis=0)

    raise TypeError("Model type not supported for prediction")


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_ultimate_actuals(
    raw_data: pd.DataFrame,
    lob: str,
    accident_year_range: Tuple[int, int] = None,
    max_dev_lag: int = 9,
) -> pd.DataFrame:
    if accident_year_range is None:
        accident_year_range = DEFAULT_AY_RANGE
    lob_data = raw_data[raw_data["lob"] == lob].copy()

    ult_rows = lob_data[
        (lob_data["development_lag"] == max_dev_lag)
        & (lob_data["accident_year"] >= accident_year_range[0])
        & (lob_data["accident_year"] <= accident_year_range[1])
    ][["group_code", "accident_year", "cumulative_paid_loss", "earned_premium_net"]].copy()

    ult_rows = ult_rows.rename(columns={"cumulative_paid_loss": "ultimate_actual_raw"})

    return ult_rows.reset_index(drop=True)


def extract_observed_cumulative(
    raw_data: pd.DataFrame,
    lob: str,
    test_calendar_year: int = None,
    accident_year_range: Tuple[int, int] = None,
) -> pd.DataFrame:
    if test_calendar_year is None:
        test_calendar_year = _DEFAULT_TEST_CAL
    if accident_year_range is None:
        accident_year_range = DEFAULT_AY_RANGE
    lob_data = raw_data[raw_data["lob"] == lob].copy()

    last_cal_year = test_calendar_year - 1

    rows = []
    for (gc, ay), grp in lob_data.groupby(["group_code", "accident_year"]):
        if not (accident_year_range[0] <= ay <= accident_year_range[1]):
            continue

        obs = grp[grp["calendar_year"] <= last_cal_year].copy()
        if obs.empty:
            rows.append({
                "group_code": gc,
                "accident_year": int(ay),
                "observed_cumulative_raw": 0.0,
                "last_observed_lag": 0,
            })
            continue

        max_lag_row = obs.loc[obs["development_lag"].idxmax()]
        rows.append({
            "group_code": gc,
            "accident_year": int(ay),
            "observed_cumulative_raw": float(max_lag_row["cumulative_paid_loss"]),
            "last_observed_lag": int(max_lag_row["development_lag"]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def compute_mape_rmspe(
    model,
    test_data: Dict[str, Any],
    test_metadata: pd.DataFrame,
    raw_data: pd.DataFrame = None,
    lob: str = None,
    focus_lag: int = 9,
    accident_year_range: Tuple[int, int] = None,
    outlier_threshold: float = 10.0,
    test_calendar_year: int = None,
) -> Dict[str, float]:
    if accident_year_range is None:
        accident_year_range = DEFAULT_AY_RANGE
    if test_calendar_year is None:
        test_calendar_year = _DEFAULT_TEST_CAL
    paid_pred_norm = predict_paid_output(model, test_data["x"])
    paid_pred_norm = np.squeeze(paid_pred_norm, axis=-1)

    n_samples = paid_pred_norm.shape[0]
    assert len(test_metadata) == n_samples, (
        f"test_metadata rows ({len(test_metadata)}) != prediction rows ({n_samples})"
    )

    ep = test_metadata["earned_premium_net"].values
    paid_pred_dollars = paid_pred_norm * ep[:, None]

    if raw_data is not None and lob is not None:
        return _evaluate_mode_a(
            paid_pred_dollars=paid_pred_dollars,
            test_metadata=test_metadata,
            raw_data=raw_data,
            lob=lob,
            focus_lag=focus_lag,
            accident_year_range=accident_year_range,
            outlier_threshold=outlier_threshold,
            test_calendar_year=test_calendar_year,
        )

    return _evaluate_mode_b(
        paid_pred_dollars=paid_pred_dollars,
        test_metadata=test_metadata,
        accident_year_range=accident_year_range,
        outlier_threshold=outlier_threshold,
    )


def _evaluate_mode_a(
    paid_pred_dollars: np.ndarray,
    test_metadata: pd.DataFrame,
    raw_data: pd.DataFrame,
    lob: str,
    focus_lag: int,
    accident_year_range: tuple,
    outlier_threshold: float,
    test_calendar_year: int = None,
) -> Dict[str, float]:
    actual_ult_df = extract_ultimate_actuals(
        raw_data, lob, accident_year_range, focus_lag
    )
    obs_cum_df = extract_observed_cumulative(
        raw_data, lob, test_calendar_year=test_calendar_year,
        accident_year_range=accident_year_range,
    )

    actual_lookup = {
        (str(r.group_code), int(r.accident_year)): float(r.ultimate_actual_raw)
        for _, r in actual_ult_df.iterrows()
    }
    obs_lookup = {
        (str(r.group_code), int(r.accident_year)): (
            float(r.observed_cumulative_raw),
            int(r.last_observed_lag),
        )
        for _, r in obs_cum_df.iterrows()
    }

    results = []
    for i in range(len(test_metadata)):
        row = test_metadata.iloc[i]
        gc = str(row["group_code"])
        ay = int(row["accident_year"])

        if not (accident_year_range[0] <= ay <= accident_year_range[1]):
            continue

        key = (gc, ay)
        if key not in actual_lookup:
            continue

        actual_ult = actual_lookup[key]
        obs_cum, last_obs_lag = obs_lookup.get(key, (0.0, 0))

        remaining_lags = focus_lag - last_obs_lag
        n_pred = min(remaining_lags, paid_pred_dollars.shape[1])
        pred_increments_sum = float(paid_pred_dollars[i, :n_pred].sum())
        predicted_ult = obs_cum + pred_increments_sum

        results.append({
            "group_code": gc,
            "accident_year": ay,
            "actual_ultimate": actual_ult,
            "predicted_ultimate": predicted_ult,
        })

    if not results:
        return {"mape": float("nan"), "rmspe": float("nan"), "n_companies": 0, "n_filtered": 0}

    df = pd.DataFrame(results)
    df = df.groupby(["group_code", "accident_year"]).mean().reset_index()

    return _compute_metrics(df, outlier_threshold)


def _evaluate_mode_b(
    paid_pred_dollars: np.ndarray,
    test_metadata: pd.DataFrame,
    accident_year_range: tuple,
    outlier_threshold: float,
) -> Dict[str, float]:
    results = []
    for i in range(len(test_metadata)):
        row = test_metadata.iloc[i]
        ay = int(row["accident_year"])
        if not (accident_year_range[0] <= ay <= accident_year_range[1]):
            continue

        predicted_ult = float(paid_pred_dollars[i].sum())
        results.append({
            "group_code": str(row["group_code"]),
            "accident_year": ay,
            "actual_ultimate": float("nan"),
            "predicted_ultimate": predicted_ult,
        })

    return {"mape": float("nan"), "rmspe": float("nan"), "n_companies": 0, "n_filtered": 0}


def _compute_metrics(
    df: pd.DataFrame,
    outlier_threshold: float,
) -> Dict[str, float]:
    denominator = np.where(
        np.abs(df["actual_ultimate"].values) < EPSILON,
        EPSILON,
        np.abs(df["actual_ultimate"].values),
    )
    pct_error = (
        (df["predicted_ultimate"].values - df["actual_ultimate"].values) / denominator
    )
    df = df.copy()
    df["pct_error"] = pct_error

    df = df[np.isfinite(df["pct_error"])]

    outlier_mask = np.abs(df["pct_error"]) <= outlier_threshold
    n_filtered = int((~outlier_mask).sum())
    df = df[outlier_mask]

    if len(df) == 0:
        return {
            "mape": float("nan"), "rmspe": float("nan"),
            "n_companies": 0, "n_filtered": n_filtered,
        }

    company_metrics = df.groupby("group_code").agg(
        abs_pct_error=("pct_error", lambda x: np.mean(np.abs(x))),
        sq_pct_error=("pct_error", lambda x: np.mean(x ** 2)),
    ).reset_index()

    result = {
        "mape": float(company_metrics["abs_pct_error"].mean()),
        "rmspe": float(np.sqrt(company_metrics["sq_pct_error"].mean())),
        "n_companies": len(company_metrics),
        "n_filtered": n_filtered,
    }
    # Per-company breakdown for EDA (Tier 3 analysis)
    result["per_company_mape"] = {
        str(row.group_code): round(float(row.abs_pct_error), 6)
        for _, row in company_metrics.iterrows()
    }
    result["mape_std"] = float(company_metrics["abs_pct_error"].std())
    result["mape_p25"] = float(company_metrics["abs_pct_error"].quantile(0.25))
    result["mape_p75"] = float(company_metrics["abs_pct_error"].quantile(0.75))
    return result



