"""
Maturity Analysis — DeepTriangle v2.

Evaluates how model performance varies with accident year maturity at the
test diagonal (calendar_year = 2011).

Background
----------
The test diagonal is at calendar_year = 2011.  For a given accident year AY,
the development lag at the test diagonal is:

    test_lag = 2011 - AY

  AY=2002 => test_lag=9  (fully mature — only 1 future lag to predict)
  AY=2003 => test_lag=8
  ...
  AY=2007 => test_lag=4
  AY=2008 => test_lag=3
  AY=2009 => test_lag=2
  AY=2010 => test_lag=1  (very immature — must predict lags 2-9)

Maturity groups
---------------
  "mature"   : AY 2002-2006  (test_lag 9-5; ≥5 observed lags, ≤4 to predict)
  "immature" : AY 2007-2010  (test_lag 4-1; ≤4 observed lags, ≥5 to predict)

This split is motivated by the actuarial intuition that neural network
reserving models are expected to outperform chain-ladder on immature claims
(where loss development patterns are more variable) but may offer less
advantage on mature claims (where CL extrapolation is more reliable).

Methodology
-----------
1. Retrain 1 model per architecture (seed=0) on WC data (Phase 1 HPs).
2. Run inference on test set, retrieving per-(company, accident_year) predictions.
3. Split predictions by maturity group using the accident_year column in test_metadata.
4. Compute MAPE for each group independently.
5. Optionally compare to Mack CL broken down by maturity group.

Note: We retrain rather than loading saved models because run_phase1.py
does not save model weights (only metrics).  One seed is sufficient for
directional insight; cross-seed variability analysis can be added later.

Usage
-----
    python analyze_maturity.py                      # all 3 architectures, WC
    python analyze_maturity.py --archs gru_baseline # single arch
    python analyze_maturity.py --lob private_passenger_auto
    python analyze_maturity.py --seed 42            # use a different seed

Results saved to: results/maturity_analysis/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from data_prep import DataManager
from models import build_model, ARCH_NAMES
from train import TrainConfig, train_model, history_summary
from evaluate import (
    extract_ultimate_actuals,
    extract_observed_cumulative,
    _compute_metrics,
    predict_paid_output,
)

DATA_DIR    = PROJECT_DIR / "data"
_BASE_RESULTS = Path(os.environ.get("DEEPTRIANGLE_RESULTS", str(PROJECT_DIR / "results")))
RESULTS_DIR = _BASE_RESULTS / "maturity_analysis"

TRI_PATH = str(DATA_DIR / "triangle_sample.csv")
CO_PATH  = str(DATA_DIR / "triangle_company_info.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase 1 fixed HPs (identical to run_phase1.py)
FIXED_HP = dict(
    gru_units=128,
    dropout_rate=0.10,
    dense_units=64,
    learning_rate=5e-4,
    batch_size=512,
    epochs=1000,
    es_patience=200,
    min_delta=0.001,
    lr_patience=50,
    lr_factor=0.5,
    min_lr=1e-6,
)

# Maturity group definitions (accident year ranges)
# "mature"   = AY 2002-2006 (test_lag 9 down to 5: 1-4 future lags)
# "immature" = AY 2007-2010 (test_lag 4 down to 1: 5-8 future lags)
MATURITY_GROUPS = {
    "mature":   (2002, 2006),  # test_lag in {9, 8, 7, 6, 5}
    "immature": (2007, 2010),  # test_lag in {4, 3, 2, 1}
}

# Test calendar year (env-var-aware for CAS mode)
TEST_CAL_YEAR = int(os.environ.get("DEEPTRIANGLE_TEST_CAL", 2011))
FULL_AY_RANGE = (
    int(os.environ.get("DEEPTRIANGLE_AY_MIN", 2002)),
    int(os.environ.get("DEEPTRIANGLE_TEST_MAX_AY", 2010)),
)


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-(company, accident_year) prediction extraction
# ---------------------------------------------------------------------------

def get_per_company_ay_predictions(
    model,
    test_data: Dict[str, Any],
    test_metadata: pd.DataFrame,
    raw_data: pd.DataFrame,
    lob: str,
    outlier_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Compute per-(company, accident_year) predicted and actual ultimates.

    This is the same logic as evaluate._evaluate_mode_a, but returns
    individual rows rather than aggregated MAPE so we can stratify by
    maturity group afterwards.

    Parameters
    ----------
    model         : trained Keras model
    test_data     : dict with key 'x'
    test_metadata : DataFrame from DataManager.get_test_metadata(lob)
    raw_data      : full prepared DataFrame (DataManager.data)
    lob           : line of business string
    outlier_threshold : remove |pct_error| > threshold for diagnostics

    Returns
    -------
    pd.DataFrame with columns:
        ['group_code', 'accident_year', 'test_lag',
         'actual_ultimate', 'predicted_ultimate', 'pct_error',
         'abs_pct_error', 'maturity_group']
    """
    # --- Forward pass ---
    paid_pred_norm = predict_paid_output(model, test_data["x"])
    paid_pred_norm = np.squeeze(paid_pred_norm, axis=-1)  # (N, 9)

    # --- De-normalize ---
    ep = test_metadata["earned_premium_net"].values
    paid_pred_dollars = paid_pred_norm * ep[:, None]      # (N, 9)

    # --- Actual ultimates and observed cumulatives ---
    actual_ult_df = extract_ultimate_actuals(raw_data, lob, FULL_AY_RANGE, max_dev_lag=9)
    obs_cum_df    = extract_observed_cumulative(raw_data, lob, accident_year_range=FULL_AY_RANGE)

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

    rows = []
    for i in range(len(test_metadata)):
        row = test_metadata.iloc[i]
        gc  = str(row["group_code"])
        ay  = int(row["accident_year"])
        dev_lag = int(row["development_lag"])   # test diagonal lag = TEST_CAL_YEAR - AY

        if not (FULL_AY_RANGE[0] <= ay <= FULL_AY_RANGE[1]):
            continue

        key = (gc, ay)
        if key not in actual_lookup:
            continue

        actual_ult = actual_lookup[key]
        obs_cum, last_obs_lag = obs_lookup.get(key, (0.0, 0))

        # Predicted ultimate = observed + sum of predicted future increments
        remaining_lags = 9 - last_obs_lag
        n_pred = min(remaining_lags, paid_pred_dollars.shape[1])
        predicted_ult = obs_cum + float(paid_pred_dollars[i, :n_pred].sum())

        # Assign maturity group based on accident year
        maturity_group = _get_maturity_group(ay)

        rows.append({
            "group_code":        gc,
            "accident_year":     ay,
            "test_lag":          dev_lag,   # = TEST_CAL_YEAR - AY
            "actual_ultimate":   actual_ult,
            "predicted_ultimate": predicted_ult,
            "maturity_group":    maturity_group,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "group_code", "accident_year", "test_lag",
            "actual_ultimate", "predicted_ultimate",
            "pct_error", "abs_pct_error", "maturity_group",
        ])

    df = pd.DataFrame(rows)

    # Average across duplicate (gc, ay) entries if any
    df = df.groupby(
        ["group_code", "accident_year", "test_lag", "maturity_group"]
    ).mean().reset_index()

    # Compute percentage errors
    EPSILON = 1e-8
    denom = np.where(
        np.abs(df["actual_ultimate"].values) < EPSILON,
        EPSILON,
        np.abs(df["actual_ultimate"].values),
    )
    df["pct_error"]     = (df["predicted_ultimate"].values - df["actual_ultimate"].values) / denom
    df["abs_pct_error"] = np.abs(df["pct_error"].values)

    # Flag outliers (keep them in output for diagnostics, flag them)
    df["is_outlier"] = df["abs_pct_error"] > outlier_threshold

    return df


def _get_maturity_group(ay: int) -> str:
    """Map accident year to 'mature', 'immature', or 'unknown'."""
    for group, (lo, hi) in MATURITY_GROUPS.items():
        if lo <= ay <= hi:
            return group
    return "unknown"


# ---------------------------------------------------------------------------
# Maturity-stratified MAPE computation
# ---------------------------------------------------------------------------

def compute_maturity_metrics(
    pred_df: pd.DataFrame,
    outlier_threshold: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compute MAPE and RMSPE for each maturity group and overall.

    Follows Kuo (2019) per-company averaging methodology:
      1. Within each (company, maturity_group): average |pct_error|
      2. Average across companies within group

    Parameters
    ----------
    pred_df           : output of get_per_company_ay_predictions
    outlier_threshold : remove |pct_error| > threshold before computing metrics

    Returns
    -------
    dict  keyed by 'overall', 'mature', 'immature' (and any other groups present)
    Each value: {'mape': ..., 'rmspe': ..., 'n_companies': ..., 'n_ay': ...,
                 'n_outliers': ...}
    """
    if pred_df.empty:
        return {}

    results = {}
    groups_to_eval = ["overall"] + list(MATURITY_GROUPS.keys())

    for group in groups_to_eval:
        if group == "overall":
            subset = pred_df[~pred_df["is_outlier"]].copy()
        else:
            subset = pred_df[
                (pred_df["maturity_group"] == group) & (~pred_df["is_outlier"])
            ].copy()

        n_outliers = int(pred_df[
            pred_df["maturity_group"] == group
            if group != "overall"
            else pd.Series([True] * len(pred_df))
        ]["is_outlier"].sum()) if group != "overall" else int(pred_df["is_outlier"].sum())

        if subset.empty:
            results[group] = {
                "mape": float("nan"), "rmspe": float("nan"),
                "n_companies": 0, "n_ay": 0, "n_outliers": n_outliers,
            }
            continue

        # Per-company aggregation (Kuo 2019 methodology)
        company_metrics = subset.groupby("group_code").agg(
            abs_pct_error=("abs_pct_error", "mean"),
            sq_pct_error=("pct_error", lambda x: np.mean(x ** 2)),
            n_ay=("accident_year", "count"),
        ).reset_index()

        results[group] = {
            "mape":         float(company_metrics["abs_pct_error"].mean()),
            "rmspe":        float(np.sqrt(company_metrics["sq_pct_error"].mean())),
            "n_companies":  len(company_metrics),
            "n_ay":         int(subset["accident_year"].nunique()),
            "n_outliers":   n_outliers,
        }

    return results


# ---------------------------------------------------------------------------
# Mack CL maturity analysis
# ---------------------------------------------------------------------------

def run_mack_maturity_analysis(lob: str) -> Dict[str, Dict[str, float]]:
    """
    Run Mack CL and compute maturity-stratified MAPE for comparison.

    Uses the same LOB and accident year ranges as the DT models.

    Parameters
    ----------
    lob : str  line of business

    Returns
    -------
    dict with same structure as compute_maturity_metrics output
    """
    try:
        import chainladder as cl
    except ImportError:
        print("  chainladder-python not installed; skipping Mack maturity analysis.")
        return {}

    raw = pd.read_csv(TRI_PATH)
    co  = pd.read_csv(CO_PATH)
    raw = raw.merge(co[["group_code"]], on="group_code", how="left")
    raw = raw.sort_values(["lob", "group_code", "accident_year", "development_lag"])

    lob_data   = raw[raw["lob"] == lob].copy()
    train_data = lob_data[lob_data["calendar_year"] <= 2010].copy()
    train_data["development_year"] = train_data["accident_year"] + train_data["development_lag"]

    EPSILON = 1e-8
    OUTLIER_THRESHOLD = 10.0

    # Actual ultimates at lag 9
    actual_ult = (
        lob_data[
            (lob_data["development_lag"] == 9)
            & (lob_data["accident_year"] >= FULL_AY_RANGE[0])
            & (lob_data["accident_year"] <= FULL_AY_RANGE[1])
        ][["group_code", "accident_year", "cumulative_paid_loss"]]
        .rename(columns={"cumulative_paid_loss": "actual_ult"})
        .copy()
    )

    try:
        tri_obj = cl.Triangle(
            data=train_data,
            origin="accident_year",
            development="development_year",
            columns=["cumulative_paid_loss"],
            index=["group_code"],
            cumulative=True,
        )
        dev      = cl.Development().fit_transform(tri_obj)
        cl_model = cl.Chainladder().fit(dev)
        cl_ult   = cl_model.ultimate_

        ult_frame  = cl_ult.to_frame().reset_index()
        period_cols = [c for c in ult_frame.columns if isinstance(c, pd.Period)]
        if not period_cols:
            period_cols = [c for c in ult_frame.columns if c != "group_code"]

        melted = ult_frame.melt(
            id_vars=["group_code"],
            value_vars=period_cols,
            var_name="accident_year",
            value_name="pred_ult",
        )
        melted["accident_year"] = melted["accident_year"].apply(
            lambda p: p.year if hasattr(p, "year") else int(p)
        )
        melted = melted[melted["pred_ult"].notna() & (melted["pred_ult"] > 0)]
        melted = melted[
            (melted["accident_year"] >= FULL_AY_RANGE[0])
            & (melted["accident_year"] <= FULL_AY_RANGE[1])
        ]
        melted["group_code"]    = melted["group_code"].astype(str)
        actual_ult["group_code"] = actual_ult["group_code"].astype(str)

        merged = melted.merge(actual_ult, on=["group_code", "accident_year"], how="inner")
        merged = merged.dropna(subset=["pred_ult", "actual_ult"])
        if merged.empty:
            return {}

        # Compute pct errors
        denom = np.where(
            np.abs(merged["actual_ult"].values) < EPSILON,
            EPSILON,
            np.abs(merged["actual_ult"].values),
        )
        merged = merged.copy()
        merged["pct_error"]     = (merged["pred_ult"].values - merged["actual_ult"].values) / denom
        merged["abs_pct_error"] = np.abs(merged["pct_error"].values)
        merged["test_lag"]      = TEST_CAL_YEAR - merged["accident_year"]
        merged["maturity_group"] = merged["accident_year"].apply(_get_maturity_group)
        merged["is_outlier"]    = merged["abs_pct_error"] > OUTLIER_THRESHOLD

        results = {}
        for group in ["overall"] + list(MATURITY_GROUPS.keys()):
            if group == "overall":
                subset = merged[~merged["is_outlier"]]
            else:
                subset = merged[
                    (merged["maturity_group"] == group) & (~merged["is_outlier"])
                ]
            if subset.empty:
                results[group] = {
                    "mape": float("nan"), "rmspe": float("nan"),
                    "n_companies": 0, "n_ay": 0,
                }
                continue

            company_metrics = subset.groupby("group_code").agg(
                abs_pct_error=("abs_pct_error", "mean"),
                sq_pct_error=("pct_error", lambda x: np.mean(x ** 2)),
            ).reset_index()

            results[group] = {
                "mape":        float(company_metrics["abs_pct_error"].mean()),
                "rmspe":       float(np.sqrt(company_metrics["sq_pct_error"].mean())),
                "n_companies": len(company_metrics),
                "n_ay":        int(subset["accident_year"].nunique()),
            }

        return results

    except Exception as exc:
        print(f"  Mack maturity analysis failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Per-lag MAPE breakdown
# ---------------------------------------------------------------------------

def compute_per_lag_mape(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAPE broken down by test_lag (1 through 9).

    This shows exactly how prediction accuracy varies with how many lags
    the model must forecast from the test diagonal.

    Parameters
    ----------
    pred_df : output of get_per_company_ay_predictions

    Returns
    -------
    pd.DataFrame with columns ['test_lag', 'accident_year', 'mape', 'n_companies']
    sorted by test_lag descending (lag=9 = most mature first).
    """
    if pred_df.empty:
        return pd.DataFrame()

    rows = []
    for lag in sorted(pred_df["test_lag"].unique(), reverse=True):
        subset = pred_df[
            (pred_df["test_lag"] == lag) & (~pred_df["is_outlier"])
        ]
        if subset.empty:
            continue

        # Per-company MAPE for this lag
        company_metrics = subset.groupby("group_code").agg(
            abs_pct_error=("abs_pct_error", "mean"),
        ).reset_index()

        ay_for_lag = int(TEST_CAL_YEAR - lag)
        rows.append({
            "test_lag":      lag,
            "accident_year": ay_for_lag,
            "n_companies":   len(company_metrics),
            "mape":          float(company_metrics["abs_pct_error"].mean()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def print_cached_maturity_analysis(lob: str, seed: int) -> bool:
    """Print shipped maturity-analysis summaries without requiring raw data."""
    summary_path = RESULTS_DIR / f"maturity_summary_{lob}_seed{seed:02d}.csv"
    if not summary_path.exists():
        return False

    summary_df = pd.read_csv(summary_path)
    all_results: Dict[str, Any] = {}
    mack_maturity: Dict[str, Any] = {}

    for arch, arch_df in summary_df.groupby("arch"):
        metrics: Dict[str, Any] = {}
        for _, row in arch_df.iterrows():
            metrics[str(row["group"])] = {
                "mape": float(row["mape"]),
                "rmspe": float(row["rmspe"]),
                "n_companies": int(row["n_companies"]),
                "n_ay": int(row["n_ay"]),
                "n_outliers": int(row.get("n_outliers", 0)),
            }
        if arch == "mack_cl":
            mack_maturity = metrics
        else:
            all_results[str(arch)] = {"maturity_metrics": metrics}

    print("[Maturity Analysis] Raw data files are not present; using shipped pre-computed results.")
    print_maturity_table(all_results, mack_maturity, lob, seed)

    lag_path = RESULTS_DIR / f"per_lag_mape_{lob}_seed{seed:02d}.csv"
    if lag_path.exists() and all_results:
        lag_df = pd.read_csv(lag_path)
        _print_per_lag_table(lag_df, list(all_results.keys()), mack_maturity)

    print(f"\n[Maturity Analysis] Cached summary loaded from {summary_path}")
    return True


def run_maturity_analysis(
    archs: List[str],
    lob: str,
    seed: int,
    run_mack: bool = True,
    resume: bool = True,
) -> None:
    """
    Retrain one model per architecture, compute maturity-stratified MAPE.

    Parameters
    ----------
    archs    : list of architecture names
    lob      : line of business (default 'workers_compensation')
    seed     : random seed for training (default 0)
    run_mack : whether to also run Mack CL for comparison
    resume   : if True and result JSON exists, skip retraining
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Maturity Analysis]  LOB={lob}  seed={seed}  archs={archs}")

    # Public verification mode: raw Schedule P CSVs are not redistributed.
    # If they are absent, print the shipped pre-computed maturity analysis
    # instead of retraining models from raw data.
    if not (Path(TRI_PATH).exists() and Path(CO_PATH).exists()):
        if print_cached_maturity_analysis(lob, seed):
            return
        missing = [str(p) for p in [Path(TRI_PATH), Path(CO_PATH)] if not p.exists()]
        raise FileNotFoundError(
            "Maturity analysis requires raw triangle data or shipped cached results. "
            f"Missing: {', '.join(missing)}"
        )

    # --- Load data ---
    print("[Maturity Analysis] Loading dataset ...")
    dm = DataManager(TRI_PATH, CO_PATH)
    dm.load()
    dm.prepare()
    print(f"  vocab_size={dm.vocab_size}, LOBs={dm.available_lobs()}")

    if lob not in dm.available_lobs():
        print(f"  ERROR: LOB '{lob}' not available. Available: {dm.available_lobs()}")
        return

    # --- Training config (Phase 1 HPs) ---
    config = TrainConfig(
        learning_rate=FIXED_HP["learning_rate"],
        batch_size=FIXED_HP["batch_size"],
        epochs=FIXED_HP["epochs"],
        es_patience=FIXED_HP["es_patience"],
        min_delta=FIXED_HP["min_delta"],
        lr_patience=FIXED_HP["lr_patience"],
        lr_factor=FIXED_HP["lr_factor"],
        min_lr=FIXED_HP["min_lr"],
        verbose=0,
    )

    # --- Mack CL maturity analysis (run once, not per-arch) ---
    mack_maturity = {}
    mack_lag_df   = pd.DataFrame()
    if run_mack:
        mack_result_path = RESULTS_DIR / f"mack_{lob}_maturity.json"
        if resume and mack_result_path.exists():
            with open(mack_result_path) as f:
                mack_maturity = json.load(f)
            print(f"  [Mack] Loaded cached maturity results.")
        else:
            print(f"\n[Mack] Computing maturity-stratified MAPE for {lob} ...")
            t0 = time.time()
            mack_maturity = run_mack_maturity_analysis(lob)
            with open(mack_result_path, "w") as f:
                json.dump(mack_maturity, f, indent=2)
            print(f"  [Mack] Done in {time.time()-t0:.1f}s")
            if mack_maturity:
                for grp, m in mack_maturity.items():
                    print(
                        f"    {grp:12s}  MAPE={m['mape']:.4f}  "
                        f"N_companies={m['n_companies']}  N_AY={m['n_ay']}"
                    )

    # --- Per-architecture analysis ---
    all_results = {}  # keyed by arch

    for arch in archs:
        print(f"\n[{arch}] Training seed={seed} on {lob} ...")

        # Resume check
        result_json_path = RESULTS_DIR / f"{arch}_{lob}_seed{seed:02d}.json"
        pred_csv_path = RESULTS_DIR / f"{arch}_{lob}_seed{seed:02d}_predictions.csv"

        if resume and result_json_path.exists():
            print(f"  Loading cached results from {result_json_path.name}")
            with open(result_json_path) as f:
                arch_results = json.load(f)
            if pred_csv_path.exists():
                pred_df = pd.read_csv(str(pred_csv_path))
                print(f"  Loaded {len(pred_df)} predictions from cache.")
            else:
                pred_df = pd.DataFrame()
        else:
            # --- Train ---
            set_seeds(seed)

            train_data = dm.get(lob, "full_training_data")
            val_data   = dm.get(lob, "validation_data")
            test_data  = dm.get(lob, "test_data")
            test_meta  = dm.get_test_metadata(lob)

            model = build_model(
                arch,
                vocab_size=dm.vocab_size,
                gru_units=FIXED_HP["gru_units"],
                dropout_rate=FIXED_HP["dropout_rate"],
                dense_units=FIXED_HP["dense_units"],
            )

            t0 = time.time()
            history, t_sec = train_model(model, train_data, val_data, config)
            summ = history_summary(history)
            print(
                f"  Training: epochs={summ['epochs_trained']}  "
                f"best_val={summ['best_val_loss']:.5f}  "
                f"time={t_sec:.0f}s"
            )

            # --- Get per-(company, AY) predictions ---
            print("  Computing per-(company, AY) predictions ...")
            pred_df = get_per_company_ay_predictions(
                model, test_data, test_meta, dm.data, lob
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Compute maturity metrics ---
            maturity_metrics = compute_maturity_metrics(pred_df)

            # --- Per-lag breakdown ---
            lag_df = compute_per_lag_mape(pred_df)

            # --- Assemble result dict ---
            arch_results = {
                "arch":            arch,
                "lob":             lob,
                "seed":            seed,
                "epochs_trained":  summ["epochs_trained"],
                "best_val_loss":   summ["best_val_loss"],
                "training_time":   round(t_sec, 2),
                "maturity_metrics": maturity_metrics,
                "per_lag_mape":    lag_df.to_dict(orient="records") if not lag_df.empty else [],
            }

            # Save
            with open(result_json_path, "w") as f:
                json.dump(arch_results, f, indent=2)
            if not pred_df.empty:
                pred_df.to_csv(str(pred_csv_path), index=False)
                print(f"  Saved predictions to {pred_csv_path.name}")

            print(f"  Saved results to {result_json_path.name}")

        all_results[arch] = arch_results

        # Print maturity breakdown
        print(f"\n  [{arch}] Maturity breakdown (LOB={lob}, seed={seed}):")
        maturity_data = arch_results.get("maturity_metrics", {})
        for grp in ["overall", "mature", "immature"]:
            m = maturity_data.get(grp, {})
            mape = m.get("mape", float("nan"))
            rmspe = m.get("rmspe", float("nan"))
            n_co  = m.get("n_companies", 0)
            n_ay  = m.get("n_ay", 0)
            mape_str  = f"{mape:.4f}" if np.isfinite(mape) else "nan"
            rmspe_str = f"{rmspe:.4f}" if np.isfinite(rmspe) else "nan"
            print(
                f"    {grp:12s}  MAPE={mape_str}  RMSPE={rmspe_str}  "
                f"N_companies={n_co}  N_AY={n_ay}"
            )

    # --- Print consolidated comparison table ---
    print_maturity_table(all_results, mack_maturity, lob, seed)

    # --- Save summary CSV ---
    summary_rows = []
    for arch, arch_res in all_results.items():
        mm = arch_res.get("maturity_metrics", {})
        for grp in ["overall", "mature", "immature"]:
            m = mm.get(grp, {})
            summary_rows.append({
                "arch":        arch,
                "lob":         lob,
                "seed":        seed,
                "group":       grp,
                "mape":        m.get("mape", float("nan")),
                "rmspe":       m.get("rmspe", float("nan")),
                "n_companies": m.get("n_companies", 0),
                "n_ay":        m.get("n_ay", 0),
                "n_outliers":  m.get("n_outliers", 0),
            })

    # Add Mack results
    for grp, m in mack_maturity.items():
        summary_rows.append({
            "arch":        "mack_cl",
            "lob":         lob,
            "seed":        -1,
            "group":       grp,
            "mape":        m.get("mape", float("nan")),
            "rmspe":       m.get("rmspe", float("nan")),
            "n_companies": m.get("n_companies", 0),
            "n_ay":        m.get("n_ay", 0),
            "n_outliers":  0,
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / f"maturity_summary_{lob}_seed{seed:02d}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[Maturity Analysis] Summary saved to {csv_path}")

    # --- Save per-lag breakdown ---
    lag_rows = []
    for arch, arch_res in all_results.items():
        for row in arch_res.get("per_lag_mape", []):
            lag_rows.append({"arch": arch, "lob": lob, "seed": seed, **row})
    if lag_rows:
        lag_df = pd.DataFrame(lag_rows)
        lag_csv = RESULTS_DIR / f"per_lag_mape_{lob}_seed{seed:02d}.csv"
        lag_df.to_csv(lag_csv, index=False)
        print(f"[Maturity Analysis] Per-lag MAPE saved to {lag_csv}")
        _print_per_lag_table(lag_df, archs, mack_maturity)


def print_maturity_table(
    all_results: Dict[str, Any],
    mack_maturity: Dict[str, Any],
    lob: str,
    seed: int,
) -> None:
    """Print a formatted comparison table of MAPE by architecture and maturity group."""
    archs = list(all_results.keys())

    print("\n" + "=" * 70)
    print(f"Maturity Analysis — {lob}  (seed={seed})")
    print(f"Mature   = AY {MATURITY_GROUPS['mature'][0]}-{MATURITY_GROUPS['mature'][1]}   (test_lag 5-9, 1-4 future lags)")
    print(f"Immature = AY {MATURITY_GROUPS['immature'][0]}-{MATURITY_GROUPS['immature'][1]}  (test_lag 1-4, 5-8 future lags)")
    print("=" * 70)

    header = f"{'Method':<25}  {'Overall':>8}  {'Mature':>8}  {'Immature':>8}  {'Ratio':>8}"
    print(header)
    print("-" * 70)

    # DT architectures
    for arch in archs:
        mm = all_results[arch].get("maturity_metrics", {})
        overall  = mm.get("overall",  {}).get("mape", float("nan"))
        mature   = mm.get("mature",   {}).get("mape", float("nan"))
        immature = mm.get("immature", {}).get("mape", float("nan"))
        ratio    = immature / mature if (np.isfinite(mature) and np.isfinite(immature) and mature > 0) else float("nan")

        def _fmt(x: float) -> str:
            return f"{x:.4f}" if np.isfinite(x) else "  nan "

        print(
            f"{arch:<25}  {_fmt(overall):>8}  {_fmt(mature):>8}  {_fmt(immature):>8}  "
            f"{'x'+f'{ratio:.2f}':>8}"
        )

    # Mack CL
    if mack_maturity:
        print("-" * 70)
        overall  = mack_maturity.get("overall",  {}).get("mape", float("nan"))
        mature   = mack_maturity.get("mature",   {}).get("mape", float("nan"))
        immature = mack_maturity.get("immature", {}).get("mape", float("nan"))
        ratio    = immature / mature if (np.isfinite(mature) and np.isfinite(immature) and mature > 0) else float("nan")
        print(
            f"{'mack_cl':<25}  {overall:.4f}  {mature:.4f}  {immature:.4f}  "
            f"{'x'+f'{ratio:.2f}':>8}"
        )

    print("=" * 70)
    print("Ratio = immature_MAPE / mature_MAPE  (>1 means harder for immature AYs)")


def _print_per_lag_table(
    lag_df: pd.DataFrame,
    archs: List[str],
    mack_maturity: Dict[str, Any],
) -> None:
    """Print MAPE by individual test lag for each architecture."""
    print("\n--- Per-lag MAPE (test_lag = TEST_CAL_YEAR - accident_year) ---")
    print(f"  test_lag=9 means AY=2002 (most mature); test_lag=1 means AY=2010 (least mature)")

    all_lags = sorted(lag_df["test_lag"].unique(), reverse=True)
    arch_col_width = max(len(a) for a in archs) + 2 if archs else 20

    header_parts = [f"{'test_lag':>9}", f"{'accident_year':>14}"]
    for arch in archs:
        header_parts.append(f"{arch:>{arch_col_width}}")
    print("  " + "  ".join(header_parts))

    for lag in all_lags:
        ay = TEST_CAL_YEAR - lag
        row_parts = [f"{lag:>9}", f"{ay:>14}"]
        for arch in archs:
            sub = lag_df[(lag_df["arch"] == arch) & (lag_df["test_lag"] == lag)]
            if sub.empty:
                row_parts.append(f"{'—':>{arch_col_width}}")
            else:
                mape_val = sub["mape"].iloc[0]
                row_parts.append(f"{mape_val:>{arch_col_width}.4f}")
        print("  " + "  ".join(row_parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Maturity analysis: MAPE by accident year maturity group"
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=list(ARCH_NAMES),
        choices=list(ARCH_NAMES),
        help="Architectures to run (default: all 3)",
    )
    parser.add_argument(
        "--lob",
        default="workers_compensation",
        choices=["workers_compensation", "private_passenger_auto"],
        help="Line of business (default: workers_compensation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for model training (default: 0)",
    )
    parser.add_argument(
        "--no-mack",
        action="store_true",
        help="Skip Mack CL maturity analysis",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force re-run even if result files exist",
    )
    args = parser.parse_args()

    run_maturity_analysis(
        archs=args.archs,
        lob=args.lob,
        seed=args.seed,
        run_mack=not args.no_mack,
        resume=not args.no_resume,
    )
