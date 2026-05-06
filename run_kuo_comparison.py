"""
Kuo (2019) Comparison Experiments — DeepTriangle v2.

Four experiments decomposing the factors that differ between our Phase 1
setup and Kuo (2019), to validate the relative improvement pattern:

  E0 (baseline)  : Phase 1 gru_baseline on ALL WC companies, batch=512,
                   20-seed mean/std.  Loaded from existing Phase 1 JSON files.

  E1 (subsample) : Randomly subsample 50 WC companies (seed=2024), matching
                   Kuo's company count.  Run GRU baseline with Phase 1 HPs
                   (batch=512), 20 seeds (0-19).

  E2 (ensemble)  : Using the 20 trained models from E1, average raw paid
                   predictions before computing MAPE.  Approximates Kuo's
                   100-model ensemble with 20 models.

  E3 (Kuo batch) : Same 50-company subsample as E1, but batch_size=2250
                   (Kuo's exact value).  20 seeds, then ensemble-average.

Mack CL is computed on the 50-company subset for comparison.

Key design decisions
--------------------
- vocab_size for embeddings = dm.vocab_size (ALL companies = 434).
  The LabelEncoder is fit on the full dataset; subsampling only changes
  which rows are fed to the model, not the embedding table size.  This is
  necessary because group_code indices must be consistent with the
  pre-fitted encoder.
- Ensemble averaging: collect raw paid_pred_norm arrays from each model,
  average them, THEN de-normalize and compute MAPE.  This is the correct
  way to ensemble — averaging in the normalized prediction space before
  converting to dollar ultimates.
- The 50-company subsample is applied at the DataManager level: after
  dm.load(), we filter dm.data to only the 50 companies BEFORE calling
  dm.prepare().  This ensures the train/val/test splits are all restricted
  to the subsample.

Published results for reference (Kuo 2019, Table 2)
----------------------------------------------------
  Workers' Compensation  : DT=0.046, Mack=0.053, AutoML=0.067, ODP=0.105
  Data: 1988-1997, 50 companies/LOB, development lags 1-10
  Our data covers different years/companies, so exact replication is not
  expected; the goal is to validate the relative improvement pattern
  and understand the directional impact of each factor.

Usage
-----
    python run_kuo_comparison.py                # all experiments
    python run_kuo_comparison.py --exp e1 e2    # specific experiments
    python run_kuo_comparison.py --seeds 0 1 2  # specific seeds
    python run_kuo_comparison.py --resume       # skip completed runs

Results saved to: results/kuo_comparison/
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
# Path setup — same pattern as run_phase1.py
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from data_prep import DataManager, load_and_prepare_data, create_group_code_encoder, prepare_all_data
from models import build_model
from train import TrainConfig, train_model, history_summary
from evaluate import (
    compute_mape_rmspe,
    extract_ultimate_actuals,
    extract_observed_cumulative,
    _compute_metrics,
    predict_paid_output,
)

DATA_DIR    = PROJECT_DIR / "data"
_BASE_RESULTS = Path(os.environ.get("DEEPTRIANGLE_RESULTS", str(PROJECT_DIR / "results")))
RESULTS_DIR = _BASE_RESULTS / "kuo_comparison"
PHASE1_DIR  = _BASE_RESULTS / "phase1"

TRI_PATH = str(DATA_DIR / "triangle_sample.csv")
CO_PATH  = str(DATA_DIR / "triangle_company_info.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOB = "workers_compensation"
SUBSAMPLE_SEED = 2024          # reproducible 50-company draw
N_COMPANIES    = 50            # Kuo (2019) company count
N_SEEDS        = 20            # seeds 0-19
ARCH           = "gru_baseline"

# Phase 1 fixed HPs
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

# Kuo's original batch size
KUO_BATCH_SIZE = 2250

# Published Kuo (2019) results
KUO_PUBLISHED = {
    "workers_compensation": {"mape": 0.046, "mack": 0.053, "automl": 0.067, "odp": 0.105},
}


# ---------------------------------------------------------------------------
# Seed control (identical to run_phase1.py)
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 50-company subsample DataManager
# ---------------------------------------------------------------------------

def build_subsampled_dm(companies_50: List) -> DataManager:
    """
    Build a DataManager whose internal `data` is filtered to the 50
    specified companies BEFORE calling prepare().

    The LabelEncoder is fit ONLY on the subsampled data's group_codes,
    but the vocab_size reflects the full dataset (434) because the
    LabelEncoder is replaced post-hoc with the one from the full DM.

    IMPORTANT: We do NOT use this approach.  Instead, we load the full DM
    (to get the correct encoder), then filter `dm.data` before calling
    `prepare_all_data()`.  This preserves the original integer codes for
    every group_code so that embedding lookup indices are consistent.

    This function returns a DataManager-like object with all the same
    attributes but restricted to the 50-company subset.

    Parameters
    ----------
    companies_50 : list
        List of group_code values (strings or ints) for the 50 companies.

    Returns
    -------
    DataManager with .data filtered; .encoder and .splits set from full data.
    """
    # 1. Load full dataset
    dm_full = DataManager(TRI_PATH, CO_PATH)
    dm_full.load()

    # 2. Fit encoder on FULL data (so integer codes for all 434 companies are stable)
    dm_full.encoder = create_group_code_encoder(dm_full.data)
    full_vocab_size = len(dm_full.encoder.classes_)

    # 3. Filter dm.data to the 50-company subset
    companies_50_str = [str(c) for c in companies_50]
    mask = dm_full.data["group_code"].astype(str).isin(companies_50_str)
    dm_full.data = dm_full.data[mask].copy().reset_index(drop=True)

    # 4. Build splits using the FULL encoder (preserves original integer codes)
    #    and the active test calendar year (CAS mode sets this via env vars).
    dm_full.splits = prepare_all_data(
        dm_full.data,
        dm_full.encoder,
        test_calendar_year=dm_full.test_min_calendar_year,
    )

    print(
        f"  Subsampled DataManager: "
        f"{len(dm_full.data)} rows, "
        f"{len(companies_50)} companies, "
        f"vocab_size={dm_full.vocab_size} (uses full-data encoder)"
    )
    return dm_full


def select_50_companies(dm_full: DataManager, seed: int = SUBSAMPLE_SEED) -> List:
    """
    Randomly select N_COMPANIES companies from the WC LOB.

    Uses the full DataManager's data (already filtered to WC when used here).
    The same companies are used across all seeds of E1/E2/E3.

    Parameters
    ----------
    dm_full : DataManager  (already loaded, NOT yet filtered)
    seed    : int          RNG seed for the company draw (default 2024)

    Returns
    -------
    List of N_COMPANIES group_code values (sorted for reproducibility).
    """
    wc_data = dm_full.data[dm_full.data["lob"] == LOB]
    all_companies = sorted(wc_data["group_code"].unique().tolist())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_companies, size=min(N_COMPANIES, len(all_companies)), replace=False)
    return sorted(chosen.tolist())


# ---------------------------------------------------------------------------
# Raw prediction extraction (for ensemble averaging)
# ---------------------------------------------------------------------------

def get_raw_paid_predictions(
    model,
    test_data: Dict[str, Any],
) -> np.ndarray:
    """
    Run a forward pass and return the raw normalized paid predictions.

    Parameters
    ----------
    model     : trained PyTorch model
    test_data : dict with key 'x'

    Returns
    -------
    np.ndarray of shape (N, 9) — normalized incremental paid loss ratios.
    """
    paid_pred_norm = predict_paid_output(model, test_data["x"])
    return np.squeeze(paid_pred_norm, axis=-1)


def compute_mape_from_raw_preds(
    paid_pred_norm: np.ndarray,
    test_metadata: pd.DataFrame,
    raw_data: pd.DataFrame,
    lob: str,
    accident_year_range: Tuple[int, int] = (2002, 2010),
    outlier_threshold: float = 10.0,
) -> Dict[str, float]:
    """
    Compute MAPE from a normalized paid prediction array.

    This is the same logic as evaluate._evaluate_mode_a, factored out so
    we can use it with ensemble-averaged predictions.

    Parameters
    ----------
    paid_pred_norm : (N, 9) normalized incremental paid predictions
    test_metadata  : DataFrame from DataManager.get_test_metadata(lob)
    raw_data       : full prepared DataFrame (DataManager.data)
    lob            : line of business string
    accident_year_range : (min_ay, max_ay) inclusive
    outlier_threshold   : remove |pct_error| > threshold

    Returns
    -------
    dict with keys 'mape', 'rmspe', 'n_companies', 'n_filtered'
    """
    ep = test_metadata["earned_premium_net"].values       # (N,)
    paid_pred_dollars = paid_pred_norm * ep[:, None]      # (N, 9)

    actual_ult_df = extract_ultimate_actuals(
        raw_data, lob, accident_year_range, max_dev_lag=9
    )
    obs_cum_df = extract_observed_cumulative(
        raw_data, lob, accident_year_range=accident_year_range
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
        gc  = str(row["group_code"])
        ay  = int(row["accident_year"])
        if not (accident_year_range[0] <= ay <= accident_year_range[1]):
            continue
        key = (gc, ay)
        if key not in actual_lookup:
            continue

        actual_ult = actual_lookup[key]
        obs_cum, last_obs_lag = obs_lookup.get(key, (0.0, 0))
        remaining_lags = 9 - last_obs_lag
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
    metrics = _compute_metrics(df, outlier_threshold)
    # _compute_metrics returns 'n_companies' implicitly through len(company_metrics)
    # but the key is 'mape', 'rmspe', 'n_companies', 'n_filtered'
    return metrics


# ---------------------------------------------------------------------------
# Mack CL on a filtered company subset
# ---------------------------------------------------------------------------

def run_mack_on_subset(companies_50: List) -> Dict[str, float]:
    """
    Run Mack Chain-Ladder (chainladder-python) on the 50-company WC subset.

    Uses calendar_year <= 2010 as training data (no test-year leakage),
    then evaluates against actual ultimates at development lag 9.

    Parameters
    ----------
    companies_50 : list of group_code values

    Returns
    -------
    dict with 'mape', 'rmspe', 'n'
    """
    try:
        import chainladder as cl
    except ImportError:
        print("  chainladder-python not installed; skipping Mack for subset.")
        return {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

    raw = pd.read_csv(TRI_PATH)
    co  = pd.read_csv(CO_PATH)
    raw = raw.merge(co[["group_code"]], on="group_code", how="left")
    raw = raw.sort_values(["lob", "group_code", "accident_year", "development_lag"])

    companies_50_str = [str(c) for c in companies_50]
    lob_data = raw[
        (raw["lob"] == LOB)
        & raw["group_code"].astype(str).isin(companies_50_str)
    ].copy()

    _test_cal = int(os.environ.get("DEEPTRIANGLE_TEST_CAL", 2011))
    ACCIDENT_YEAR_RANGE = (
        int(os.environ.get("DEEPTRIANGLE_AY_MIN", 2002)),
        int(os.environ.get("DEEPTRIANGLE_TEST_MAX_AY", 2010)),
    )
    EPSILON = 1e-8
    OUTLIER_THRESHOLD = 10.0

    # Training data: calendar_year <= test_cal - 1
    train_data = lob_data[lob_data["calendar_year"] <= _test_cal - 1].copy()
    # Add development_year for chainladder (requires date-like development index)
    train_data = train_data.copy()
    train_data["development_year"] = train_data["accident_year"] + train_data["development_lag"]

    # Actual ultimates at lag 9
    actual_ult = (
        lob_data[
            (lob_data["development_lag"] == 9)
            & (lob_data["accident_year"] >= ACCIDENT_YEAR_RANGE[0])
            & (lob_data["accident_year"] <= ACCIDENT_YEAR_RANGE[1])
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
        dev   = cl.Development().fit_transform(tri_obj)
        cl_model = cl.Chainladder().fit(dev)
        cl_ult = cl_model.ultimate_

        # Extract predictions to DataFrame
        ult_frame = cl_ult.to_frame().reset_index()
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
            (melted["accident_year"] >= ACCIDENT_YEAR_RANGE[0])
            & (melted["accident_year"] <= ACCIDENT_YEAR_RANGE[1])
        ]
        melted["group_code"] = melted["group_code"].astype(str)
        actual_ult["group_code"] = actual_ult["group_code"].astype(str)

        merged = melted.merge(actual_ult, on=["group_code", "accident_year"], how="inner")
        merged = merged.dropna(subset=["pred_ult", "actual_ult"])

        if merged.empty:
            return {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

        denom = np.where(
            np.abs(merged["actual_ult"].values) < EPSILON,
            EPSILON,
            np.abs(merged["actual_ult"].values),
        )
        merged = merged.copy()
        merged["pct_error"] = (merged["pred_ult"].values - merged["actual_ult"].values) / denom
        merged = merged[np.abs(merged["pct_error"]) <= OUTLIER_THRESHOLD]

        company_metrics = merged.groupby("group_code").agg(
            abs_pct_error=("pct_error", lambda x: np.mean(np.abs(x))),
            sq_pct_error=("pct_error", lambda x: np.mean(x ** 2)),
        ).reset_index()

        return {
            "mape":  float(company_metrics["abs_pct_error"].mean()),
            "rmspe": float(np.sqrt(company_metrics["sq_pct_error"].mean())),
            "n":     len(company_metrics),
        }

    except Exception as exc:
        print(f"  Mack on subset failed: {exc}")
        return {"mape": float("nan"), "rmspe": float("nan"), "n": 0}


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def result_path(exp: str, seed: int) -> Path:
    return RESULTS_DIR / exp / f"run_{seed:03d}.json"


def save_result(exp: str, seed: int, data: Dict[str, Any]) -> None:
    p = result_path(exp, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def load_result(exp: str, seed: int) -> Optional[Dict[str, Any]]:
    p = result_path(exp, seed)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# E0: Load Phase 1 baseline results for WC gru_baseline
# ---------------------------------------------------------------------------

def load_e0_results(n_seeds: int = N_SEEDS) -> List[Dict[str, Any]]:
    """
    Load Phase 1 gru_baseline WC results for seeds 0..n_seeds-1.

    Returns list of dicts with 'seed', 'mape', 'rmspe'.
    Missing seeds are silently omitted.
    """
    results = []
    for seed in range(n_seeds):
        path = PHASE1_DIR / ARCH / LOB / f"run_{seed:03d}.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            results.append({"seed": seed, "mape": r["mape"], "rmspe": r["rmspe"]})
        else:
            print(f"  [E0] Phase 1 result missing: seed={seed} — run run_phase1.py first")
    return results


# ---------------------------------------------------------------------------
# Single-seed training + evaluation helper
# ---------------------------------------------------------------------------

def train_and_eval_seed(
    seed: int,
    dm: DataManager,
    config: TrainConfig,
    collect_raw_preds: bool = False,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Train one GRU baseline model on the given DataManager and evaluate.

    Parameters
    ----------
    seed              : int  random seed
    dm                : DataManager (already loaded+prepared; possibly subsampled)
    config            : TrainConfig
    collect_raw_preds : bool  if True, also return the raw (N,9) paid_pred_norm array

    Returns
    -------
    result : dict  with 'seed', 'mape', 'rmspe', 'n_companies', 'training_time',
                   'epochs_trained', 'best_val_loss'
    raw_preds : np.ndarray (N,9) or None
    """
    set_seeds(seed)

    train_data = dm.get(LOB, "full_training_data")
    val_data   = dm.get(LOB, "validation_data")
    test_data  = dm.get(LOB, "test_data")
    test_meta  = dm.get_test_metadata(LOB)

    # Build model — vocab_size uses FULL encoder size (434) for consistent embeddings
    model = build_model(
        ARCH,
        vocab_size=dm.vocab_size,
        gru_units=FIXED_HP["gru_units"],
        dropout_rate=FIXED_HP["dropout_rate"],
        dense_units=FIXED_HP["dense_units"],
    )

    history, t_sec = train_model(model, train_data, val_data, config)
    summ = history_summary(history)

    metrics = compute_mape_rmspe(
        model, test_data, test_meta,
        raw_data=dm.data,
        lob=LOB,
    )

    raw_preds = None
    if collect_raw_preds:
        raw_preds = get_raw_paid_predictions(model, test_data)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "seed":           seed,
        "mape":           metrics["mape"],
        "rmspe":          metrics["rmspe"],
        "n_companies":    metrics["n_companies"],
        "n_filtered":     metrics["n_filtered"],
        "training_time":  round(t_sec, 2),
        "epochs_trained": summ["epochs_trained"],
        "best_val_loss":  summ["best_val_loss"],
    }
    return result, raw_preds


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_e1(
    dm_50: DataManager,
    seeds: List[int],
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    E1: 50-company subsample, batch=512, seeds 0-19, individual evaluation.

    Each seed is trained independently.  MAPE is computed on each seed
    individually (no ensemble averaging).

    Parameters
    ----------
    dm_50  : DataManager pre-filtered to 50 companies (see build_subsampled_dm)
    seeds  : list of ints
    resume : if True, skip seeds whose JSON already exists

    Returns
    -------
    list of result dicts
    """
    print(f"\n[E1] 50-company subsample, batch=512, {len(seeds)} seeds")

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

    results = []
    for i, seed in enumerate(seeds):
        # Resume check
        existing = load_result("e1", seed)
        if resume and existing is not None:
            print(f"  [E1] seed={seed:02d}  (resumed)  MAPE={existing['mape']:.4f}")
            results.append(existing)
            continue

        print(f"  [E1] seed={seed:02d}  [{i+1}/{len(seeds)}]  ...", end="", flush=True)
        t0 = time.time()
        result, _ = train_and_eval_seed(seed, dm_50, config, collect_raw_preds=False)
        result["exp"] = "e1"
        save_result("e1", seed, result)
        results.append(result)
        print(f"  MAPE={result['mape']:.4f}  epochs={result['epochs_trained']}  t={time.time()-t0:.0f}s")

    return results


def run_e2_and_e3(
    dm_50: DataManager,
    seeds: List[int],
    resume: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    E2 (ensemble with batch=512) and E3 (ensemble with batch=2250).

    Both experiments collect raw paid_pred_norm arrays from 20 models and
    average them before computing MAPE.  E2 uses the same models as E1
    (batch=512); E3 retrains with batch=2250.

    Strategy
    --------
    For E2: check if all E1 raw predictions are cached; if yes, load and
    average.  If not (raw arrays were not saved), retrain and collect.
    To keep things simple and self-contained, we always retrain for both
    E2 and E3 (collecting raw preds in-memory), since raw pred arrays are
    large and not saved by E1.  E2 results will be consistent with E1
    seed-by-seed results because set_seeds() is called before each model.

    Parameters
    ----------
    dm_50  : DataManager (50-company subset)
    seeds  : list of ints
    resume : if True and ensemble result JSON exists, skip

    Returns
    -------
    e2_metrics : dict  ensemble MAPE/RMSPE (batch=512)
    e3_metrics : dict  ensemble MAPE/RMSPE (batch=2250)
    """
    # --- E2: ensemble of batch=512 models ---
    e2_path = RESULTS_DIR / "e2_ensemble.json"
    if resume and e2_path.exists():
        with open(e2_path) as f:
            e2_metrics = json.load(f)
        print(f"\n[E2] Loaded cached ensemble result: MAPE={e2_metrics['mape']:.4f}")
    else:
        print(f"\n[E2] Ensemble averaging ({len(seeds)} models, batch=512)")
        e2_config = TrainConfig(
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
        e2_metrics = _run_ensemble(dm_50, seeds, e2_config, exp_tag="e2")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(e2_path, "w") as f:
            json.dump(e2_metrics, f, indent=2)
        print(f"  [E2] Ensemble MAPE={e2_metrics['mape']:.4f}  RMSPE={e2_metrics['rmspe']:.4f}")

    # --- E3: ensemble of batch=2250 models ---
    e3_path = RESULTS_DIR / "e3_ensemble.json"
    if resume and e3_path.exists():
        with open(e3_path) as f:
            e3_metrics = json.load(f)
        print(f"\n[E3] Loaded cached ensemble result: MAPE={e3_metrics['mape']:.4f}")
    else:
        print(f"\n[E3] Ensemble averaging ({len(seeds)} models, batch={KUO_BATCH_SIZE})")
        e3_config = TrainConfig(
            learning_rate=FIXED_HP["learning_rate"],
            batch_size=KUO_BATCH_SIZE,
            epochs=FIXED_HP["epochs"],
            es_patience=FIXED_HP["es_patience"],
            min_delta=FIXED_HP["min_delta"],
            lr_patience=FIXED_HP["lr_patience"],
            lr_factor=FIXED_HP["lr_factor"],
            min_lr=FIXED_HP["min_lr"],
            verbose=0,
        )
        e3_metrics = _run_ensemble(dm_50, seeds, e3_config, exp_tag="e3")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(e3_path, "w") as f:
            json.dump(e3_metrics, f, indent=2)
        print(f"  [E3] Ensemble MAPE={e3_metrics['mape']:.4f}  RMSPE={e3_metrics['rmspe']:.4f}")

    return e2_metrics, e3_metrics


def _run_ensemble(
    dm: DataManager,
    seeds: List[int],
    config: TrainConfig,
    exp_tag: str,
) -> Dict[str, float]:
    """
    Train N models and average their raw paid predictions, then compute MAPE.

    The raw paid_pred_norm arrays are averaged in normalized space BEFORE
    de-normalizing with earned_premium_net.  This is the correct ensemble
    approach because de-normalization is linear, so
      mean(de_norm(pred_i)) == de_norm(mean(pred_i)).

    Parameters
    ----------
    dm       : DataManager (subsampled)
    seeds    : seeds to train
    config   : TrainConfig (batch_size differs between E2 and E3)
    exp_tag  : 'e2' or 'e3' (for per-seed raw pred caching)

    Returns
    -------
    dict with 'mape', 'rmspe', 'n_companies', 'n_filtered', 'n_models'
    """
    test_data = dm.get(LOB, "test_data")
    test_meta = dm.get_test_metadata(LOB)
    raw_data  = dm.data

    all_raw_preds = []  # list of (N,9) arrays

    for i, seed in enumerate(seeds):
        # Check for cached per-seed raw preds
        raw_pred_path = RESULTS_DIR / exp_tag / f"raw_preds_{seed:03d}.npy"
        if raw_pred_path.exists():
            raw_preds = np.load(str(raw_pred_path))
            print(f"  [{exp_tag.upper()}] seed={seed:02d}  [{i+1}/{len(seeds)}]  (cached raw preds)")
        else:
            print(f"  [{exp_tag.upper()}] seed={seed:02d}  [{i+1}/{len(seeds)}]  training ...", end="", flush=True)
            t0 = time.time()
            _, raw_preds = train_and_eval_seed(seed, dm, config, collect_raw_preds=True)
            # Cache raw preds to disk (useful if interrupted)
            raw_pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(raw_pred_path), raw_preds)
            print(f"  t={time.time()-t0:.0f}s")

        all_raw_preds.append(raw_preds)

    # Average predictions across models  (N, 9)
    stacked = np.stack(all_raw_preds, axis=0)   # (n_models, N, 9)
    ensemble_preds = stacked.mean(axis=0)        # (N, 9)

    # Compute MAPE on ensemble predictions
    metrics = compute_mape_from_raw_preds(
        ensemble_preds, test_meta, raw_data, LOB
    )
    metrics["n_models"] = len(all_raw_preds)
    return metrics


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(
    e0_results: List[Dict[str, Any]],
    e1_results: List[Dict[str, Any]],
    e2_metrics: Dict[str, float],
    e3_metrics: Dict[str, float],
    mack_subset: Dict[str, float],
) -> pd.DataFrame:
    """
    Print and return a comparison table against Kuo (2019).

    Columns: Experiment | N_companies | batch_size | ensemble | MAPE | RMSPE | MAPE_std
    """
    rows = []

    # E0: Phase 1 baseline (all companies, batch=512, no ensemble)
    e0_mapes = [r["mape"] for r in e0_results if np.isfinite(r.get("mape", float("nan")))]
    if e0_mapes:
        rows.append({
            "Experiment":   "E0: Phase 1 baseline",
            "N_companies":  "all (~170 WC)",
            "batch_size":   FIXED_HP["batch_size"],
            "ensemble":     "no (mean of 20 seeds)",
            "MAPE":         np.mean(e0_mapes),
            "MAPE_std":     np.std(e0_mapes, ddof=1) if len(e0_mapes) > 1 else float("nan"),
            "RMSPE":        float("nan"),
        })
    else:
        rows.append({
            "Experiment": "E0: Phase 1 baseline",
            "N_companies": "all (~170 WC)", "batch_size": FIXED_HP["batch_size"],
            "ensemble": "no", "MAPE": float("nan"), "MAPE_std": float("nan"), "RMSPE": float("nan"),
        })

    # E1: 50-company subsample, individual seeds
    e1_mapes = [r["mape"] for r in e1_results if np.isfinite(r.get("mape", float("nan")))]
    if e1_mapes:
        rows.append({
            "Experiment":   "E1: 50-company, no ensemble",
            "N_companies":  N_COMPANIES,
            "batch_size":   FIXED_HP["batch_size"],
            "ensemble":     "no (mean of 20 seeds)",
            "MAPE":         np.mean(e1_mapes),
            "MAPE_std":     np.std(e1_mapes, ddof=1) if len(e1_mapes) > 1 else float("nan"),
            "RMSPE":        float("nan"),
        })

    # E2: ensemble averaging, batch=512
    rows.append({
        "Experiment":   "E2: 50-company, ensemble (batch=512)",
        "N_companies":  N_COMPANIES,
        "batch_size":   FIXED_HP["batch_size"],
        "ensemble":     f"yes ({e2_metrics.get('n_models', len(e1_results))} models)",
        "MAPE":         e2_metrics.get("mape", float("nan")),
        "MAPE_std":     float("nan"),
        "RMSPE":        e2_metrics.get("rmspe", float("nan")),
    })

    # E3: ensemble averaging, batch=2250
    rows.append({
        "Experiment":   "E3: 50-company, ensemble (batch=2250)",
        "N_companies":  N_COMPANIES,
        "batch_size":   KUO_BATCH_SIZE,
        "ensemble":     f"yes ({e3_metrics.get('n_models', len(e1_results))} models)",
        "MAPE":         e3_metrics.get("mape", float("nan")),
        "MAPE_std":     float("nan"),
        "RMSPE":        e3_metrics.get("rmspe", float("nan")),
    })

    # Mack CL on 50-company subset
    rows.append({
        "Experiment":   "Mack CL (50-company subset)",
        "N_companies":  N_COMPANIES,
        "batch_size":   "N/A",
        "ensemble":     "N/A",
        "MAPE":         mack_subset.get("mape", float("nan")),
        "MAPE_std":     float("nan"),
        "RMSPE":        mack_subset.get("rmspe", float("nan")),
    })

    # Kuo (2019) published results
    kuo = KUO_PUBLISHED["workers_compensation"]
    rows.append({
        "Experiment":   "Kuo (2019) DeepTriangle",
        "N_companies":  50,
        "batch_size":   2250,
        "ensemble":     "yes (100 models)",
        "MAPE":         kuo["mape"],
        "MAPE_std":     float("nan"),
        "RMSPE":        float("nan"),
    })
    rows.append({
        "Experiment":   "Kuo (2019) Mack CL",
        "N_companies":  50,
        "batch_size":   "N/A",
        "ensemble":     "N/A",
        "MAPE":         kuo["mack"],
        "MAPE_std":     float("nan"),
        "RMSPE":        float("nan"),
    })

    df = pd.DataFrame(rows)

    # Format for display
    float_cols = ["MAPE", "MAPE_std", "RMSPE"]
    display_df = df.copy()
    for col in float_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if np.isfinite(x) else "—"
        )

    print("\n" + "=" * 85)
    print(f"Kuo (2019) Comparison — Workers' Compensation ({LOB})")
    print("=" * 85)
    print(display_df.to_string(index=False))
    print("=" * 85)
    print(
        "\nNotes:"
        "\n  E0 MAPE = mean across 20 seeds (not ensemble-averaged)"
        "\n  E1 MAPE = mean across 20 seeds (not ensemble-averaged)"
        "\n  E2/E3   = single MAPE from ensemble-averaged predictions"
        "\n  Kuo (2019) uses 1988-1997 data; our data uses a different period"
        "\n  Ensemble approximated with 20 models vs Kuo's 100"
    )

    return df


def print_cached_results_table() -> bool:
    """
    Print the shipped Kuo comparison table without requiring raw Schedule P data.

    The public repository intentionally does not include triangle_sample.csv or
    triangle_company_info.csv. Default replication/verification should therefore
    validate the pre-computed paper artifacts, not attempt to rebuild the Kuo
    comparison from raw proprietary data.
    """
    csv_path = RESULTS_DIR / "kuo_comparison_table.csv"
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    display_df = df.copy()
    for col in ["MAPE", "MAPE_std", "RMSPE"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{float(x):.4f}" if pd.notna(x) and np.isfinite(float(x)) else "—"
            )

    print("[Kuo Comparison] Raw data files are not present; using shipped pre-computed results.")
    print("=" * 85)
    print(f"Kuo (2019) Comparison — Workers' Compensation ({LOB})")
    print("=" * 85)
    print(display_df.to_string(index=False))
    print("=" * 85)
    print(f"\n[Kuo Comparison] Cached results loaded from {csv_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    run_exps: List[str],
    seeds: List[int],
    resume: bool,
) -> None:
    """
    Main experiment driver.

    Parameters
    ----------
    run_exps : list of experiment codes, e.g. ['e0', 'e1', 'e2', 'e3', 'mack']
    seeds    : list of seed integers (default 0-19)
    resume   : skip experiments whose result files already exist
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # Public verification mode: the raw S&P/NAIC Schedule P CSVs are not
    # redistributable, so a clean public checkout should not fail simply because
    # data/triangle_sample.csv is absent. In that case, print the shipped table
    # and exit successfully. Full reruns still work once the data files exist
    # or after `python replicate.py --cas` creates public CAS-derived inputs.
    if not (Path(TRI_PATH).exists() and Path(CO_PATH).exists()):
        if print_cached_results_table():
            return
        missing = [str(p) for p in [Path(TRI_PATH), Path(CO_PATH)] if not p.exists()]
        raise FileNotFoundError(
            "Kuo comparison requires raw triangle data or shipped cached results. "
            f"Missing: {', '.join(missing)}"
        )

    # --- Load full DataManager once ---
    print("[Kuo Comparison] Loading full dataset ...")
    dm_full = DataManager(TRI_PATH, CO_PATH)
    dm_full.load()
    dm_full.prepare()
    print(f"  Full dataset: vocab_size={dm_full.vocab_size}, LOBs={dm_full.available_lobs()}")

    # --- Select 50 WC companies (deterministic) ---
    companies_50 = select_50_companies(dm_full, seed=SUBSAMPLE_SEED)
    print(f"  Selected {len(companies_50)} WC companies (seed={SUBSAMPLE_SEED})")
    # Save company list for reproducibility
    comp_path = RESULTS_DIR / "companies_50.json"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump([str(c) for c in companies_50], f, indent=2)

    # --- E0: load Phase 1 results ---
    e0_results = []
    if "e0" in run_exps:
        print(f"\n[E0] Loading Phase 1 gru_baseline WC results (seeds 0-{len(seeds)-1}) ...")
        e0_results = load_e0_results(n_seeds=len(seeds))
        e0_mapes = [r["mape"] for r in e0_results if np.isfinite(r.get("mape", float("nan")))]
        if e0_mapes:
            print(f"  [E0] N={len(e0_mapes)} seeds  MAPE={np.mean(e0_mapes):.4f} ± {np.std(e0_mapes, ddof=1):.4f}")
        else:
            print("  [E0] No Phase 1 results found — run run_phase1.py first.")

    # --- Build 50-company DataManager (for E1/E2/E3) ---
    dm_50 = None
    if any(e in run_exps for e in ["e1", "e2", "e3"]):
        print(f"\n[Setup] Building 50-company subsampled DataManager ...")
        dm_50 = build_subsampled_dm(companies_50)

    # --- E1: individual seeds, batch=512 ---
    e1_results = []
    if "e1" in run_exps:
        e1_results = run_e1(dm_50, seeds, resume=resume)
        e1_mapes = [r["mape"] for r in e1_results if np.isfinite(r.get("mape", float("nan")))]
        if e1_mapes:
            print(
                f"\n  [E1] Summary: N={len(e1_mapes)} seeds  "
                f"MAPE={np.mean(e1_mapes):.4f} ± {np.std(e1_mapes, ddof=1):.4f}"
            )
    else:
        # Try to load from disk for table construction
        e1_results = [
            r for seed in seeds
            for r in [load_result("e1", seed)]
            if r is not None
        ]

    # --- E2/E3: ensemble experiments ---
    e2_metrics = {"mape": float("nan"), "rmspe": float("nan")}
    e3_metrics = {"mape": float("nan"), "rmspe": float("nan")}
    if "e2" in run_exps or "e3" in run_exps:
        e2_metrics, e3_metrics = run_e2_and_e3(dm_50, seeds, resume=resume)
    else:
        # Try to load cached ensemble results
        e2_path = RESULTS_DIR / "e2_ensemble.json"
        e3_path = RESULTS_DIR / "e3_ensemble.json"
        if e2_path.exists():
            with open(e2_path) as f:
                e2_metrics = json.load(f)
        if e3_path.exists():
            with open(e3_path) as f:
                e3_metrics = json.load(f)

    # --- Mack CL on 50-company subset ---
    mack_subset = {"mape": float("nan"), "rmspe": float("nan"), "n": 0}
    if "mack" in run_exps:
        print(f"\n[Mack] Running Mack CL on {len(companies_50)}-company WC subset ...")
        mack_subset = run_mack_on_subset(companies_50)
        print(f"  Mack MAPE={mack_subset['mape']:.4f}  RMSPE={mack_subset['rmspe']:.4f}  N={mack_subset['n']}")
        mack_path = RESULTS_DIR / "mack_subset.json"
        with open(mack_path, "w") as f:
            json.dump(mack_subset, f, indent=2)
    else:
        mack_path = RESULTS_DIR / "mack_subset.json"
        if mack_path.exists():
            with open(mack_path) as f:
                mack_subset = json.load(f)

    # --- Print and save results table ---
    results_df = print_results_table(e0_results, e1_results, e2_metrics, e3_metrics, mack_subset)
    csv_path = RESULTS_DIR / "kuo_comparison_table.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[Kuo Comparison] Results saved to {csv_path}")
    print(f"[Kuo Comparison] Total wall time: {(time.time()-t_total)/60:.1f} min")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kuo (2019) comparison experiments: E0, E1, E2, E3, Mack subset"
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        default=["e0", "e1", "e2", "e3", "mack"],
        choices=["e0", "e1", "e2", "e3", "mack"],
        help="Experiments to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(N_SEEDS)),
        help=f"Seeds to use (default: 0-{N_SEEDS-1})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force re-run even if result files already exist",
    )
    args = parser.parse_args()

    main(
        run_exps=args.exp,
        seeds=args.seeds,
        resume=not args.no_resume,
    )
