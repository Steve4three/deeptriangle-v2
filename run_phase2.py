"""
Phase 2 Sensitivity Runner — DeepTriangle v2.

Workers' Compensation only.
100 random HP configurations × GRU Baseline.

The public paper package keeps Phase 2 focused on GRU Baseline tuning.

HP search space:
  dropout_rate : [0.01, 0.30]  log-uniform
  lr           : [1e-4, 5e-3]  log-uniform
  batch_size   : {256, 512, 1024}
  max_epochs   : {500, 1000}
  gru_units    : {64, 128, 256}
  dense_units  : {32, 64, 128}

Fixed (not varied):
  es_patience  = 200
  min_delta    = 0.001
  lr_patience  = 50
  lr_factor    = 0.5
  min_lr       = 1e-6

Results
-------
  results/phase2/workers_compensation/<arch>/hp_<config_idx>.json  — per-run
  results/phase2/workers_compensation/phase2_summary.csv           — all runs
  results/phase2/workers_compensation/<arch>_rf_importance.json    — RF feature importance

Usage
-----
    python run_phase2.py
    python run_phase2.py --archs gru_baseline gru_attention
    python run_phase2.py --configs 0 1 2         # specific HP indices
    python run_phase2.py --no-resume              # force re-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from data_prep import DataManager
from models import build_model, ARCH_NAMES
from train import TrainConfig, train_model, history_summary
from evaluate import compute_mape_rmspe

DATA_DIR = PROJECT_DIR / "data"
_BASE_RESULTS = Path(os.environ.get("DEEPTRIANGLE_RESULTS", str(PROJECT_DIR / "results")))
# Only Workers' Compensation for the main Phase 2 sweep.
PHASE2_LOB = "workers_compensation"
RESULTS_DIR = _BASE_RESULTS / "phase2" / PHASE2_LOB

# Fixed HP not varied in sensitivity
FIXED_HP = dict(
    es_patience=200,
    min_delta=0.001,
    lr_patience=50,
    lr_factor=0.5,
    min_lr=1e-6,
)

# Number of HP configs and seeds per config
N_CONFIGS = 100
SEED_PER_CONFIG = 1   # one seed per config (use config_idx as seed)


# ---------------------------------------------------------------------------
# HP grid definition
# ---------------------------------------------------------------------------

def sample_hp_configs(
    n: int = N_CONFIGS,
    seed: int = 2024,
) -> List[Dict[str, Any]]:
    """
    Sample N random hyperparameter configurations.

    The SAME configs are shared across all architectures to enable
    a fair architectural comparison on the sensitivity landscape.

    Sampling rules
    --------------
    - dropout_rate : log-uniform in [0.01, 0.30]
      sampled as exp(uniform(log(0.01), log(0.30)))
    - lr           : log-uniform in [1e-4, 5e-3]
      sampled as exp(uniform(log(1e-4), log(5e-3)))
    - batch_size   : uniform choice from {256, 512, 1024}
    - max_epochs   : uniform choice from {500, 1000}
    - gru_units    : uniform choice from {64, 128, 256}
    - dense_units  : uniform choice from {32, 64, 128}

    Parameters
    ----------
    n    : int  number of configs to sample
    seed : int  numpy random seed for reproducibility

    Returns
    -------
    list of dicts, each containing all HP keys needed by TrainConfig + build_model
    """
    rng = np.random.default_rng(seed)

    configs = []
    for i in range(n):
        dropout = float(np.exp(rng.uniform(np.log(0.01), np.log(0.30))))
        lr = float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-3))))
        batch_size = int(rng.choice([256, 512, 1024]))
        max_epochs = int(rng.choice([500, 1000]))
        gru_units = int(rng.choice([64, 128, 256]))
        dense_units = int(rng.choice([32, 64, 128]))

        configs.append({
            "config_idx": i,
            "dropout_rate": round(dropout, 5),
            "learning_rate": round(lr, 7),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "gru_units": gru_units,
            "dense_units": dense_units,
        })

    return configs


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def result_path(arch: str, config_idx: int) -> Path:
    return RESULTS_DIR / arch / f"hp_{config_idx:03d}.json"


def save_result(result: Dict[str, Any]) -> None:
    path = result_path(result["arch"], result["config_idx"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(arch: str, config_idx: int) -> Optional[Dict[str, Any]]:
    path = result_path(arch, config_idx)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    arch: str,
    config: Dict[str, Any],
    data_manager: DataManager,
) -> Dict[str, Any]:
    """
    Train and evaluate one model with a specific HP config.

    Uses config_idx as the random seed for reproducibility.

    Parameters
    ----------
    arch         : str
    config       : dict from sample_hp_configs
    data_manager : DataManager

    Returns
    -------
    dict with all HP params + metrics
    """
    seed = config["config_idx"]  # deterministic seed per config
    set_seeds(seed)

    train_data = data_manager.get(PHASE2_LOB, "full_training_data")
    val_data = data_manager.get(PHASE2_LOB, "validation_data")
    test_data = data_manager.get(PHASE2_LOB, "test_data")
    test_meta = data_manager.get_test_metadata(PHASE2_LOB)

    # --- Build ---
    model = build_model(
        arch,
        vocab_size=data_manager.vocab_size,
        gru_units=config["gru_units"],
        dropout_rate=config["dropout_rate"],
        dense_units=config["dense_units"],
    )

    # --- Train ---
    train_cfg = TrainConfig(
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        epochs=config["max_epochs"],
        es_patience=FIXED_HP["es_patience"],
        min_delta=FIXED_HP["min_delta"],
        lr_patience=FIXED_HP["lr_patience"],
        lr_factor=FIXED_HP["lr_factor"],
        min_lr=FIXED_HP["min_lr"],
        verbose=0,
    )
    history, t_sec = train_model(model, train_data, val_data, train_cfg)
    summ = history_summary(history)

    # --- Evaluate ---
    metrics = compute_mape_rmspe(
        model, test_data, test_meta,
        raw_data=data_manager.data,
        lob=PHASE2_LOB,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "arch": arch,
        "lob": PHASE2_LOB,
        **config,
        "mape": metrics["mape"],
        "rmspe": metrics["rmspe"],
        "n_companies": metrics["n_companies"],
        "n_filtered": metrics["n_filtered"],
        "mape_std": metrics.get("mape_std"),
        "mape_p25": metrics.get("mape_p25"),
        "mape_p75": metrics.get("mape_p75"),
        "training_time": round(t_sec, 2),
        "epochs_trained": summ["epochs_trained"],
        "best_val_loss": summ["best_val_loss"],
        # Loss curves for learning dynamics EDA
        "train_loss_curve": [round(x, 6) for x in history.history.get("loss", [])],
        "val_loss_curve": [round(x, 6) for x in history.history.get("val_loss", [])],
        # Per-company breakdown
        "per_company_mape": metrics.get("per_company_mape"),
    }


# ---------------------------------------------------------------------------
# Random Forest feature importance
# ---------------------------------------------------------------------------

def fit_rf_importance(arch_df: pd.DataFrame, arch: str) -> Dict[str, float]:
    """
    Fit a Random Forest regressor to predict MAPE from HP features.

    This quantifies which hyperparameters most influence model performance,
    reproducing the sensitivity analysis methodology from the enhanced_deeptriangle
    paper's Section 4.3.

    Parameters
    ----------
    arch_df : pd.DataFrame  results for one architecture
    arch    : str           architecture name (for logging)

    Returns
    -------
    dict mapping feature name -> importance score (sums to 1.0)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [
        "dropout_rate",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "gru_units",
        "dense_units",
    ]

    df = arch_df.dropna(subset=["mape"] + feature_cols).copy()
    if len(df) < 5:
        print(f"  RF importance: insufficient data for {arch} ({len(df)} rows)")
        return {}

    X = df[feature_cols].values
    y = df["mape"].values

    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importance = dict(zip(feature_cols, rf.feature_importances_.tolist()))
    print(f"\n  RF Importance ({arch}):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:20s}: {imp:.4f}")

    return importance


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_phase2(
    archs: List[str],
    config_indices: List[int],
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Phase 2 sensitivity experiments.

    Parameters
    ----------
    archs          : list of architecture names
    config_indices : list of HP config indices (subset of 0-49)
    resume         : skip completed runs
    verbose        : print per-run progress

    Returns
    -------
    pd.DataFrame — all results
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Canonical HP configs (shared across all architectures) ---
    all_configs = sample_hp_configs(N_CONFIGS)
    configs = [all_configs[i] for i in config_indices if i < len(all_configs)]

    # Save config table for reference
    config_df = pd.DataFrame(all_configs)
    config_df.to_csv(RESULTS_DIR / "hp_configs.csv", index=False)
    print(f"[Phase 2] HP configs saved to {RESULTS_DIR / 'hp_configs.csv'}")

    # --- Load data ---
    tri_path = str(DATA_DIR / "triangle_sample.csv")
    co_path = str(DATA_DIR / "triangle_company_info.csv")

    print("[Phase 2] Loading and preparing data ...")
    dm = DataManager(tri_path, co_path)
    dm.load()
    dm.prepare()
    print(f"  vocab_size = {dm.vocab_size}  LOB = {PHASE2_LOB}\n")

    total = len(archs) * len(configs)
    run_idx = 0
    completed = 0
    skipped = 0

    t_start = time.time()

    for arch in archs:
        for config in configs:
            run_idx += 1
            cidx = config["config_idx"]

            if resume and load_result(arch, cidx) is not None:
                skipped += 1
                continue

            label = (
                f"[{run_idx:3d}/{total}]  arch={arch}  "
                f"cfg={cidx:02d}  "
                f"lr={config['learning_rate']:.2e}  "
                f"drop={config['dropout_rate']:.3f}  "
                f"gru={config['gru_units']}  "
                f"bs={config['batch_size']}"
            )
            if verbose:
                print(f"{label}  ...", end="", flush=True)

            try:
                result = run_single(arch, config, dm)
                save_result(result)
                completed += 1

                if verbose:
                    print(
                        f"  MAPE={result['mape']:.4f}  "
                        f"epochs={result['epochs_trained']:4d}  "
                        f"t={result['training_time']:.0f}s"
                    )
            except Exception as e:
                print(f"  FAILED: {e}")
                save_result({
                    "arch": arch, "lob": PHASE2_LOB, **config,
                    "mape": float("nan"), "rmspe": float("nan"),
                    "n_companies": 0, "n_filtered": 0,
                    "training_time": 0, "epochs_trained": 0,
                    "best_val_loss": float("nan"),
                    "error": str(e),
                })

            if run_idx % 10 == 0 and verbose:
                elapsed = time.time() - t_start
                remaining = total - run_idx - skipped
                rate = completed / elapsed if elapsed > 0 else 1e-9
                eta = remaining / rate
                print(
                    f"  >>> {run_idx}/{total} runs  "
                    f"({completed} done, {skipped} skipped)  "
                    f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min"
                )

    # --- Aggregate results ---
    all_rows = []
    for arch in archs:
        for cidx in config_indices:
            r = load_result(arch, cidx)
            if r is not None:
                all_rows.append(r)

    summary_df = pd.DataFrame(all_rows)
    summary_path = RESULTS_DIR / "phase2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Phase 2] Summary saved to {summary_path}")

    # --- RF feature importance per architecture ---
    if len(summary_df) > 0:
        for arch in archs:
            arch_df = summary_df[summary_df["arch"] == arch]
            if len(arch_df) >= 5:
                importance = fit_rf_importance(arch_df, arch)
                imp_path = RESULTS_DIR / f"{arch}_rf_importance.json"
                with open(imp_path, "w") as f:
                    json.dump(importance, f, indent=2)
                print(f"  Saved RF importance: {imp_path}")

    return summary_df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: HP sensitivity analysis (Workers' Comp only)"
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=list(ARCH_NAMES),
        choices=list(ARCH_NAMES),
        help="Architectures to run (default: all 3)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=int,
        default=list(range(N_CONFIGS)),
        help=f"HP config indices to run (default: 0-{N_CONFIGS - 1})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force re-run of all configs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run output",
    )
    args = parser.parse_args()

    run_phase2(
        archs=args.archs,
        config_indices=args.configs,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )
