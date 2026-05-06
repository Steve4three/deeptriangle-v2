"""
Rolling-Origin Temporal Robustness Study — DeepTriangle v2.

Evaluates whether the architecture ranking (GRU Baseline > GRU+Attention masked > GRU+Attention unmasked)
is robust across different temporal windows, not just the single W3 window used
in Phase 1.

Three rolling windows, each shifted by 1 year:

  Window | Train <=  | Val       | Test diag | Mask <= | Test AYs   | N AYs
  -------|-----------|-----------|-----------|---------|------------|------
  W1     | 2006      | 2007-2008 | 2009      | 2008    | 2000-2009  | 10
  W2     | 2007      | 2008-2009 | 2010      | 2009    | 2001-2010  | 10
  W3     | 2008      | 2009-2010 | 2011      | 2010    | 2002-2011  | 10

Runs: 3 windows x 3 architectures x 10 seeds x 1 LOB (WC) = 90 runs

Note: Each window uses the FULL triangle — the most recent AY has only
lag-0 data, which is included for completeness.
Also runs Mack Chain-Ladder benchmark per window for comparison.

Usage
-----
    python run_temporal.py                  # all runs
    python run_temporal.py --seeds 0 1 2    # specific seeds
    python run_temporal.py --resume         # skip completed runs
    python run_temporal.py --windows W1     # single window

Results are written to:
    results/temporal/<window>/<arch>/run_<seed>.json  — per-run results
    results/temporal/temporal_summary.csv              — aggregated summary
    results/temporal/benchmarks.json                   — Mack CL per window
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
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
from evaluate import compute_mape_rmspe

DATA_DIR = PROJECT_DIR / "data"
_BASE_RESULTS = Path(os.environ.get("DEEPTRIANGLE_RESULTS", str(PROJECT_DIR / "results")))
RESULTS_DIR = _BASE_RESULTS / "temporal"

LOB = "workers_compensation"

# ---------------------------------------------------------------------------
# Fixed hyperparameters (same as Phase 1)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Window definitions
# ---------------------------------------------------------------------------

@dataclass
class WindowConfig:
    name: str
    train_ranges: List[Tuple[Optional[int], int]]
    validation_ranges: List[Tuple[int, int]]
    test_min_calendar_year: int
    test_max_accident_year: int
    accident_year_range: Tuple[int, int]  # for evaluation


# Proprietary data windows (test diagonals 2009, 2010, 2011)
_WINDOWS_PROPRIETARY = {
    "W1": WindowConfig(
        name="W1",
        train_ranges=[(None, 2006)],
        validation_ranges=[(2007, 2008)],
        test_min_calendar_year=2009,
        test_max_accident_year=2009,
        accident_year_range=(2000, 2009),
    ),
    "W2": WindowConfig(
        name="W2",
        train_ranges=[(None, 2007)],
        validation_ranges=[(2008, 2009)],
        test_min_calendar_year=2010,
        test_max_accident_year=2010,
        accident_year_range=(2001, 2010),
    ),
    "W3": WindowConfig(
        name="W3",
        train_ranges=[(None, 2008)],
        validation_ranges=[(2009, 2010)],
        test_min_calendar_year=2011,
        test_max_accident_year=2011,
        accident_year_range=(2002, 2011),
    ),
}

# CAS public data windows (AY 1998-2007, test diagonals 2006, 2007, 2008)
_WINDOWS_CAS = {
    "W1": WindowConfig(
        name="W1",
        train_ranges=[(None, 2003)],
        validation_ranges=[(2004, 2005)],
        test_min_calendar_year=2006,
        test_max_accident_year=2007,
        accident_year_range=(1998, 2007),
    ),
    "W2": WindowConfig(
        name="W2",
        train_ranges=[(None, 2004)],
        validation_ranges=[(2005, 2006)],
        test_min_calendar_year=2007,
        test_max_accident_year=2007,
        accident_year_range=(1998, 2007),
    ),
    "W3": WindowConfig(
        name="W3",
        train_ranges=[(None, 2005)],
        validation_ranges=[(2006, 2007)],
        test_min_calendar_year=2008,
        test_max_accident_year=2007,
        accident_year_range=(1998, 2007),
    ),
}

# Select windows based on env var
WINDOWS = _WINDOWS_CAS if os.environ.get("DEEPTRIANGLE_TEST_CAL") else _WINDOWS_PROPRIETARY


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    arch: str,
    seed: int,
    window: WindowConfig,
    data_manager: DataManager,
    config: TrainConfig,
) -> Dict[str, Any]:
    """Train and evaluate one model instance for a given window."""
    set_seeds(seed)

    train_data = data_manager.get(LOB, "full_training_data")
    val_data = data_manager.get(LOB, "validation_data")
    test_data = data_manager.get(LOB, "test_data")
    test_meta = data_manager.get_test_metadata(LOB)

    model = build_model(
        arch,
        vocab_size=data_manager.vocab_size,
        gru_units=FIXED_HP["gru_units"],
        dropout_rate=FIXED_HP["dropout_rate"],
        dense_units=FIXED_HP["dense_units"],
    )

    history, t_sec = train_model(model, train_data, val_data, config)
    summ = history_summary(history)

    metrics = compute_mape_rmspe(
        model, test_data, test_meta,
        raw_data=data_manager.data,
        lob=LOB,
        accident_year_range=window.accident_year_range,
        test_calendar_year=window.test_min_calendar_year,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "window": window.name,
        "arch": arch,
        "lob": LOB,
        "seed": seed,
        "mape": metrics["mape"],
        "rmspe": metrics["rmspe"],
        "n_companies": metrics["n_companies"],
        "n_filtered": metrics["n_filtered"],
        "training_time": round(t_sec, 2),
        "epochs_trained": summ["epochs_trained"],
        "best_val_loss": summ["best_val_loss"],
    }


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def result_path(window: str, arch: str, seed: int) -> Path:
    return RESULTS_DIR / window / arch / f"run_{seed:03d}.json"


def save_result(result: Dict[str, Any]) -> None:
    path = result_path(result["window"], result["arch"], result["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(window: str, arch: str, seed: int) -> Optional[Dict[str, Any]]:
    path = result_path(window, arch, seed)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Mack Chain-Ladder benchmark per window
# ---------------------------------------------------------------------------

def run_mack_benchmark(window: WindowConfig) -> Dict[str, float]:
    """
    Run Mack Chain-Ladder for a given window and return MAPE/RMSPE.

    Uses chainladder-python with development data up to mask_calendar_year.
    Evaluates against actual ultimates at development lag 9.
    """
    import chainladder as cl

    tri_path = str(DATA_DIR / "triangle_sample.csv")
    co_path = str(DATA_DIR / "triangle_company_info.csv")

    raw = pd.read_csv(tri_path)
    co = pd.read_csv(co_path)
    raw = raw.merge(co[["group_code"]], on="group_code", how="left")
    raw = raw.sort_values(["lob", "group_code", "accident_year", "development_lag"])

    lob_data = raw[raw["lob"] == LOB].copy()
    mask_cal_year = window.test_min_calendar_year - 1

    # Training data: up to mask_calendar_year
    train_data = lob_data[lob_data["calendar_year"] <= mask_cal_year].copy()

    # Build triangle
    tri_obj = cl.Triangle(
        data=train_data,
        origin="accident_year",
        development="development_year",
        columns=["cumulative_paid_loss"],
        index=["group_code"],
        cumulative=True,
    )

    # Fit chain-ladder
    dev = cl.Development().fit_transform(tri_obj)
    cl_model = cl.Chainladder().fit(dev)
    cl_ult = cl_model.ultimate_

    # Extract predictions
    ult_df = cl_ult.to_frame().reset_index()
    period_cols = [c for c in ult_df.columns if isinstance(c, pd.Period)]
    if not period_cols:
        period_cols = [c for c in ult_df.columns if c != "group_code"]

    melted = ult_df.melt(
        id_vars=["group_code"],
        value_vars=period_cols,
        var_name="accident_year",
        value_name="pred_ult",
    )
    melted["accident_year"] = melted["accident_year"].apply(
        lambda p: p.year if hasattr(p, "year") else int(p)
    )
    melted = melted[melted["pred_ult"].notna() & (melted["pred_ult"] > 0)]
    ay_min, ay_max = window.accident_year_range
    melted = melted[
        (melted["accident_year"] >= ay_min)
        & (melted["accident_year"] <= ay_max)
    ]

    # Get actual ultimates
    actual_ult = (
        lob_data[
            (lob_data["development_lag"] == 9)
            & (lob_data["accident_year"] >= ay_min)
            & (lob_data["accident_year"] <= ay_max)
        ][["group_code", "accident_year", "cumulative_paid_loss"]]
        .rename(columns={"cumulative_paid_loss": "actual_ult"})
    )

    merged = melted.merge(actual_ult, on=["group_code", "accident_year"], how="inner")
    merged = merged.dropna(subset=["pred_ult", "actual_ult"])

    if merged.empty:
        return {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

    EPSILON = 1e-8
    denom = np.where(
        np.abs(merged["actual_ult"].values) < EPSILON,
        EPSILON,
        np.abs(merged["actual_ult"].values),
    )
    merged = merged.copy()
    merged["pct_error"] = (merged["pred_ult"].values - merged["actual_ult"].values) / denom
    merged = merged[np.abs(merged["pct_error"]) <= 10.0]

    company_metrics = merged.groupby("group_code").agg(
        abs_pct_error=("pct_error", lambda x: np.mean(np.abs(x))),
        sq_pct_error=("pct_error", lambda x: np.mean(x ** 2)),
    ).reset_index()

    return {
        "mape": float(company_metrics["abs_pct_error"].mean()),
        "rmspe": float(np.sqrt(company_metrics["sq_pct_error"].mean())),
        "n": len(company_metrics),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(
    windows: List[str], archs: List[str], seeds: List[int]
) -> pd.DataFrame:
    rows = []
    for w in windows:
        for arch in archs:
            for seed in seeds:
                r = load_result(w, arch, seed)
                if r is not None:
                    rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_temporal(
    window_names: List[str],
    archs: List[str],
    seeds: List[int],
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the temporal robustness experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tri_path = str(DATA_DIR / "triangle_sample.csv")
    co_path = str(DATA_DIR / "triangle_company_info.csv")

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

    # --- Enumerate all runs ---
    all_runs = [
        (w, arch, seed)
        for w in window_names
        for arch in archs
        for seed in seeds
    ]
    total = len(all_runs)

    print(f"[Temporal] Total runs planned: {total}")
    print(f"[Temporal] Windows       : {window_names}")
    print(f"[Temporal] Architectures : {archs}")
    print(f"[Temporal] Seeds         : {min(seeds)} - {max(seeds)}")
    print(f"[Temporal] LOB           : {LOB}\n")

    # --- Run Mack CL benchmarks ---
    benchmarks = {}
    for wname in window_names:
        w = WINDOWS[wname]
        print(f"[Temporal] Running Mack CL for {wname} (train <= {w.test_min_calendar_year - 1}) ...", end="")
        try:
            bm = run_mack_benchmark(w)
            benchmarks[wname] = bm
            print(f"  MAPE={bm['mape']:.4f}  RMSPE={bm['rmspe']:.4f}  N={bm['n']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            benchmarks[wname] = {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

    bm_path = RESULTS_DIR / "benchmarks.json"
    with open(bm_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"[Temporal] Benchmarks saved to {bm_path}\n")

    # --- Load data per window and run ---
    # Pre-load DataManagers for each window to avoid reloading
    data_managers: Dict[str, DataManager] = {}
    for wname in window_names:
        w = WINDOWS[wname]
        print(f"[Temporal] Loading data for {wname} ...", end="")
        dm = DataManager(
            tri_path, co_path,
            train_ranges=w.train_ranges,
            validation_ranges=w.validation_ranges,
            test_min_calendar_year=w.test_min_calendar_year,
            test_max_accident_year=w.test_max_accident_year,
        )
        dm.load()
        dm.prepare()
        data_managers[wname] = dm
        n_train = dm.get(LOB, "full_training_data")["x"]["ay_seq_input"].shape[0]
        n_test = dm.get(LOB, "test_data")["x"]["ay_seq_input"].shape[0]
        print(f"  train={n_train}  test={n_test}  vocab={dm.vocab_size}")

    print()

    completed = 0
    skipped = 0
    t_start = time.time()

    for run_idx, (wname, arch, seed) in enumerate(all_runs, 1):
        if resume and load_result(wname, arch, seed) is not None:
            skipped += 1
            continue

        run_label = f"[{run_idx:3d}/{total}]  {wname}  arch={arch:15s}  seed={seed:03d}"
        if verbose:
            print(f"{run_label}  ...", end="", flush=True)

        try:
            result = run_single(
                arch, seed, WINDOWS[wname], data_managers[wname], config
            )
            save_result(result)
            completed += 1

            if verbose:
                print(
                    f"  MAPE={result['mape']:.4f}  "
                    f"RMSPE={result['rmspe']:.4f}  "
                    f"epochs={result['epochs_trained']:4d}  "
                    f"t={result['training_time']:.0f}s"
                )
        except Exception as e:
            print(f"  FAILED: {e}")
            save_result({
                "window": wname, "arch": arch, "lob": LOB, "seed": seed,
                "mape": float("nan"), "rmspe": float("nan"),
                "n_companies": 0, "n_filtered": 0,
                "training_time": 0, "epochs_trained": 0,
                "best_val_loss": float("nan"),
                "error": str(e),
            })

        if (run_idx % 10 == 0 or run_idx == total) and verbose:
            elapsed = time.time() - t_start
            remaining = total - run_idx - skipped
            rate = completed / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"  >>> Progress: {run_idx}/{total}  "
                f"({completed} done, {skipped} skipped)  "
                f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min"
            )

    # --- Build and save summary ---
    summary_df = build_summary(window_names, archs, seeds)
    summary_path = RESULTS_DIR / "temporal_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Temporal] Summary saved to {summary_path}")

    if not summary_df.empty:
        _print_summary(summary_df, benchmarks)

    return summary_df


def _print_summary(df: pd.DataFrame, benchmarks: Dict[str, Any]) -> None:
    """Print a summary table of MAPE by window x architecture."""
    valid = df.dropna(subset=["mape"])
    if valid.empty:
        print("No valid results to summarize.")
        return

    print("\n" + "=" * 80)
    print("TEMPORAL ROBUSTNESS SUMMARY  (MAPE: mean +/- std across seeds)")
    print("=" * 80)

    pivot = valid.groupby(["window", "arch"])["mape"].agg(
        mean="mean", std="std", count="count"
    ).reset_index()

    for wname in sorted(valid["window"].unique()):
        bm = benchmarks.get(wname, {})
        mack_mape = bm.get("mape", float("nan"))
        print(f"\n  {wname}  (Mack CL: {mack_mape:.4f})")
        w_df = pivot[pivot["window"] == wname].sort_values("mean")
        for _, row in w_df.iterrows():
            marker = " *" if row["mean"] == w_df["mean"].min() else ""
            print(
                f"    {row['arch']:20s}  "
                f"MAPE={row['mean']:.4f} +/- {row['std']:.4f}  "
                f"(N={row['count']:.0f}){marker}"
            )

    # Architecture ranking consistency
    print("\n  Architecture ranking (by mean MAPE, best first):")
    for wname in sorted(valid["window"].unique()):
        w_pivot = pivot[pivot["window"] == wname].sort_values("mean")
        ranking = " > ".join(w_pivot["arch"].tolist())
        print(f"    {wname}: {ranking}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rolling-origin temporal robustness study for DeepTriangle v2"
    )
    parser.add_argument(
        "--windows", nargs="+", default=list(WINDOWS.keys()),
        choices=list(WINDOWS.keys()),
        help="Windows to run (default: all 3)",
    )
    parser.add_argument(
        "--archs", nargs="+", default=list(ARCH_NAMES),
        choices=list(ARCH_NAMES),
        help="Architectures to run (default: all 3)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(10)),
        help="Seeds to run (default: 0-9)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Force re-run even if result exists",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run output",
    )
    args = parser.parse_args()

    run_temporal(
        window_names=args.windows,
        archs=args.archs,
        seeds=args.seeds,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )
