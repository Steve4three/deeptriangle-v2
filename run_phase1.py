"""
Phase 1 Experiment Runner — DeepTriangle v2.

Fixed hyperparameters applied to all 3 architectures × N LOBs (auto-detected) × 50 seeds.

Fixed HP set (matching Kuo 2019 original training config):
  gru_units    = 128
  dropout_rate = 0.10
  lr           = 5e-4
  batch_size   = 512
  epochs       = 1000
  es_patience  = 200
  min_delta    = 0.001
  lr_patience  = 50
  dense_units  = 64

Total runs: 3 architectures × N LOBs (auto-detected) × 50 seeds = 600 runs

Usage
-----
    python run_phase1.py                        # all runs
    python run_phase1.py --lobs workers_compensation  # single LOB
    python run_phase1.py --archs gru_baseline   # single arch
    python run_phase1.py --seeds 0 1 2          # specific seeds
    python run_phase1.py --resume               # skip already-completed runs

Results are written to:
    results/phase1/<arch>/<lob>/run_<seed>.json  — per-run results
    results/phase1/phase1_summary.csv            — aggregated summary
"""

from __future__ import annotations

import argparse
import json
import os
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

from data_prep import DataManager, LOBS
from models import build_model, ARCH_NAMES
from train import TrainConfig, train_model, history_summary
from evaluate import compute_mape_rmspe

DATA_DIR = PROJECT_DIR / "data"
_BASE_RESULTS = Path(os.environ.get("DEEPTRIANGLE_RESULTS", str(PROJECT_DIR / "results")))
RESULTS_DIR = _BASE_RESULTS / "phase1"


# ---------------------------------------------------------------------------
# Fixed hyperparameters
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
# Seed control
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: Python's random module is not used here but set for completeness
    import random
    random.seed(seed)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    arch: str,
    lob: str,
    seed: int,
    data_manager: DataManager,
    config: TrainConfig,
) -> Dict[str, Any]:
    """
    Train and evaluate one model instance.

    Parameters
    ----------
    arch         : str   architecture name
    lob          : str   line of business
    seed         : int   random seed
    data_manager : DataManager  (already prepared)
    config       : TrainConfig

    Returns
    -------
    dict with keys: arch, lob, seed, mape, rmspe, n_companies, training_time,
                    epochs_trained, best_val_loss
    """
    set_seeds(seed)

    # --- Data ---
    train_data = data_manager.get(lob, "full_training_data")
    val_data = data_manager.get(lob, "validation_data")
    test_data = data_manager.get(lob, "test_data")
    test_meta = data_manager.get_test_metadata(lob)

    # --- Build model ---
    model = build_model(
        arch,
        vocab_size=data_manager.vocab_size,
        gru_units=FIXED_HP["gru_units"],
        dropout_rate=FIXED_HP["dropout_rate"],
        dense_units=FIXED_HP["dense_units"],
    )

    # --- Train ---
    history, t_sec = train_model(model, train_data, val_data, config)
    summ = history_summary(history)

    # --- Evaluate ---
    metrics = compute_mape_rmspe(
        model, test_data, test_meta,
        raw_data=data_manager.data,
        lob=lob,
    )

    # Save weights for seed 0 (reproducibility anchor)
    if seed == 0:
        weights_dir = RESULTS_DIR / arch / lob / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_dir / f"seed_{seed:03d}.pt")

    result = {
        "arch": arch,
        "lob": lob,
        "seed": seed,
        "mape": metrics["mape"],
        "rmspe": metrics["rmspe"],
        "n_companies": metrics["n_companies"],
        "n_filtered": metrics["n_filtered"],
        "training_time": round(t_sec, 2),
        "epochs_trained": summ["epochs_trained"],
        "best_val_loss": summ["best_val_loss"],
    }

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Result I/O helpers
# ---------------------------------------------------------------------------

def result_path(arch: str, lob: str, seed: int) -> Path:
    return RESULTS_DIR / arch / lob / f"run_{seed:03d}.json"


def save_result(result: Dict[str, Any]) -> None:
    path = result_path(result["arch"], result["lob"], result["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(arch: str, lob: str, seed: int) -> Optional[Dict[str, Any]]:
    path = result_path(arch, lob, seed)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(
    archs: List[str], lobs: List[str], seeds: List[int]
) -> pd.DataFrame:
    """Load all available results and build a summary CSV."""
    rows = []
    for arch in archs:
        for lob in lobs:
            for seed in seeds:
                r = load_result(arch, lob, seed)
                if r is not None:
                    rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_phase1(
    archs: List[str],
    lobs: List[str],
    seeds: List[int],
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Phase 1 experiment.

    Parameters
    ----------
    archs  : list of architecture names
    lobs   : list of LOB strings
    seeds  : list of integer seeds (e.g. range(50))
    resume : if True, skip runs where result JSON already exists
    verbose: print progress

    Returns
    -------
    pd.DataFrame — all results collected so far
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data once ---
    tri_path = str(DATA_DIR / "triangle_sample.csv")
    co_path = str(DATA_DIR / "triangle_company_info.csv")

    print("[Phase 1] Loading and preparing data ...")
    dm = DataManager(tri_path, co_path)
    dm.load()
    dm.prepare()
    print(f"  vocab_size = {dm.vocab_size}")

    # --- Training config ---
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
        (arch, lob, seed)
        for arch in archs
        for lob in lobs
        for seed in seeds
    ]
    total = len(all_runs)
    completed = 0
    skipped = 0

    print(f"[Phase 1] Total runs planned: {total}")
    print(f"[Phase 1] Architectures : {archs}")
    print(f"[Phase 1] LOBs          : {lobs}")
    print(f"[Phase 1] Seeds         : {min(seeds)} - {max(seeds)}\n")

    t_start = time.time()

    for run_idx, (arch, lob, seed) in enumerate(all_runs, 1):
        # --- Resume: skip if result exists ---
        if resume and load_result(arch, lob, seed) is not None:
            skipped += 1
            continue

        # --- Run ---
        run_label = f"[{run_idx:4d}/{total}]  arch={arch}  lob={lob}  seed={seed:03d}"
        if verbose:
            print(f"{run_label}  ...", end="", flush=True)

        try:
            result = run_single(arch, lob, seed, dm, config)
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
            # Save partial failure record so we can diagnose later
            save_result({
                "arch": arch, "lob": lob, "seed": seed,
                "mape": float("nan"), "rmspe": float("nan"),
                "n_companies": 0, "n_filtered": 0,
                "training_time": 0, "epochs_trained": 0,
                "best_val_loss": float("nan"),
                "error": str(e),
            })

        # Progress every 10 completed runs
        if (run_idx % 10 == 0 or run_idx == total) and verbose:
            elapsed = time.time() - t_start
            remaining_runs = total - run_idx - skipped
            rate = completed / elapsed if elapsed > 0 else 0
            eta = remaining_runs / rate if rate > 0 else float("inf")
            print(
                f"  >>> Progress: {run_idx}/{total} runs  "
                f"({completed} completed, {skipped} skipped)  "
                f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min"
            )

    # --- Build and save summary ---
    summary_df = build_summary(archs, lobs, seeds)
    summary_path = RESULTS_DIR / "phase1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Phase 1] Summary saved to {summary_path}")
    print(f"[Phase 1] Total completed: {len(summary_df)} runs")

    # Print aggregate MAPE by arch × LOB
    if not summary_df.empty:
        _print_phase1_summary(summary_df)

    total_elapsed = time.time() - t_start
    print(f"\n[Phase 1] Total wall time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} min)")
    print(f"[Phase 1] Avg per run: {total_elapsed/max(completed,1):.1f}s")

    # Save best model weights per arch × LOB
    if not summary_df.empty:
        _save_best_weights(summary_df, dm, config, archs, lobs)

    return summary_df


def _save_best_weights(
    summary_df: pd.DataFrame,
    dm: DataManager,
    config: TrainConfig,
    archs: list,
    lobs: list,
) -> None:
    """Re-train and save weights for the best seed per arch × LOB."""
    valid = summary_df.dropna(subset=["mape"])
    if valid.empty:
        return

    best = valid.loc[valid.groupby(["arch", "lob"])["mape"].idxmin()]
    print("\n[Phase 1] Saving best model weights...")

    for _, row in best.iterrows():
        arch, lob, seed = row["arch"], row["lob"], int(row["seed"])
        weights_dir = RESULTS_DIR / arch / lob / "weights"
        best_path = weights_dir / f"best_seed_{seed:03d}.pt"

        # Skip if seed 0 already saved or best weights exist
        if best_path.exists():
            print(f"  {arch}/{lob}: already saved (seed {seed})")
            continue

        print(f"  {arch}/{lob}: re-training seed {seed} (MAPE={row['mape']:.4f})...")
        set_seeds(seed)
        train_data = dm.get(lob, "full_training_data")
        val_data = dm.get(lob, "validation_data")
        model_hp = {k: FIXED_HP[k] for k in ("gru_units", "dropout_rate", "dense_units")}
        model = build_model(arch, vocab_size=dm.vocab_size, **model_hp)
        _, _ = train_model(model, train_data, val_data, config)
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), best_path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    saved → {best_path}")


def _print_phase1_summary(df: pd.DataFrame) -> None:
    """Print a pivot table of mean MAPE by arch and LOB."""
    valid = df.dropna(subset=["mape"])
    if valid.empty:
        print("No valid results to summarize.")
        return

    pivot = valid.groupby(["arch", "lob"])["mape"].agg(
        mean_mape="mean", std_mape="std", count="count"
    ).reset_index()

    print("\n=== Phase 1 MAPE Summary (mean ± std across seeds) ===")
    for arch in valid["arch"].unique():
        print(f"\n  Architecture: {arch}")
        arch_df = pivot[pivot["arch"] == arch]
        for _, row in arch_df.iterrows():
            print(
                f"    {row['lob']:30s}  "
                f"MAPE={row['mean_mape']:.4f} ± {row['std_mape']:.4f}  "
                f"(N={row['count']:.0f})"
            )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Fixed-HP comparison across architectures, LOBs, and seeds"
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=list(ARCH_NAMES),
        choices=list(ARCH_NAMES),
        help="Architectures to run (default: all 3)",
    )
    parser.add_argument(
        "--lobs",
        nargs="+",
        default=None,  # None = auto-detect from data
        choices=LOBS + [None],
        help="Lines of business (default: auto-detect from data file)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(50)),
        help="Seeds to run (default: 0-49)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force re-run even if result file already exists",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run output",
    )
    args = parser.parse_args()

    # Auto-detect available LOBs from data if not specified
    if args.lobs is None:
        tri_path = str(DATA_DIR / "triangle_sample.csv")
        import pandas as _pd
        _raw = _pd.read_csv(tri_path)
        available_lobs = sorted(_raw["lob"].unique().tolist())
        print(f"[Phase 1] Auto-detected LOBs: {available_lobs}")
    else:
        available_lobs = args.lobs

    run_phase1(
        archs=args.archs,
        lobs=available_lobs,
        seeds=args.seeds,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )
