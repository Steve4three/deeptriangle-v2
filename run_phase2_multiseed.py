#!/usr/bin/env python3
"""
Phase 2 Multi-Seed Validation

Two-stage protocol:
  Stage 1: Run 100 configs × 1 seed for GRU Baseline (screening)
  Stage 2: Pick top 20 configs by MAPE → run 5 additional seeds each (validation)

Usage:
    python run_phase2_multiseed.py                    # full pipeline
    python run_phase2_multiseed.py --stage 1          # screening only
    python run_phase2_multiseed.py --stage 2          # validation only (requires stage 1 results)
    python run_phase2_multiseed.py --top-k 20 --extra-seeds 5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from run_phase2 import (
    sample_hp_configs,
    set_seeds,
    run_single,
    save_result,
    load_result,
    fit_rf_importance,
    RESULTS_DIR,
    PHASE2_LOB,
    DATA_DIR,
    N_CONFIGS,
)
from data_prep import DataManager

MULTISEED_DIR = RESULTS_DIR / "multiseed"


def result_path_seeded(arch: str, config_idx: int, seed: int) -> Path:
    return MULTISEED_DIR / arch / f"hp_{config_idx:03d}_seed{seed:02d}.json"


def save_result_seeded(result: Dict[str, Any], seed: int) -> None:
    path = result_path_seeded(result["arch"], result["config_idx"], seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result_seeded(arch: str, config_idx: int, seed: int) -> Optional[Dict[str, Any]]:
    path = result_path_seeded(arch, config_idx, seed)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Stage 1: Screening — 100 configs × 1 seed
# ---------------------------------------------------------------------------

def stage1_screening(dm: DataManager, resume: bool = True) -> pd.DataFrame:
    """Run 100 HP configs with 1 seed each for GRU Baseline."""
    arch = "gru_baseline"
    configs = sample_hp_configs(100, seed=2024)

    print(f"\n{'='*60}")
    print(f"STAGE 1: Screening — 100 configs × 1 seed (GRU Baseline)")
    print(f"{'='*60}\n")

    t_start = time.time()
    completed = 0

    for i, config in enumerate(configs):
        cidx = config["config_idx"]

        if resume and load_result(arch, cidx) is not None:
            completed += 1
            continue

        print(f"[{i+1:3d}/100]  cfg={cidx:02d}  lr={config['learning_rate']:.2e}  "
              f"drop={config['dropout_rate']:.3f}  gru={config['gru_units']}  "
              f"bs={config['batch_size']}  ...", end="", flush=True)

        try:
            result = run_single(arch, config, dm)
            save_result(result)
            completed += 1
            print(f"  MAPE={result['mape']:.4f}  t={result['training_time']:.0f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            save_result({
                "arch": arch, "lob": PHASE2_LOB, **config,
                "mape": float("nan"), "rmspe": float("nan"),
                "training_time": 0, "epochs_trained": 0,
                "error": str(e),
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  >>> {i+1}/100  elapsed={elapsed/60:.1f}min")

    # Aggregate
    rows = []
    for cidx in range(100):
        r = load_result(arch, cidx)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "phase2_stage1_screening.csv", index=False)

    # RF importance on full 100
    if len(df) >= 10:
        importance = fit_rf_importance(df, arch)
        with open(RESULTS_DIR / f"{arch}_rf_importance_100.json", "w") as f:
            json.dump(importance, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nStage 1 complete: {completed}/100 configs, {elapsed/60:.1f} min total")
    print(f"Best MAPE: {df['mape'].min():.4f}  Worst: {df['mape'].max():.4f}  "
          f"Mean: {df['mape'].mean():.4f}")

    return df


# ---------------------------------------------------------------------------
# Stage 2: Validation — top K configs × extra seeds
# ---------------------------------------------------------------------------

def stage2_validation(
    dm: DataManager,
    screening_df: pd.DataFrame,
    top_k: int = 20,
    extra_seeds: int = 5,
    resume: bool = True,
) -> pd.DataFrame:
    """Run top K configs with additional seeds to validate screening results."""
    arch = "gru_baseline"

    # Pick top K by MAPE
    valid = screening_df.dropna(subset=["mape"]).sort_values("mape")
    top_configs = valid.head(top_k)
    top_indices = top_configs["config_idx"].astype(int).tolist()

    print(f"\n{'='*60}")
    print(f"STAGE 2: Validation — top {top_k} configs × {extra_seeds} extra seeds")
    print(f"{'='*60}")
    print(f"Top {top_k} config indices: {top_indices}")
    print(f"MAPE range of selected configs: {top_configs['mape'].min():.4f} - {top_configs['mape'].max():.4f}\n")

    all_configs = sample_hp_configs(100, seed=2024)
    # Seeds: 100-104 (avoid overlap with screening seed which uses config_idx 0-99)
    seeds = list(range(100, 100 + extra_seeds))

    t_start = time.time()
    total = top_k * extra_seeds
    run_idx = 0

    for cidx in top_indices:
        config = all_configs[cidx]
        for seed in seeds:
            run_idx += 1

            if resume and load_result_seeded(arch, cidx, seed) is not None:
                continue

            print(f"[{run_idx:3d}/{total}]  cfg={cidx:02d}  seed={seed}  ...", end="", flush=True)

            # Override the seed
            config_with_seed = {**config, "config_idx": cidx}
            set_seeds(seed)

            try:
                result = run_single(arch, config_with_seed, dm)
                result["seed"] = seed
                save_result_seeded(result, seed)
                print(f"  MAPE={result['mape']:.4f}  t={result['training_time']:.0f}s")
            except Exception as e:
                print(f"  FAILED: {e}")

        # Print per-config summary
        config_mapes = []
        # Include screening seed (seed = cidx)
        screening_r = load_result(arch, cidx)
        if screening_r and not np.isnan(screening_r.get("mape", float("nan"))):
            config_mapes.append(screening_r["mape"])
        for seed in seeds:
            r = load_result_seeded(arch, cidx, seed)
            if r and not np.isnan(r.get("mape", float("nan"))):
                config_mapes.append(r["mape"])

        if config_mapes:
            print(f"  cfg={cidx:02d}: {len(config_mapes)} seeds, "
                  f"MAPE={np.mean(config_mapes):.4f} ± {np.std(config_mapes):.4f}  "
                  f"(range {np.min(config_mapes):.4f}-{np.max(config_mapes):.4f})")

    # Aggregate all results
    rows = []
    for cidx in top_indices:
        # Screening seed
        r = load_result(arch, cidx)
        if r:
            r["seed"] = cidx
            r["stage"] = "screening"
            rows.append(r)
        # Validation seeds
        for seed in seeds:
            r = load_result_seeded(arch, cidx, seed)
            if r:
                r["stage"] = "validation"
                rows.append(r)

    val_df = pd.DataFrame(rows)
    val_df.to_csv(MULTISEED_DIR / "phase2_stage2_validation.csv", index=False)

    # Summary per config
    summary_rows = []
    for cidx in top_indices:
        config_rows = val_df[val_df["config_idx"] == cidx]
        mapes = config_rows["mape"].dropna()
        if len(mapes) > 0:
            summary_rows.append({
                "config_idx": cidx,
                "n_seeds": len(mapes),
                "mean_mape": mapes.mean(),
                "std_mape": mapes.std(),
                "min_mape": mapes.min(),
                "max_mape": mapes.max(),
                "screening_mape": load_result(arch, cidx).get("mape") if load_result(arch, cidx) else None,
                "learning_rate": all_configs[cidx]["learning_rate"],
                "dropout_rate": all_configs[cidx]["dropout_rate"],
                "gru_units": all_configs[cidx]["gru_units"],
            })

    summary = pd.DataFrame(summary_rows).sort_values("mean_mape")
    summary.to_csv(MULTISEED_DIR / "phase2_top20_summary.csv", index=False)

    elapsed = time.time() - t_start
    print(f"\nStage 2 complete: {run_idx} runs, {elapsed/60:.1f} min")
    print(f"\nTop 5 configs by validated mean MAPE:")
    print(summary.head().to_string(index=False))

    return val_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Multi-Seed Validation")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="Run only stage 1 or 2 (default: both)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top configs to validate (default: 20)")
    parser.add_argument("--extra-seeds", type=int, default=5,
                        help="Additional seeds per top config (default: 5)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    resume = not args.no_resume

    # Load data once
    tri_path = str(DATA_DIR / "triangle_sample.csv")
    co_path = str(DATA_DIR / "triangle_company_info.csv")
    print("Loading data...")
    dm = DataManager(tri_path, co_path)
    dm.load()
    dm.prepare()
    print(f"  vocab_size = {dm.vocab_size}  LOB = {PHASE2_LOB}\n")

    if args.stage is None or args.stage == 1:
        screening_df = stage1_screening(dm, resume=resume)
    else:
        # Load existing screening results
        screening_path = RESULTS_DIR / "phase2_stage1_screening.csv"
        if not screening_path.exists():
            # Fall back to original phase2 summary
            screening_path = RESULTS_DIR / "phase2_summary.csv"
        screening_df = pd.read_csv(screening_path)
        screening_df = screening_df[screening_df["arch"] == "gru_baseline"]

    if args.stage is None or args.stage == 2:
        stage2_validation(dm, screening_df, top_k=args.top_k,
                          extra_seeds=args.extra_seeds, resume=resume)
