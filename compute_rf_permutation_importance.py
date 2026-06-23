#!/usr/bin/env python3
"""
Compute a permutation-importance robustness check for Phase 2.

The manuscript reports impurity-based Random Forest feature importance for the
Phase 2 hyperparameter screen. Because impurity importance can favor continuous
or high-cardinality predictors, this script refits the same Random Forest
surrogate on the shipped Phase 2 summaries and computes permutation importance
using the decrease in negative-MAE score.

No neural network training is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score


FEATURE_COLS = [
    "dropout_rate",
    "learning_rate",
    "batch_size",
    "max_epochs",
    "gru_units",
    "dense_units",
]

LOBS = ["workers_compensation", "private_passenger_auto"]


def _fit_and_score(summary_path: Path, n_repeats: int, seed: int) -> Dict[str, object]:
    df = pd.read_csv(summary_path).dropna(subset=["mape"] + FEATURE_COLS)
    if df.empty:
        raise ValueError(f"No usable rows in {summary_path}")

    x = df[FEATURE_COLS].values
    y = df["mape"].values

    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x, y)

    permutation = permutation_importance(
        rf,
        x,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    total_decrease = float(permutation.importances_mean.sum())
    if total_decrease <= 0:
        raise ValueError(f"Non-positive permutation-importance total for {summary_path}")

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(rf, x, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

    return {
        "n_configs": int(len(df)),
        "train_r2": float(rf.score(x, y)),
        "cv_mae_mean": float(-cv_scores.mean()),
        "cv_mae_std": float(cv_scores.std()),
        "impurity_importance": {
            feature: float(value)
            for feature, value in zip(FEATURE_COLS, rf.feature_importances_)
        },
        "permutation_mean_decrease_mae": {
            feature: float(value)
            for feature, value in zip(FEATURE_COLS, permutation.importances_mean)
        },
        "permutation_std_decrease_mae": {
            feature: float(value)
            for feature, value in zip(FEATURE_COLS, permutation.importances_std)
        },
        "permutation_normalized": {
            feature: float(value / total_decrease)
            for feature, value in zip(FEATURE_COLS, permutation.importances_mean)
        },
    }


def compute(results_dir: Path, n_repeats: int, seed: int) -> Dict[str, object]:
    phase2_dir = results_dir / "phase2"
    output: Dict[str, object] = {
        "method": (
            "RandomForestRegressor surrogate fitted to Phase 2 screening "
            "configurations; permutation_importance with "
            "scoring=neg_mean_absolute_error. Normalized values divide each "
            "mean decrease by the sum of mean decreases within LOB."
        ),
        "feature_order": FEATURE_COLS,
        "n_repeats": int(n_repeats),
        "random_state": int(seed),
        "lobs": {},
    }

    lobs: Dict[str, object] = {}
    for lob in LOBS:
        summary_path = phase2_dir / lob / "phase2_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing Phase 2 summary: {summary_path}")
        lobs[lob] = _fit_and_score(summary_path, n_repeats=n_repeats, seed=seed)

    output["lobs"] = lobs
    return output


def _print_summary(payload: Dict[str, object]) -> None:
    lobs = payload["lobs"]
    assert isinstance(lobs, dict)
    for lob, result in lobs.items():
        assert isinstance(result, dict)
        values = result["permutation_normalized"]
        assert isinstance(values, dict)
        print(f"\n{lob}")
        for feature, value in sorted(values.items(), key=lambda item: -float(item[1])):
            print(f"  {feature:14s} {float(value):.4f}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute RF permutation-importance robustness check for Phase 2.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory containing phase2/*/phase2_summary.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "results/diagnostics/rf_permutation_importance_check.json."
        ),
    )
    parser.add_argument("--n-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    results_dir = args.results_dir.resolve()
    out_path = args.out or (results_dir / "diagnostics" / "rf_permutation_importance_check.json")

    payload = compute(results_dir=results_dir, n_repeats=args.n_repeats, seed=args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if not args.quiet:
        print(f"Wrote {out_path}")
        _print_summary(payload)


if __name__ == "__main__":
    main()
