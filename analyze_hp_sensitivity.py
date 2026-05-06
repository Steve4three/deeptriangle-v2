"""
Generate the Phase 2 partial-dependence figure used in the paper.

Reads WC GRU Baseline Phase 2 screening results and writes:

  results/figures/partial_dependence.png

The earlier diagnostic-only plots were intentionally removed from the
replication pipeline so `replicate.py` produces only paper-facing figures.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.resolve()
BASE_RESULTS = Path(os.environ.get("DEEPTRIANGLE_RESULTS", str(PROJECT_DIR / "results")))

HP_NAMES = [
    "learning_rate",
    "dropout_rate",
    "gru_units",
    "dense_units",
    "batch_size",
    "max_epochs",
]
HP_LABELS = {
    "learning_rate": "Learning Rate",
    "dropout_rate": "Dropout Rate",
    "gru_units": "GRU Units",
    "dense_units": "Dense Units",
    "batch_size": "Batch Size",
    "max_epochs": "Max Epochs",
}
CONTINUOUS_HPS = {"learning_rate", "dropout_rate"}


def load_results() -> pd.DataFrame:
    """Load WC Phase 2 screening results from the organized public layout."""
    patterns = [
        BASE_RESULTS / "phase2" / "workers_compensation" / "gru_baseline" / "hp_*.json",
        BASE_RESULTS / "phase2" / "gru_baseline" / "hp_*.json",  # legacy fallback
    ]
    files = []
    for pattern in patterns:
        files = sorted(glob.glob(str(pattern)))
        if files:
            break

    rows = []
    for file_name in files:
        with open(file_name) as fh:
            row = json.load(fh)
        if row.get("mape") is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} Phase 2 WC GRU Baseline screening results")
    return df


def plot_partial_dependence(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create the paper-facing partial-dependence figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, hp in enumerate(HP_NAMES):
        ax = axes[i]
        clean = df.dropna(subset=[hp, "mape"]).copy()
        x = clean[hp].to_numpy()
        y = clean["mape"].to_numpy()

        if hp in CONTINUOUS_HPS:
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            ax.scatter(x_sorted, y_sorted, alpha=0.45, s=28, c="#4C72B0", edgecolors="white", linewidths=0.4)

            window = max(5, len(x_sorted) // 10)
            if len(x_sorted) > window:
                from scipy.ndimage import uniform_filter1d

                trend = uniform_filter1d(y_sorted.astype(float), window)
                ax.plot(x_sorted, trend, color="#C44E52", linewidth=2.0, label="Moving average")
                ax.legend(fontsize=8, framealpha=0.9)

            if hp == "learning_rate":
                ax.set_xscale("log")
        else:
            groups = clean.groupby(hp)["mape"]
            positions = sorted(clean[hp].unique())
            box_data = [groups.get_group(p).to_numpy() for p in positions]
            bp = ax.boxplot(box_data, positions=range(len(positions)), widths=0.6, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#4C72B0")
                patch.set_alpha(0.6)
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels([str(int(p)) for p in positions])

        ax.set_xlabel(HP_LABELS[hp], fontsize=10)
        ax.set_ylabel("MAPE", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.set_title(HP_LABELS[hp], fontsize=11, fontweight="bold")
        ax.grid(linestyle=":", alpha=0.35)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "partial_dependence.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Partial dependence] Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/figures", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    df = load_results()
    if df.empty:
        raise SystemExit("No Phase 2 WC results found")

    print(f"Dataset: {len(df)} runs, MAPE range [{df.mape.min():.4f}, {df.mape.max():.4f}]")
    plot_partial_dependence(df, output_dir)


if __name__ == "__main__":
    main()
