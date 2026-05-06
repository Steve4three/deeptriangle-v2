"""
Analysis and figure generation for DeepTriangle v2.

Loads Phase 1 and Phase 2 results and produces:

  Figure 1 : Grouped box plot — MAPE by method × LOB
  Figure 2 : LaTeX summary table
  Figure 3 : Heatmap — % improvement over Chain-Ladder by arch × LOB
  Figure 4a: Random Forest feature importance (Phase 2) — bar chart
  Figure 7 : Temporal robustness

Statistical tests
-----------------
  - Welch's t-test (pairwise across architecture pairs) per LOB
  - Bonferroni correction for multiple comparisons (3 pairwise tests × 4 LOBs)

Usage
-----
    python analyze_results.py
    python analyze_results.py --output-dir results/figures
    python analyze_results.py --no-phase2       # skip Phase 2 RF importance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

RESULTS_DIR = Path(os.environ.get("DEEPTRIANGLE_RESULTS", str(PROJECT_DIR / "results")))
PHASE1_DIR = RESULTS_DIR / "phase1"
PHASE2_DIR = RESULTS_DIR / "phase2"
PHASE2_WC_DIR = PHASE2_DIR / "workers_compensation"

# Nice display names
ARCH_LABELS = {
    "gru_baseline": "GRU Baseline",
    "gru_attention": "GRU + Attention (masked)",
    "gru_attention_unmasked": "GRU + Attention (unmasked)",
}
LOB_LABELS = {
    "workers_compensation": "Workers Comp",
    "commercial_auto": "Commercial Auto",
    "private_passenger_auto": "Priv. Pass. Auto",
    "other_liability": "Other Liability",
}
METHOD_LABELS = {
    "mack": "Mack CL",
    "odp": "ODP Bootstrap",
    "bf": "BF",
    "gru_baseline": "GRU Baseline",
    "gru_attention": "GRU + Attention (masked)",
    "gru_attention_unmasked": "GRU + Attention (unmasked)",
}
ARCH_COLORS = {
    "gru_baseline": "#4C72B0",
    "gru_attention": "#DD8452",
    "gru_attention_unmasked": "#55A868",
}
BENCHMARK_COLORS = {
    "mack": "#C44E52",
    "odp": "#8172B2",
    "bf": "#937860",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase1(phase1_dir: Path = PHASE1_DIR) -> pd.DataFrame:
    """Load all Phase 1 per-run JSON files into a DataFrame."""
    summary_csv = phase1_dir / "phase1_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        print(f"[Load] Phase 1: {len(df)} runs from {summary_csv}")
        return df

    # Fall back to scanning individual JSONs
    rows = []
    for json_path in sorted(phase1_dir.rglob("run_*.json")):
        with open(json_path) as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows)
    print(f"[Load] Phase 1: {len(df)} runs from individual JSON files")
    return df


def load_phase2(phase2_dir: Path = PHASE2_WC_DIR) -> pd.DataFrame:
    """Load WC Phase 2 per-run JSON files into a DataFrame.

    Supports both the current organized layout
    ``results/phase2/workers_compensation`` and the earlier flat
    ``results/phase2`` layout for backward compatibility.
    """
    candidate_dirs = [phase2_dir]
    if phase2_dir != PHASE2_DIR:
        candidate_dirs.append(PHASE2_DIR)

    for candidate in candidate_dirs:
        summary_csv = candidate / "phase2_summary.csv"
        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            print(f"[Load] Phase 2: {len(df)} runs from {summary_csv}")
            return df

    rows = []
    for candidate in candidate_dirs:
        for json_path in sorted((candidate / "gru_baseline").glob("hp_*.json")):
            with open(json_path) as f:
                rows.append(json.load(f))
        if rows:
            break
    df = pd.DataFrame(rows)
    print(f"[Load] Phase 2: {len(df)} runs from individual JSON files")
    return df


def load_benchmarks(results_dir: Path = RESULTS_DIR) -> Optional[pd.DataFrame]:
    """Load benchmark results CSV (produced by benchmarks.py)."""
    for bench_csv in [results_dir / "phase1" / "benchmark_results.csv", results_dir / "benchmark_results.csv"]:
        if bench_csv.exists():
            df = pd.read_csv(bench_csv)
            print(f"[Load] Benchmarks: {len(df)} rows from {bench_csv}")
            return df
    print("[Load] No benchmark_results.csv found — benchmark figures will be skipped")
    return None


def load_rf_importance(
    phase2_dir: Path = PHASE2_WC_DIR,
) -> Dict[str, Dict[str, float]]:
    """Load WC Phase 2 RF feature importance JSONs."""
    importance = {}
    candidate_dirs = [phase2_dir]
    if phase2_dir != PHASE2_DIR:
        candidate_dirs.append(PHASE2_DIR)
    for arch in ("gru_baseline", "gru_attention", "gru_attention_unmasked"):
        for candidate in candidate_dirs:
            path = candidate / f"{arch}_rf_importance.json"
            if path.exists():
                with open(path) as f:
                    importance[arch] = json.load(f)
                break
    return importance


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def pairwise_welch_tests(
    df: pd.DataFrame,
    archs: List[str],
    lobs: List[str],
    metric: str = "mape",
) -> pd.DataFrame:
    """
    Pairwise Welch's t-tests between architectures, per LOB.

    Applies Bonferroni correction across all tests (n_arch_pairs × n_lobs).

    Parameters
    ----------
    df     : Phase 1 DataFrame with columns [arch, lob, mape, ...]
    archs  : list of architecture names
    lobs   : list of LOB names
    metric : column to test (default 'mape')

    Returns
    -------
    pd.DataFrame with columns:
        [lob, arch1, arch2, t_stat, p_value, p_adjusted, significant]
    """
    pairs = list(combinations(archs, 2))
    n_tests = len(pairs) * len(lobs)
    rows = []

    for lob in lobs:
        for arch1, arch2 in pairs:
            x1 = df[(df["arch"] == arch1) & (df["lob"] == lob)][metric].dropna().values
            x2 = df[(df["arch"] == arch2) & (df["lob"] == lob)][metric].dropna().values
            if len(x1) < 2 or len(x2) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False)
            rows.append({
                "lob": lob,
                "arch1": arch1,
                "arch2": arch2,
                "mean1": float(np.mean(x1)),
                "mean2": float(np.mean(x2)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "n1": len(x1),
                "n2": len(x2),
            })

    if not rows:
        return pd.DataFrame()

    test_df = pd.DataFrame(rows)
    # Bonferroni correction
    test_df["p_adjusted"] = (test_df["p_value"] * n_tests).clip(upper=1.0)
    test_df["significant"] = test_df["p_adjusted"] < 0.05

    return test_df




# ---------------------------------------------------------------------------
# Figure 1: Grouped box plot
# ---------------------------------------------------------------------------

def plot_boxplot_mape(
    phase1_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    out_path: Path,
    lobs: Optional[List[str]] = None,
    metric: str = "mape",
) -> None:
    """
    Grouped box plot of MAPE distributions by method × LOB.

    Neural arch distributions are shown as box plots (50 seeds each).
    Benchmark point estimates are shown as horizontal markers.
    """
    if lobs is None:
        lobs = sorted(phase1_df["lob"].unique())

    archs = [a for a in ("gru_baseline", "gru_attention", "gru_attention_unmasked")
              if a in phase1_df["arch"].unique()]

    n_lobs = len(lobs)
    fig, axes = plt.subplots(1, n_lobs, figsize=(4.5 * n_lobs, 5), sharey=True)
    if n_lobs == 1:
        axes = [axes]

    for ax, lob in zip(axes, lobs):
        lob_df = phase1_df[phase1_df["lob"] == lob]

        positions = np.arange(len(archs))
        box_data = [lob_df[lob_df["arch"] == a][metric].dropna().values for a in archs]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, arch in zip(bp["boxes"], archs):
            patch.set_facecolor(ARCH_COLORS.get(arch, "#888888"))
            patch.set_alpha(0.75)

        # Add benchmark markers
        if benchmark_df is not None:
            bench_lob = benchmark_df[benchmark_df["lob"] == lob]
            bench_methods = [m for m in ("mack", "odp", "bf")
                             if m in bench_lob["method"].values]
            for bm in bench_methods:
                bm_val = bench_lob[bench_lob["method"] == bm][metric].values
                if len(bm_val) > 0 and np.isfinite(bm_val[0]):
                    ax.axhline(
                        bm_val[0],
                        color=BENCHMARK_COLORS.get(bm, "gray"),
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.8,
                        label=METHOD_LABELS.get(bm, bm),
                    )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [ARCH_LABELS.get(a, a) for a in archs],
            rotation=25,
            ha="right",
            fontsize=9,
        )
        ax.set_title(LOB_LABELS.get(lob, lob), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel("MAPE", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    handles = [
        plt.Line2D([0], [0], color=BENCHMARK_COLORS.get(m, "gray"),
                   linestyle="--", linewidth=1.5,
                   label=METHOD_LABELS.get(m, m))
        for m in ("mack", "odp", "bf")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        title="Benchmarks",
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 1] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: LaTeX summary table
# ---------------------------------------------------------------------------

def generate_latex_table(
    phase1_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    out_path: Path,
    metric: str = "mape",
) -> None:
    """
    Generate a LaTeX-formatted summary table.

    Rows: methods (neural archs + benchmarks)
    Columns: LOBs
    Cells: mean ± std for neural arches, single value for benchmarks
    Bold: best performer per LOB
    """
    lobs = sorted(phase1_df["lob"].unique())
    archs = [a for a in ("gru_baseline", "gru_attention", "gru_attention_unmasked")
              if a in phase1_df["arch"].unique()]

    lob_headers = " & ".join([f"\\textbf{{{LOB_LABELS.get(l, l)}}}" for l in lobs])

    # Collect all cell values to determine bold
    cell_means: Dict[str, Dict[str, float]] = {}

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Mean MAPE (\\%) by method and line of business. "
                 "Neural architecture results reported as mean $\\pm$ std across 50 seeds. "
                 "\\textbf{Bold} = best per column. $^*$ = statistically significant "
                 "improvement over GRU Baseline (Bonferroni-corrected Welch $t$-test, $p<0.05$).}")
    lines.append("\\label{tab:phase1_results}")
    lines.append(f"\\begin{{tabular}}{{l{'c' * len(lobs)}}}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Method}} & {lob_headers} \\\\")
    lines.append("\\midrule")

    # Neural architectures
    for arch in archs:
        arch_df = phase1_df[phase1_df["arch"] == arch]
        cells = []
        for lob in lobs:
            vals = arch_df[arch_df["lob"] == lob][metric].dropna().values
            if len(vals) > 0:
                m = np.mean(vals) * 100
                s = np.std(vals) * 100
                cell_means.setdefault(arch, {})[lob] = np.mean(vals)
                cells.append(f"${m:.2f} \\pm {s:.2f}$")
            else:
                cells.append("---")

        # Add significance marker vs. baseline
        if arch != "gru_baseline" and test_df is not None and len(test_df) > 0:
            row_label = ARCH_LABELS.get(arch, arch)
            sig_markers = []
            for lob in lobs:
                sub = test_df[
                    (test_df["lob"] == lob)
                    & (
                        ((test_df["arch1"] == arch) & (test_df["arch2"] == "gru_baseline"))
                        | ((test_df["arch1"] == "gru_baseline") & (test_df["arch2"] == arch))
                    )
                ]
                is_sig = len(sub) > 0 and bool(sub["significant"].iloc[0])
                is_better = (
                    len(sub) > 0 and
                    cell_means.get(arch, {}).get(lob, 1.0) <
                    cell_means.get("gru_baseline", {}).get(lob, 1.0)
                )
                sig_markers.append("$^*$" if (is_sig and is_better) else "")

            # Annotate cells
            annotated = []
            for cell, marker in zip(cells, sig_markers):
                annotated.append(cell + marker)
            cells = annotated
        else:
            row_label = ARCH_LABELS.get(arch, arch)

        row_label = ARCH_LABELS.get(arch, arch)
        lines.append(f"{row_label} & " + " & ".join(cells) + " \\\\")

    lines.append("\\midrule")

    # Benchmark methods
    if benchmark_df is not None:
        for bm in ("mack", "odp", "bf"):
            bm_rows = benchmark_df[benchmark_df["method"] == bm]
            if bm_rows.empty:
                continue
            cells = []
            for lob in lobs:
                lob_row = bm_rows[bm_rows["lob"] == lob]
                if len(lob_row) > 0:
                    v = float(lob_row[metric].iloc[0]) * 100
                    cell_means.setdefault(bm, {})[lob] = float(lob_row[metric].iloc[0])
                    cells.append(f"${v:.2f}$")
                else:
                    cells.append("---")
            row_label = METHOD_LABELS.get(bm, bm)
            lines.append(f"{row_label} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"[Fig 2] LaTeX table saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4a: RF Feature Importance
# ---------------------------------------------------------------------------

def plot_rf_importance(
    importance_dict: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    """
    Bar chart of RF feature importance per architecture (Phase 2).
    """
    if not importance_dict:
        print("[Fig 4a] No RF importance data — skipping")
        return

    archs = list(importance_dict.keys())
    n_archs = len(archs)
    # All archs should have same features
    all_feats = list(list(importance_dict.values())[0].keys())

    fig, axes = plt.subplots(1, n_archs, figsize=(5 * n_archs, 4), sharey=True)
    if n_archs == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        imp = importance_dict[arch]
        feats = sorted(imp, key=lambda k: -imp[k])
        vals = [imp[f] for f in feats]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feats)))[::-1]

        bars = ax.barh(feats, vals, color=colors, edgecolor="white")
        ax.set_xlabel("Importance", fontsize=10)
        if n_archs > 1:
            ax.set_title(ARCH_LABELS.get(arch, arch), fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
            )
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 4a] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Attention bimodal collapse histogram
# ---------------------------------------------------------------------------

# Naive mean predictor MAPE by LOB (the "collapse sentinel")
COLLAPSE_SENTINEL = {
    "workers_compensation": 0.249,
    "private_passenger_auto": 0.165,
}

# Short labels for fig2 (matching the original figure style)
_FIG2_LABELS = {
    "gru_baseline": "GRU Baseline",
    "gru_attention": "GRU + Attn (masked)",
    "gru_attention_unmasked": "GRU + Attention",
}


def plot_attention_bimodal(
    phase1_df: pd.DataFrame,
    out_path: Path,
    metric: str = "mape",
) -> None:
    """Histogram of MAPE showing bimodal attention collapse with sentinel line."""
    archs = [a for a in ("gru_baseline", "gru_attention_unmasked", "gru_attention")
             if a in phase1_df["arch"].unique()]
    # WC first, PPA second (matching paper figure order)
    _LOB_ORDER = ["workers_compensation", "private_passenger_auto"]
    lobs = [l for l in _LOB_ORDER if l in phase1_df["lob"].unique()]

    fig, axes = plt.subplots(1, len(lobs), figsize=(5 * len(lobs), 4), sharey=True)
    if len(lobs) == 1:
        axes = [axes]

    for ax, lob in zip(axes, lobs):
        for arch in archs:
            vals = phase1_df[
                (phase1_df["arch"] == arch) & (phase1_df["lob"] == lob)
            ][metric].dropna().values
            if len(vals) > 0:
                ax.hist(
                    vals, bins=20, alpha=0.5,
                    color=ARCH_COLORS.get(arch, "#888888"),
                    label=_FIG2_LABELS.get(arch, arch),
                    edgecolor="white",
                )

        # Collapse sentinel line
        sentinel = COLLAPSE_SENTINEL.get(lob)
        if sentinel is not None:
            ax.axvline(sentinel, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.annotate(
                f"Collapse\nsentinel\n({sentinel * 100:.1f}%)",
                xy=(sentinel, ax.get_ylim()[1] * 0.85),
                fontsize=8, color="red", ha="left",
                xytext=(5, 0), textcoords="offset points",
            )

        ax.set_xlabel("MAPE (%)", fontsize=10)
        _FIG2_LOB_LABELS = {
            "workers_compensation": "Workers' Compensation",
            "private_passenger_auto": "Private Passenger Auto",
        }
        ax.set_title(
            _FIG2_LOB_LABELS.get(lob, LOB_LABELS.get(lob, lob)),
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    if len(axes) > 0:
        axes[0].set_ylabel("Count (out of 50 seeds)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 2] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 7: Temporal robustness
# ---------------------------------------------------------------------------

ARCH_LABELS_TEMPORAL = {
    "gru_baseline": "GRU Baseline",
    "gru_attention": "GRU + Attention (masked)",
    "gru_attention_unmasked": "GRU + Attention (unmasked)",
}
ARCH_COLORS_TEMPORAL = {
    "gru_baseline": "#2196F3",
    "gru_attention": "#FF9800",
    "gru_attention_unmasked": "#4CAF50",
}


def plot_temporal_robustness(
    out_path: Path = None,
    metric: str = "mape",
) -> None:
    """
    Generate Figure 7: temporal robustness grouped bar chart.

    Reads results/temporal/temporal_summary.csv and benchmarks.json.
    """
    temporal_dir = RESULTS_DIR / "temporal"
    summary_path = temporal_dir / "temporal_summary.csv"
    bm_path = temporal_dir / "benchmarks.json"

    if not summary_path.exists():
        print(f"[Fig7] temporal_summary.csv not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    if "arch" in df.columns:
        df = df[df["arch"] == "gru_baseline"].copy()
    valid = df.dropna(subset=[metric])
    if valid.empty:
        print("[Fig7] No valid temporal results")
        return

    # Load benchmarks
    benchmarks = {}
    if bm_path.exists():
        with open(bm_path) as f:
            benchmarks = json.load(f)

    # Aggregate: mean +/- std by window x arch
    agg = valid.groupby(["window", "arch"])[metric].agg(
        mean="mean", std="std", count="count"
    ).reset_index()

    windows = sorted(agg["window"].unique())
    archs = ["gru_baseline", "gru_attention", "gru_attention_unmasked"]
    archs = [a for a in archs if a in agg["arch"].unique()]

    n_windows = len(windows)
    n_archs = len(archs)
    bar_width = 0.18
    x = np.arange(n_windows)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot bars for each architecture
    for i, arch in enumerate(archs):
        arch_data = agg[agg["arch"] == arch].set_index("window")
        means = [arch_data.loc[w, "mean"] * 100 if w in arch_data.index else 0 for w in windows]
        stds = [arch_data.loc[w, "std"] * 100 if w in arch_data.index else 0 for w in windows]
        offset = (i - (n_archs - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=3,
            label=ARCH_LABELS_TEMPORAL.get(arch, arch),
            color=ARCH_COLORS_TEMPORAL.get(arch, f"C{i}"),
            edgecolor="white", linewidth=0.5,
            alpha=0.85,
        )

    # Add Mack CL horizontal markers per window
    for j, w in enumerate(windows):
        bm = benchmarks.get(w, {})
        mack_val = bm.get(metric, None)
        if mack_val is not None and not np.isnan(mack_val):
            line_x_start = j - (n_archs * bar_width) / 2 - 0.02
            line_x_end = j + (n_archs * bar_width) / 2 + 0.02
            ax.hlines(
                mack_val * 100, line_x_start, line_x_end,
                colors="red", linestyles="--", linewidth=1.5,
                label="Mack CL" if j == 0 else None,
            )

    # Window labels with train/test info
    window_labels = {
        "W1": "W1\n(train$\\leq$2006,\ntest 2009)",
        "W2": "W2\n(train$\\leq$2007,\ntest 2010)",
        "W3": "W3\n(train$\\leq$2008,\ntest 2011)",
    }

    ax.set_xticks(x)
    ax.set_xticklabels([window_labels.get(w, w) for w in windows], fontsize=8)
    ax.set_ylabel(f"{metric.upper()} (%)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # y-axis formatting
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[Fig7] Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master analysis function
# ---------------------------------------------------------------------------

def run_analysis(
    output_dir: Optional[Path] = None,
    run_phase2_figs: bool = True,
    metric: str = "mape",
) -> None:
    """
    Load all results and generate all figures and tables.

    Parameters
    ----------
    output_dir     : Path  where to save figures (default: results/figures/)
    run_phase2_figs: bool  whether to generate Phase 2 figures
    metric         : str   'mape' or 'rmspe'
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Analysis] Output directory: {output_dir}\n")

    # --- Load data ---
    phase1_df = load_phase1()
    benchmark_df = load_benchmarks()

    if phase1_df.empty:
        print("[Analysis] No Phase 1 results found. Run run_phase1.py first.")
        return

    lobs = sorted(phase1_df["lob"].unique())
    archs = [a for a in ("gru_baseline", "gru_attention", "gru_attention_unmasked")
              if a in phase1_df["arch"].unique()]

    # --- Summary statistics ---
    print("\n=== Phase 1 Summary ===")
    summary = phase1_df.groupby(["arch", "lob"])[metric].agg(
        ["mean", "std", "median", "count"]
    ).round(4)
    print(summary.to_string())

    # --- Statistical tests ---
    print("\n=== Welch's t-tests (Bonferroni-corrected) ===")
    test_df = pairwise_welch_tests(phase1_df, archs, lobs, metric=metric)
    if not test_df.empty:
        print(test_df[["lob", "arch1", "arch2", "p_value", "p_adjusted", "significant"]].to_string())
        test_csv = PHASE1_DIR / "phase1_ttest_results.csv"
        test_df.to_csv(test_csv, index=False)
        print(f"T-test results saved to {test_csv}")
    else:
        test_df = None
        print("Insufficient data for t-tests")

    # --- Figures ---
    plot_boxplot_mape(
        phase1_df,
        benchmark_df,
        out_path=output_dir / "mape_boxplot.pdf",
        lobs=lobs,
        metric=metric,
    )
    plot_attention_bimodal(
        phase1_df,
        out_path=output_dir / "attention_bimodal.pdf",
        metric=metric,
    )
    generate_latex_table(
        phase1_df,
        benchmark_df,
        test_df,
        out_path=output_dir / "tab1_results.tex",
        metric=metric,
    )

    if run_phase2_figs:
        phase2_df = load_phase2()
        importance_dict = load_rf_importance()

        if not phase2_df.empty:
            plot_rf_importance(
                importance_dict,
                out_path=output_dir / "rf_importance.pdf",
            )
        else:
            print("[Analysis] No Phase 2 data found — skipping Phase 2 figures")

    plot_temporal_robustness(
        out_path=output_dir / "temporal_robustness.pdf",
        metric=metric,
    )
    print(f"\n[Analysis] Complete. All outputs in {output_dir}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate analysis figures and tables for DeepTriangle v2"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: results/figures/)",
    )
    parser.add_argument(
        "--no-phase2",
        action="store_true",
        help="Skip Phase 2 RF importance figure",
    )
    parser.add_argument(
        "--metric",
        choices=("mape", "rmspe"),
        default="mape",
        help="Metric to analyze (default: mape)",
    )
    args = parser.parse_args()

    run_analysis(
        output_dir=args.output_dir,
        run_phase2_figs=not args.no_phase2,
        metric=args.metric,
    )
