"""
Build the Reviewer 2 attention-collapse diagnostic figure.

By default, this script can retrain two representative Workers' Compensation
Phase-1 attention runs:
  - seed 0: collapsed at the naive-mean sentinel
  - seed 26: converged lower-mode run

For the revision figure, use --from-archive-artifacts. That path loads
representative archived loss curves and archived Phase-1 attention weights,
which avoids relying on current-runtime seed reproducibility for the
lower-mode example.

It saves a compact 2x2 figure showing training/validation loss curves and
actual-vs-predicted ultimate paid-loss ratios for the two runs.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_prep import DataManager
from evaluate import (
    extract_observed_cumulative,
    extract_ultimate_actuals,
    predict_paid_output,
)
from models import build_model
from run_phase1 import FIXED_HP, set_seeds
from train import TrainConfig, history_summary, train_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PRIVATE_DATA = (
    PROJECT_DIR
    / "archive"
    / "2026-05-05_public_sync"
    / "private_only"
    / "data"
)
DEFAULT_OUT = PROJECT_DIR / "results" / "revision_diagnostics"
DEFAULT_ARCHIVE_RESULTS = (
    PROJECT_DIR
    / "archive"
    / "2026-05-05_public_sync"
    / "private_only"
    / "results"
)
DEFAULT_PLOT_DATA = (
    PROJECT_DIR
    / "results"
    / "diagnostics"
    / "attention_collapse_diagnostic_data.json"
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_data(data_dir: Path) -> DataManager:
    tri = data_dir / "triangle_sample.csv"
    co = data_dir / "triangle_company_info.csv"
    if not tri.exists() or not co.exists():
        raise FileNotFoundError(f"Missing proprietary CSVs under {data_dir}")
    dm = DataManager(str(tri), str(co))
    dm.load()
    dm.prepare()
    return dm


def train_attention_run(
    dm: DataManager,
    lob: str,
    seed: int,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_seeds(seed)
    model = build_model(
        "gru_attention",
        vocab_size=dm.vocab_size,
        gru_units=FIXED_HP["gru_units"],
        dropout_rate=FIXED_HP["dropout_rate"],
        dense_units=FIXED_HP["dense_units"],
    )
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
        device=device,
    )
    history, elapsed = train_model(
        model,
        dm.get(lob, "full_training_data"),
        dm.get(lob, "validation_data"),
        config,
    )
    summary = history_summary(history)
    return model, {
        "seed": seed,
        "history": history.history,
        "training_time": elapsed,
        **summary,
    }


def ultimate_prediction_frame(
    model: torch.nn.Module,
    dm: DataManager,
    lob: str,
    device: str,
    accident_year_range: tuple[int, int] = (2002, 2010),
    focus_lag: int = 9,
    outlier_threshold: float = 10.0,
) -> pd.DataFrame:
    test_data = dm.get(lob, "test_data")
    test_meta = dm.get_test_metadata(lob)

    paid_pred_norm = predict_paid_output(
        model,
        test_data["x"],
        device=device,
    ).squeeze(-1)
    paid_pred_dollars = paid_pred_norm * test_meta["earned_premium_net"].values[:, None]

    actual_ult = extract_ultimate_actuals(
        dm.data,
        lob,
        accident_year_range=accident_year_range,
        max_dev_lag=focus_lag,
    )
    observed = extract_observed_cumulative(
        dm.data,
        lob,
        test_calendar_year=2011,
        accident_year_range=accident_year_range,
    )

    actual_lookup = {
        (str(r.group_code), int(r.accident_year)): float(r.ultimate_actual_raw)
        for _, r in actual_ult.iterrows()
    }
    obs_lookup = {
        (str(r.group_code), int(r.accident_year)): (
            float(r.observed_cumulative_raw),
            int(r.last_observed_lag),
        )
        for _, r in observed.iterrows()
    }

    rows = []
    for i, row in test_meta.iterrows():
        gc = str(row["group_code"])
        ay = int(row["accident_year"])
        if not (accident_year_range[0] <= ay <= accident_year_range[1]):
            continue

        key = (gc, ay)
        if key not in actual_lookup:
            continue

        ep = float(row["earned_premium_net"])
        obs_cum, last_obs_lag = obs_lookup.get(key, (0.0, 0))
        n_pred = min(focus_lag - last_obs_lag, paid_pred_dollars.shape[1])
        pred_ult = obs_cum + float(paid_pred_dollars[i, :n_pred].sum())
        actual = actual_lookup[key]

        rows.append(
            {
                "group_code": gc,
                "accident_year": ay,
                "actual_ultimate_ratio": actual / ep,
                "predicted_ultimate_ratio": pred_ult / ep,
            }
        )

    df = pd.DataFrame(rows)
    df = (
        df.groupby(["group_code", "accident_year"], as_index=False)
        .mean(numeric_only=True)
    )
    denom = np.maximum(np.abs(df["actual_ultimate_ratio"].values), 1e-8)
    df["pct_error"] = (
        df["predicted_ultimate_ratio"].values
        - df["actual_ultimate_ratio"].values
    ) / denom
    df = df[np.isfinite(df["pct_error"])]
    df = df[np.abs(df["pct_error"]) <= outlier_threshold].copy()
    return df


def company_mape(df: pd.DataFrame) -> float:
    group_col = "group_code" if "group_code" in df.columns else "company_id"
    by_company = df.groupby(group_col)["pct_error"].apply(lambda x: np.mean(np.abs(x)))
    return float(by_company.mean())


def load_plot_data(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    """Load de-identified diagnostic plot data."""
    with open(path) as f:
        payload = json.load(f)

    runs = payload["runs"]
    pred_frames: dict[str, pd.DataFrame] = {}
    for key, values in payload["predictions"].items():
        df = pd.DataFrame(values)
        denom = np.maximum(np.abs(df["actual_ultimate_ratio"].values), 1e-8)
        df["pct_error"] = (
            df["predicted_ultimate_ratio"].values
            - df["actual_ultimate_ratio"].values
        ) / denom
        pred_frames[key] = df
    return runs, pred_frames


def write_plot_data(
    path: Path,
    runs: dict[str, dict[str, Any]],
    pred_frames: dict[str, pd.DataFrame],
) -> None:
    """Write de-identified plot data so the diagnostic figure is reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "De-identified data for the Reviewer 2 attention-collapse "
            "diagnostic figure. Coordinates are ultimate paid-loss ratios; "
            "company IDs are anonymized and used only to reproduce the "
            "company-mean MAPE annotation."
        ),
        "runs": runs,
        "predictions": {},
    }

    for key, df in pred_frames.items():
        company_codes = sorted(df["group_code"].astype(str).unique())
        company_map = {code: f"G{i:03d}" for i, code in enumerate(company_codes)}
        out = df.copy()
        out["company_id"] = out["group_code"].astype(str).map(company_map)
        payload["predictions"][key] = out[
            ["company_id", "actual_ultimate_ratio", "predicted_ultimate_ratio"]
        ].to_dict(orient="records")

    with open(path, "w") as f:
        json.dump(_jsonify(payload), f, indent=2)


def save_figure(
    runs: dict[str, dict[str, Any]],
    pred_frames: dict[str, pd.DataFrame],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "attention_collapse_diagnostic.pdf"
    png_path = out_dir / "attention_collapse_diagnostic.png"

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    labels = {
        "collapsed": runs["collapsed"].get("label", "Collapsed attention run"),
        "converged": runs["converged"].get("label", "Converged attention run"),
    }
    colors = {
        "collapsed": "#C44E52",
        "converged": "#4C72B0",
    }

    for key in ("collapsed", "converged"):
        hist = runs[key]["history"]
        epochs = np.arange(1, len(hist["loss"]) + 1)
        axes[0, 0].plot(
            epochs,
            hist["loss"],
            color=colors[key],
            linewidth=1.2,
            label=labels[key],
        )
        axes[0, 1].plot(
            epochs,
            hist["val_loss"],
            color=colors[key],
            linewidth=1.2,
            label=labels[key],
        )

    axes[0, 0].set_title("(a) Training loss")
    axes[0, 1].set_title("(b) Validation loss")
    for ax in axes[0]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked MSE")
        ax.set_yscale("log")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.legend(frameon=False)

    all_ratios = pd.concat(
        [
            pred_frames["collapsed"][
                ["actual_ultimate_ratio", "predicted_ultimate_ratio"]
            ],
            pred_frames["converged"][
                ["actual_ultimate_ratio", "predicted_ultimate_ratio"]
            ],
        ],
        axis=0,
    )
    upper = float(
        np.nanpercentile(
            all_ratios[["actual_ultimate_ratio", "predicted_ultimate_ratio"]].values,
            99,
        )
    )
    upper = max(upper, 0.75)

    scatter_specs = [
        (
            "collapsed",
            axes[1, 0],
            runs["collapsed"].get(
                "scatter_title",
                "(c) Collapsed seed: ultimate predictions",
            ),
        ),
        (
            "converged",
            axes[1, 1],
            runs["converged"].get(
                "scatter_title",
                "(d) Converged seed: ultimate predictions",
            ),
        ),
    ]
    for key, ax, title in scatter_specs:
        df = pred_frames[key]
        mape = company_mape(df) * 100
        ax.scatter(
            df["actual_ultimate_ratio"],
            df["predicted_ultimate_ratio"],
            s=8,
            alpha=0.45,
            color=colors[key],
            edgecolors="none",
        )
        ax.plot([0, upper], [0, upper], color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Actual ultimate paid loss ratio")
        ax.set_ylabel("Predicted ultimate paid loss ratio")
        ax.set_title(title)
        ax.text(
            0.03,
            0.95,
            f"MAPE = {mape:.1f}%\nN = {len(df):,} company-years",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.3),
        )
        ax.grid(True, linewidth=0.3, alpha=0.35)

    fig.suptitle(
        "Diagnostic view of attention collapse (Workers' Compensation)",
        fontsize=10,
    )
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def archive_history_runs(archive_results: Path) -> dict[str, dict[str, Any]]:
    """Load representative loss curves from the archived attention HP sweep."""
    summary_path = archive_results / "phase2_attention" / "phase2_attention_summary.csv"
    df = pd.read_csv(summary_path)
    low = df.loc[df["mape"].idxmin()]
    high = df.loc[df["mape"].idxmax()]

    def _row_to_run(row: pd.Series, label: str, scatter_title: str) -> dict[str, Any]:
        return {
            "label": label,
            "scatter_title": scatter_title,
            "config_idx": int(row["config_idx"]),
            "mape": float(row["mape"]),
            "epochs_trained": int(row["epochs_trained"]),
            "best_val_loss": float(row["best_val_loss"]),
            "history": {
                "loss": ast.literal_eval(row["train_loss_curve"]),
                "val_loss": ast.literal_eval(row["val_loss_curve"]),
            },
        }

    return {
        "collapsed": _row_to_run(
            high,
            "Sentinel-mode attention run",
            "(c) Collapsed attention prediction example",
        ),
        "converged": _row_to_run(
            low,
            "Lower-mode attention run",
            "(d) Lower-mode attention prediction example",
        ),
    }


def archive_prediction_frames(
    dm: DataManager,
    lob: str,
    archive_results: Path,
    device: str,
) -> dict[str, pd.DataFrame]:
    """Load archived Phase-1 masked-attention weights and compute predictions."""
    weights_dir = (
        archive_results
        / "phase1"
        / "gru_attention_masked"
        / lob
        / "weights"
    )
    paths = {
        "collapsed": weights_dir / "seed_000.pt",
        "converged": weights_dir / "best_seed_005.pt",
    }

    frames = {}
    for key, path in paths.items():
        model = build_model(
            "gru_attention",
            vocab_size=dm.vocab_size,
            gru_units=FIXED_HP["gru_units"],
            dropout_rate=FIXED_HP["dropout_rate"],
            dense_units=FIXED_HP["dense_units"],
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        frames[key] = ultimate_prediction_frame(model, dm, lob, device)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_PRIVATE_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--archive-results", type=Path, default=DEFAULT_ARCHIVE_RESULTS)
    parser.add_argument("--plot-data", type=Path, default=DEFAULT_PLOT_DATA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lob", default="workers_compensation")
    parser.add_argument(
        "--from-archive-artifacts",
        action="store_true",
        help=(
            "Use archived Phase-2 attention loss curves and archived Phase-1 "
            "masked-attention weights instead of retraining."
        ),
    )
    parser.add_argument(
        "--write-plot-data",
        action="store_true",
        help="When using archive artifacts or retraining, write de-identified plot data.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.from_archive_artifacts and args.plot_data.exists():
        print(f"[diagnostic] Loading de-identified plot data from {args.plot_data}")
        runs, pred_frames = load_plot_data(args.plot_data)
        pdf_path, png_path = save_figure(runs, pred_frames, args.out_dir)
        print(f"[diagnostic] Saved {pdf_path}")
        print(f"[diagnostic] Saved {png_path}")
        return

    print(f"[diagnostic] Loading data from {args.data_dir}")
    dm = load_data(args.data_dir)

    if args.from_archive_artifacts:
        print(f"[diagnostic] Loading archived results from {args.archive_results}")
        runs = archive_history_runs(args.archive_results)
        pred_frames = archive_prediction_frames(
            dm,
            args.lob,
            args.archive_results,
            args.device,
        )
        for key in ("collapsed", "converged"):
            runs[key]["prediction_mape"] = company_mape(pred_frames[key])
            runs[key]["n_company_years"] = int(len(pred_frames[key]))
            pred_frames[key].to_csv(
                args.out_dir / f"{key}_archive_weight_predictions.csv",
                index=False,
            )
            print(
                "[diagnostic] "
                f"{key}: curve_mape={runs[key]['mape']:.6f} "
                f"prediction_mape={runs[key]['prediction_mape']:.6f} "
                f"epochs={runs[key]['epochs_trained']}"
            )

        with open(args.out_dir / "attention_collapse_diagnostic_runs.json", "w") as f:
            json.dump(_jsonify(runs), f, indent=2)
        if args.write_plot_data:
            write_plot_data(args.plot_data, runs, pred_frames)
            print(f"[diagnostic] Saved de-identified plot data to {args.plot_data}")
        pdf_path, png_path = save_figure(runs, pred_frames, args.out_dir)
        print(f"[diagnostic] Saved {pdf_path}")
        print(f"[diagnostic] Saved {png_path}")
        return

    run_specs = {
        "collapsed": 0,
        "converged": 26,
    }
    runs: dict[str, dict[str, Any]] = {}
    pred_frames: dict[str, pd.DataFrame] = {}

    for key, seed in run_specs.items():
        print(f"[diagnostic] Training {key} attention run, seed={seed}")
        model, run = train_attention_run(dm, args.lob, seed, args.device)
        pred = ultimate_prediction_frame(model, dm, args.lob, args.device)
        run["mape"] = company_mape(pred)
        run["n_company_years"] = int(len(pred))
        runs[key] = run
        pred_frames[key] = pred
        pred.to_csv(args.out_dir / f"{key}_seed_{seed:03d}_predictions.csv", index=False)
        print(
            "[diagnostic] "
            f"{key} seed={seed} MAPE={run['mape']:.6f} "
            f"epochs={run['epochs_trained']} "
            f"best_val_loss={run['best_val_loss']:.6g}"
        )

    with open(args.out_dir / "attention_collapse_diagnostic_runs.json", "w") as f:
        json.dump(_jsonify(runs), f, indent=2)
    if args.write_plot_data:
        write_plot_data(args.plot_data, runs, pred_frames)
        print(f"[diagnostic] Saved de-identified plot data to {args.plot_data}")

    pdf_path, png_path = save_figure(runs, pred_frames, args.out_dir)
    print(f"[diagnostic] Saved {pdf_path}")
    print(f"[diagnostic] Saved {png_path}")


if __name__ == "__main__":
    main()
