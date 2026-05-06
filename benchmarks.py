"""
Actuarial and AutoML benchmark models for DeepTriangle v2.

Benchmark methods
-----------------
1. Mack Chain-Ladder (Mack 1993)       — chainladder-python MackChainladder
2. ODP Bootstrap (England & Verrall 1999) — chainladder-python BootstrapODPSample
3. Bornhuetter-Ferguson (1972)         — chainladder-python BornhuetterFerguson
4. H2O AutoML                          — 5-minute budget, features = group_code + 9 lag ratios

All benchmarks are evaluated on the same test period as the DT models:
  - Accident years 2002-2010
  - Calendar year 2011 diagonal (out-of-sample)

References
----------
Mack, T. (1993). Distribution-free calculation of the standard error of chain
    ladder reserve estimates. ASTIN Bulletin, 23(2), 213-225.
England, P. D., & Verrall, R. J. (1999). Analytic and bootstrap estimates of
    prediction errors in claims reserving. IME, 25(3), 281-293.
Bornhuetter, R. L., & Ferguson, R. E. (1972). The actuary and IBNR. PCAS, 59.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# NOTE: The actual dataset only contains 'private_passenger_auto' and
# 'workers_compensation'. benchmarks.py filters to LOBs present in data.
LOBS = [
    "workers_compensation",
    "commercial_auto",
    "private_passenger_auto",
    "other_liability",
    "medical_malpractice",
    "product_liability"
]
EPSILON = 1e-8
ACCIDENT_YEAR_RANGE = (
    int(os.environ.get("DEEPTRIANGLE_AY_MIN", 2002)),
    int(os.environ.get("DEEPTRIANGLE_TEST_MAX_AY", 2010)),
)
OUTLIER_THRESHOLD = 10.0   # |pct_error| > 10 (1000%) removed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_errors(
    predicted: np.ndarray,
    actual: np.ndarray,
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> np.ndarray:
    """Compute percentage errors with outlier removal."""
    denom = np.where(np.abs(actual) < EPSILON, EPSILON, np.abs(actual))
    pct = (predicted - actual) / denom
    valid = np.isfinite(pct) & (np.abs(pct) <= outlier_threshold)
    return pct[valid]


def _mape_rmspe(pct: np.ndarray) -> Dict[str, float]:
    if len(pct) == 0:
        return {"mape": float("nan"), "rmspe": float("nan"), "n": 0}
    return {
        "mape": float(np.mean(np.abs(pct))),
        "rmspe": float(np.sqrt(np.mean(pct ** 2))),
        "n": len(pct),
    }


def _load_raw_data(triangle_file: str, company_file: str) -> pd.DataFrame:
    """Load and merge triangle + company CSVs."""
    tri = pd.read_csv(triangle_file)
    co = pd.read_csv(company_file)
    data = tri.merge(co[["group_code"]], on="group_code", how="left")
    data = data.sort_values(["lob", "group_code", "accident_year", "development_lag"])
    return data


# ---------------------------------------------------------------------------
# Chain-Ladder benchmarks
# ---------------------------------------------------------------------------

def _run_odp_per_company(
    train_data: pd.DataFrame,
    actual_ult: pd.DataFrame,
    accident_year_range: tuple,
    n_sims: int = 100,
) -> Dict[str, Any]:
    """
    Run ODP Bootstrap per-company (BootstrapODPSample requires single-index).

    For each company, build a single triangle, run ODP bootstrap, take mean
    ultimate across simulations.  Collect across companies and compute metrics.
    """
    import chainladder as cl

    companies = train_data["group_code"].unique()
    rows = []

    for gc in companies:
        gc_data = train_data[train_data["group_code"] == gc]
        try:
            gc_tri = cl.Triangle(
                data=gc_data,
                origin="accident_year",
                development="development_year",
                columns=["cumulative_paid_loss"],
                cumulative=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                odp = cl.BootstrapODPSample(n_sims=n_sims, random_state=42)
                odp_tri = odp.fit_transform(gc_tri)
                odp_dev = cl.Development().fit_transform(odp_tri)
                odp_model = cl.Chainladder().fit(odp_dev)
                # Mean ultimate across simulations
                mean_ult = odp_model.ultimate_.mean("index")
                # to_frame() gives DatetimeIndex rows (origins) and a single value column
                ult_df = mean_ult.to_frame()
                val_col = ult_df.columns[0]
                for idx_val, row_val in ult_df[val_col].items():
                    if pd.notna(row_val) and row_val > 0:
                        # idx_val is a Timestamp — extract year
                        ay = idx_val.year if hasattr(idx_val, "year") else int(idx_val)
                        rows.append({
                            "group_code": gc,
                            "accident_year": ay,
                            "pred_ult": float(row_val),
                        })
        except Exception:
            continue  # Skip companies with insufficient data

    if not rows:
        return {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df[
        (pred_df["accident_year"] >= accident_year_range[0])
        & (pred_df["accident_year"] <= accident_year_range[1])
    ]
    merged = pred_df.merge(actual_ult, on=["group_code", "accident_year"], how="inner")
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
        "mape": float(company_metrics["abs_pct_error"].mean()),
        "rmspe": float(np.sqrt(company_metrics["sq_pct_error"].mean())),
        "n": len(company_metrics),
    }


def run_chainladder_benchmarks(
    triangle_file: str,
    company_file: str,
    lobs: Optional[List[str]] = None,
    accident_year_range: tuple = ACCIDENT_YEAR_RANGE,
    max_dev_lag: int = 9,
    test_calendar_year: int = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run Chain-Ladder, ODP Bootstrap, and Bornhuetter-Ferguson benchmarks.

    Uses chainladder-python (pip install chainladder).

    Note: We use cl.Chainladder() for the basic chain-ladder point estimates
    instead of cl.MackChainladder(), which has a dimension bug with
    multi-company triangles in chainladder 0.9.x.  Mack and basic CL
    produce identical point estimates; Mack only adds standard errors.

    Important: The Triangle must use 'development_year' (not 'development_lag')
    because chainladder requires date-like development periods.

    Parameters
    ----------
    triangle_file  : str   path to triangle_sample.csv
    company_file   : str   path to triangle_company_info.csv
    lobs           : list  LOBs to run (default: auto-detect from data)
    accident_year_range : tuple  (min, max) accident years to evaluate
    max_dev_lag    : int   final development lag = ultimate (default 9)

    Returns
    -------
    dict  keyed by lob, each value is a dict keyed by method name:
        {'mack': {'mape': ..., 'rmspe': ..., 'n': ...},
         'odp':  {'mape': ..., 'rmspe': ..., 'n': ...},
         'bf':   {'mape': ..., 'rmspe': ..., 'n': ...}}
    """
    try:
        import chainladder as cl
    except ImportError:
        raise ImportError(
            "chainladder-python is required. Install with: pip install chainladder"
        )

    if lobs is None:
        lobs = LOBS

    if test_calendar_year is None:
        test_calendar_year = int(os.environ.get("DEEPTRIANGLE_TEST_CAL", 2011))
    train_cutoff_cal = test_calendar_year - 1

    raw = _load_raw_data(triangle_file, company_file)
    available_lobs = raw["lob"].unique().tolist()
    lobs = [l for l in lobs if l in available_lobs]
    print(f"[CL benchmarks] LOBs to process: {lobs}")
    results: Dict[str, Dict[str, Any]] = {}

    for lob in lobs:
        print(f"\n[CL benchmarks] LOB: {lob}")
        lob_data = raw[raw["lob"] == lob].copy()

        # Use data up to calendar_year <= train_cutoff (no test-year leakage)
        train_data = lob_data[lob_data["calendar_year"] <= train_cutoff_cal].copy()

        # --- Build cl.Triangle using development_year (not development_lag) ---
        try:
            tri_obj = cl.Triangle(
                data=train_data,
                origin="accident_year",
                development="development_year",
                columns=["cumulative_paid_loss"],
                index=["group_code"],
                cumulative=True,
            )
        except Exception as e:
            print(f"  Failed to create triangle for {lob}: {e}")
            results[lob] = {m: {"mape": float("nan"), "rmspe": float("nan"), "n": 0}
                            for m in ("mack", "odp", "bf")}
            continue

        # --- Get actual ultimates from the full data (including 2011) ---
        actual_ult = (
            lob_data[
                (lob_data["development_lag"] == max_dev_lag)
                & (lob_data["accident_year"] >= accident_year_range[0])
                & (lob_data["accident_year"] <= accident_year_range[1])
            ][["group_code", "accident_year", "cumulative_paid_loss"]]
            .rename(columns={"cumulative_paid_loss": "actual_ult"})
        )

        lob_results: Dict[str, Any] = {}

        # -- 1. Chain-Ladder (Mack point estimates) --
        try:
            dev = cl.Development().fit_transform(tri_obj)
            cl_model = cl.Chainladder().fit(dev)
            cl_ult = cl_model.ultimate_
            _record_cl_metrics(cl_ult, actual_ult, "mack", lob_results, accident_year_range)
        except Exception as e:
            print(f"  Mack/CL failed: {e}")
            lob_results["mack"] = {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

        # -- 2. ODP Bootstrap --
        # BootstrapODPSample requires single-index triangles, so we loop
        # per company and collect results.
        try:
            odp_results = _run_odp_per_company(
                train_data, actual_ult, accident_year_range, n_sims=100
            )
            lob_results["odp"] = odp_results
            print(
                f"  ODP  : MAPE={odp_results['mape']:.4f}  "
                f"RMSPE={odp_results['rmspe']:.4f}  N={odp_results['n']}"
            )
        except Exception as e:
            print(f"  ODP Bootstrap failed: {e}")
            lob_results["odp"] = {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

        # -- 3. Bornhuetter-Ferguson --
        try:
            dev3 = cl.Development().fit_transform(tri_obj)
            # Earned premium triangle for sample_weight
            ep_tri = cl.Triangle(
                data=train_data,
                origin="accident_year",
                development="development_year",
                columns=["earned_premium_net"],
                index=["group_code"],
                cumulative=False,
            )
            # CL-derived apriori loss ratio
            cl_model3 = cl.Chainladder().fit(dev3)
            try:
                apriori_ratio = float(
                    (cl_model3.ultimate_ / ep_tri.latest_diagonal).mean().values[0][0]
                )
            except Exception:
                apriori_ratio = 0.65
            bf = cl.BornhuetterFerguson(apriori=apriori_ratio).fit(
                dev3, sample_weight=ep_tri.latest_diagonal
            )
            bf_ult = bf.ultimate_
            _record_cl_metrics(bf_ult, actual_ult, "bf", lob_results, accident_year_range)
        except Exception as e:
            print(f"  BF failed: {e}")
            lob_results["bf"] = {"mape": float("nan"), "rmspe": float("nan"), "n": 0}

        results[lob] = lob_results

    return results


def _record_cl_metrics(
    ultimate_triangle,
    actual_ult: pd.DataFrame,
    method: str,
    out: dict,
    accident_year_range: tuple,
) -> None:
    """
    Extract ultimate predictions from a chainladder Triangle and compute metrics.

    The ultimate_.to_frame() produces a wide DataFrame with Period-indexed columns
    for each origin year.  We melt it to long format and merge with actuals.

    Follows Kuo (2019) methodology: per-company average, then across companies.
    """
    try:
        ult_df = ultimate_triangle.to_frame().reset_index()

        # Columns are: group_code, Period('1987','Y-DEC'), Period('1988','Y-DEC'), ...
        # Melt Period columns to long format
        period_cols = [c for c in ult_df.columns if isinstance(c, pd.Period)]
        if not period_cols:
            # Fallback: try string columns that look like years
            period_cols = [c for c in ult_df.columns if c != "group_code"]

        melted = ult_df.melt(
            id_vars=["group_code"],
            value_vars=period_cols,
            var_name="accident_year",
            value_name="pred_ult",
        )
        # Convert Period to int year
        melted["accident_year"] = melted["accident_year"].apply(
            lambda p: p.year if hasattr(p, "year") else int(p)
        )

        # Filter to valid predictions
        melted = melted[melted["pred_ult"].notna() & (melted["pred_ult"] > 0)]
        melted = melted[
            (melted["accident_year"] >= accident_year_range[0])
            & (melted["accident_year"] <= accident_year_range[1])
        ]

        # Merge with actuals
        merged = melted.merge(actual_ult, on=["group_code", "accident_year"], how="inner")
        merged = merged.dropna(subset=["pred_ult", "actual_ult"])

        if merged.empty:
            out[method] = {"mape": float("nan"), "rmspe": float("nan"), "n": 0}
            print(f"  {method.upper():5s}: no valid predictions to evaluate")
            return

        # Per-company MAPE (Kuo 2019 methodology)
        denom = np.where(
            np.abs(merged["actual_ult"].values) < EPSILON,
            EPSILON,
            np.abs(merged["actual_ult"].values),
        )
        merged = merged.copy()
        merged["pct_error"] = (merged["pred_ult"].values - merged["actual_ult"].values) / denom

        # Outlier removal
        merged = merged[np.abs(merged["pct_error"]) <= OUTLIER_THRESHOLD]

        # Per-company aggregation then cross-company mean
        company_metrics = merged.groupby("group_code").agg(
            abs_pct_error=("pct_error", lambda x: np.mean(np.abs(x))),
            sq_pct_error=("pct_error", lambda x: np.mean(x ** 2)),
        ).reset_index()

        metrics = {
            "mape": float(company_metrics["abs_pct_error"].mean()),
            "rmspe": float(np.sqrt(company_metrics["sq_pct_error"].mean())),
            "n": len(company_metrics),
        }
        out[method] = metrics
        print(
            f"  {method.upper():5s}: MAPE={metrics['mape']:.4f}  "
            f"RMSPE={metrics['rmspe']:.4f}  N={metrics['n']}"
        )
    except Exception as e:
        print(f"  {method} metric extraction failed: {e}")
        out[method] = {"mape": float("nan"), "rmspe": float("nan"), "n": 0}


# ---------------------------------------------------------------------------
# H2O AutoML benchmark
# ---------------------------------------------------------------------------

def run_automl_benchmark(
    triangle_file: str,
    company_file: str,
    lobs: Optional[List[str]] = None,
    max_runtime_secs: int = 300,
    accident_year_range: tuple = ACCIDENT_YEAR_RANGE,
    max_dev_lag: int = 9,
    seed: int = 42,
    test_calendar_year: int = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run H2O AutoML benchmark for loss reserving.

    Feature engineering
    -------------------
    Features for each (company, accident_year) at a given development lag k:
      - group_code          : categorical company id
      - lag_1 ... lag_k     : observed incremental paid ratios (normalized)
      - lag_{k+1} ... lag_9 : set to 0 (unknown future) — model learns to ignore
    Target: incremental_paid at lag k+1 (next step prediction)

    Forecasting strategy
    --------------------
    We use an iterative 1-step-ahead forecast:
      for k in range(current_lag, max_dev_lag):
          predict lag k+1 using model on current feature vector
          append prediction to feature vector

    This approach matches the DT model's task: predict future incremental
    payments given observed history.

    Parameters
    ----------
    triangle_file      : str
    company_file       : str
    lobs               : list of str
    max_runtime_secs   : int   AutoML time budget per LOB (default 300 = 5 min)
    accident_year_range: tuple
    max_dev_lag        : int
    seed               : int

    Returns
    -------
    dict keyed by lob:
        {'automl': {'mape': ..., 'rmspe': ..., 'n': ..., 'best_model': ...}}
    """
    if test_calendar_year is None:
        test_calendar_year = int(os.environ.get("DEEPTRIANGLE_TEST_CAL", 2011))
    train_cutoff_cal = test_calendar_year - 1

    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError:
        raise ImportError(
            "h2o is required. Install with: pip install h2o"
        )

    if lobs is None:
        lobs = LOBS

    # --- Load data first to filter to available LOBs ---
    raw_check = _load_raw_data(triangle_file, company_file)
    available_lobs = raw_check["lob"].unique().tolist()
    lobs = [l for l in lobs if l in available_lobs]
    print(f"[AutoML] LOBs to process: {lobs}")

    # --- Initialize H2O (suppress verbose output) ---
    h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)
    h2o.no_progress()

    raw = _load_raw_data(triangle_file, company_file)
    results: Dict[str, Dict[str, Any]] = {}

    for lob in lobs:
        print(f"\n[AutoML] LOB: {lob}")
        lob_data = raw[raw["lob"] == lob].copy()

        # Normalize by earned premium to get loss ratios
        lob_data["inc_paid_ratio"] = (
            lob_data["incremental_paid_loss"] / lob_data["earned_premium_net"]
        )

        # --- Build wide feature table ---
        # One row per (group_code, accident_year, development_lag)
        # with lag_1...lag_9 as columns (filled with observed or 0)
        wide_rows = []
        for (gc, ay), grp in lob_data.groupby(["group_code", "accident_year"]):
            grp = grp.sort_values("development_lag")
            lag_vals = {f"lag_{r['development_lag']}": r["inc_paid_ratio"]
                        for _, r in grp.iterrows()}
            wide_rows.append({"group_code": str(gc), "accident_year": int(ay), **lag_vals})

        wide_df = pd.DataFrame(wide_rows).fillna(0.0)

        # Ensure all lag columns exist
        for k in range(1, max_dev_lag + 1):
            if f"lag_{k}" not in wide_df.columns:
                wide_df[f"lag_{k}"] = 0.0

        lag_cols = [f"lag_{k}" for k in range(1, max_dev_lag + 1)]

        # --- Training set: build one row per (gc, ay, lag) for multi-step training ---
        train_records = []
        for _, row in wide_df.iterrows():
            gc = row["group_code"]
            ay = int(row["accident_year"])
            # Use data up to 2010 (matching CL benchmarks) for fair comparison.
            # The DT models train on <=2008 + validate on 2009-2010, but AutoML
            # uses 5-fold CV internally, so giving it data up to 2010 is fairer.
            ay_in_train = lob_data[
                (lob_data["group_code"].astype(str) == gc)
                & (lob_data["accident_year"] == ay)
                & (lob_data["calendar_year"] <= train_cutoff_cal)
            ]
            if ay_in_train.empty:
                continue

            max_observed_lag = int(ay_in_train["development_lag"].max())
            for k in range(1, max_observed_lag):
                # Features: lags 1..k observed, k+1..9 = 0
                feat = {"group_code": gc}
                for j in range(1, max_dev_lag + 1):
                    feat[f"lag_{j}"] = float(row[f"lag_{j}"]) if j <= k else 0.0
                feat["target"] = float(row[f"lag_{k + 1}"])
                train_records.append(feat)

        if len(train_records) < 10:
            print(f"  Insufficient training records: {len(train_records)}")
            results[lob] = {"automl": {"mape": float("nan"), "rmspe": float("nan"),
                                        "n": 0, "best_model": "none"}}
            continue

        train_pd = pd.DataFrame(train_records)

        # --- H2O frames ---
        train_h2o = h2o.H2OFrame(train_pd)
        train_h2o["group_code"] = train_h2o["group_code"].asfactor()

        feature_cols = ["group_code"] + lag_cols
        target_col = "target"

        # --- AutoML ---
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=seed,
            verbosity=None,
            nfolds=5,
        )
        aml.train(x=feature_cols, y=target_col, training_frame=train_h2o)
        best_model_name = aml.leader.model_id if aml.leader else "none"
        print(f"  Best model: {best_model_name}")

        # --- Iterative forecasting on test companies ---
        test_ay_lob = lob_data[
            (lob_data["accident_year"] >= accident_year_range[0])
            & (lob_data["accident_year"] <= accident_year_range[1])
            & (lob_data["calendar_year"] <= train_cutoff_cal)  # use only observed history
        ]

        preds_list = []
        actuals_list = []

        for (gc, ay), grp in test_ay_lob.groupby(["group_code", "accident_year"]):
            grp = grp.sort_values("development_lag")
            max_obs_lag = int(grp["development_lag"].max())
            if max_obs_lag >= max_dev_lag:
                continue  # already at ultimate, no forecasting needed

            # Observed incremental paid ratios
            obs_lags: Dict[int, float] = {
                int(r["development_lag"]): float(r["inc_paid_ratio"])
                for _, r in grp.iterrows()
                if not pd.isna(r["inc_paid_ratio"])
            }

            # Build current feature vector
            current_lags = {k: obs_lags.get(k, 0.0) for k in range(1, max_dev_lag + 1)}

            # Iteratively predict from max_obs_lag+1 to max_dev_lag
            pred_future = []
            for k in range(max_obs_lag + 1, max_dev_lag + 1):
                feat_row = {"group_code": str(gc)}
                for j in range(1, max_dev_lag + 1):
                    feat_row[f"lag_{j}"] = current_lags[j]

                row_h2o = h2o.H2OFrame(pd.DataFrame([feat_row]))
                row_h2o["group_code"] = row_h2o["group_code"].asfactor()

                p = float(aml.leader.predict(row_h2o).as_data_frame()["predict"].iloc[0])
                p = max(0.0, p)  # clip negative predictions
                pred_future.append(p)
                current_lags[k] = p  # feed prediction back as feature

            # Actual ultimate = sum of all incremental paid at development lag 9
            actual_row = lob_data[
                (lob_data["group_code"].astype(str) == str(gc))
                & (lob_data["accident_year"] == ay)
                & (lob_data["development_lag"] == max_dev_lag)
            ]
            if actual_row.empty:
                continue

            ep = float(actual_row["earned_premium_net"].iloc[0])
            actual_ult = float(actual_row["cumulative_paid_loss"].iloc[0])

            # Predicted ultimate = sum of observed + sum of predicted future increments
            obs_cumulative = float(grp[grp["development_lag"] == max_obs_lag][
                "cumulative_paid_loss"
            ].iloc[0]) if max_obs_lag > 0 else 0.0

            # Convert predicted future ratios to dollar amounts
            pred_future_dollars = [p * ep for p in pred_future]
            predicted_ult = obs_cumulative + sum(pred_future_dollars)

            preds_list.append(predicted_ult)
            actuals_list.append(actual_ult)

        if len(preds_list) == 0:
            print("  No valid forecasts generated")
            results[lob] = {"automl": {"mape": float("nan"), "rmspe": float("nan"),
                                        "n": 0, "best_model": best_model_name}}
            continue

        pct = _pct_errors(np.array(preds_list), np.array(actuals_list))
        metrics = _mape_rmspe(pct)
        metrics["best_model"] = best_model_name
        results[lob] = {"automl": metrics}
        print(
            f"  AutoML: MAPE={metrics['mape']:.4f}  "
            f"RMSPE={metrics['rmspe']:.4f}  N={metrics['n']}"
        )

    h2o.shutdown(prompt=False)
    return results


# ---------------------------------------------------------------------------
# Convenience: run all benchmarks and return a tidy DataFrame
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    triangle_file: str,
    company_file: str,
    run_automl: bool = True,
    automl_max_secs: int = 300,
    lobs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run all benchmarks (CL + AutoML) and return a combined tidy DataFrame.

    Parameters
    ----------
    triangle_file : str
    company_file  : str
    run_automl    : bool  whether to run H2O AutoML (slow; default True)
    automl_max_secs : int  AutoML budget per LOB in seconds (default 300)
    lobs          : list  subset of LOBs (default all 4)

    Returns
    -------
    pd.DataFrame with columns ['lob', 'method', 'mape', 'rmspe', 'n']
    """
    rows = []

    # --- Chain-ladder benchmarks ---
    cl_results = run_chainladder_benchmarks(triangle_file, company_file, lobs=lobs)
    for lob, methods in cl_results.items():
        for method, metrics in methods.items():
            rows.append({
                "lob": lob,
                "method": method,
                "mape": metrics.get("mape"),
                "rmspe": metrics.get("rmspe"),
                "n": metrics.get("n", 0),
            })

    # --- AutoML benchmark ---
    if run_automl:
        aml_results = run_automl_benchmark(
            triangle_file, company_file,
            lobs=lobs,
            max_runtime_secs=automl_max_secs,
        )
        for lob, methods in aml_results.items():
            for method, metrics in methods.items():
                rows.append({
                    "lob": lob,
                    "method": method,
                    "mape": metrics.get("mape"),
                    "rmspe": metrics.get("rmspe"),
                    "n": metrics.get("n", 0),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, json

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    TRI = os.path.join(DATA_DIR, "triangle_sample.csv")
    CO = os.path.join(DATA_DIR, "triangle_company_info.csv")
    BASE_RESULTS = os.environ.get("DEEPTRIANGLE_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    OUT_DIR = os.path.join(BASE_RESULTS, "phase1")
    os.makedirs(OUT_DIR, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser(description="Run benchmarks for DeepTriangle v2")
    parser.add_argument("--no-automl", action="store_true", help="Skip H2O AutoML")
    parser.add_argument("--automl-secs", type=int, default=300, help="AutoML budget per LOB")
    parser.add_argument("--lobs", nargs="+", default=None, help="Subset of LOBs")
    args = parser.parse_args()

    results_df = run_all_benchmarks(
        TRI,
        CO,
        run_automl=not args.no_automl,
        automl_max_secs=args.automl_secs,
        lobs=args.lobs,
    )

    print("\n=== Benchmark Results ===")
    print(results_df.to_string(index=False))

    # Save
    csv_path = os.path.join(OUT_DIR, "benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")
