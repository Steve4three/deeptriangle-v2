"""
Data preparation module for DeepTriangle v2.

Simplified from enhanced_deeptriangle/data_prep_original.py:
  - Only encodes group_code (no other categorical or numeric company features)
  - Same train/val/test temporal splits: train cal_year <= 2008, val 2009-2010, test 2011
  - Same normalization: divide by earned_premium_net
  - Same sequence generation: timesteps=9, mask=-99.0
  - Input shape: (batch, 9, 2)  [paid_lags, case_lags]
  - Output shape: (batch, 9, 1) x2  [paid_target, case_target]
  - Company input: (batch, 1)  [group_code integer]

Reference: Kuo (2019) "DeepTriangle: A Deep Learning Approach to Loss Reserving"
"""

import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any, Tuple, Union
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
# NOTE: The actual dataset (triangle_sample.csv) contains only 2 LOBs:
#   'private_passenger_auto' and 'workers_compensation'
# The full CAS dataset has 6 LOBs but this extract covers only 2.
# LOBS is kept as the canonical expected list; DataManager.available_lobs()
# returns only LOBs actually present in the loaded data.
LOBS = [
    "workers_compensation",
    "commercial_auto",
    "private_passenger_auto",
    "other_liability",
    "medical_malpractice",
    "product_liability"
]
MASK_VALUE = -99.0
TIMESTEPS = 9


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------

def assign_bucket(
    row: pd.Series,
    train_ranges: List[Tuple[int, int]] = [(None, 2008)],
    validation_ranges: List[Tuple[int, int]] = [(2009, 2010)],
    test_min_calendar_year: int = 2011,
    test_max_accident_year: int = 2010,
) -> Union[str, float]:
    """
    Assign a temporal bucket to each observation.

    Parameters
    ----------
    row : pd.Series
        Must contain 'calendar_year', 'development_lag', 'accident_year'.
    train_ranges : list of (min_year, max_year)
        Use None for no lower bound on the min side.
    validation_ranges : list of (min_year, max_year)
    test_min_calendar_year : int
    test_max_accident_year : int

    Returns
    -------
    'train', 'validation', 'test', or np.nan
    """
    calendar_year = row["calendar_year"]
    development_lag = row["development_lag"]
    accident_year = row["accident_year"]

    # Test: calendar_year == test_min AND accident_year is within scope
    if calendar_year == test_min_calendar_year and accident_year <= test_max_accident_year:
        return "test"

    # Observations with development_lag <= 0 are not usable sequences
    if development_lag <= 0:
        return np.nan

    # Training ranges
    for min_year, max_year in train_ranges:
        if min_year is None:
            if calendar_year <= max_year:
                return "train"
        else:
            if min_year <= calendar_year <= max_year:
                return "train"

    # Validation ranges
    for min_year, max_year in validation_ranges:
        if min_year is None:
            if calendar_year <= max_year:
                return "validation"
        else:
            if min_year <= calendar_year <= max_year:
                return "validation"

    return np.nan


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

def load_and_prepare_data(
    triangle_file: str,
    company_file: str,
    train_ranges: List[Tuple[int, int]] = [(None, 2008)],
    validation_ranges: List[Tuple[int, int]] = [(2009, 2010)],
    test_min_calendar_year: int = 2011,
    test_max_accident_year: int = 2010,
) -> pd.DataFrame:
    """
    Load raw CSV files, engineer features, assign buckets, and normalize.

    Normalization: incremental_paid, cumulative_paid, and case_reserves are
    divided by earned_premium_net to make loss ratios comparable across
    companies of different sizes (matching Kuo 2019).

    Parameters
    ----------
    triangle_file : str
        Path to triangle_sample.csv
    company_file : str
        Path to triangle_company_info.csv

    Returns
    -------
    pd.DataFrame with columns including 'incremental_paid', 'case_reserves',
    'bucket', 'earned_premium_net', 'group_code', 'lob', etc.
    """
    # --- Load ---
    data = pd.read_csv(triangle_file)
    company_info = pd.read_csv(company_file)

    # Merge on group_code (left join so all triangle rows are kept)
    data = data.merge(company_info, on="group_code", how="left")

    # --- Derived features ---
    data["case_reserves"] = data["incurred_loss"] - data["cumulative_paid_loss"]

    # Sort for consistent group-by ordering
    data = data.sort_values(["lob", "group_code", "accident_year", "development_lag"])

    # --- Conditional features (mask future calendar years) ---
    # mask_calendar_year: last calendar year where features are observable.
    # For the default setup (test diagonal 2011), this is 2010.
    mask_calendar_year = test_min_calendar_year - 1

    def _create_conditional_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()

        group["incremental_paid_actual"] = group["incremental_paid_loss"]
        group["incremental_paid"] = np.where(
            group["calendar_year"] <= mask_calendar_year, group["incremental_paid_actual"], np.nan
        )

        group["cumulative_paid_actual"] = group["cumulative_paid_loss"]
        group["cumulative_paid"] = np.where(
            group["calendar_year"] <= mask_calendar_year, group["cumulative_paid_actual"], np.nan
        )

        group["case_reserves_actual"] = group["case_reserves"]
        group["case_reserves"] = np.where(
            group["calendar_year"] <= mask_calendar_year, group["case_reserves_actual"], np.nan
        )

        return group

    # pandas >=3.0 excludes grouping columns from apply(); preserve them via index
    group_cols = ["lob", "group_code", "accident_year"]
    data = data.set_index(group_cols)
    data = (
        data.groupby(level=group_cols, group_keys=False)
        .apply(_create_conditional_features)
        .reset_index()
    )

    # --- Bucket assignment ---
    data["bucket"] = data.apply(
        lambda row: assign_bucket(
            row,
            train_ranges=train_ranges,
            validation_ranges=validation_ranges,
            test_min_calendar_year=test_min_calendar_year,
            test_max_accident_year=test_max_accident_year,
        ),
        axis=1,
    )

    # --- Normalize by earned premium ---
    for col in [
        "incremental_paid",
        "incremental_paid_actual",
        "cumulative_paid",
        "cumulative_paid_actual",
        "case_reserves",
        "case_reserves_actual",
    ]:
        assert (data["earned_premium_net"] > 0).all(), (
            "earned_premium_net must be positive for all rows"
        )
        data[col] = data[col] / data["earned_premium_net"]

    return data


# ---------------------------------------------------------------------------
# Group code encoder (only categorical feature used in DT v2)
# ---------------------------------------------------------------------------

def create_group_code_encoder(data: pd.DataFrame) -> LabelEncoder:
    """
    Fit a LabelEncoder on group_code using all rows in data.

    Returns
    -------
    sklearn.preprocessing.LabelEncoder fitted on group_code
    """
    encoder = LabelEncoder()
    encoder.fit(data["group_code"].astype(str))
    return encoder


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def make_series(
    values: List[float],
    start_offset: int,
    end_offset: int,
    na_pad: float = MASK_VALUE,
    timesteps: int = TIMESTEPS,
) -> List[List[float]]:
    """
    Create fixed-length sequence windows from a time-series vector.

    Follows the R make_series logic from Kuo (2019):
      - start_offset < 0  => lag features, pre-padded with na_pad
      - start_offset >= 0 => target features, post-padded with na_pad

    Parameters
    ----------
    values : list of floats
        The raw time-series for one accident year (length = development lags).
    start_offset, end_offset : int
        Defines the window [start_offset, end_offset] relative to current
        position (1-based R convention).
    na_pad : float
        Padding / masking value (default -99.0).
    timesteps : int
        Length of the output window.

    Returns
    -------
    List[List[float]] of length len(values), each inner list of length timesteps.
    """

    def _prepad(v: list, length: int = timesteps) -> list:
        diff = length - len(v)
        return [na_pad] * diff + v if diff > 0 else v

    def _postpad(v: list, length: int = timesteps) -> list:
        diff = length - len(v)
        return v + [na_pad] * diff if diff > 0 else v

    result = []
    for i in range(len(values)):
        # Convert to R-style 1-based index, then build window
        start = i + 1 + start_offset  # R 1-based
        end = i + 1 + end_offset      # R 1-based (inclusive)

        start_idx = start - 1  # Python 0-based
        end_idx = end          # Python slice end (exclusive)

        window = []
        for idx in range(start_idx, end_idx):
            if idx < 0 or idx >= len(values):
                window.append(na_pad)
            else:
                val = values[idx]
                window.append(na_pad if pd.isna(val) else val)

        if start_offset < 0:
            window = _prepad(window)
        else:
            window = _postpad(window)

        result.append(window)

    return result


def _mutate_series(data: pd.DataFrame, timesteps: int = TIMESTEPS) -> pd.DataFrame:
    """
    Add paid_lags, case_lags, paid_target, case_target sequence columns.

    Operates group-by (lob, group_code, accident_year).
    """

    def _process_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("development_lag")
        inc_paid = group["incremental_paid"].tolist()
        case_res = group["case_reserves"].tolist()

        group["paid_lags"] = make_series(inc_paid, -timesteps, -1, timesteps=timesteps)
        group["case_lags"] = make_series(case_res, -timesteps, -1, timesteps=timesteps)
        group["paid_target"] = make_series(inc_paid, 0, timesteps - 1, timesteps=timesteps)
        group["case_target"] = make_series(case_res, 0, timesteps - 1, timesteps=timesteps)

        return group

    # pandas >=3.0 excludes grouping columns from apply(); preserve them via index
    _grp_cols = ["lob", "group_code", "accident_year"]
    data = data.set_index(_grp_cols)
    result = (
        data.groupby(level=_grp_cols, group_keys=False)
        .apply(_process_group)
        .reset_index()
    )
    return result


# ---------------------------------------------------------------------------
# Per-LOB model input preparation (SIMPLIFIED: only group_code)
# ---------------------------------------------------------------------------

def prep_lob_model_data(
    data: pd.DataFrame,
    group_code_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """
    Convert a filtered/mutated DataFrame into a model-input dict.

    (Historical note: previously named ``prep_keras_data`` when DeepTriangle
    was a Keras/TensorFlow codebase. Renamed 2026-04-10 during dead-code
    audit — the function is framework-agnostic and had nothing Keras-specific.)

    Inputs
    ------
    - ay_seq_input  : (N, 9, 2)  — stacked [paid_lags, case_lags]
    - group_code_input : (N, 1)  — integer-encoded group_code

    Targets
    -------
    - paid_output           : (N, 9, 1)
    - case_reserves_output  : (N, 9, 1)

    Parameters
    ----------
    data : pd.DataFrame
        Rows must already have paid_lags, case_lags, paid_target, case_target
        columns (i.e., _mutate_series has already been applied).
    group_code_encoder : LabelEncoder
        Fitted on the full dataset's group_code column.

    Returns
    -------
    dict with keys 'x' and 'y', each a dict of named arrays.
    """
    # --- Sequence inputs ---
    paid_lags = np.array(data["paid_lags"].tolist(), dtype=np.float32)   # (N, 9)
    case_lags = np.array(data["case_lags"].tolist(), dtype=np.float32)   # (N, 9)
    ay_seq = np.stack([paid_lags, case_lags], axis=-1)                   # (N, 9, 2)

    # --- Group code input ---
    gc_encoded = group_code_encoder.transform(data["group_code"].astype(str))
    group_code_arr = gc_encoded.reshape(-1, 1).astype(np.int32)          # (N, 1)

    # --- Targets ---
    paid_target = np.array(data["paid_target"].tolist(), dtype=np.float32)
    case_target = np.array(data["case_target"].tolist(), dtype=np.float32)

    paid_target = paid_target.reshape(paid_target.shape[0], paid_target.shape[1], 1)
    case_target = case_target.reshape(case_target.shape[0], case_target.shape[1], 1)

    return {
        "x": {
            "ay_seq_input": ay_seq,
            "group_code_input": group_code_arr,
        },
        "y": {
            "paid_output": paid_target,
            "case_reserves_output": case_target,
        },
    }


# ---------------------------------------------------------------------------
# Full pipeline: prepare all splits for all LOBs
# ---------------------------------------------------------------------------

def prepare_all_data(
    data_with_features: pd.DataFrame,
    group_code_encoder: LabelEncoder,
    test_calendar_year: int = 2011,
) -> Dict[str, Dict[str, Any]]:
    """
    Build train, validation, and test Keras dicts for all four LOBs.

    Parameters
    ----------
    data_with_features : pd.DataFrame
    group_code_encoder : LabelEncoder
    test_calendar_year : int
        Calendar year of the test diagonal (default 2011).

    Returns
    -------
    dict with keys 'full_training_data', 'validation_data', 'test_data'.
    Each value is a dict keyed by LOB name.

    Notes
    -----
    - Validation sequences: sequence context includes train + validation rows
      (so that sequences at the validation boundary have full lag history),
      but only validation-bucketed rows are returned as targets.
    - Test sequences: context includes all rows up to test_calendar_year;
      only rows with bucket == 'test' and calendar_year == test_calendar_year
      are returned.
    """
    datasets: Dict[str, Dict[str, Any]] = {
        "full_training_data": {},
        "validation_data": {},
        "test_data": {},
    }

    # --- Validation ---
    val_context = data_with_features[
        data_with_features["bucket"].isin(["train", "validation"])
        | (data_with_features["development_lag"] == 0)
    ].copy()
    val_context = _mutate_series(val_context)

    val_rows = val_context[val_context["bucket"] == "validation"].copy()
    for lob in val_rows["lob"].unique():
        lob_df = val_rows[val_rows["lob"] == lob]
        datasets["validation_data"][lob] = prep_lob_model_data(lob_df, group_code_encoder)

    # --- Training ---
    train_context = data_with_features[
        data_with_features["bucket"].isin(["train", "validation"])
        | (data_with_features["development_lag"] == 0)
    ].copy()
    train_context = _mutate_series(train_context)

    train_rows = train_context[train_context["bucket"] == "train"].copy()
    for lob in train_rows["lob"].unique():
        lob_df = train_rows[train_rows["lob"] == lob]
        datasets["full_training_data"][lob] = prep_lob_model_data(lob_df, group_code_encoder)

    # --- Test ---
    test_context = data_with_features[
        data_with_features["calendar_year"] <= test_calendar_year
    ].copy()
    test_context = _mutate_series(test_context)

    test_rows = test_context[
        (test_context["bucket"] == "test")
        & (test_context["calendar_year"] == test_calendar_year)
    ].copy()
    for lob in test_rows["lob"].unique():
        lob_df = test_rows[test_rows["lob"] == lob]
        datasets["test_data"][lob] = prep_lob_model_data(lob_df, group_code_encoder)

    return datasets


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

class DataManager:
    """
    Orchestrates data loading, encoding, and split preparation for DT v2.

    Example
    -------
    >>> dm = DataManager(
    ...     "/path/to/triangle_sample.csv",
    ...     "/path/to/triangle_company_info.csv"
    ... )
    >>> dm.load()
    >>> dm.prepare()
    >>> train = dm.get("workers_compensation", "full_training_data")
    >>> val   = dm.get("workers_compensation", "validation_data")
    >>> test  = dm.get("workers_compensation", "test_data")
    """

    # Default temporal splits (proprietary data)
    _DEFAULT_TRAIN_CUTOFF = 2008
    _DEFAULT_VAL_START = 2009
    _DEFAULT_VAL_END = 2010
    _DEFAULT_TEST_CAL = 2011
    _DEFAULT_TEST_MAX_AY = 2010

    def __init__(
        self,
        triangle_file: str,
        company_file: str,
        train_ranges: List[Tuple[int, int]] = None,
        validation_ranges: List[Tuple[int, int]] = None,
        test_min_calendar_year: int = None,
        test_max_accident_year: int = None,
    ):
        self.triangle_file = triangle_file
        self.company_file = company_file

        # Allow env vars to override defaults (set by replicate.py --cas)
        train_cutoff = int(os.environ.get(
            "DEEPTRIANGLE_TRAIN_CUTOFF", self._DEFAULT_TRAIN_CUTOFF))
        val_start = int(os.environ.get(
            "DEEPTRIANGLE_VAL_START", self._DEFAULT_VAL_START))
        val_end = int(os.environ.get(
            "DEEPTRIANGLE_VAL_END", self._DEFAULT_VAL_END))
        default_test_cal = int(os.environ.get(
            "DEEPTRIANGLE_TEST_CAL", self._DEFAULT_TEST_CAL))
        default_test_max_ay = int(os.environ.get(
            "DEEPTRIANGLE_TEST_MAX_AY", self._DEFAULT_TEST_MAX_AY))

        self.train_ranges = train_ranges or [(None, train_cutoff)]
        self.validation_ranges = validation_ranges or [(val_start, val_end)]
        self.test_min_calendar_year = test_min_calendar_year or default_test_cal
        self.test_max_accident_year = test_max_accident_year or default_test_max_ay
        self.data: pd.DataFrame = None
        self.encoder: LabelEncoder = None
        self.splits: Dict[str, Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def load(self) -> "DataManager":
        """Load CSVs, engineer features, assign buckets, normalize."""
        self.data = load_and_prepare_data(
            self.triangle_file,
            self.company_file,
            train_ranges=self.train_ranges,
            validation_ranges=self.validation_ranges,
            test_min_calendar_year=self.test_min_calendar_year,
            test_max_accident_year=self.test_max_accident_year,
        )
        return self

    # ------------------------------------------------------------------
    def prepare(self) -> "DataManager":
        """Fit encoder and build all Keras splits."""
        if self.data is None:
            raise RuntimeError("Call .load() before .prepare()")
        self.encoder = create_group_code_encoder(self.data)
        self.splits = prepare_all_data(
            self.data,
            self.encoder,
            test_calendar_year=self.test_min_calendar_year,
        )
        return self

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Number of distinct group_code values (= embedding vocab size)."""
        if self.encoder is None:
            raise RuntimeError("Call .prepare() first")
        return len(self.encoder.classes_)

    # ------------------------------------------------------------------
    def get(self, lob: str, split: str) -> Dict[str, Any]:
        """
        Retrieve Keras data dict for a specific LOB and split.

        Parameters
        ----------
        lob : str
            e.g. 'workers_compensation'
        split : str
            One of 'full_training_data', 'validation_data', 'test_data'
        """
        if self.splits is None:
            raise RuntimeError("Call .prepare() first")
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}'. Choose from {list(self.splits)}")
        if lob not in self.splits[split]:
            raise ValueError(f"LOB '{lob}' not in split '{split}'")
        return self.splits[split][lob]

    # ------------------------------------------------------------------
    def available_lobs(self, split: str = "full_training_data") -> List[str]:
        """Return LOBs available in a given split."""
        if self.splits is None:
            raise RuntimeError("Call .prepare() first")
        return list(self.splits[split].keys())

    # ------------------------------------------------------------------
    def bucket_distribution(self) -> pd.Series:
        """Show how many rows fall into each bucket."""
        if self.data is None:
            raise RuntimeError("Call .load() first")
        return self.data["bucket"].value_counts().sort_index()

    # ------------------------------------------------------------------
    def get_test_metadata(self, lob: str) -> pd.DataFrame:
        """
        Return a lightweight DataFrame with group_code and earned_premium_net
        for the test split of a given LOB.  Used by evaluate.py to de-normalize.
        """
        if self.data is None:
            raise RuntimeError("Call .load() first")
        if self.splits is None:
            raise RuntimeError("Call .prepare() first")

        test_cal_year = self.test_min_calendar_year
        test_context = self.data[self.data["calendar_year"] <= test_cal_year].copy()
        # Re-apply series mutation to get same row ordering as test split
        test_context = _mutate_series(test_context)
        test_rows = test_context[
            (test_context["bucket"] == "test")
            & (test_context["calendar_year"] == test_cal_year)
            & (test_context["lob"] == lob)
        ].copy()

        return test_rows[
            ["group_code", "accident_year", "development_lag", "earned_premium_net"]
        ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    tri = os.path.join(DATA_DIR, "triangle_sample.csv")
    co = os.path.join(DATA_DIR, "triangle_company_info.csv")

    print("Loading data ...")
    dm = DataManager(tri, co)
    dm.load()
    dm.prepare()

    print(f"Vocab size (group_code): {dm.vocab_size}")
    print(f"Bucket distribution:\n{dm.bucket_distribution()}")
    print(f"Available LOBs (train): {dm.available_lobs()}")

    for lob in dm.available_lobs():
        tr = dm.get(lob, "full_training_data")
        va = dm.get(lob, "validation_data")
        te = dm.get(lob, "test_data")
        print(
            f"  {lob:30s}  "
            f"train={tr['x']['ay_seq_input'].shape[0]:5d}  "
            f"val={va['x']['ay_seq_input'].shape[0]:5d}  "
            f"test={te['x']['ay_seq_input'].shape[0]:5d}"
        )

    print("data_prep.py OK")
