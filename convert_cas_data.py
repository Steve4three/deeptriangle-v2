#!/usr/bin/env python3
"""
Convert CAS Loss Reserving Database CSVs into the format expected by data_prep.py.

Downloads (or reads local copies of) the six CAS Schedule P CSV files and produces
    data/triangle_sample.csv
    data/triangle_company_info.csv

CAS data source:
    https://www.casact.org/publications-research/research/research-resources/
        loss-reserving-data-pulled-naic-schedule-p

Usage:
    python convert_cas_data.py                  # download from CAS website
    python convert_cas_data.py --local DIR      # read CSVs from a local directory
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"

# CAS CSV URLs (December 2025 release)
CAS_BASE = "https://www.casact.org/sites/default/files/2026-03"
CAS_FILES = {
    "private_passenger_auto": f"{CAS_BASE}/ppauto_pos98-07%20%281%29.csv",
    "workers_compensation":   f"{CAS_BASE}/wkcomp_pos_98-07.csv",
    "commercial_auto":        f"{CAS_BASE}/comauto_pos_98-07.csv",
    "medical_malpractice":    f"{CAS_BASE}/medmal_pos_98-07.csv",
    "product_liability":      f"{CAS_BASE}/prodliab_pos_98-07.csv",
    "other_liability":        f"{CAS_BASE}/othliab_pos_98-07.csv",
}

# Map CAS filenames (without path) to LOB names for --local mode
CAS_FILENAME_TO_LOB = {
    "ppauto":   "private_passenger_auto",
    "wkcomp":   "workers_compensation",
    "comauto":  "commercial_auto",
    "medmal":   "medical_malpractice",
    "prodliab": "product_liability",
    "othliab":  "other_liability",
}

# Column rename: CAS → our schema
COLUMN_MAP = {
    "GRCODE":         "group_code",
    "GRNAME":         "group_name",
    "AccidentYear":   "accident_year",
    "DevelopmentYear": "development_year",
    "DevelopmentLag":  "development_lag",
    "IncurredLosses":  "incurred_loss",       # Note: CAS calls it "IncurredLosses"
    "CumPaidLoss":     "cumulative_paid_loss",
    "BulkLoss":        "bulk_loss",
    "EarnedPremDIR":   "earned_premium_dir",
    "EarnedPremCeded": "earned_premium_ceded",
    "EarnedPremNet":   "earned_premium_net",
    "Single":          "single",
    "PostedReserves2007": "posted_reserves_2007",
}


def load_cas_csv(lob, url, local_dir=None):
    """Load a single CAS CSV, either from URL or local file."""
    if local_dir:
        # Find matching file in local directory
        candidates = list(Path(local_dir).glob("*.csv"))
        matched = None
        for c in candidates:
            stem = c.stem.lower().split("_")[0]
            if stem in CAS_FILENAME_TO_LOB and CAS_FILENAME_TO_LOB[stem] == lob:
                matched = c
                break
        if matched is None:
            print(f"  ⚠ No local file found for {lob}, skipping")
            return pd.DataFrame()
        print(f"  Loading {lob} from {matched}")
        df = pd.read_csv(matched)
    else:
        print(f"  Downloading {lob} ...")
        df = pd.read_csv(url)

    # Rename columns
    df = df.rename(columns=COLUMN_MAP)

    # Add LOB column
    df["lob"] = lob

    # CAS DevelopmentLag is 1-indexed; our code expects 0-indexed
    df["development_lag"] = df["development_lag"] - 1

    # Compute calendar_year = accident_year + development_lag
    df["calendar_year"] = df["accident_year"] + df["development_lag"]

    # Compute incremental paid losses from cumulative
    df = df.sort_values(["group_code", "accident_year", "development_lag"])
    df["incremental_paid_loss"] = (
        df.groupby(["group_code", "accident_year"])["cumulative_paid_loss"]
        .diff()
        .fillna(df["cumulative_paid_loss"])  # lag 0 = first cumulative value
    )

    # Compute paid and incurred loss ratios
    df["paid_LR"] = df["cumulative_paid_loss"] / df["earned_premium_net"] * 100
    df["incurred_LR"] = df["incurred_loss"] / df["earned_premium_net"] * 100

    # Set data_year = max calendar year in the upper triangle
    # CAS data: accident years 1998-2007, dev lags 0-9 (after reindex)
    # Upper triangle observed as of year-end 2007
    df["data_year"] = df["accident_year"] + 9  # each AY has 10 dev lags

    # Prefix group_code to avoid collisions across LOBs
    df["group_code"] = "C" + df["group_code"].astype(str)

    return df


def build_company_info(df: pd.DataFrame) -> pd.DataFrame:
    """Build minimal triangle_company_info.csv from the triangle data."""
    info = (
        df[["group_code"]]
        .drop_duplicates()
        .sort_values("group_code")
        .reset_index(drop=True)
    )
    return info


def main():
    parser = argparse.ArgumentParser(description="Convert CAS data to our format")
    parser.add_argument("--local", type=str, default=None,
                        help="Local directory containing CAS CSVs")
    parser.add_argument("--lobs", nargs="+",
                        default=["private_passenger_auto", "workers_compensation"],
                        help="LOBs to include (default: PP Auto + WC)")
    args = parser.parse_args()

    print("=== CAS → DeepTriangle v2 Data Converter ===\n")

    frames = []
    for lob in args.lobs:
        if lob not in CAS_FILES:
            print(f"  ⚠ Unknown LOB '{lob}', skipping")
            continue
        df = load_cas_csv(lob, CAS_FILES[lob], args.local)
        if len(df) > 0:
            frames.append(df)
            print(f"    → {len(df)} rows, AY {df['accident_year'].min()}-{df['accident_year'].max()}, "
                  f"{df['group_code'].nunique()} companies")

    if not frames:
        print("\n❌ No data loaded. Check URLs or --local path.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Select columns matching our schema
    output_cols = [
        "group_code", "lob", "data_year", "accident_year", "development_year",
        "development_lag", "earned_premium_net", "paid_LR", "incurred_LR",
        "incurred_loss", "cumulative_paid_loss", "calendar_year", "incremental_paid_loss",
    ]
    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in combined.columns]
    combined = combined[output_cols]

    # Filter out rows with zero or negative earned premium
    before = len(combined)
    combined = combined[combined["earned_premium_net"] > 0]
    if len(combined) < before:
        print(f"\n  Filtered {before - len(combined)} rows with non-positive earned premium")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    tri_path = DATA_DIR / "triangle_sample.csv"
    combined.to_csv(tri_path, index=False)
    print(f"\n✅ Saved {tri_path} ({len(combined)} rows, {combined['lob'].nunique()} LOBs)")

    # Company info (minimal)
    info = build_company_info(combined)
    info_path = DATA_DIR / "triangle_company_info.csv"
    info.to_csv(info_path, index=False)
    print(f"✅ Saved {info_path} ({len(info)} companies)")

    # Summary
    print(f"\n=== Summary ===")
    for lob in combined["lob"].unique():
        sub = combined[combined["lob"] == lob]
        print(f"  {lob}: {sub['group_code'].nunique()} companies, "
              f"AY {sub['accident_year'].min()}-{sub['accident_year'].max()}, "
              f"dev lags {sub['development_lag'].min()}-{sub['development_lag'].max()}")


if __name__ == "__main__":
    main()
