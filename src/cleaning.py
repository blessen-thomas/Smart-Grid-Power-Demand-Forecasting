"""
Data Cleaning and Validation Module for Smart Grid Power Demand Forecasting.

Adheres strictly to project constraints:
1. Does NOT alter raw data files.
2. Documents all preprocessing decisions without silent removals.
3. Validates chronological continuity with zero missing hours.
4. Handles and explains edge-case values (e.g. negative solar readings).
"""

from typing import Dict, Any
import json
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_CSV,
    PROCESSED_DATA_PARQUET,
    COL_TIMESTAMP,
    NUMERIC_COLUMNS,
    COL_SOLAR,
    COL_DEMAND,
    COL_TOTAL_GEN,
    COL_WIND,
)
from src.data_loader import load_raw_dataset


def clean_grid_data(
    raw_path=RAW_DATA_PATH,
    save_processed: bool = True,
) -> pd.DataFrame:
    """
    Cleans raw grid data, converts data types, audits continuity and values,
    and documents every transformation decision.

    Returns
    -------
    pd.DataFrame
        Cleaned, chronologically ordered hourly dataset.
    """
    raw_hourly_df, footer_df = load_raw_dataset(raw_path)

    # Preprocessing Decision 1: Separation of Report Footer
    # Raw sheet contains 6 metadata rows at the bottom (Minimum, Maximum, Average, Sum, etc.)
    # These must be separated from time-series observations.
    df = raw_hourly_df.copy()

    # Preprocessing Decision 2: Timestamp Parsing and Ordering
    # Parse format DD-MM-YYYY HH:MM:SS to standard datetime
    df[COL_TIMESTAMP] = pd.to_datetime(
        df[COL_TIMESTAMP], format="%d-%m-%Y %H:%M:%S"
    )
    df = df.sort_values(COL_TIMESTAMP).reset_index(drop=True)

    # Preprocessing Decision 3: Numerical Type Conversion
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Preprocessing Decision 4: Continuity Verification
    min_time = df[COL_TIMESTAMP].min()
    max_time = df[COL_TIMESTAMP].max()
    full_range = pd.date_range(start=min_time, end=max_time, freq="h")

    if len(full_range) != len(df):
        missing = set(full_range) - set(df[COL_TIMESTAMP])
        raise ValueError(
            f"Chronological continuity check failed! Missing {len(missing)} hours: {sorted(list(missing))[:5]}"
        )

    # Preprocessing Decision 5: Solar Negative Readings Audit
    # The dataset contains 499 hourly records where Solar (MW) drops slightly below zero (min: -69.76 MW).
    # Engineering hypothesis: In power systems, negative solar readings commonly stem from nighttime
    # auxiliary consumption (inverter/transformer standby loads) or sensor/transducer calibration offset.
    # CRITICAL AUDIT RULE: Because the dataset source provides no calibration logs or sensor metadata,
    # this hypothesis cannot be treated as verified fact.
    # Therefore, the original measurements in 'Solar (MW)' are strictly preserved without alteration.
    # An experimental derived feature 'Solar_Calibrated (MW)' (clipped at 0) is created for analytical
    # comparison only, and is explicitly NOT treated as an established physical ground-truth correction.
    df["Solar_Calibrated (MW)"] = np.maximum(df[COL_SOLAR], 0.0)

    # Preprocessing Decision 6: Derived Physical Aggregations & Anti-Leakage Notice
    # Definitions:
    # - Renewable_Generation (MW) = Wind (MW) + Solar_Calibrated (MW)
    # - Net_Demand (MW) = Demand (MW) - Renewable_Generation (MW)
    # PREDICTOR JUSTIFICATION & ANTI-LEAKAGE RULE:
    # In a 24-hour ahead demand forecasting pipeline operating at origin t:
    # Future wind, solar, and total generation at t+1 ... t+24 are UNKNOWN in real operations
    # unless separate external forecast models exist. Using concurrent future values as predictors
    # constitutes direct data leakage.
    # Therefore, Renewable_Generation and Net_Demand are computed solely for historical exploratory
    # analysis and must NEVER be supplied as concurrent future predictors for t+k.
    df["Renewable_Generation (MW)"] = df[COL_WIND] + df["Solar_Calibrated (MW)"]
    df["Net_Demand (MW)"] = df[COL_DEMAND] - df["Renewable_Generation (MW)"]

    if save_processed:
        PROCESSED_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_CSV, index=False)
        try:
            df.to_parquet(PROCESSED_DATA_PARQUET, index=False)
        except Exception as e:
            print(f"Parquet export skipped or failed ({e}); CSV saved successfully.")

        # Document decisions to JSON
        decisions_log = generate_cleaning_audit_report(df, footer_df)
        log_path = PROCESSED_DATA_CSV.parent / "preprocessing_audit_report.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(decisions_log, f, indent=4, default=str)

    return df


def generate_cleaning_audit_report(
    cleaned_df: pd.DataFrame, footer_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Creates an explicit audit report detailing all data characteristics and decisions.
    """
    neg_solar_count = int((cleaned_df[COL_SOLAR] < 0).sum())
    min_solar = float(cleaned_df[COL_SOLAR].min())

    report = {
        "dataset_name": "Indian Electricity Grid Hourly Data (Jan 2024 - Jun 2025)",
        "total_raw_rows": len(cleaned_df) + len(footer_df),
        "cleaned_hourly_rows": len(cleaned_df),
        "start_timestamp": str(cleaned_df[COL_TIMESTAMP].min()),
        "end_timestamp": str(cleaned_df[COL_TIMESTAMP].max()),
        "expected_hourly_count": len(cleaned_df),
        "missing_timestamps_count": 0,
        "duplicate_timestamps_count": int(
            cleaned_df[COL_TIMESTAMP].duplicated().sum()
        ),
        "null_values_per_column": cleaned_df.isnull().sum().to_dict(),
        "unusual_readings": {
            "negative_solar_readings_count": neg_solar_count,
            "min_solar_mw": min_solar,
            "hypothesis_and_status": (
                "Nighttime solar values drop slightly below zero (up to -69.76 MW, 499 occurrences). "
                "While power system literature suggests this frequently arises from inverter standby loads "
                "or CT/PT transducer calibration offset, no metadata is provided to establish this as verified fact. "
                "The original readings are strictly preserved in 'Solar (MW)'. "
                "'Solar_Calibrated (MW)' is provided purely as an experimental derived feature, not an established correction."
            ),
        },
        "anti_leakage_policy": {
            "predictor_justification": (
                "Actual future wind, solar, and generation measurements at t+1...t+24 are not available at forecast origin t. "
                "Therefore, Renewable_Generation and Net_Demand cannot be used as future concurrent predictors without external forecasts. "
                "Only historical observations up to forecast origin t are admissible."
            )
        },
        "footer_metadata_rows_isolated": len(footer_df),
        "footer_metadata_content": footer_df.to_dict(orient="records"),
        "preprocessing_decisions": [
            {
                "decision": "Footer Isolation",
                "reason": "Rows 12984-12989 contained summary statistics (Min, Max, Avg, Sum) appended by the export system.",
            },
            {
                "decision": "Timestamp Parsing",
                "format": "%d-%m-%Y %H:%M:%S",
                "reason": "Standardizing to ISO-compatible pandas datetime and sorting chronologically.",
            },
            {
                "decision": "Continuity Validation",
                "outcome": "Strictly 12,984 consecutive hours from 2024-01-01 00:00 to 2025-06-24 23:00 without any gap.",
            },
            {
                "decision": "No Data Invention / No Silent Deletion",
                "outcome": "Zero imputation was needed because there are zero missing records. No rows deleted.",
            },
        ],
    }
    return report


if __name__ == "__main__":
    cleaned = clean_grid_data()
    print("Data cleaning completed successfully.")
    print("Cleaned shape:", cleaned.shape)
    print("Head:\n", cleaned.head(3))

