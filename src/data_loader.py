"""
Data loading module for Smart Grid Power Demand Forecasting.
Loads raw hourly grid data without modifying the source file.
Identifies and separates data records from report summary rows.
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    RAW_SHEET_NAME,
    COL_TIMESTAMP,
    NUMERIC_COLUMNS,
)


def load_raw_dataset(
    file_path: Path = RAW_DATA_PATH,
    sheet_name: str = RAW_SHEET_NAME,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw Excel workbook, cleanly separating the hourly time-series records
    from trailing summary/footer metadata rows.

    Parameters
    ----------
    file_path : Path
        Path to the raw Excel spreadsheet.
    sheet_name : str
        Sheet name containing the grid data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - raw_hourly_df: DataFrame containing the hourly time series records.
        - footer_metadata_df: DataFrame containing any trailing summary rows (Min, Max, Avg, Sum).
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at: {file_path}. "
            "Please ensure 'January 2024- June 2025.xlsx' is placed in dataset/raw/."
        )

    # Read the sheet preserving original values as strings/objects to avoid silent coercion
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Ensure expected columns are present
    expected_cols = [COL_TIMESTAMP] + NUMERIC_COLUMNS
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    # Inspect the Timestamp column to identify actual hourly records vs report summaries
    # Format expected: DD-MM-YYYY HH:MM:SS
    parsed_timestamps = pd.to_datetime(
        df[COL_TIMESTAMP], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )

    is_valid_timestamp = parsed_timestamps.notnull()
    hourly_records_df = df[is_valid_timestamp].copy()
    footer_metadata_df = df[~is_valid_timestamp].copy()

    return hourly_records_df, footer_metadata_df


def inspect_raw_structure(file_path: Path = RAW_DATA_PATH) -> Dict[str, Any]:
    """
    Generates an inspection summary of the raw Excel file.
    """
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names

    hourly_df, footer_df = load_raw_dataset(file_path)

    summary = {
        "file_path": str(file_path),
        "sheets": sheet_names,
        "total_raw_rows": len(hourly_df) + len(footer_df),
        "valid_hourly_rows": len(hourly_df),
        "footer_metadata_rows": len(footer_df),
        "footer_sample": footer_df.to_dict(orient="records"),
        "columns": list(hourly_df.columns),
    }
    return summary


if __name__ == "__main__":
    info = inspect_raw_structure()
    print("Raw Data Inspection:")
    for k, v in info.items():
        if k != "footer_sample":
            print(f"  {k}: {v}")
    print(f"  footer_metadata_rows count: {len(info['footer_sample'])}")

