"""
Configuration settings for Smart Grid Power Demand Forecasting.
Defines directory paths, column definitions, time-series splitting parameters,
and modeling constraints.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Raw Data Specification
RAW_FILE_NAME = "January 2024- June 2025.xlsx"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_FILE_NAME
RAW_SHEET_NAME = "Report"

# Geographical and Scope Note
# The raw dataset provides hourly grid measurements across 5 telemetry channels.
# Because the source workbook lacks explicit metadata on regional versus national boundaries
# (e.g., specific RLDC or NLDC dispatch boundaries), no geographical boundaries are assumed.

# Column Definitions
COL_TIMESTAMP = "Timestamp"
COL_DEMAND = "Demand (MW)"
COL_WIND = "Wind (MW)"
COL_SOLAR = "Solar (MW)"
COL_TOTAL_GEN = "Total Generation (MW)"

NUMERIC_COLUMNS = [COL_DEMAND, COL_WIND, COL_SOLAR, COL_TOTAL_GEN]
TARGET_COLUMN = COL_DEMAND

# Cleaned Data Output
PROCESSED_DATA_CSV = PROCESSED_DATA_DIR / "cleaned_hourly_grid_data.csv"
PROCESSED_DATA_PARQUET = PROCESSED_DATA_DIR / "cleaned_hourly_grid_data.parquet"

# Forecasting Constraints
FORECAST_HORIZON_HOURS = 24  # Next 24 hours forecasting objective

# Chronological Time-Series Splitting Parameters
# Strictly temporal: Train -> Validation -> Test (No shuffling, no leakage)
# Total hourly observations: 12,984 (2024-01-01 00:00:00 to 2025-06-24 23:00:00)
# Train: 2024-01-01 to 2024-12-31 (8,784 hours, ~67.65%)
# Validation: 2025-01-01 to 2025-03-31 (2,160 hours, ~16.64%)
# Test: 2025-04-01 to 2025-06-24 (2,040 hours, ~15.71%)
TRAIN_END_DATE = "2024-12-31 23:00:00"
VAL_END_DATE = "2025-03-31 23:00:00"

# Project Metadata
PROJECT_INFO = {
    "Project": "Smart Grid Power Demand Forecasting",
    "Description": "Machine Learning-Based Electricity Demand Forecasting",
}

