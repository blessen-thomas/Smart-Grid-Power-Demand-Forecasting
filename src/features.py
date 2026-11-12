"""
Feature Engineering Module for Smart Grid Power Demand Forecasting.

Constructs anti-leakage time-series features for 24-hour ahead direct multi-horizon forecasting:
- Temporal calendar features (hour, day, weekday, month, weekend)
- Cyclical trigonometric encodings (sin/cos of hour, weekday, month)
- Historical demand lags known at forecast origin t (t, t-1, t-2, t-23, t-47, t-167)
- Past rolling statistics (mean, std, min, max over 24h and 168h windows) strictly closed on the past
- Historical telemetry lags (Wind, Solar, Total Generation at origin t)
- Direct multi-horizon targets: [Demand(t+1), Demand(t+2), ..., Demand(t+24)]
"""

from typing import Tuple, List, Dict, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_CSV,
    PROCESSED_DATA_DIR,
    COL_TIMESTAMP,
    COL_DEMAND,
    COL_WIND,
    COL_SOLAR,
    COL_TOTAL_GEN,
    FORECAST_HORIZON_HOURS,
    TRAIN_END_DATE,
    VAL_END_DATE,
)


def create_forecasting_dataset(
    df: pd.DataFrame = None,
    horizon: int = FORECAST_HORIZON_HOURS,
    save_features: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Constructs feature matrix X and multi-horizon target matrix Y for each forecast origin t.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Cleaned hourly DataFrame. If None, loaded from PROCESSED_DATA_CSV.
    horizon : int
        Number of forward hours to predict (default 24).
    save_features : bool
        Whether to export the resulting dataset to disk.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        - dataset_df: complete DataFrame containing features and 24-hour targets
        - feature_cols: list of input feature column names
        - target_cols: list of target column names [Demand_lead_1, ..., Demand_lead_24]
    """
    if df is None:
        if not PROCESSED_DATA_CSV.exists():
            from src.cleaning import clean_grid_data
            df = clean_grid_data()
        else:
            df = pd.read_csv(PROCESSED_DATA_CSV)
            df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])

    df = df.sort_values(COL_TIMESTAMP).reset_index(drop=True)

    # 1. Temporal & Calendar Features at origin t
    df["hour"] = df[COL_TIMESTAMP].dt.hour
    df["day"] = df[COL_TIMESTAMP].dt.day
    df["dayofweek"] = df[COL_TIMESTAMP].dt.dayofweek
    df["month"] = df[COL_TIMESTAMP].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 2. Cyclical Trigonometric Encodings
    # Captures periodic continuity (e.g. 23:00 to 00:00, Sunday to Monday, Dec to Jan)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    # 3. Historical Demand Lags at forecast origin t
    # Lag 0 is current demand at time t; lag k is demand at t-k
    lags = [0, 1, 2, 3, 23, 24, 47, 48, 167, 168]
    lag_feature_cols = []
    for lag in lags:
        col_name = f"demand_lag_{lag}h"
        df[col_name] = df[COL_DEMAND].shift(lag)
        lag_feature_cols.append(col_name)

    # 4. Past Rolling Statistics (Strictly closed on the past, shift(1) or current lag 0)
    # Using shift(0) closed on [t-23, t] (past 24h) and [t-167, t] (past 168h / 7 days)
    df["demand_rolling_mean_24h"] = df[COL_DEMAND].rolling(window=24, closed="right").mean()
    df["demand_rolling_std_24h"] = df[COL_DEMAND].rolling(window=24, closed="right").std()
    df["demand_rolling_min_24h"] = df[COL_DEMAND].rolling(window=24, closed="right").min()
    df["demand_rolling_max_24h"] = df[COL_DEMAND].rolling(window=24, closed="right").max()

    df["demand_rolling_mean_168h"] = df[COL_DEMAND].rolling(window=168, closed="right").mean()
    df["demand_rolling_std_168h"] = df[COL_DEMAND].rolling(window=168, closed="right").std()

    # 5. Historical Telemetry Lags (Observed at or before origin t)
    # Never using future generation!
    df["wind_lag_0h"] = df[COL_WIND]
    df["wind_lag_24h"] = df[COL_WIND].shift(24)
    df["solar_lag_0h"] = df[COL_SOLAR]
    df["solar_lag_24h"] = df[COL_SOLAR].shift(24)
    df["total_gen_lag_0h"] = df[COL_TOTAL_GEN]
    df["total_gen_lag_24h"] = df[COL_TOTAL_GEN].shift(24)

    # 6. Direct Multi-Horizon Forecast Targets
    # For forecast origin t, predict Demand(t + h) for h in 1..24
    target_cols = []
    for h in range(1, horizon + 1):
        target_name = f"target_demand_h{h}"
        df[target_name] = df[COL_DEMAND].shift(-h)
        target_cols.append(target_name)

    # Define full feature column list
    feature_cols = [
        "hour",
        "day",
        "dayofweek",
        "month",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dayofweek",
        "cos_dayofweek",
        "sin_month",
        "cos_month",
        "demand_rolling_mean_24h",
        "demand_rolling_std_24h",
        "demand_rolling_min_24h",
        "demand_rolling_max_24h",
        "demand_rolling_mean_168h",
        "demand_rolling_std_168h",
        "wind_lag_0h",
        "wind_lag_24h",
        "solar_lag_0h",
        "solar_lag_24h",
        "total_gen_lag_0h",
        "total_gen_lag_24h",
    ] + lag_feature_cols

    # Drop warm-up rows (first 168 hours due to 168h rolling/lags)
    # and drop final 24 rows where future targets are beyond dataset end
    valid_mask = df[feature_cols + target_cols].notnull().all(axis=1)
    dataset_df = df[valid_mask].copy().reset_index(drop=True)

    if save_features:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        feat_path = PROCESSED_DATA_DIR / "forecasting_features.csv"
        dataset_df.to_csv(feat_path, index=False)
        print(f"Features dataset successfully saved to: {feat_path}")
        print(f"Total valid samples: {len(dataset_df)} rows, {len(feature_cols)} features, {len(target_cols)} targets.")

    return dataset_df, feature_cols, target_cols


def split_chronological_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    train_end: str = TRAIN_END_DATE,
    val_end: str = VAL_END_DATE,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Performs strictly chronological train/validation/test splitting.
    Scales features by fitting StandardScaler strictly on the training set.

    Returns dictionary with:
    - X_train, y_train, timestamps_train
    - X_val, y_val, timestamps_val
    - X_test, y_test, timestamps_test
    - scaler, feature_cols, target_cols
    """
    train_mask = df[COL_TIMESTAMP] <= train_end
    val_mask = (df[COL_TIMESTAMP] > train_end) & (df[COL_TIMESTAMP] <= val_end)
    test_mask = df[COL_TIMESTAMP] > val_end

    X_train_raw = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, target_cols].values
    ts_train = df.loc[train_mask, COL_TIMESTAMP].values

    X_val_raw = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, target_cols].values
    ts_val = df.loc[val_mask, COL_TIMESTAMP].values

    X_test_raw = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, target_cols].values
    ts_test = df.loc[test_mask, COL_TIMESTAMP].values

    scaler = None
    if scale_features:
        # Fit scaler ONLY on training data to prevent leakage!
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    data_bundle = {
        "X_train": X_train,
        "y_train": y_train,
        "timestamps_train": ts_train,
        "X_val": X_val,
        "y_val": y_val,
        "timestamps_val": ts_val,
        "X_test": X_test,
        "y_test": y_test,
        "timestamps_test": ts_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "shapes": {
            "train": X_train.shape,
            "val": X_val.shape,
            "test": X_test.shape,
        },
    }
    return data_bundle


if __name__ == "__main__":
    df_feat, feats, targets = create_forecasting_dataset()
    splits = split_chronological_data(df_feat, feats, targets)
    print("Split Shapes:")
    for k, v in splits["shapes"].items():
        print(f"  {k}: {v}")

