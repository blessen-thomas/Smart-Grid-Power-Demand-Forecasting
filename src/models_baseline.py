"""
Baseline and Classical Machine Learning Forecasting Models for 24-Hour Smart Grid Demand.

Models implemented:
1. Seasonal Naive 24h (Persistence benchmark predicting same diurnal hour from yesterday)
2. Seasonal Naive 168h (Weekly benchmark predicting same hour from last week)
3. Ridge Linear Regression (L2 regularized direct multi-horizon linear model)
4. Random Forest Regressor (Non-linear ensemble multi-output trees)
5. XGBoost Regressor (Multi-target gradient boosted decision trees)
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.config import MODELS_DIR, RESULTS_DIR
from src.features import create_forecasting_dataset, split_chronological_data
from src.evaluate import calculate_metrics, analyze_peak_demand_errors


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Baseline forecaster for direct 24-hour horizon.
    Predicts horizon h in 1..24 using historical demand from lag (24 - h) or weekly lag.
    """
    def __init__(self, lag_period: int = 24):
        self.lag_period = lag_period

    def predict(self, df_raw_with_lags: pd.DataFrame, mask: pd.Series) -> np.ndarray:
        """
        Extracts diurnal naive predictions from precomputed lags.
        For horizon h in 1..24 at origin t:
        If lag_period == 24: prediction is Demand(t + h - 24) = Demand(t - (24 - h)).
        For example:
        h=1  -> lag 23h (Demand at t-23)
        h=24 -> lag 0h  (Demand at t)
        """
        preds = []
        subset = df_raw_with_lags.loc[mask]
        for h in range(1, 25):
            lag_needed = self.lag_period - h
            col_name = f"demand_lag_{lag_needed}h"
            if col_name in subset.columns:
                preds.append(subset[col_name].values)
            else:
                # Fallback to closest available lag
                preds.append(subset["demand_lag_0h"].values)
        return np.column_stack(preds)


def train_evaluate_baselines_and_ml(
    splits: Dict[str, Any],
    df_feat: pd.DataFrame,
    save_models: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Trains and validates baseline and ML models. Evaluates on validation and final test sets.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, np.ndarray]]
        - results_dict: comprehensive metric logs
        - predictions_dict: test set predictions for all models
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    val_mask = (df_feat["Timestamp"] > splits["timestamps_train"][-1]) & (df_feat["Timestamp"] <= splits["timestamps_val"][-1])
    test_mask = df_feat["Timestamp"] > splits["timestamps_val"][-1]

    results = {}
    test_predictions = {}

    # 1. Seasonal Naive 24h Baseline
    print("Evaluating Model 1: Seasonal Naive 24h Baseline...")
    naive24 = SeasonalNaiveForecaster(lag_period=24)
    val_pred_naive = naive24.predict(df_feat, val_mask)
    test_pred_naive = naive24.predict(df_feat, test_mask)

    test_predictions["Seasonal_Naive_24h"] = test_pred_naive
    results["Seasonal_Naive_24h"] = {
        "val_metrics": calculate_metrics(y_val, val_pred_naive),
        "val_peak": analyze_peak_demand_errors(y_val, val_pred_naive),
        "test_metrics": calculate_metrics(y_test, test_pred_naive),
        "test_peak": analyze_peak_demand_errors(y_test, test_pred_naive),
    }

    # 2. Linear Regression (Ridge)
    print("Training Model 2: Ridge Linear Regression...")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train, y_train)
    val_pred_ridge = ridge.predict(X_val)
    test_pred_ridge = ridge.predict(X_test)

    if save_models:
        joblib.dump(ridge, MODELS_DIR / "ridge_regression.joblib")

    test_predictions["Ridge_Linear_Regression"] = test_pred_ridge
    results["Ridge_Linear_Regression"] = {
        "val_metrics": calculate_metrics(y_val, val_pred_ridge),
        "val_peak": analyze_peak_demand_errors(y_val, val_pred_ridge),
        "test_metrics": calculate_metrics(y_test, test_pred_ridge),
        "test_peak": analyze_peak_demand_errors(y_test, test_pred_ridge),
    }

    # 3. Random Forest Regressor
    print("Training Model 3: Random Forest Regressor (100 trees)...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    val_pred_rf = rf.predict(X_val)
    test_pred_rf = rf.predict(X_test)

    if save_models:
        joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    test_predictions["Random_Forest"] = test_pred_rf
    results["Random_Forest"] = {
        "val_metrics": calculate_metrics(y_val, val_pred_rf),
        "val_peak": analyze_peak_demand_errors(y_val, val_pred_rf),
        "test_metrics": calculate_metrics(y_test, test_pred_rf),
        "test_peak": analyze_peak_demand_errors(y_test, test_pred_rf),
    }

    # 4. XGBoost Regressor
    print("Training Model 4: XGBoost Regressor (150 estimators)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred_xgb = xgb_model.predict(X_val)
    test_pred_xgb = xgb_model.predict(X_test)

    if save_models:
        joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")

    test_predictions["XGBoost"] = test_pred_xgb
    results["XGBoost"] = {
        "val_metrics": calculate_metrics(y_val, val_pred_xgb),
        "val_peak": analyze_peak_demand_errors(y_val, val_pred_xgb),
        "test_metrics": calculate_metrics(y_test, test_pred_xgb),
        "test_peak": analyze_peak_demand_errors(y_test, test_pred_xgb),
    }

    return results, test_predictions


if __name__ == "__main__":
    df_feat, feats, targets = create_forecasting_dataset()
    splits = split_chronological_data(df_feat, feats, targets)
    results, preds = train_evaluate_baselines_and_ml(splits, df_feat)
    print("\n--- Model Evaluation Summary (Test Set) ---")
    for model_name, res in results.items():
        tm = res["test_metrics"]
        tp = res["test_peak"]
        print(f"{model_name:25s} | MAE: {tm['MAE_MW']:8.2f} MW | RMSE: {tm['RMSE_MW']:8.2f} MW | MAPE: {tm['MAPE_percent']:5.2f}% | R2: {tm['R2_score']:.4f} | Peak MAE: {tp['Peak_Demand_MAE_MW']:8.2f} MW")

