"""
Technical Verification Test Suite for Smart Grid Power Demand Forecasting.
Verifies data integrity, anti-leakage compliance, 5-model multi-horizon predictions,
metric alignment, and dashboard assets.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_CSV,
    RESULTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    COL_TIMESTAMP,
    COL_DEMAND,
    COL_SOLAR,
    COL_WIND,
    COL_TOTAL_GEN,
    TRAIN_END_DATE,
    VAL_END_DATE,
    FORECAST_HORIZON_HOURS,
)
from src.data_loader import load_raw_dataset
from src.cleaning import clean_grid_data
from src.features import create_forecasting_dataset, split_chronological_data
from src.evaluate import calculate_metrics, analyze_peak_demand_errors


class TestDataIntegrity:
    """Verifies raw data preservation and cleaning accuracy."""

    def test_raw_dataset_unchanged(self):
        assert RAW_DATA_PATH.exists(), f"Raw data file missing at {RAW_DATA_PATH}"
        hourly_df, footer_df = load_raw_dataset(RAW_DATA_PATH)
        assert len(hourly_df) == 12984, f"Expected 12,984 hourly rows, found {len(hourly_df)}"
        assert len(footer_df) == 6, f"Expected 6 footer summary rows, found {len(footer_df)}"

    def test_cleaned_data_continuity_and_nulls(self):
        df = clean_grid_data(save_processed=False)
        assert len(df) == 12984
        assert df.isnull().sum().sum() == 0, "Cleaned dataset contains unexpected nulls"

        # Continuity check
        start_ts = df[COL_TIMESTAMP].min()
        end_ts = df[COL_TIMESTAMP].max()
        expected_range = pd.date_range(start=start_ts, end=end_ts, freq="h")
        assert len(expected_range) == len(df), "Hourly timeline has gaps"
        assert df[COL_TIMESTAMP].duplicated().sum() == 0, "Duplicate timestamps found"

    def test_solar_readings_preserved_and_calibrated(self):
        df = clean_grid_data(save_processed=False)
        neg_solar_count = (df[COL_SOLAR] < 0).sum()
        assert neg_solar_count == 499, f"Expected 499 negative solar readings, found {neg_solar_count}"
        assert (df["Solar_Calibrated (MW)"] >= 0).all(), "Solar_Calibrated contains negative values"


class TestAntiLeakageAndFeatures:
    """Verifies strict adherence to time-series anti-leakage rules."""

    def test_no_future_generation_leakage_in_features(self):
        df_feat, feature_cols, target_cols = create_forecasting_dataset(save_features=False)

        # Ensure target horizon is 24
        assert len(target_cols) == FORECAST_HORIZON_HOURS
        assert target_cols[0] == "target_demand_h1"
        assert target_cols[-1] == "target_demand_h24"

        # Check feature definitions: NO feature must be a future target or future generation
        for feat in feature_cols:
            assert not feat.startswith("target_"), f"Target leaked into features: {feat}"
            assert "lead" not in feat.lower(), f"Lead feature leaked: {feat}"
            # Ensure wind and solar features are strictly past/current lags
            if "wind" in feat or "solar" in feat or "gen" in feat:
                assert "lag" in feat, f"Unlagged or future generation feature: {feat}"
                # Must be lag 0h or older (e.g. lag 24h)
                assert feat.endswith("0h") or feat.endswith("24h"), f"Unsafe generation lag: {feat}"

    def test_chronological_splits_and_scaler_isolation(self):
        df_feat, feature_cols, target_cols = create_forecasting_dataset(save_features=False)
        splits = split_chronological_data(df_feat, feature_cols, target_cols)

        # Temporal ordering
        ts_train = pd.to_datetime(splits["timestamps_train"])
        ts_val = pd.to_datetime(splits["timestamps_val"])
        ts_test = pd.to_datetime(splits["timestamps_test"])

        assert ts_train.max() <= pd.to_datetime(TRAIN_END_DATE)
        assert ts_val.min() > pd.to_datetime(TRAIN_END_DATE)
        assert ts_val.max() <= pd.to_datetime(VAL_END_DATE)
        assert ts_test.min() > pd.to_datetime(VAL_END_DATE)

        # Scaler check: mean of training set should be ~0 after scaling
        X_train_mean = np.mean(splits["X_train"], axis=0)
        np.testing.assert_allclose(X_train_mean, 0.0, atol=1e-3, err_msg="X_train is not properly standardized")

        # Validation set mean should NOT be forced to 0 (since fitted strictly on train!)
        X_val_mean = np.mean(splits["X_val"], axis=0)
        assert not np.allclose(X_val_mean, 0.0, atol=1e-2), "Validation data was leaked into scaler fitting"


class TestModelsAndMetricsVerification:
    """Verifies all five models produce valid 24h predictions matching reported metrics."""

    @pytest.fixture
    def cache_and_comparison(self):
        json_path = RESULTS_DIR / "model_comparison.json"
        csv_path = RESULTS_DIR / "model_comparison.csv"
        cache_path = RESULTS_DIR / "dashboard_forecast_cache.json"

        assert json_path.exists(), "results/model_comparison.json missing"
        assert csv_path.exists(), "results/model_comparison.csv missing"
        assert cache_path.exists(), "results/dashboard_forecast_cache.json missing"

        with open(json_path, "r", encoding="utf-8") as f:
            comp_json = json.load(f)
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_json = json.load(f)
        comp_df = pd.read_csv(csv_path)

        return comp_json, comp_df, cache_json

    def test_all_five_models_present(self, cache_and_comparison):
        comp_json, comp_df, cache_json = cache_and_comparison
        expected_models = {
            "Seasonal_Naive_24h",
            "Ridge_Linear_Regression",
            "Random_Forest",
            "XGBoost",
            "LSTM",
        }
        assert set(comp_json.keys()) == expected_models
        assert set(comp_df["Model"].values) == expected_models
        assert set(cache_json["models"]) == expected_models

    def test_prediction_shapes_and_values(self, cache_and_comparison):
        _, _, cache_json = cache_and_comparison
        y_test = np.array(cache_json["y_test"])
        assert y_test.ndim == 2
        assert y_test.shape[1] == 24, "Target window is not 24 hours"

        for model_name in cache_json["models"]:
            preds = np.array(cache_json["predictions"][model_name])
            assert preds.shape == y_test.shape, f"{model_name} shape {preds.shape} != target {y_test.shape}"
            assert not np.isnan(preds).any(), f"{model_name} predictions contain NaNs"
            assert not np.isinf(preds).any(), f"{model_name} predictions contain Infs"
            assert (preds > 0).all(), f"{model_name} predicted negative demand values"

    def test_metrics_consistency_between_json_and_csv(self, cache_and_comparison):
        comp_json, comp_df, _ = cache_and_comparison
        for _, row in comp_df.iterrows():
            model_name = row["Model"]
            json_metrics = comp_json[model_name]["test_metrics"]
            assert pytest.approx(row["Test_MAE_MW"], rel=1e-3) == json_metrics["MAE_MW"]
            assert pytest.approx(row["Test_RMSE_MW"], rel=1e-3) == json_metrics["RMSE_MW"]
            assert pytest.approx(row["Test_MAPE_%"], rel=1e-3) == json_metrics["MAPE_percent"]
            assert pytest.approx(row["Test_R2"], rel=1e-3) == json_metrics["R2_score"]

    def test_xgboost_is_top_performer(self, cache_and_comparison):
        _, comp_df, _ = cache_and_comparison
        best_model = comp_df.loc[comp_df["Test_MAE_MW"].idxmin(), "Model"]
        assert best_model == "XGBoost", f"Expected XGBoost to be top model, found {best_model}"
        best_mape = comp_df.loc[comp_df["Model"] == "XGBoost", "Test_MAPE_%"].values[0]
        assert best_mape < 3.0, f"XGBoost MAPE {best_mape}% is unexpectedly high"


class TestDashboardAssets:
    """Verifies all assets and figures required by the Streamlit dashboard exist."""

    def test_evaluation_figures_exist(self):
        expected_figures = [
            "01_demand_time_series_and_splits.png",
            "02_diurnal_hourly_demand_profile.png",
            "03_weekly_and_monthly_demand_patterns.png",
            "04_generation_mix_and_correlation.png",
            "05_model_actual_vs_predicted.png",
            "06_model_error_distributions.png",
            "07_forecast_next_24_hours.png",
            "08_peak_demand_analysis.png",
        ]
        for fig in expected_figures:
            fig_path = FIGURES_DIR / fig
            assert fig_path.exists(), f"Dashboard figure missing: {fig_path}"
            assert fig_path.stat().st_size > 10000, f"Figure file seems empty or corrupted: {fig}"

    def test_trained_model_artifacts_exist(self):
        expected_artifacts = [
            "ridge_regression.joblib",
            "random_forest.joblib",
            "xgboost.joblib",
            "lstm_demand_forecaster.pt",
        ]
        for art in expected_artifacts:
            art_path = MODELS_DIR / art
            assert art_path.exists(), f"Model artifact missing: {art_path}"
            assert art_path.stat().st_size > 1000, f"Model file seems empty: {art}"

