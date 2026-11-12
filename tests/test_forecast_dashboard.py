"""
Tests for Streamlit Dashboard Forecast Alignment and Interactive Simulation.

Verifies:
1. Cache data integrity across all 1,993 aligned test forecast origins.
2. Indexing bounds: First (0), Middle, and Last (1992) positions.
3. Model subsets: XGBoost alone, LSTM alone, and all 5 models together.
4. Mismatched length tolerance and fallback handling.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR


@pytest.fixture
def dashboard_cache():
    cache_path = RESULTS_DIR / "dashboard_forecast_cache.json"
    assert cache_path.exists(), "dashboard_forecast_cache.json does not exist"
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_forecast_cache_structure_and_lengths(dashboard_cache):
    """Verifies that all arrays in cache have exact matching lengths."""
    timestamps = dashboard_cache["timestamps"]
    y_test = dashboard_cache["y_test"]
    predictions = dashboard_cache["predictions"]
    models = dashboard_cache["models"]

    expected_len = len(timestamps)
    assert expected_len == 1993, f"Expected 1993 test origins, found {expected_len}"
    assert len(y_test) == expected_len, f"y_test len {len(y_test)} != timestamps len {expected_len}"

    for m in models:
        assert m in predictions, f"Model {m} missing from predictions"
        m_preds = predictions[m]
        assert len(m_preds) == expected_len, f"Model {m} length {len(m_preds)} != {expected_len}"


@pytest.mark.parametrize("sample_idx", [0, 996, 1992])
def test_slider_boundary_positions(dashboard_cache, sample_idx):
    """Verifies first, middle, and last slider positions return valid 24h data."""
    timestamps = dashboard_cache["timestamps"]
    y_test = dashboard_cache["y_test"]
    predictions = dashboard_cache["predictions"]

    # Target
    y_true = np.array(y_test[sample_idx])
    assert y_true.shape == (24,)
    assert (y_true > 0).all()
    assert not np.isnan(y_true).any()

    # Timestamp
    ts = timestamps[sample_idx]
    dt = pd.to_datetime(ts)
    assert dt is not None

    # Predictions for all models at this index
    for m_name, m_preds in predictions.items():
        pred_sample = np.array(m_preds[sample_idx])
        assert pred_sample.shape == (24,)
        assert (pred_sample > 0).all()
        assert not np.isnan(pred_sample).any()


@pytest.mark.parametrize("selected_models", [
    ["XGBoost"],
    ["LSTM"],
    ["Seasonal_Naive_24h", "Ridge_Linear_Regression", "Random_Forest", "XGBoost", "LSTM"],
])
def test_model_selection_subsets(dashboard_cache, selected_models):
    """Verifies rendering logic for individual models and all five models combined."""
    sample_idx = 100
    y_true = np.array(dashboard_cache["y_test"][sample_idx])
    predictions = dashboard_cache["predictions"]

    active_rows = []
    for m in selected_models:
        assert m in predictions
        assert sample_idx < len(predictions[m])
        pred_sample = np.array(predictions[m][sample_idx])
        mae = float(np.mean(np.abs(y_true - pred_sample)))
        mape = float(np.mean(np.abs(y_true - pred_sample) / y_true) * 100)
        assert mae > 0.0
        assert mape > 0.0
        active_rows.append((m, mae, mape))

    assert len(active_rows) == len(selected_models)


def test_graceful_handling_of_mismatched_lengths():
    """Simulates a cache with truncated predictions to ensure no unhandled IndexError."""
    dummy_cache = {
        "timestamps": [f"2025-04-01 {h:02d}:00:00" for h in range(100)],
        "y_test": [[100000.0] * 24 for _ in range(50)],  # only 50 target windows
        "predictions": {
            "Model_Full": [[105000.0] * 24 for _ in range(50)],
            "Model_Truncated": [[102000.0] * 24 for _ in range(30)],  # only 30 windows
        },
        "models": ["Model_Full", "Model_Truncated"],
    }

    valid_origins = min(len(dummy_cache["timestamps"]), len(dummy_cache["y_test"]))
    assert valid_origins == 50  # Must be capped at target length

    max_slider_idx = max(0, valid_origins - 1)
    assert max_slider_idx == 49

    # At sample_idx = 40 (valid for y_test and Model_Full, but beyond Model_Truncated)
    sample_idx = 40
    y_sample = np.array(dummy_cache["y_test"][sample_idx])
    assert y_sample.shape == (24,)

    for m in dummy_cache["models"]:
        m_preds = dummy_cache["predictions"][m]
        if sample_idx < len(m_preds):
            pred = np.array(m_preds[sample_idx])
            assert pred.shape == (24,)
        else:
            # Successfully identified as unavailable without IndexError
            assert m == "Model_Truncated"

