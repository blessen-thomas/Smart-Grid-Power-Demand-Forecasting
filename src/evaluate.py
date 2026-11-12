"""
Evaluation metrics and utility module for 24-Hour Smart Grid Demand Forecasting.

Implements fair, common metrics:
- MAE (Mean Absolute Error, MW)
- RMSE (Root Mean Squared Error, MW)
- MAPE (Mean Absolute Percentage Error, %) with documented limitations
- R2 (Coefficient of Determination)
- Peak Demand Analysis (Peak Hour MAE, Peak Under/Over-prediction)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes global evaluation metrics across all 24 horizons.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth demand values (shape: [N, 24] or [N*24]).
    y_pred : np.ndarray
        Predicted demand values (shape: [N, 24] or [N*24]).

    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, RMSE, MAPE, and R2.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # MAPE: Handled carefully. Since grid demand is strictly >= 97,000 MW,
    # division by zero is not an issue, but percentage error is computed reliably.
    epsilon = 1e-8
    mape = float(np.mean(np.abs((y_t - y_p) / (y_t + epsilon))) * 100.0)

    r2 = float(r2_score(y_t, y_p))

    return {
        "MAE_MW": round(mae, 2),
        "RMSE_MW": round(rmse, 2),
        "MAPE_percent": round(mape, 3),
        "R2_score": round(r2, 4),
    }


def analyze_peak_demand_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates forecasting accuracy specifically during daily peak demand periods.
    For each 24-hour day forecast, finds the true peak hour and compares with prediction.
    """
    y_true_2d = np.asarray(y_true)
    y_pred_2d = np.asarray(y_pred)

    if y_true_2d.ndim == 1:
        y_true_2d = y_true_2d.reshape(-1, 24)
        y_pred_2d = y_pred_2d.reshape(-1, 24)

    # True daily maximums and corresponding predictions
    true_daily_peaks = np.max(y_true_2d, axis=1)
    true_peak_indices = np.argmax(y_true_2d, axis=1)

    # Predicted value at the exact hour of true peak
    pred_at_true_peaks = np.array([y_pred_2d[i, true_peak_indices[i]] for i in range(len(true_daily_peaks))])
    
    # Peak predicted by the model anywhere in the 24h window
    pred_daily_peaks = np.max(y_pred_2d, axis=1)
    pred_peak_indices = np.argmax(y_pred_2d, axis=1)

    peak_mae = float(np.mean(np.abs(true_daily_peaks - pred_at_true_peaks)))
    peak_bias = float(np.mean(pred_at_true_peaks - true_daily_peaks))  # Negative = underestimating peaks
    peak_timing_mae_hours = float(np.mean(np.abs(true_peak_indices - pred_peak_indices)))

    return {
        "Peak_Demand_MAE_MW": round(peak_mae, 2),
        "Peak_Demand_Bias_MW": round(peak_bias, 2),
        "Peak_Timing_MAE_Hours": round(peak_timing_mae_hours, 2),
    }


def compute_horizon_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Computes MAE, RMSE, and MAPE for each individual horizon step h=1..24."""
    y_true_2d = np.asarray(y_true)
    y_pred_2d = np.asarray(y_pred)

    rows = []
    for h in range(y_true_2d.shape[1]):
        step_true = y_true_2d[:, h]
        step_pred = y_pred_2d[:, h]
        metrics = calculate_metrics(step_true, step_pred)
        metrics["Horizon_Hour"] = h + 1
        rows.append(metrics)

    return pd.DataFrame(rows)

