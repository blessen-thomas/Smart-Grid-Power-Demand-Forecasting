"""
Comprehensive Model Comparison and Evaluation Runner for Smart Grid Power Demand Forecasting.

Benchmarks:
1. Seasonal Naive 24h
2. Ridge Linear Regression
3. Random Forest Regressor
4. XGBoost Regressor
5. LSTM Neural Network

Outputs:
- results/model_comparison.csv
- results/model_comparison.json
- results/figures/05_model_actual_vs_predicted.png
- results/figures/06_model_error_distributions.png
- results/figures/07_forecast_next_24_hours.png
- results/figures/08_peak_demand_analysis.png
"""

from typing import Dict, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RESULTS_DIR, FIGURES_DIR
from src.features import create_forecasting_dataset, split_chronological_data
from src.models_baseline import train_evaluate_baselines_and_ml
from src.models_lstm import train_lstm_model
from src.evaluate import calculate_metrics, analyze_peak_demand_errors


def run_full_model_comparison() -> pd.DataFrame:
    """Executes training, evaluation, comparison, and visual reporting for all models."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Preparing Features and Chronological Splits...")
    df_feat, feats, targets = create_forecasting_dataset()
    splits = split_chronological_data(df_feat, feats, targets)

    print("\nStep 2: Training & Evaluating Baselines, Ridge, Random Forest, and XGBoost...")
    results_dict, test_preds = train_evaluate_baselines_and_ml(splits, df_feat)

    print("\nStep 3: Training & Evaluating Deep Learning LSTM Model...")
    lstm_results, lstm_preds = train_lstm_model(splits, epochs=15)
    results_dict["LSTM"] = lstm_results["LSTM"]
    
    # LSTM sequence length offset alignment for test set
    lstm_offset = lstm_results["LSTM"]["eval_offset"]
    y_test_full = splits["y_test"]
    y_test_aligned = y_test_full[lstm_offset:]
    
    # Store aligned predictions
    aligned_preds = {}
    for name, pred in test_preds.items():
        aligned_preds[name] = pred[lstm_offset:]
    aligned_preds["LSTM"] = lstm_preds

    print("\nStep 4: Compiling Benchmark Metrics...")
    table_rows = []
    for model_name, info in results_dict.items():
        val_m = info["val_metrics"]
        test_m = info["test_metrics"]
        test_p = info["test_peak"]
        table_rows.append({
            "Model": model_name,
            "Val_MAE_MW": val_m["MAE_MW"],
            "Val_RMSE_MW": val_m["RMSE_MW"],
            "Val_MAPE_%": val_m["MAPE_percent"],
            "Val_R2": val_m["R2_score"],
            "Test_MAE_MW": test_m["MAE_MW"],
            "Test_RMSE_MW": test_m["RMSE_MW"],
            "Test_MAPE_%": test_m["MAPE_percent"],
            "Test_R2": test_m["R2_score"],
            "Test_Peak_MAE_MW": test_p["Peak_Demand_MAE_MW"],
            "Test_Peak_Bias_MW": test_p["Peak_Demand_Bias_MW"],
            "Test_Peak_Timing_MAE_h": test_p["Peak_Timing_MAE_Hours"],
        })

    comparison_df = pd.DataFrame(table_rows).sort_values("Test_MAE_MW").reset_index(drop=True)
    
    # Export CSV & JSON
    csv_path = RESULTS_DIR / "model_comparison.csv"
    json_path = RESULTS_DIR / "model_comparison.json"
    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Comparison saved to:\n  {csv_path}\n  {json_path}")

    print("\nStep 5: Generating Evaluation Figures...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Actual vs Predicted (Sample 7 days / 168 hours from test set)
    # Target horizon h=1 (1 hour ahead) and h=24 (24 hours ahead)
    sample_window = 168
    sample_start = 100
    sample_end = sample_start + sample_window
    
    plt.figure(figsize=(15, 6))
    actual_sample = y_test_aligned[sample_start:sample_end, 0] / 1000.0  # GW
    plt.plot(range(sample_window), actual_sample, label="Actual Demand", color="black", lw=2.5)

    colors = {
        "Seasonal_Naive_24h": "#7f7f7f",
        "Ridge_Linear_Regression": "#2ca02c",
        "Random_Forest": "#ff7f0e",
        "XGBoost": "#d62728",
        "LSTM": "#9467bd",
    }
    for name, pred in aligned_preds.items():
        pred_sample = pred[sample_start:sample_end, 0] / 1000.0
        plt.plot(range(sample_window), pred_sample, label=name.replace("_", " "), color=colors.get(name, "blue"), lw=1.5, alpha=0.85)

    plt.title("Actual vs. Predicted Grid Demand (1-Hour Ahead, 7-Day Test Window)", fontsize=14, fontweight="bold")
    plt.xlabel("Test Hour Index", fontsize=12)
    plt.ylabel("Demand (GW)", fontsize=12)
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_model_actual_vs_predicted.png", dpi=300)
    plt.close()

    # 2. Error Distributions Boxplot
    plt.figure(figsize=(12, 6))
    error_data = []
    for name, pred in aligned_preds.items():
        errors = (pred - y_test_aligned).ravel() / 1000.0  # Error in GW
        for err in errors[::20]:  # Sample for clean plotting
            error_data.append({"Model": name.replace("_", " "), "Error_GW": err})
    err_df = pd.DataFrame(error_data)

    sns.boxplot(data=err_df, x="Model", y="Error_GW", hue="Model", palette="Set2", legend=False)
    plt.axhline(0, color="red", linestyle="--", lw=1.5)
    plt.title("Prediction Error Distributions Across All Horizons (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Forecasting Model", fontsize=12)
    plt.ylabel("Prediction Error (GW: Pred - Actual)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_model_error_distributions.png", dpi=300)
    plt.close()

    # 3. Next 24 Hours Detailed Forecast Profile (Single origin snapshot)
    plt.figure(figsize=(12, 6))
    origin_idx = 200  # snapshot sample
    actual_24h = y_test_aligned[origin_idx, :] / 1000.0
    plt.plot(range(1, 25), actual_24h, marker="o", color="black", lw=2.5, label="Actual Ground Truth")

    for name, pred in aligned_preds.items():
        pred_24h = pred[origin_idx, :] / 1000.0
        plt.plot(range(1, 25), pred_24h, marker="s", label=name.replace("_", " "), color=colors.get(name, "blue"), lw=1.8, alpha=0.85)

    plt.title("24-Hour Forward Demand Forecast Snapshot (Forecast Origin t)", fontsize=14, fontweight="bold")
    plt.xlabel("Forecast Horizon Hour (t + 1 through t + 24)", fontsize=12)
    plt.ylabel("Demand (GW)", fontsize=12)
    plt.xticks(range(1, 25))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_forecast_next_24_hours.png", dpi=300)
    plt.close()

    # 4. Peak Demand Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Peak MAE Bar Chart
    models_clean = [m.replace("_", " ") for m in comparison_df["Model"]]
    axes[0].bar(models_clean, comparison_df["Test_Peak_MAE_MW"] / 1000.0, color="#3498db")
    axes[0].set_title("Peak Demand MAE (GW)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Peak MAE (GW)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=25)

    # Actual vs Predicted Daily Peak for Best Model (e.g. XGBoost)
    best_pred = aligned_preds["XGBoost"]
    true_peaks = np.max(y_test_aligned, axis=1) / 1000.0
    pred_peaks = np.max(best_pred, axis=1) / 1000.0
    
    axes[1].scatter(true_peaks, pred_peaks, alpha=0.35, color="#e74c3c", edgecolors="none")
    diag_line = np.linspace(true_peaks.min(), true_peaks.max(), 100)
    axes[1].plot(diag_line, diag_line, color="black", linestyle="--", lw=1.5, label="Perfect Forecast")
    axes[1].set_title("Actual vs. Predicted Daily Peaks (XGBoost, Test Set)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Actual Daily Peak (GW)", fontsize=11)
    axes[1].set_ylabel("Predicted Daily Peak (GW)", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_peak_demand_analysis.png", dpi=300)
    plt.close()
    print("All figures successfully saved in results/figures/.")

    # Save aligned test sample forecast data for dashboard quick-loading
    dashboard_data = {
        "models": list(aligned_preds.keys()),
        "timestamps": [str(ts) for ts in splits["timestamps_test"][lstm_offset:]],
        "y_test": np.round(y_test_aligned, 2).tolist(),
        "predictions": {k: np.round(v, 2).tolist() for k, v in aligned_preds.items()},
    }
    dash_cache = RESULTS_DIR / "dashboard_forecast_cache.json"
    with open(dash_cache, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f)
    print(f"Dashboard cache saved to: {dash_cache}")

    return comparison_df


if __name__ == "__main__":
    df_comp = run_full_model_comparison()
    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON TABLE (TEST SET)")
    print("="*80)
    print(df_comp.to_string(index=False))
