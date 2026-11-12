"""
Exploratory Data Analysis (EDA) module for Smart Grid Power Demand Forecasting.
Generates comprehensive distribution stats, time series profiles, seasonal patterns,
and publication-ready figures.
"""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_CSV,
    FIGURES_DIR,
    RESULTS_DIR,
    COL_TIMESTAMP,
    COL_DEMAND,
    COL_WIND,
    COL_SOLAR,
    COL_TOTAL_GEN,
    TRAIN_END_DATE,
    VAL_END_DATE,
)


def load_processed_data() -> pd.DataFrame:
    """Loads the cleaned dataset from processed CSV."""
    if not PROCESSED_DATA_CSV.exists():
        from src.cleaning import clean_grid_data
        return clean_grid_data()

    df = pd.read_csv(PROCESSED_DATA_CSV)
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
    return df


def compute_statistical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes distribution and descriptive metrics for all numeric features."""
    numeric_cols = [
        COL_DEMAND,
        COL_WIND,
        COL_SOLAR,
        COL_TOTAL_GEN,
        "Renewable_Generation (MW)",
        "Net_Demand (MW)",
    ]

    summary = {}
    for col in numeric_cols:
        series = df[col]
        summary[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p10": float(series.quantile(0.10)),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

    # Chronological Split Stats
    train_mask = df[COL_TIMESTAMP] <= TRAIN_END_DATE
    val_mask = (df[COL_TIMESTAMP] > TRAIN_END_DATE) & (df[COL_TIMESTAMP] <= VAL_END_DATE)
    test_mask = df[COL_TIMESTAMP] > VAL_END_DATE

    summary["splits"] = {
        "train": {
            "start": str(df[train_mask][COL_TIMESTAMP].min()),
            "end": str(df[train_mask][COL_TIMESTAMP].max()),
            "hours": int(train_mask.sum()),
            "percentage": round(float(train_mask.mean() * 100), 2),
            "demand_mean": float(df[train_mask][COL_DEMAND].mean()),
            "demand_std": float(df[train_mask][COL_DEMAND].std()),
        },
        "validation": {
            "start": str(df[val_mask][COL_TIMESTAMP].min()),
            "end": str(df[val_mask][COL_TIMESTAMP].max()),
            "hours": int(val_mask.sum()),
            "percentage": round(float(val_mask.mean() * 100), 2),
            "demand_mean": float(df[val_mask][COL_DEMAND].mean()),
            "demand_std": float(df[val_mask][COL_DEMAND].std()),
        },
        "test": {
            "start": str(df[test_mask][COL_TIMESTAMP].min()),
            "end": str(df[test_mask][COL_TIMESTAMP].max()),
            "hours": int(test_mask.sum()),
            "percentage": round(float(test_mask.mean() * 100), 2),
            "demand_mean": float(df[test_mask][COL_DEMAND].mean()),
            "demand_std": float(df[test_mask][COL_DEMAND].std()),
        },
    }

    return summary


def plot_all_eda_figures(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Generates and saves publication-quality exploratory visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Full Time-Series & Chronological Splits
    plt.figure(figsize=(15, 6))
    plt.plot(df[COL_TIMESTAMP], df[COL_DEMAND] / 1000.0, label="Demand (GW)", color="#1f77b4", lw=0.9, alpha=0.85)

    train_end_dt = pd.to_datetime(TRAIN_END_DATE)
    val_end_dt = pd.to_datetime(VAL_END_DATE)

    plt.axvline(train_end_dt, color="red", linestyle="--", lw=2, label=f"Train Split End ({TRAIN_END_DATE[:10]})")
    plt.axvline(val_end_dt, color="darkgreen", linestyle="--", lw=2, label=f"Val Split End ({VAL_END_DATE[:10]})")

    plt.axvspan(df[COL_TIMESTAMP].min(), train_end_dt, color="#3498db", alpha=0.1, label="Train Set (~67.6%)")
    plt.axvspan(train_end_dt, val_end_dt, color="#f39c12", alpha=0.1, label="Validation Set (~16.6%)")
    plt.axvspan(val_end_dt, df[COL_TIMESTAMP].max(), color="#2ecc71", alpha=0.1, label="Test Set (~15.7%)")

    plt.title("Indian Electricity Grid Hourly Demand (Jan 2024 – Jun 2025) with Chronological Splits", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Demand (GW)", fontsize=12)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    fig1_path = output_dir / "01_demand_time_series_and_splits.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Feature enrichments for temporal analysis
    df["Hour"] = df[COL_TIMESTAMP].dt.hour
    df["DayOfWeek"] = df[COL_TIMESTAMP].dt.day_name()
    df["DayOfWeekNum"] = df[COL_TIMESTAMP].dt.dayofweek
    df["YearMonth"] = df[COL_TIMESTAMP].dt.to_period("M").astype(str)

    # 2. Diurnal Cycle (24-Hour Profile)
    plt.figure(figsize=(12, 6))
    hourly_stats = df.groupby("Hour")[COL_DEMAND].agg(["mean", "median", lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)])
    hourly_stats.columns = ["mean", "median", "p10", "p90"]

    plt.plot(hourly_stats.index, hourly_stats["mean"] / 1000.0, marker="o", color="#d62728", lw=2.5, label="Hourly Mean Demand")
    plt.plot(hourly_stats.index, hourly_stats["median"] / 1000.0, linestyle="--", color="#ff7f0e", label="Hourly Median Demand")
    plt.fill_between(hourly_stats.index, hourly_stats["p10"] / 1000.0, hourly_stats["p90"] / 1000.0, color="#d62728", alpha=0.15, label="10th–90th Percentile Band")

    plt.title("24-Hour Diurnal Demand Profile (Average Daily Rhythm)", fontsize=14, fontweight="bold")
    plt.xlabel("Hour of Day (0 to 23)", fontsize=12)
    plt.ylabel("Demand (GW)", fontsize=12)
    plt.xticks(range(0, 24))
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig2_path = output_dir / "02_diurnal_hourly_demand_profile.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Weekly & Monthly Demand Variation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.boxplot(ax=axes[0], data=df, x="DayOfWeek", y=df[COL_DEMAND] / 1000.0, order=day_order, hue="DayOfWeek", palette="Blues", legend=False)
    axes[0].set_title("Demand Distribution by Day of Week", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Day of Week", fontsize=11)
    axes[0].set_ylabel("Demand (GW)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # Monthly Trend
    monthly_order = sorted(df["YearMonth"].unique())
    sns.barplot(ax=axes[1], data=df, x="YearMonth", y=df[COL_DEMAND] / 1000.0, order=monthly_order, color="#4575b4", errorbar=None)
    axes[1].set_title("Monthly Mean Demand (Seasonal Peaks)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Year-Month", fontsize=11)
    axes[1].set_ylabel("Mean Demand (GW)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig3_path = output_dir / "03_weekly_and_monthly_demand_patterns.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 4. Renewable Generation & Correlation Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Diurnal Solar & Wind profile
    solar_diurnal = df.groupby("Hour")["Solar_Calibrated (MW)"].mean() / 1000.0
    wind_diurnal = df.groupby("Hour")[COL_WIND].mean() / 1000.0
    demand_scaled = df.groupby("Hour")[COL_DEMAND].mean() / 1000.0

    axes[0].plot(solar_diurnal.index, solar_diurnal, label="Solar Generation (GW)", color="#e67e22", lw=2, marker="s")
    axes[0].plot(wind_diurnal.index, wind_diurnal, label="Wind Generation (GW)", color="#16a085", lw=2, marker="^")
    axes[0].plot(demand_scaled.index, demand_scaled / 5.0, label="Demand / 5 (GW, scaled)", color="#2c3e50", lw=2, linestyle=":")
    axes[0].set_title("Diurnal Generation vs Demand Profiles", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Hour of Day", fontsize=11)
    axes[0].set_ylabel("Generation (GW)", fontsize=11)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(loc="upper left")

    # Correlation Matrix
    corr_cols = [COL_DEMAND, COL_TOTAL_GEN, COL_WIND, "Solar_Calibrated (MW)", "Net_Demand (MW)"]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, ax=axes[1], annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation"})
    axes[1].set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig4_path = output_dir / "04_generation_mix_and_correlation.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")


def run_full_eda() -> Dict[str, Any]:
    """Runs complete EDA pipeline, writes summary JSON, and generates figures."""
    df = load_processed_data()
    summary = compute_statistical_summary(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary metrics saved to: {summary_path}")

    plot_all_eda_figures(df)
    return summary


if __name__ == "__main__":
    run_full_eda()
