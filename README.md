# Smart Grid Power Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)

> **Direct 24-Hour Ahead Multi-Horizon Electricity Demand Prediction Using Machine Learning and Deep Learning on Indian Power Grid Telemetry.**  
> **Author:** Blessen Thomas  
> **Repository:** [blessen-thomas/Smart-Grid-Power-Demand-Forecasting](https://github.com/blessen-thomas/Smart-Grid-Power-Demand-Forecasting)

---

## Table of Contents

1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [System Architecture & Methodology](#2-system-architecture--methodology)
3. [Dataset Specification & Placement](#3-dataset-specification--placement)
4. [Feature Engineering & Anti-Leakage Protocol](#4-feature-engineering--anti-leakage-protocol)
5. [Model Descriptions](#5-model-descriptions)
6. [Experimental Results & Evaluation](#6-experimental-results--evaluation)
7. [Streamlit Interactive Dashboard](#7-streamlit-interactive-dashboard)
8. [Cross-Platform Setup & Execution Guide](#8-cross-platform-setup--execution-guide)
9. [Automated Testing & Verification](#9-automated-testing--verification)
10. [Troubleshooting & FAQ](#10-troubleshooting--faq)
11. [Limitations & Future Improvements](#11-limitations--future-improvements)

---

## 1. Project Overview & Objectives

Accurate short-term load forecasting (STLF) is fundamental to modern electrical utility operations. With increasing penetration of non-dispatchable renewable energy (solar and wind), the net load profile exhibits sharp ramps and steep diurnal variations. System operators require reliable day-ahead forecasts to:
- Schedule economical generator dispatch (Unit Commitment).
- Procure adequate spinning and non-spinning reserves.
- Regulate grid frequency and prevent transmission congestion or load shedding.

### Primary Objective
To construct an automated, leakage-free forecasting pipeline that predicts **hourly electricity demand (Demand in MW) for the upcoming 24 hours** (`t + 1` through `t + 24`) from telemetry observed at forecast origin $t$.

---

## 2. System Architecture & Methodology

```text
Smart-Grid-Power-Demand-Forecasting/
├── .gitignore                      # Excludes raw data, virtual environments, models, and caches
├── LICENSE                         # MIT License
├── README.md                       # Complete documentation and execution guide
├── requirements.txt                # Python package dependencies
├── pytest.ini                      # Pytest discovery configuration
├── dashboard/                      # Web user interface
│   ├── app.py                      # Interactive Streamlit application
│   └── .gitkeep
├── dataset/
│   ├── raw/                        # Original immutable raw Excel file
│   │   └── January 2024- June 2025.xlsx  (Local placement)
│   └── processed/                  # Cleaned, standardized time-series datasets
│       ├── cleaned_hourly_grid_data.csv
│       ├── forecasting_features.csv
│       └── preprocessing_audit_report.json
├── models/                         # Serialized weights and model artifacts
│   ├── ridge_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── lstm_demand_forecaster.pt
├── notebooks/                      # Interactive Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
├── results/                        # Generated figures, metrics, and evaluation summaries
│   ├── eda_summary.json
│   ├── model_comparison.csv
│   ├── model_comparison.json
│   ├── dashboard_forecast_cache.json
│   └── figures/
│       ├── 01_demand_time_series_and_splits.png
│       ├── 02_diurnal_hourly_demand_profile.png
│       ├── 03_weekly_and_monthly_demand_patterns.png
│       ├── 04_generation_mix_and_correlation.png
│       ├── 05_model_actual_vs_predicted.png
│       ├── 06_model_error_distributions.png
│       ├── 07_forecast_next_24_hours.png
│       └── 08_peak_demand_analysis.png
├── src/                            # Modular, production-grade Python package
│   ├── __init__.py
│   ├── config.py                   # Central paths, constants, and split boundary definitions
│   ├── data_loader.py              # Raw data loading, schema inspection, and metadata isolation
│   ├── cleaning.py                 # Data validation, type conversion, continuity verification
│   ├── eda.py                      # Statistical summaries, time-series profiling, visual figure export
│   ├── features.py                 # Anti-leakage lag, rolling, and cyclical feature engineering
│   ├── models_baseline.py          # Naive, Linear, and classical tree models (Ridge, RF, XGBoost)
│   ├── models_lstm.py              # PyTorch 24-step multi-horizon LSTM model
│   ├── evaluate.py                 # Multi-metric evaluation (MAE, RMSE, MAPE, R2, Peak Error)
│   └── compare_models.py           # Benchmark runner, figure exporter, and cache builder
└── tests/                          # Automated verification test suite
    ├── test_verification.py        # System integrity and anti-leakage tests
    └── test_forecast_dashboard.py  # Forecast indexing and dashboard simulation tests
```

### Direct Multi-Horizon Forecasting
Rather than iterative autoregression (which compounds forecast errors at each forward step), this pipeline adopts **Direct Multi-Horizon Forecasting**. At forecast origin $t$, models map known historical states directly to all 24 forward horizons:
$$\hat{\mathbf{Y}}_t = [\hat{D}_{t+1}, \hat{D}_{t+2}, \dots, \hat{D}_{t+24}]^\top \in \mathbb{R}^{24}$$

### Chronological Splitting Protocol
Random shuffling (e.g., standard K-fold cross-validation or random `train_test_split`) violates the temporal causality of time series. Data is strictly partitioned chronologically:
- **Train:** `2024-01-01 00:00` to `2024-12-31 23:00` (8,616 samples after 7-day warm-up, ~67.4%)
- **Validation:** `2025-01-01 00:00` to `2025-03-31 23:00` (2,160 samples, ~16.9%)
- **Test:** `2025-04-01 00:00` to `2025-06-24 23:00` (1,993 aligned samples, ~15.7%, untouched until final evaluation)

---

## 3. Dataset Specification & Placement

### Telemetry Channels
The dataset comprises hourly Indian power grid records from January 1, 2024 to June 24, 2025:
- `Timestamp`: Hourly reading timestamp in format `DD-MM-YYYY HH:MM:SS`
- `Demand (MW)`: Actual electricity load / grid demand in megawatts (Target)
- `Wind (MW)`: Wind power generation in megawatts
- `Solar (MW)`: Solar power generation in megawatts
- `Total Generation (MW)`: Total utility generation in megawatts

### Exact File Name & Placement
Place the raw Excel spreadsheet at:
```text
dataset/raw/January 2024- June 2025.xlsx
```

### Preprocessing & Audit Log
1. **Metadata Footer Isolation:** Rows 12,984 through 12,989 contained summary statistics (`Minimum`, `Maximum`, `Average`, `Sum`). The data loader separates these from actual hourly readings, restoring proper `float64` data types.
2. **Timeline Continuity:** Verified exactly **12,984 consecutive hours** with **0 missing intervals** and **0 duplicate timestamps**.
3. **Negative Solar Readings:** 499 hourly records exhibit negative solar readings (minimum: `-69.76 MW`). While literature hypothesizes nighttime inverter auxiliary standby consumption or sensor calibration offsets, no calibration metadata exists to verify this as fact. Consequently, original values in `Solar (MW)` are strictly preserved, and an experimental derived feature `Solar_Calibrated (MW)` (clipped at 0) is provided.
4. **No Geographical Assumptions:** The source file does not specify particular Regional Load Despatch Centre (RLDC) boundaries. No geographical assumptions are made.

---

## 4. Feature Engineering & Anti-Leakage Protocol

All 33 input features are constructed **exclusively from information available at or before forecast origin $t$**:
1. **Calendar Harmonics:** `hour`, `day`, `dayofweek`, `month`, `is_weekend`.
2. **Cyclical Encodings:** Continuous periodic transformations ensuring smooth transitions across midnight and calendar boundaries:
   $$\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \sin\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right), \quad \cos\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right), \quad \sin\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right), \quad \cos\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right)$$
3. **Historical Demand Lags:** $t$, $t-1$, $t-2$, $t-3$, $t-23$, $t-24$ (yesterday same hour), $t-47$, $t-48$, $t-167$, and $t-168$ (last week same hour).
4. **Historical Rolling Statistics:** 24-hour and 168-hour rolling mean, standard deviation, minimum, and maximum strictly closed on the past (`closed='right'`).
5. **Historical Telemetry Lags:** Wind, Solar, and Total Generation at $t$ and $t-24$.

### Anti-Leakage Guarantee:
- **No Future Telemetry:** Concurrent future wind, solar, or generation at $t+1 \dots t+24$ are **strictly excluded**. In real-time grid dispatch, future renewable generation is unknown without separate forecast models. Using concurrent future values constitutes direct target leakage.
- **Scaler Isolation:** Feature scalers (`StandardScaler`) are fitted **strictly on the training partition** and transformed on validation and test sets.

---

## 5. Model Descriptions

1. **Seasonal Naive 24h Baseline:**  
   Diurnal persistence benchmark. Predicts each forward hour $h \in [1..24]$ using the observed demand from yesterday's corresponding hour ($t + h - 24$).
2. **Ridge Linear Regression:**  
   Direct multi-horizon linear model with L2 regularization ($\alpha=10.0$) preventing multicollinearity among dense historical lags.
3. **Random Forest Regressor:**  
   Non-linear ensemble consisting of 100 decision trees (`max_depth=16`, `min_samples_split=5`), capturing non-linear interactions across calendar and rolling features.
4. **XGBoost Regressor:**  
   Gradient boosted decision trees natively supporting multi-target output (150 trees, `learning_rate=0.07`, `max_depth=6`, `subsample=0.85`).
5. **PyTorch LSTM Neural Network:**  
   Recurrent deep learning architecture featuring a 2-layer LSTM encoder with dropout (0.15) and a dense multi-horizon projection head mapping the final hidden state directly to the 24-hour target vector.

---

## 6. Experimental Results & Evaluation

Evaluated on the held-out test partition (**April 1, 2025 to June 24, 2025**):

| Model | Test MAE (MW) | Test RMSE (MW) | Test MAPE (%) | Test $R^2$ | Peak MAE (MW) | Peak Bias (MW) | Peak Timing MAE (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **5,371.01** | **7,459.84** | **2.700%** | **0.7037** | **6,126.43** | -5,101.24 | 4.72 h |
| **Random Forest Regressor** | 5,888.08 | 8,032.59 | 2.952% | 0.6565 | 6,996.44 | -6,274.13 | 4.95 h |
| **Ridge Linear Regression** | 7,046.84 | 9,651.78 | 3.527% | 0.5041 | 8,057.52 | -6,773.40 | 5.47 h |
| **PyTorch LSTM** | 8,169.85 | 10,618.33 | 4.018% | 0.4019 | 12,474.81 | -12,268.16 | 6.54 h |
| **Seasonal Naive 24h Baseline** | 10,832.86 | 13,836.39 | 5.411% | -0.0192 | 13,892.42 | -13,272.70 | 8.91 h |

### Analytical Insights:
- **Superiority of Gradient Boosted Trees:** XGBoost attained the lowest error across all metrics (**2.70% MAPE**, **5,371 MW MAE**), achieving a **50.4% error reduction** compared to the naive baseline.
- **MAPE Properties:** Because demand consistently exceeds $97,000\text{ MW}$, MAPE remains numerically stable without division-by-zero distortion.
- **Peak Under-Forecasting Bias:** All models exhibited a negative Peak Bias (e.g. -5,101 MW for XGBoost). Standard regression loss functions optimize for expected mean value, which inherently attenuates peak heatwave demand spikes. System operators should integrate safety reserve margins.

---

## 7. Streamlit Interactive Dashboard

An interactive web application is available in `dashboard/app.py`:
- **📌 Project Overview:** Executive summary, problem definition, and methodological guarantees.
- **📊 Historical Grid Telemetry:** Exploratory analysis of diurnal rhythms, seasonal peaks, and renewable correlations.
- **⚡ 24-Hour Forward Forecast Simulation:** Interactive slider covering all 1,993 test origins. Displays forward timestamps, ground truth, selected model predictions, and window performance metrics.
- **🏆 Model Benchmark Comparison:** Interactive sortable metrics table and comparative charts.
- **🎯 Peak Demand Analysis:** Daily peak scatter plots and dispatch reliability indicators.

---

## 8. Cross-Platform Setup & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.
- Git.

### 1. Clone the Repository
```bash
git clone https://github.com/blessen-thomas/Smart-Grid-Power-Demand-Forecasting.git
cd Smart-Grid-Power-Demand-Forecasting
```

### 2. Set Up Virtual Environment

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional PyTorch CPU note: On Linux and Windows, PyTorch CPU wheels can be installed explicitly if desired:)*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### Execution Paths

#### Path A: Quick-Start (Launch Dashboard Immediately)
The repository includes precomputed benchmark forecasts and evaluation figures in `results/`. You can launch the interactive dashboard immediately on a fresh clone without retraining:
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

#### Path B: Full Pipeline Reproduction (From Raw Data)
Place `January 2024- June 2025.xlsx` in `dataset/raw/`, then run:

```bash
# Step 1: Clean and validate raw grid data
python -m src.cleaning

# Step 2: Generate exploratory data analysis figures and summary statistics
python -m src.eda

# Step 3: Run feature engineering, train all 5 models, and evaluate benchmarks
python -m src.compare_models

# Step 4: Launch interactive dashboard
streamlit run dashboard/app.py
```

---

## 9. Automated Testing & Verification

Run the complete test suite (19 test cases covering data integrity, anti-leakage compliance, model validity, metric alignment, and dashboard simulation):

```bash
python -m pytest tests/ -v
```

---

## 10. Troubleshooting & FAQ

1. **`FileNotFoundError: Raw dataset file not found at: dataset/raw/...`**  
   *Fix:* Ensure `January 2024- June 2025.xlsx` is placed in `dataset/raw/`. Alternatively, you can run the dashboard directly using Path A (Quick-Start).

2. **Streamlit Port Already in Use:**  
   *Fix:* Launch Streamlit on an alternate port:
   ```bash
   streamlit run dashboard/app.py --server.port 8502
   ```

3. **`ModuleNotFoundError: No module named 'src'` during test runs:**  
   *Fix:* Ensure you execute pytest via `python -m pytest tests/ -v`, which automatically includes the root directory in `sys.path`.

---

## 11. Limitations & Future Improvements

### Current Limitations
- **Absence of Weather Telemetry:** Ambient temperature and humidity are primary drivers of cooling loads during Indian summers. In the absence of weather data, models rely entirely on calendar and autoregressive proxies.
- **Regional Disaggregation:** Telemetry is aggregated across the grid without regional bus-level breakdown.
- **Peak Attenuation:** Symmetrical MSE loss penalizes over-prediction and under-prediction equally, resulting in moderate peak demand underestimation.

### Future Improvements
- **Probabilistic Forecasting:** Implement Quantile Gradient Boosted Trees or DeepAR to generate 90% and 95% prediction intervals for risk-aware unit commitment.
- **External NWP Integration:** Incorporate numerical weather predictions (temperature, dew point, solar irradiance) when telemetry feeds become accessible.
- **Cloud Deployment:** Containerize the application using Docker and deploy the Streamlit dashboard on a cloud platform (e.g. Streamlit Cloud, AWS ECS, or GCP Cloud Run).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
