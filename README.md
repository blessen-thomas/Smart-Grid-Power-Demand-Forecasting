# Group 4 – Smart Grid Power Demand Forecasting

A machine learning and deep learning time-series engineering project for predicting short-term electricity demand on the Indian electricity grid.

---

## Team Information (Group No. 4)

| USN | Student Name | Role |
| :--- | :--- | :--- |
| **1AT24CG017** | **Blessen Thomas** | **Team Leader** |
| **1AT24CG015** | Bhavani Shankar T | Core ML & Pipeline Engineering |
| **1AT24CG016** | Bheemesh V | Data Cleaning & Evaluation |
| **1AT24CG018** | C Vedasree | Feature Engineering & Visualizations |

---

## 1. Project Overview & Objective

The primary objective of this project is to build an accurate, robust forecasting model that predicts **hourly electricity demand (Demand in MW) 24 hours into the future** (`t + 1` through `t + 24`).

### Core Problem
Reliable short-term electricity demand forecasting is essential for:
- Grid stability and frequency regulation.
- Unit commitment and dispatch planning.
- Integrating variable renewable energy sources (solar and wind) while minimizing spinning reserve costs and load shedding.

---

## 2. Tech Stack

- **Data Processing & Scientific Computing:** Python 3.13, Pandas, NumPy, SciPy, OpenPyXL, PyArrow
- **Exploratory Data Analysis & Visualization:** Matplotlib, Seaborn
- **Machine Learning & Time-Series Evaluation:** Scikit-learn, XGBoost
- **Deep Learning:** PyTorch (LSTM Recurrent Neural Networks)
- **Interactive Web Dashboard:** Streamlit

---

## 3. Project Directory Structure

```text
Smart-Grid-Power-Demand-Forecasting/
├── .gitignore                      # Excludes raw data, virtual environments, models, and caches
├── LICENSE                         # Project license
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
├── dashboard/                      # Dashboard UI components (Streamlit)
│   ├── app.py                      # Main interactive Streamlit application
│   └── .gitkeep
├── dataset/
│   ├── raw/                        # Original, immutable raw Excel datasets (git-ignored)
│   │   └── January 2024- June 2025.xlsx
│   └── processed/                  # Cleaned, standardized time-series datasets (git-ignored)
│       ├── cleaned_hourly_grid_data.csv
│       ├── cleaned_hourly_grid_data.parquet
│       ├── forecasting_features.csv
│       └── preprocessing_audit_report.json
├── models/                         # Trained model artifacts and weights (git-ignored)
│   └── .gitkeep
├── notebooks/                      # Jupyter notebooks for interactive analysis and experimentation
│   └── 01_exploratory_data_analysis.ipynb
├── results/                        # Generated figures, metrics, and evaluation summaries
│   ├── eda_summary.json
│   ├── model_comparison.json
│   ├── model_comparison.csv
│   └── figures/
│       ├── 01_demand_time_series_and_splits.png
│       ├── 02_diurnal_hourly_demand_profile.png
│       ├── 03_weekly_and_monthly_demand_patterns.png
│       ├── 04_generation_mix_and_correlation.png
│       ├── 05_model_actual_vs_predicted.png
│       ├── 06_model_error_distributions.png
│       ├── 07_forecast_next_24_hours.png
│       └── 08_peak_demand_analysis.png
└── src/                            # Modular, production-grade Python package
    ├── __init__.py
    ├── config.py                   # Central paths, constants, and split boundary definitions
    ├── data_loader.py              # Raw data loading, schema inspection, and metadata isolation
    ├── cleaning.py                 # Data validation, type conversion, continuity verification
    ├── eda.py                      # Statistical summaries, time-series profiling, visual figure export
    ├── features.py                 # Anti-leakage lag, rolling, and cyclical feature engineering
    ├── models_baseline.py          # Naive, Linear, and classical tree models (Ridge, RF, XGBoost)
    ├── models_lstm.py              # PyTorch 24-step multi-horizon LSTM model
    └── evaluate.py                 # Multi-metric evaluation (MAE, RMSE, MAPE, R2, Peak Error)
```

---

## 4. Dataset Specification & Preprocessing Decisions

### Data Characteristics & Scope
The dataset consists of hourly grid measurements from January 1, 2024 to June 24, 2025:
- `Timestamp`: Hourly reading timestamp in format `DD-MM-YYYY HH:MM:SS`
- `Demand (MW)`: Grid load / electricity demand in megawatts (Target)
- `Wind (MW)`: Wind generation in megawatts
- `Solar (MW)`: Solar generation in megawatts
- `Total Generation (MW)`: Total utility generation in megawatts

> **Important Geographical Notice:** The source dataset does not specify specific regional or national dispatch boundaries (e.g., specific State / Regional Load Despatch Centre). Accordingly, no assumptions are made regarding specific sub-regions or geographical bounds.

### Preprocessing & Audit Decisions
1. **Raw Data Immutability:** The raw workbook `January 2024- June 2025.xlsx` in `dataset/raw/` remains byte-exact and untouched.
2. **Metadata Footer Isolation:** The raw sheet contains 6 metadata rows at the bottom (Rows 12984–12989: `Minimum`, `Maximum`, `Average`, `Sum`). These caused Excel readers to type all numeric columns as generic `object` strings. The data loader separates these summary rows from the time-series observations.
3. **Chronological Continuity Validation:** The time series covers exactly **12,984 continuous hours** from `2024-01-01 00:00:00` to `2025-06-24 23:00:00`. Analysis verified that **0 hours are missing** and **0 duplicate timestamps exist**.
4. **Handling of Unusual Readings (Nighttime Solar):**
   - The dataset contains 499 hourly records where `Solar (MW)` is slightly negative (minimum: `-69.76 MW`).
   - *Status:* In power engineering literature, negative solar readings are commonly hypothesized to reflect telemetry transducer calibration drift or inverter/transformer standby power drawn from the grid at night. However, because the dataset source provides no sensor calibration metadata, this explanation is treated as a plausible hypothesis rather than verified ground truth.
   - *Policy:* The raw measurements in `Solar (MW)` are strictly preserved without alteration. An engineered feature `Solar_Calibrated (MW)` (clipped at 0) is provided strictly as an **experimental derived feature**, not an established correction.
5. **Anti-Leakage Rule for Renewable Generation:**
   - Definitions: `Renewable_Generation (MW) = Wind (MW) + Solar_Calibrated (MW)`, `Net_Demand (MW) = Demand (MW) - Renewable_Generation (MW)`.
   - *Forecasting restriction:* In real-time 24-hour ahead demand forecasting issued at time $t$, actual renewable generation at future hours $t+1 \dots t+24$ is unknown without separate wind/solar forecast models. Feeding actual future generation into the demand forecast would be direct data leakage. Consequently, concurrent future generation is strictly prohibited as a predictor. Only historical lags known at or before forecast origin $t$ are permitted.
6. **No Weather Data Assumed:** No external temperature, humidity, or irradiance data is assumed or hallucinated. Features are derived strictly from calendar harmonics, past lags, and historical telemetry.

---

## 5. Time-Series Splitting & Anti-Leakage Protocol

To ensure models generalize to real-world operations, **random shuffling (e.g., standard train_test_split or K-Fold) is strictly prohibited**. 

The dataset is partitioned chronologically:
| Partition | Date Range | Duration | Proportion |
| :--- | :--- | :--- | :--- |
| **Train** | `2024-01-01 00:00` – `2024-12-31 23:00` | 8,784 hours (full calendar year) | 67.65% |
| **Validation** | `2025-01-01 00:00` – `2025-03-31 23:00` | 2,160 hours (Q1 winter/spring) | 16.64% |
| **Test** | `2025-04-01 00:00` – `2025-06-24 23:00` | 2,040 hours (Peak summer period) | 15.71% |

### Anti-Leakage Safeguards:
- Rolling statistics and lag features are computed using only past timestamps (`closed='right'` strictly on historical lags $\le t$).
- Feature and target scalers (`StandardScaler`) are fitted **exclusively on the training partition**.
- Target vector $Y(t) = [Demand(t+1), Demand(t+2), \dots, Demand(t+24)] \in \mathbb{R}^{24}$ represents direct multi-horizon forecasting, predicting all 24 forward hours simultaneously without accumulating iterative feedback errors.

---

## 6. Feature Engineering

Features are constructed exclusively from information available at or before the forecast origin $t$:
1. **Calendar Harmonics:** `hour`, `day`, `dayofweek`, `month`, `is_weekend`.
2. **Cyclical Encodings:** $\sin$ and $\cos$ transformations for hour ($\frac{2\pi \cdot h}{24}$), weekday ($\frac{2\pi \cdot d}{7}$), and month ($\frac{2\pi \cdot (m-1)}{12}$) ensuring continuous transitions across midnight and calendar boundaries.
3. **Historical Demand Lags:** Known demand at $t$, $t-1$, $t-2$, $t-3$, $t-23$, $t-24$ (yesterday same hour), $t-47$, $t-48$, $t-167$, and $t-168$ (last week same hour).
4. **Historical Rolling Statistics:** 24-hour and 168-hour rolling mean, standard deviation, min, and max of demand over past observations.
5. **Historical Telemetry Lags:** Recorded Wind, Solar, and Total Generation at $t$ and $t-24$ (future concurrent values are strictly omitted).

---

## 7. Experimental Results & Model Benchmark Comparison

All 5 models were evaluated on the held-out test partition (**April 1, 2025 to June 24, 2025**):

| Model | Test MAE (MW) | Test RMSE (MW) | Test MAPE (%) | Test $R^2$ | Test Peak MAE (MW) | Peak Bias (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **5,371.01** | **7,459.84** | **2.700%** | **0.7037** | **6,126.43** | -5,101.24 |
| **Random Forest Regressor** | 5,888.08 | 8,032.59 | 2.952% | 0.6565 | 6,996.44 | -6,274.13 |
| **Ridge Linear Regression** | 7,046.84 | 9,651.78 | 3.527% | 0.5041 | 8,057.52 | -6,773.40 |
| **PyTorch LSTM** | 8,169.85 | 10,618.33 | 4.018% | 0.4019 | 12,474.81 | -12,268.16 |
| **Seasonal Naive 24h Baseline** | 10,832.86 | 13,836.39 | 5.411% | -0.0192 | 13,892.42 | -13,272.70 |

### Key Observations & Viva Points:
1. **Dominance of Gradient Boosted Trees:** **XGBoost achieved the lowest error across all metrics (2.70% MAPE and 5,371 MW MAE)**, outperforming the baseline by **50.4% in MAE reduction**.
2. **MAPE Interpretation & Stability:** Because grid demand is consistently large ($\ge 97,000$ MW), MAPE remains mathematically stable without risk of division-by-zero explosions. A 2.70% MAPE demonstrates exceptional multi-horizon operational accuracy for power systems dispatch.
3. **Peak Underestimation Bias:** All models exhibited a negative Peak Bias (e.g., -5,101 MW for XGBoost). In machine learning, loss functions optimize for mean performance, inherently smoothing extreme seasonal peak spikes (such as June 2025 heatwaves). For grid operations, safety margins or asymmetric loss functions (e.g. pinball/quantile loss) are recommended.
4. **Deep Learning vs. Tree Ensembles:** On this tabular time-series structure with engineered lags and harmonics, XGBoost converged faster and generalized better than LSTM, while LSTM demonstrated smooth temporal coherence across the 24 horizons.

---

## 8. Streamlit Web Dashboard

An interactive dashboard is available in `dashboard/app.py`:
- **Real-Time 24h Forecast:** Slide through any hour in the test set to view simulated multi-model predictions versus ground truth.
- **Historical Telemetry:** Inspect diurnal curves, summer peaks, and correlation heatmaps.
- **Model Comparison Table:** Interactive sortable metrics with MAE and $R^2$ bar charts.
- **Peak Demand Risk Analysis:** Scatter plot of actual vs predicted daily peaks and bias indicators.

To run the dashboard:
```bash
streamlit run dashboard/app.py
```

---

## 9. How to Reproduce

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Data Cleaning & Audit
```bash
python -m src.cleaning
```

### 3. Generate Exploratory Visualizations
```bash
python -m src.eda
```

### 4. Execute Feature Engineering, Model Training & Benchmarking
```bash
python -m src.compare_models
```

### 5. Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 10. Limitations & Future Scope

### Limitations
- **Absence of Weather Forecasts:** Temperature and humidity strongly drive air-conditioning loads during Indian summers. Because weather telemetry was not present in the dataset, models rely entirely on calendar and autoregressive demand proxies.
- **Lack of Grid Regional Despatch Metadata:** The dataset does not specify whether telemetry represents national or specific regional grid jurisdictions.
- **Extreme Peak Attenuation:** Standard MSE-trained models moderately underestimate maximum extreme daily peaks.

### Future Scope
- Implementation of **Quantile Regression (Gradient Boosted Trees / DeepAR)** to generate 90% confidence intervals for risk-aware unit commitment.
- Integration of numerical weather prediction (NWP) forecasts when available.
- Deployment of the Streamlit dashboard as a cloud microservice.

