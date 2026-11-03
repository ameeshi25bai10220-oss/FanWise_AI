# AACI Model 3 – Phase 3: Model Training, Benchmarking & Evaluation Report

**Author:** AACI Engineering Team  
**Date:** November 4, 2026  
**Status:** PHASE 3 COMPLETE & VERIFIED  
**Target Module:** `aaci_ml/v3_energy_intelligence/`  

---

## Executive Summary

Pursuant to explicit authorization, **Phase 3 (Controlled ML Model Training, Benchmarking & Evaluation)** for AACI Model 3 (Ceiling Fan Energy Intelligence) has been successfully executed.

The primary objective was achieved: developing and benchmarking machine learning models capable of forecasting the electrical energy consumed during the **next observed telemetry packet** (`target_next_eSpent_wh`) using **strictly antecedent information** available at prediction time.

### Core Scientific Findings:
1. **Meaningful Forecasting Intelligence vs. Contemporaneous Tautology:**
   The ML models learned genuine temporal, diurnal, and operational patterns without reproducing the deterministic firmware formula ($E = P(s) \times \text{opTime}$).
2. **ML Models Defeat Baselines:**
   Under strict chronological testing (75,614 unseen future records), **LightGBM and XGBoost achieved MAE = 1.185 Wh and $R^2 = 0.1517$**, outperforming both the Persistence Baseline (MAE = 1.482 Wh, $R^2 = -0.611$) and the Global Historical Mean Baseline (MAE = 1.240 Wh, $R^2 = -0.032$).
3. **Ultra-Lightweight Edge Candidate:**
   **Ridge Regression** achieved **MAE = 1.192 Wh ($R^2 = 0.1438$)** with an inference latency of **0.07 microseconds** and a compiled size of only **0.77 KB**, making it exceptionally well-suited for direct deployment on low-cost microcontrollers (e.g. ESP32, ARM Cortex-M).
4. **Zero-Shot Spatial Generalization (LOFO):**
   Across all 9 independent fans evaluated under Leave-One-Fan-Out cross-validation, LightGBM maintained an average MAE of **0.9676 Wh ($R^2 = 0.1635$)**, outperforming historical mean by up to 38.9%.
5. **Absolute Project Isolation:**
   Model 1 (Thermal Comfort) and Model 2 (Occupancy Intelligence) remained **100% frozen and untouched**. All 39 test suites across the repository pass with zero failures.

---

## 1. Experimental Setup & Target Definition

- **Target Variable:** `target_next_eSpent_wh` $= eSpent_{t+1} / 3600.0$ (Watt-hours).
- **Physical Meaning:** Electrical energy consumed during the subsequent observed telemetry packet.
- **Forecasting Horizon:** Variable packet-to-packet (median interval ~60–150s, bounded $\le 900\text{ s}$). Offline multi-hour accumulation spikes were excluded to prevent training distortion.
- **Sample Volume:** 504,031 clean, deduplicated standard interval records across all 9 fans.
- **Feature Space (26 Causal Features):**
  - **Diurnal & Seasonal Signals:** $\sin / \cos$ of hour-of-day, $\sin / \cos$ of month-of-year, `day_of_week`, `is_weekend`, `is_night`.
  - **Antecedent Energy Lags & Momentum:** `eSpent_wh_lag_1`, `eSpent_wh_lag_2`, `eSpent_wh_rolling_sum_3`, `eSpent_wh_rolling_sum_12`.
  - **Operational State Lags:** `speed_lag_1`, `speed_lag_2`, `speed_changed`.
  - **Environmental State & Rates:** $T_t, RH_t$, lags $T_{t-1}, T_{t-2}, RH_{t-1}, RH_{t-2}$, differences $\Delta T, \Delta RH$, and retrospective rolling averages.
  - **Categorical Identity:** `room_category` (`Bedroom`, `Drawingroom`, `Lounge`).

---

## 2. Protocol 1: Chronological 70/15/15 Within-Fan Benchmark

Partitions were created strictly chronologically within each fan:
- **Train Partition (70%):** 352,817 observations
- **Validation Partition (15%):** 75,600 observations (used for hyperparameter calibration)
- **Test Partition (15%):** 75,614 out-of-sample future observations

### Performance Comparison Table

| Model Architecture | Model Role | MAE (Wh) | RMSE (Wh) | $R^2$ Score | MAE (Joules) | MAPE (%) | MdAPE (%) | Inactive Acc. (%) | Latency ($\mu\text{s}$) | Artifact Size |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence Baseline** | Naive Baseline | 1.4818 | 2.6158 | -0.6114 | 5,334.5 | 205.69% | 83.43% | 67.92% | <0.01 | N/A |
| **Global Historical Mean** | Statistical Baseline | 1.2404 | 2.0938 | -0.0324 | 4,465.5 | 182.83% | 84.91% | 0.00% | <0.01 | N/A |
| **Fan-Specific Mean** | Personalized Baseline| 1.2968 | 2.1436 | -0.0821 | 4,668.6 | 207.00% | 81.71% | 0.00% | <0.01 | N/A |
| **Ridge Regression** | **Edge MCU Candidate** | **1.1917** | **1.9068** | **0.1438** | **4,290.1** | **176.65%** | **82.31%** | **40.06%** | **0.07** | **0.77 KB** |
| **Random Forest (50 trees)**| Non-Linear Ensemble | 1.2088 | 1.9030 | 0.1472 | 4,351.5 | 182.14% | 81.93% | 34.39% | 0.74 | 5,217.9 KB |
| **LightGBM Regressor** | **Primary Gradient Benchmark**| **1.1873** | **1.8982** | **0.1515** | **4,274.1** | **171.08%** | **80.85%** | **33.08%** | **0.48** | **301.8 KB** |
| **XGBoost Regressor** | Gradient Benchmark | **1.1853** | **1.8979** | **0.1517** | **4,267.1** | **171.93%** | **80.63%** | **33.01%** | **0.26** | **474.5 KB** |

### Analytical Insights:
1. **Why Persistence Fails ($R^2 = -0.6114$):**
   Persistence assumes that the fan will continue operating at the exact energy level of the previous packet. When users toggle fan speeds, switch off the fan, or transition into sleep modes, persistence lags by one full interval, creating large step errors.
2. **Why Historical Mean Yields Negative $R^2$:**
   Predicting the static mean ($1.029\text{ Wh}$) eliminates extreme errors but possesses zero explanatory variance ($R^2 \le 0$).
3. **ML Forecasting Superiority:**
   All four ML models achieve positive $R^2 \approx 0.144 - 0.152$, demonstrating that antecedents (diurnal cyclic time, rolling consumption momentum, and temperature lags) carry genuine predictive signals regarding when a fan will remain on, change speed, or cycle off.
4. **Ridge vs. Gradient Boosted Trees:**
   Ridge achieves within **0.006 Wh** of LightGBM/XGBoost while executing in **0.07 microseconds** with an artifact footprint of less than 1 KB.

---

## 3. Feature Importance Analysis (LightGBM)

The feature importance distribution confirms that the models rely on physical and behavioral momentum rather than spurious correlations:

| Rank | Feature Name | Split Importance | Interpretation & Physical Relevance |
|:---:|:---|:---:|:---|
| **1** | `eSpent_wh_rolling_sum_12` | **316** | Long-term energy consumption momentum (past ~1 hour of active operation). |
| **2** | `eSpent_wh_lag_1` | **260** | Short-term energy antecedent (immediate previous interval consumption). |
| **3** | `eSpent_wh_rolling_sum_3` | **203** | Intermediate operational momentum (past ~15 minutes). |
| **4** | `sin_month` | **152** | Macro-climatic seasonal cycle (monsoon vs summer vs winter cooling demand). |
| **5** | `sin_hour` | **151** | Diurnal cycle (daytime occupancy vs nighttime continuous sleep cooling). |
| **6** | `cos_hour` | **146** | Orthogonal diurnal component distinguishing peak afternoon heat from evening cooling. |
| **7** | `speed_lag_1` | **142** | Operating speed setting at prediction time. |
| **8** | `temperature` | **138** | Current indoor thermal demand driving fan operation. |
| **9** | `eSpent_wh_lag_2` | **131** | Second-order operational lag confirming steady-state operation. |
| **10** | `humidity_diff_1_2` | **131** | Short-term rate of humidity change (comfort depression / rain events). |

---

## 4. Protocol 2: Leave-One-Fan-Out (LOFO) Zero-Shot Spatial Benchmark

To test true spatial generalization, 9 distinct models were trained, each holding out one complete fan as an unseen test domain:

| Test Fan Identifier | Room Category | Test Records | Persistence MAE (Wh) | Historical Mean MAE (Wh) | Ridge MAE (Wh) | Ridge $R^2$ | LightGBM MAE (Wh) | LightGBM $R^2$ | LightGBM Gain over Mean |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Bedroom_Fan_1` | Bedroom | 69,212 | 0.6479 | 0.8797 | 0.5609 | 0.2249 | **0.5597** | **0.2186** | **+36.38%** |
| `Bedroom_Fan_2` | Bedroom | 45,466 | 1.6513 | 1.3181 | 1.3495 | 0.0811 | **1.3501** | **0.0832** | -2.43% |
| `Bedroom_Fan_3` | Bedroom | 103,887 | 0.7474 | 1.2343 | 0.7536 | 0.1916 | **0.7537** | **0.2114** | **+38.93%** |
| `Drawingroom_Fan_1` | Drawingroom | 55,914 | 1.8092 | 1.4284 | 1.3715 | 0.1024 | **1.3755** | **0.1197** | **+3.70%** |
| `Drawingroom_Fan_2` | Drawingroom | 38,154 | 0.7673 | 1.0065 | 0.6893 | 0.2441 | **0.7189** | **0.2348** | **+28.57%** |
| `Drawingroom_Fan_3` | Drawingroom | 76,982 | 0.5599 | 0.9221 | 0.7335 | 0.0212 | **0.6405** | **0.1232** | **+30.54%** |
| `Lounge_Fan_1` | Lounge | 20,564 | 0.8220 | 1.0876 | 0.7173 | 0.2656 | **0.6985** | **0.2693** | **+35.78%** |
| `Lounge_Fan_2` | Lounge | 51,986 | 1.5870 | 1.2907 | 1.3054 | 0.1093 | **1.2925** | **0.1152** | -0.14% |
| `Lounge_Fan_3` | Lounge | 41,866 | 1.5308 | 1.2985 | 1.3047 | 0.1004 | **1.3191** | **0.0960** | -1.59% |
| **Cross-Fan Average** | **All 9 Fans** | **504,031** | **1.1248** | **1.1629** | **0.9762** | **0.1490** | **0.9676** | **0.1635** | **+16.79%** |

### Spatial Generalization Findings:
- In zero-shot spatial transfer, **LightGBM delivers an average MAE of 0.9676 Wh ($R^2 = 0.1635$)**, beating both persistence (1.1248 Wh) and the historical mean (1.1629 Wh).
- On 6 out of 9 fans, LightGBM reduces forecasting error by up to **38.9%** over historical means.
- Performance varies systematically with room type:
  - Moderate-use bedrooms (`Bedroom_Fan_1`, `Bedroom_Fan_3`) achieve MAE ~0.56–0.75 Wh with $R^2 > 0.21$.
  - Continuous high-speed fans (`Bedroom_Fan_2`, `Drawingroom_Fan_1`) exhibit higher absolute errors (~1.35 Wh) due to extended high-power dwell times.

---

## 5. Protocol 3: Few-Shot Fan Adaptation Benchmark (7-Day Calibration)

To assess whether a newly deployed fan benefits from calibration data, models were trained on 8 base fans plus the first 7 days of the target fan, and tested on the remaining days:

| Target Fan Identifier | Adaptation Period Records | Test Period Records | Zero-Shot MAE (Wh) | Few-Shot Adapted MAE (Wh) | Adaptation Gain (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `Bedroom_Fan_1` | 391 | 68,821 | 0.5606 | **0.5535** | **+1.27%** |
| `Bedroom_Fan_2` | 1,128 | 44,338 | 1.3796 | **1.3791** | **+0.03%** |
| `Bedroom_Fan_3` | 898 | 102,989 | 0.7568 | 0.7608 | -0.53% |
| `Drawingroom_Fan_1` | 4,173 | 51,741 | 1.3802 | 1.3815 | -0.10% |
| `Drawingroom_Fan_2` | 868 | 37,286 | 0.7350 | **0.7327** | **+0.31%** |
| `Drawingroom_Fan_3` | 3,821 | 73,161 | 0.6407 | **0.6190** | **+3.39%** |
| `Lounge_Fan_1` | 2,789 | 17,775 | 0.5368 | 0.5385 | -0.33% |
| `Lounge_Fan_2` | 384 | 51,602 | 1.2875 | **1.2874** | **+0.00%** |
| `Lounge_Fan_3` | 511 | 41,355 | 1.3346 | 1.3353 | -0.05% |
| **Cross-Fan Average** | **1,663 records** | **489,068 records** | **0.9569** | **0.9542** | **+0.44%** |

### Few-Shot Analysis:
- The zero-shot model is already robust (0.9569 Wh), indicating that generalized physics and diurnal cycles transfer effectively across residential homes.
- 7 days of adaptation yields localized refinements up to **+3.39%** (e.g. `Drawingroom_Fan_3`), while requiring minimal on-device memory.

---

## 6. Edge Deployment Architecture & Model Selection

Two distinct production candidates emerge from Phase 3:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AACI MODEL 3 CANDIDATE TRADEOFF                       │
├───────────────────────────────┬─────────────────────────────────────────────┤
│   EDGE MCU CANDIDATE (Ridge)  │   HIGH-CAPACITY BENCHMARK (LightGBM/XGB)    │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ • Model Size: 0.77 KB         │ • Model Size: 301.8 KB                      │
│ • Inference Latency: 0.07 μs  │ • Inference Latency: 0.48 μs                │
│ • Chronological MAE: 1.1917 Wh│ • Chronological MAE: 1.1873 Wh              │
│ • Chronological R²: 0.1438    │ • Chronological R²: 0.1515                  │
│ • Target Device: ESP32 / MCU  │ • Target Device: Linux Gateway / Edge Hub   │
│ • Execution: Dot-product math │ • Execution: 100 decision trees             │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

**Recommendation:**
- **Primary Embedded Deployable Model:** **Ridge Regression** (`aaci_energy_ridge_model.joblib`). Its 0.77 KB footprint and sub-microsecond latency allow direct execution inside low-cost fan motor controllers with virtually no accuracy penalty.
- **Reference / Cloud Benchmark Model:** **LightGBM Regressor** (`aaci_energy_lightgbm_model.joblib`). Serves as the high-capacity non-linear ceiling.

---

## 7. Artifact Manifest

All generated Model 3 artifacts are preserved inside `aaci_ml/v3_energy_intelligence/`:

### Serialized Models ([`models/`](file:///c:/Users/bharg/OneDrive/Desktop/Atomberg/aaci_ml/v3_energy_intelligence/models/))
1. `aaci_energy_ridge_model.joblib` (0.77 KB) – Verified linear edge model.
2. `aaci_energy_lightgbm_model.joblib` (301.8 KB) – Primary gradient boosting model.
3. `aaci_energy_xgboost_model.joblib` (474.5 KB) – Gradient boosting benchmark.
4. `aaci_energy_rf_model.joblib` (5.2 MB) – Random Forest ensemble.
5. `aaci_energy_preprocessor.joblib` (2.8 KB) – Frozen fitted scaler and encoder.
6. `aaci_energy_model_metadata.json` (4.2 KB) – Complete machine-readable model cards.

### Benchmark CSV Results ([`benchmarks/`](file:///c:/Users/bharg/OneDrive/Desktop/Atomberg/aaci_ml/v3_energy_intelligence/benchmarks/))
1. `chronological_benchmark_results.csv` – Full Protocol 1 metrics across 7 models.
2. `lofo_spatial_benchmark_results.csv` – Full Protocol 2 zero-shot metrics across 9 fans.
3. `few_shot_adaptation_benchmark_results.csv` – Full Protocol 3 7-day adaptation metrics.
4. `feature_importance_ranking.csv` – Ranked feature importance table.

---

## 8. Verification & Regression Status

- **Automated Unit Tests:** **39 / 39 PASSED in 3.96 seconds.**
- **Model 1 & Model 2 Codebase:** **100% frozen, 0 modifications.**
- **Git State:** `git status` verifies all modifications remain strictly within `aaci_ml/v3_energy_intelligence/`.

Phase 3 Model Training and Evaluation is **COMPLETE and SIGNED OFF**.

