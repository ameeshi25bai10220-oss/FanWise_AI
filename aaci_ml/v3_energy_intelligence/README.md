# AACI Model 3 – Ceiling Fan Energy Intelligence

**Atomberg Adaptive Comfort Intelligence (AACI)**  
*Model 3: Energy Intelligence, Deterministic Physics & Consumption Forecasting*

---

## 1. Project Overview

Model 3 is the third and final machine learning model in the AACI ecosystem. Its objective is to provide deterministic ceiling fan energy modeling, historical consumption auditing, next-interval energy forecasting, and energy-aware fan speed recommendation.

### Core Ecosystem Architecture
- **Model 1 (Thermal Comfort Intelligence):** Predicts occupant thermal comfort (`Cold` / `Neutral` / `Hot`) from indoor environment variables and air velocity. **[FROZEN]**
- **Model 2 (Occupancy Intelligence):** Predicts room occupancy (`Occupied` / `Unoccupied`) and spatial headcount from multimodal sensor streams. **[FROZEN]**
- **Model 3 (Energy Intelligence):** Evaluates electrical power, computes deterministic energy consumption, forecasts future packet energy, and bridges air velocity to Model 1 for Pareto-optimal comfort-energy fan control. **[PHASE 3 COMPLETE]**

---

## 2. Directory Layout

```
aaci_ml/v3_energy_intelligence/
├── README.md
├── train_energy_models.py        # Master training and benchmarking script
├── src/
│   ├── __init__.py               # Package exports
│   ├── energy_physics.py         # Power ratings, deterministic equations, unit conversions, provisional Dryad airflow map
│   ├── data_loader.py            # 9-fan Mendeley telemetry loader, deduplication, sentinel filtering, 3-room mapping
│   ├── preprocessing.py          # CausalSensorImputer, train-isolated scalers, categorical encoding
│   ├── feature_engineering.py    # Retrospective lags, cyclic temporal features, variable-horizon target
│   ├── validation.py             # Chronological 70/15/15 splits, LOFO spatial splits, few-shot splits, leakage audits
│   └── model_trainer.py          # Phase 3 training engine, metric calculators, and benchmark protocols
├── models/
│   ├── aaci_energy_ridge_model.joblib      # Edge MCU linear model (0.77 KB, 0.07 μs)
│   ├── aaci_energy_lightgbm_model.joblib   # Primary gradient boosting model (301 KB, 0.48 μs)
│   ├── aaci_energy_xgboost_model.joblib    # Gradient boosting benchmark (474 KB, 0.26 μs)
│   ├── aaci_energy_rf_model.joblib         # Random Forest ensemble (5.2 MB, 0.74 μs)
│   ├── aaci_energy_preprocessor.joblib     # Fitted preprocessor and scaler
│   └── aaci_energy_model_metadata.json     # Complete model cards and metadata
├── benchmarks/
│   ├── chronological_benchmark_results.csv # Protocol 1 results across 7 models
│   ├── lofo_spatial_benchmark_results.csv  # Protocol 2 zero-shot cross-fan results across 9 fans
│   ├── few_shot_adaptation_benchmark_results.csv # Protocol 3 7-day adaptation results
│   └── feature_importance_ranking.csv      # Ranked feature importances
├── tests/
│   ├── __init__.py
│   └── test_energy_pipeline.py   # Comprehensive automated unit test suite (20 tests)
└── audit/
    ├── dataset_audit_report.md
    ├── phase1_5_verification_report.md
    ├── phase2_pipeline_implementation_report.md
    ├── phase2_5_final_verification_report.md
    ├── phase3_model_evaluation_report.md
    ├── pre_training_readiness_report.md
    ├── dryad_dataset_audit.json
    ├── dryad_air_velocity_by_speed.csv
    ├── mendeley_dataset_audit.json
    ├── file_by_file_duplicate_analysis.csv
    ├── sensor_error_analysis.csv
    └── energy_calculation_verification.csv
```

---

## 3. Phase 3 Benchmark Highlights

### Chronological Test Performance (75,614 Out-of-Sample Records)
| Model | MAE (Wh) | RMSE (Wh) | $R^2$ Score | Latency ($\mu\text{s}$) | Footprint | Primary Role |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Persistence Baseline** | 1.4818 | 2.6158 | -0.6114 | <0.01 | N/A | Naive Baseline |
| **Historical Mean** | 1.2404 | 2.0938 | -0.0324 | <0.01 | N/A | Statistical Baseline |
| **Ridge Regression** | **1.1917** | **1.9068** | **0.1438** | **0.07** | **0.77 KB** | **Recommended Edge MCU Deployable** |
| **Random Forest** | 1.2088 | 1.9030 | 0.1472 | 0.74 | 5.2 MB | Non-linear Ensemble |
| **LightGBM Regressor** | **1.1873** | **1.8982** | **0.1515** | **0.48** | **301.8 KB** | **Primary Gradient Benchmark** |
| **XGBoost Regressor** | **1.1853** | **1.8979** | **0.1517** | **0.26** | **474.5 KB** | Gradient Benchmark |

### Leave-One-Fan-Out (LOFO) Zero-Shot Generalization
- Mean LOFO LightGBM MAE: **0.9676 Wh ($R^2 = 0.1635$)** across all 9 unseen fans.
- Outperformed historical mean by up to **38.9%** on zero-shot fans.

---

## 4. Running the Training Pipeline

Execute the master training and benchmarking script:
```bash
python aaci_ml/v3_energy_intelligence/train_energy_models.py
```

Execute all unit test suites:
```bash
python -m pytest aaci_ml/v3_energy_intelligence/tests/ aaci_ml/occupancy_intelligence/tests/ -v
```

---

## 5. Development History & Status

- **Phase 1 (Initial Dataset Audit):** Complete & Signed off.
- **Phase 1.5 (Critical Verification Audit):** Complete & Signed off.
- **Phase 2 (Data Pipeline & Pre-Training Validation):** Complete & Signed off.
- **Phase 2.5 (Final Pre-Training Scientific Verification):** Complete & Signed off.
- **Phase 3 (Model Training, Benchmarking & Evaluation):** **COMPLETE & VERIFIED.**
