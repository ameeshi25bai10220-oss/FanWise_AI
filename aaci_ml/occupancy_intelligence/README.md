# AACI Model 2: Occupancy & Spatial Intelligence

**Atomberg Adaptive Comfort Intelligence (AACI)** – Supervised Machine Learning & Spatial Telemetry Suite for Room Occupancy Detection, Headcount Estimation, and Fan Energy Intelligence.

---

## Architecture Overview

AACI Model 2 operates as a completely decoupled, multi-module intelligence system residing in `aaci_ml/occupancy_intelligence/`:

```
aaci_ml/occupancy_intelligence/
├── README.md                          # Top-level spatial intelligence documentation
├── audit/                             # Module 1 cross-dataset audits & compatibility matrices
├── datasets/                          # Machine-readable inventory of all datasets
├── documentation/                     # Architecture, evaluation strategies & roadmaps
├── src/                               # Module 1 occupancy detection training and inference
├── models/                            # Module 1 trained binary classifiers (12 configurations)
├── evaluation/                        # Module 1 ROC/PR curves, thresholds & benchmark reports
├── tests/                             # Module 1 unit test suite (19 tests)
│
├── module2/                           # MODULE 2: Spatial Headcount & Pattern Intelligence
│   ├── README.md                      # Module 2 documentation & quickstart
│   ├── src/                           # ROBOD data loader, LORO training, and pattern analytics
│   ├── models/                        # Headcount regressors & multimodal scalers
│   ├── evaluation/                    # Headcount benchmarks, LORO evaluation & error plots
│   ├── audit_v2/                      # Independent verification audit & freeze addendum
│   └── tests/                         # Module 2 unit test suite (10 tests)
│
└── module3/                           # MODULE 3: Fan Telemetry, Usage Pattern & Energy Intelligence
    ├── README.md                      # Module 3 documentation & quickstart
    ├── audit/                         # RDD data audit & initial data quality report
    ├── src/                           # Telemetry loader, usage analytics & energy predictors
    ├── models/                        # Trained Random Forest energy and duration pipelines
    ├── evaluation/                    # Diurnal profiles, room comparisons & benchmark reports
    ├── final_audit/                   # Authoritative technical audit, reconciliation & limitations
    └── tests/                         # Module 3 unit test suite (10 tests)
```

---

## Module Index

### 1. Module 1: Room Occupancy Detection (UCI IoT Dataset)
- **Objective:** Binary detection of room occupancy (`0 = Vacant`, `1 = Occupied`) using ambient environmental sensors.
- **Dataset:** UCI Occupancy Detection (`archive (2)/`).
- **Feature Sets:** Set A (Temp, Humidity, Light, CO2, HumidityRatio), Set B (Low-BOM: Temp, Humidity, HumidityRatio, Hour), Set C (Temp, Humidity, Light, CO2).
- **Top Models:** Random Forest Set A (99.15% Accuracy, 0.9800 F1), Logistic Regression Set A (99.00% Accuracy, 1.7 $\mu$s latency).
- **Status:** **FROZEN RESEARCH BASELINE**.

### 2. Module 2: Spatial Headcount & Pattern Intelligence (ROBOD Dataset)
- **Objective:** Continuous occupant headcount estimation ($N \ge 0$) and room utilization analytics.
- **Dataset:** NUS SDE4 ROBOD (`SupplementaryData/`, 52,128 records across 5 commercial rooms).
- **Validation:** Leave-One-Room-Out (LORO) cross-validation and chronological forward testing.
- **Top Benchmark:** Multimodal Random Forest (MAE: 1.28 persons with Wi-Fi; 2.67 persons without Wi-Fi).
- **Status:** **FROZEN RESEARCH BASELINE** (with audit errata and limitations documented in `audit_v2/`).

### 3. Module 3: Fan Telemetry, Usage Pattern & Energy Intelligence (RDD Dataset)
- **Objective:** Smart fan operational physics, operating duration, baseline induction fan savings, and human usage dynamics.
- **Dataset:** Atomberg Residential Telemetry (`RDD Datset/`, 675,332 raw rows, 534,857 clean rows across 9 fans).
- **Key Discovery:** Telemetry energy variables (`eSpent`, `eSaved`) follow exact speed-indexed deterministic power constants: BLDC power [0, 10, 15, 20, 27, 48] W vs Induction Baseline [0, 24, 34, 55, 64, 96] W.
- **Status:** **AUDITED & FROZEN RESEARCH BASELINE** (with audit errata and limitations documented in `final_audit/`).

---

## Automated Test Suite

AACI Model 2 includes 39 automated unit and regression tests covering all three modules:

```bash
python -m pytest aaci_ml/occupancy_intelligence/ -v
```

```
collected 39 items
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py .......... 19 passed
aaci_ml/occupancy_intelligence/module2/tests/test_module2_pipeline.py .. 10 passed
aaci_ml/occupancy_intelligence/module3/tests/test_module3_pipeline.py .. 10 passed
============================= 39 passed in 5.59s =============================
```

---

## Strict Preservation & Governance

Existing baseline systems—including **Thermal Comfort Intelligence (Model 1 V1/V2/V3/V3.1)**—remain completely isolated and untouched. No cross-module fusion or dashboard controller integrations are initiated without explicit authorization.
