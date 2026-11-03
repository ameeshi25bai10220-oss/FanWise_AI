# AACI Model 3 – Phase 2: Data Pipeline Implementation & Pre-Training Validation Report

**Author:** AACI Engineering Team  
**Date:** November 4, 2026 (Updated with Phase 2.5 Scientific Verification)  
**Status:** COMPLETE & VERIFIED (Pre-Training Gate Active)  
**Target Module:** `aaci_ml/v3_energy_intelligence/`  

---

## Executive Summary

Phase 2 of **AACI Model 3 (Ceiling Fan Energy Intelligence)** has been successfully implemented and verified. In accordance with strict project guidelines, **zero machine learning models were trained** during this phase. The objective was the construction, testing, and validation of an independent, leak-free data ingestion, preprocessing, deterministic physics, and feature engineering pipeline.

Following the Phase 2.5 scientific audit, the authoritative room mapping (3 categories: Bedroom, Drawingroom, Lounge across 9 fans) was formalized, the Dryad airflow mapping was designated as a provisional surrogate, causal train-set sensor imputation was enforced via `CausalSensorImputer`, and target forecasting was formalized as variable-horizon packet prediction.

All deliverables reside inside `aaci_ml/v3_energy_intelligence/`. All 20 automated Model 3 unit tests and 19 Model 2 regression tests pass with zero warnings, and cross-module smoke tests confirm 100% compatibility with Model 1 without modifying a single line of Model 1 or Model 2 code.

---

## 1. Strict Project Isolation Verification

- **Model 1 (Thermal Comfort Intelligence):** Completely frozen. Unmodified in `aaci_ml/`.
- **Model 2 (Occupancy Intelligence):** Completely frozen. All 19 Module 1 / Module 2 test suites continue to pass with zero regressions.
- **Git State:** `git status` verifies that all newly created and modified files reside strictly within `aaci_ml/v3_energy_intelligence/`.

---

## 2. Implemented Architecture & Source Modules

The Model 3 pipeline is organized into modular, independently testable components under `aaci_ml/v3_energy_intelligence/src/`:

```
aaci_ml/v3_energy_intelligence/
├── src/
│   ├── __init__.py               # Package exports
│   ├── energy_physics.py         # Deterministic physics, power constants, unit conversions, provisional Dryad mapping
│   ├── data_loader.py            # 9-fan ingestion, day-first parsing, content deduplication, 3-room mapping
│   ├── preprocessing.py          # CausalSensorImputer, train-fitted scalers, categorical encoding
│   ├── feature_engineering.py    # Retrospective lags, cyclic temporal signals, variable-horizon target
│   └── validation.py             # Chronological 70/15/15 splits, LOFO cross-fan splits, few-shot adaptation
├── tests/
│   ├── __init__.py
│   └── test_energy_pipeline.py   # Comprehensive 20-test pytest suite
└── audit/
    ├── dataset_audit_report.md
    ├── phase1_5_verification_report.md
    ├── phase2_pipeline_implementation_report.md
    ├── phase2_5_final_verification_report.md
    ├── pre_training_readiness_report.md
    ├── dryad_dataset_audit.json
    ├── dryad_air_velocity_by_speed.csv
    ├── mendeley_dataset_audit.json
    ├── file_by_file_duplicate_analysis.csv
    ├── sensor_error_analysis.csv
    └── energy_calculation_verification.csv
```

---

## 3. Detailed Component Technical Specifications

### 3.1. `energy_physics.py` – Deterministic Energy & Aerodynamics

1. **Physical Power Constants (Watts):**
   - **BLDC Motor:** `{0: 0.0W, 1: 10.0W, 2: 15.0W, 3: 20.0W, 4: 27.0W, 5: 48.0W}`
   - **Baseline Induction:** `{0: 0.0W, 1: 24.0W, 2: 34.0W, 3: 55.0W, 4: 64.0W, 5: 96.0W}`
   - **Modeled Power Saved:** `{0: 0.0W, 1: 14.0W, 2: 19.0W, 3: 35.0W, 4: 37.0W, 5: 48.0W}`
   - Satisfies identity: $P_{\text{Saved}}(s) \equiv P_{\text{Baseline}}(s) - P_{\text{BLDC}}(s)$ across all $s \in [0, 5]$.

2. **Deterministic Energy Equations:**
   $$E_{\text{BLDC}} = P_{\text{BLDC}}(\text{speed}) \times \text{opTime} \quad [\text{Joules}]$$
   $$E_{\text{Baseline}} = P_{\text{Baseline}}(\text{speed}) \times \text{opTime} \quad [\text{Joules}]$$
   $$E_{\text{Saved}} = P_{\text{Saved}}(\text{speed}) \times \text{opTime} \quad [\text{Joules}]$$

3. **Exact Unit Conversions:**
   - $\text{Wh} = \text{Joules} / 3600.0$
   - $\text{kWh} = \text{Joules} / 3600000.0$

4. **Provisional Dryad Aerodynamic Transfer Function:**
   - Source: Liu et al. (2018) / UC Berkeley CBE ceiling fan benchmark (`doi_10_6078_D1V67R__v20180717`).
   - Whole-room spatial mean air velocities across 180 grid points at 1.1m seated height:
     - **Speed 0:** $0.050\text{ m/s}$ (Assumed ASHRAE 55 indoor still-air baseline floor; chamber fan-off mean was $0.012\text{ m/s}$)
     - **Speed 1:** $0.112\text{ m/s}$ (Provisional spatial surrogate)
     - **Speed 2:** $0.239\text{ m/s}$ (Provisional spatial surrogate)
     - **Speed 3:** $0.293\text{ m/s}$ (Provisional spatial surrogate)
     - **Speed 4:** $0.410\text{ m/s}$ (Provisional spatial surrogate)
     - **Speed 5:** $0.527\text{ m/s}$ (Provisional spatial surrogate)
   - Status: Declared as `IS_AIR_VELOCITY_MAPPING_PROVISIONAL = True` for simulation purposes only.

---

### 3.2. `data_loader.py` – Ingestion & Authoritative Metadata

1. **Authoritative 9-Fan Room Metadata:**
   - 3 Bedrooms: `Bedroom_Fan_1`, `Bedroom_Fan_2`, `Bedroom_Fan_3`
   - 3 Drawing rooms: `Drawingroom_Fan_1`, `Drawingroom_Fan_2`, `Drawingroom_Fan_3`
   - 3 Lounges: `Lounge_Fan_1`, `Lounge_Fan_2`, `Lounge_Fan_3`
   - Erroneous references to "Diningroom, Kitchen, Hall" removed.
2. **Timestamp Parsing:**
   - Strict `dayfirst=True` parsing (`DD-MM-YYYY HH:MM`) with chronological sorting by `(datetime, S.No.)`.
3. **Content-Based Deduplication:**
   - Deduplicates across the 8 telemetry content columns (`datetime`, `temperature`, `humidity`, `speed`, `mode`, `eSpent`, `eSaved`, `opTime`), eliminating database insertion repeats (139,787 rows) while preserving legitimate intra-minute physical state updates.
4. **Sensor Sentinel & Boot Error Handling:**
   - Identifies I2C timeout sentinels (`255`) and boot zero readings (`0.0`), with options to filter (`drop_errors=True`) or mask with `NaN` for imputation.
5. **Backlog Tagging:**
   - Flags reconnection accumulation packets where $\text{opTime} > 900\text{ s}$ (`is_backlog = True`).

---

### 3.3. `preprocessing.py` – Train-Set Isolation & Imputation

1. **`CausalSensorImputer`:**
   - Scalers and medians estimated strictly from the training partition.
   - For Zero-Shot LOFO, unseen fans fall back to the training global median without ever accessing the held-out fan's complete dataset.
   - For Few-Shot LOFO, only the first 7 days are permitted for calibration.
2. **`EnergyPreprocessor` Class:**
   - Implements `fit(X_train)`, `transform(X)`, `fit_transform(X_train)`.
   - Scaler statistics (means, standard deviations) are computed strictly on training partitions.
   - Adversarial future test perturbation test confirms 100.00% parameter and training feature invariance.

---

### 3.4. `feature_engineering.py` – Causal Features & Leakage Prevention

1. **Target Formulation & Variable Horizon:**
   - Primary Target: `target_next_eSpent_wh` $= eSpent_{t+1} / 3600.0$.
   - Represents the electrical energy consumed during the **next observed telemetry packet**.
   - Horizon is intrinsically variable (ranging from ~30s to 300s); not a fixed time step.
   - Zero-speed periods are represented as 0.0 Wh.
   - Offline accumulation spikes ($\text{opTime} > 900\text{ s}$) can be excluded via `exclude_backlog_targets=True`.
2. **Prevention of Contemporaneous Tautology:**
   - Contemporaneous `speed`, `opTime`, `eSpent`, and `eSaved` are forbidden from feature sets predicting interval energy.
3. **Causal Retrospective Features:**
   - Environmental Lags: $T_{t-1}, T_{t-2}, RH_{t-1}, RH_{t-2}$, differences $\Delta T, \Delta RH$, and rolling retrospective means.
   - Cyclic Temporal Signals: $\sin / \cos$ of hour-of-day, $\sin / \cos$ of month-of-year, `day_of_week`, `is_weekend`, `is_night`.
   - Operational Lags: $s_{t-1}, s_{t-2}$, state transition flag, mode lag, lagged energy $E_{t-1}$ (Wh), rolling energy sums.
4. **Candidate Speed Evaluation Matrix:**
   - Generates the 6-row evaluation table across speeds 0–5 for multi-objective simulation.

---

### 3.5. `validation.py` – Partitioning & Leakage Auditing

1. **Chronological 70/15/15 Split:**
   - Partitions time series chronologically within each fan without time overlaps.
2. **Leave-One-Fan-Out (LOFO) Split:**
   - 8 fans for training, 1 fan completely held out as an unseen zero-shot test set (generator yields all 9 folds).
3. **Few-Shot Adaptation Split:**
   - 8 base fans + first 7 days of target fan for adaptation; remaining days for testing.
4. **Automated Leakage Validators:**
   - `validate_no_temporal_leakage()` and `validate_no_feature_leakage()` prevent leaks.

---

## 4. Automated Test Suite Results

Test execution command:
```bash
python -m pytest aaci_ml/v3_energy_intelligence/tests/ aaci_ml/occupancy_intelligence/tests/ -v
```

**Results:**
- **Model 3 Tests:** **20 / 20 PASSED** (0 warnings, 0 failures)
- **Model 2 Regression Tests:** **19 / 19 PASSED** (0 regressions)
- **Total Combined:** **39 / 39 PASSED in 3.77 seconds**

---

## 5. Pre-Training Phase 2 Sign-off & Gate Status

| Verification Criterion | Status | Evidence |
|:---|:---|:---|
| Absolute isolation of Models 1 & 2 | PASSED | Git status confirms no edits outside `v3_energy_intelligence/` |
| Authoritative 9-fan room metadata | PASSED | 3 room categories (Bedroom, Drawingroom, Lounge) verified |
| Day-first timestamp parsing | PASSED | `parse_and_sort_telemetry()` verified |
| Content-based deduplication | PASSED | Preserves physical intra-minute packets, drops DB repeats |
| Sensor sentinel handling | PASSED | Sentinels (255) and boot zeros (0) tagged & filtered |
| Backlog record tagging | PASSED | `opTime > 900s` flagged |
| Physics power and energy equations | PASSED | BLDC, induction, saved constants & conversions verified |
| Dryad aerodynamic mapping | PASSED | Properly designated as provisional surrogate for simulation |
| Preprocessing & Imputation isolation | PASSED | `CausalSensorImputer` verified via adversarial perturbation test |
| Target definition & variable horizon | PASSED | Next-observed packet energy formalized; no fixed interval claims |
| Strict causality & no future leakage | PASSED | Lags strictly retrospective, shifts within fan groups |
| No contemporaneous tautology | PASSED | Contemporaneous $s_t$, $\text{opTime}_t$ excluded from energy predictors |
| 70/15/15 temporal splitting | PASSED | Strict chronological separation verified |
| LOFO cross-fan splitting | PASSED | Full zero-shot cross-validation generator operational |
| Automated unit tests | PASSED | 39 / 39 tests passing across entire repository |
| **Model Training Execution Gate** | **ACTIVE** | **NO ML MODELS HAVE BEEN TRAINED.** Awaiting approval. |
