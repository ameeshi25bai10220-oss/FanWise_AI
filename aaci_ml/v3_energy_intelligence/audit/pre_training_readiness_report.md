# AACI Model 3 – Pre-Training Readiness Report & Technical Sign-off

**Project:** Atomberg Adaptive Comfort Intelligence (AACI)  
**Component:** Model 3 – Ceiling Fan Energy Intelligence  
**Phase:** Phase 2.5 Completion & Pre-Training Execution Gate  
**Date:** November 4, 2026  
**Status:** READY FOR PHASE 3 AUTHORIZATION  

---

## 1. Executive Declaration

The data ingestion, physics, preprocessing, feature engineering, and validation infrastructure for **AACI Model 3** is fully implemented, scientifically audited, and test-covered.

In compliance with the project charter:
> **NO MACHINE LEARNING MODELS WERE TRAINED, FITTED, OR PERSISTED DURING PHASE 2 OR PHASE 2.5.**
> All work has been strictly restricted to pipeline construction, physics modeling, data hygiene, anti-leakage verification, and automated unit testing.

The system is paused at the **Pre-Training Execution Gate**, awaiting explicit user approval before executing any machine learning model training in Phase 3.

---

## 2. Pre-Training Readiness Checklist

| Category | Component / Feature | Implementation Location | Verification Status |
|:---|:---|:---|:---:|
| **Project Isolation** | Zero modifications to Model 1 | `aaci_ml/predict.py` | **VERIFIED (Unchanged)** |
| **Project Isolation** | Zero modifications to Model 2 | `aaci_ml/occupancy_intelligence/` | **VERIFIED (Unchanged)** |
| **Regression Tests** | Model 2 test suite execution | `occupancy_intelligence/tests/` | **PASSED (19/19 tests)** |
| **Data Ingestion** | 9-fan Mendeley telemetry loader | `src/data_loader.py` | **VERIFIED** |
| **Metadata Hygiene** | Authoritative 3-room mapping (Bedroom, Drawingroom, Lounge) | `src/data_loader.py` | **VERIFIED** |
| **Data Ingestion** | Day-first chronological sorting | `src/data_loader.py` | **VERIFIED** |
| **Data Hygiene** | Content-based deduplication (8 cols)| `src/data_loader.py` | **VERIFIED** |
| **Data Hygiene** | Sentinel (255) & boot zero handling | `src/data_loader.py` | **VERIFIED** |
| **Data Hygiene** | Reconnection backlog tagging (>900s) | `src/data_loader.py` | **VERIFIED** |
| **Physics Engine** | BLDC & Induction power ratings | `src/energy_physics.py` | **VERIFIED** |
| **Physics Engine** | Joules $\leftrightarrow$ Wh $\leftrightarrow$ kWh conversions | `src/energy_physics.py` | **VERIFIED** |
| **Physics Engine** | Dryad seated airflow velocity mapping | `src/energy_physics.py` | **VERIFIED (Provisional)** |
| **Preprocessing** | Train-set isolation & CausalSensorImputer | `src/preprocessing.py` | **VERIFIED** |
| **Preprocessing** | Zero-shot LOFO global median fallback | `src/preprocessing.py` | **VERIFIED** |
| **Anti-Leakage** | Adversarial future perturbation invariance | `tests/test_energy_pipeline.py` | **VERIFIED** |
| **Features** | Retrospective environmental lags | `src/feature_engineering.py` | **VERIFIED** |
| **Features** | Cyclic diurnal & seasonal signals | `src/feature_engineering.py` | **VERIFIED** |
| **Features** | Retrospective operational lags | `src/feature_engineering.py` | **VERIFIED** |
| **Target Definition**| Next-observed packet energy (variable horizon) | `src/feature_engineering.py` | **VERIFIED** |
| **Leakage Barrier** | No contemporaneous tautology | `src/feature_engineering.py` | **VERIFIED** |
| **Validation** | Chronological 70/15/15 per-fan split | `src/validation.py` | **VERIFIED** |
| **Validation** | Leave-One-Fan-Out (LOFO) splits | `src/validation.py` | **VERIFIED** |
| **Validation** | Few-shot fan adaptation split (7-day)| `src/validation.py` | **VERIFIED** |
| **Integration** | Model 1 comfort query smoke test | `tests/test_energy_pipeline.py` | **PASSED (Simulation)** |
| **Test Suite** | Automated unit tests (pytest) | `tests/test_energy_pipeline.py` | **PASSED (20/20 tests)** |

---

## 3. Detailed Data Pipeline Audit Summary

1. **Ingestion Volume & Fan Balance:**
   - 9 Mendeley CSV files comprising 675,332 raw rows across 3 room categories (Bedroom, Drawingroom, Lounge).
   - Ingestion preserves explicit `fan_id` and authoritative `room_category` labels.

2. **Deduplication Resolution:**
   - Database insertion duplicates (139,787 identical repeats) are cleanly filtered out using the 8 telemetry content columns (`['datetime', 'temperature', 'humidity', 'speed', 'mode', 'eSpent', 'eSaved', 'opTime']`).
   - Intra-minute physical packets where fan speed or power changed within the same calendar minute are 100% preserved.

3. **Sensor Sentinel & Boot Error Policy:**
   - Microcontroller I2C timeout sentinels (`temperature == 255`, `humidity == 255`, 662 coupled occurrences) and boot zeros (`0.0`, 22 occurrences) are identified and cleanly handled via `filter_sensor_sentinels()` and `CausalSensorImputer`.
   - Causal forward-filling within each fan prevents both data leakage and artificial step artifacts.

4. **Target Horizon Formalization:**
   - The primary regression target `target_next_eSpent_wh` models the electrical energy consumed during the **next observed telemetry packet**.
   - Because telemetry arrivals are irregular, the forecasting horizon is **variable** (packet-to-packet) and is explicitly documented as such.

---

## 4. Physics Engine & Dryad Aerodynamic Integration

1. **Analytical Power Constants:**
   - BLDC Power by Speed: `[0W, 10W, 15W, 20W, 27W, 48W]`
   - Induction Baseline: `[0W, 24W, 34W, 55W, 64W, 96W]`
   - Energy Savings: `[0W, 14W, 19W, 35W, 37W, 48W]`

2. **Dryad Seated Air Velocity Profile (Height = 1.1m):**
   - Speed 0: $0.050\text{ m/s}$ (Assumed ASHRAE 55 still-air baseline floor; chamber fan-off mean was $0.012\text{ m/s}$)
   - Speed 1: $0.112\text{ m/s}$ (Provisional spatial surrogate)
   - Speed 2: $0.239\text{ m/s}$ (Provisional spatial surrogate)
   - Speed 3: $0.293\text{ m/s}$ (Provisional spatial surrogate)
   - Speed 4: $0.410\text{ m/s}$ (Provisional spatial surrogate)
   - Speed 5: $0.527\text{ m/s}$ (Provisional spatial surrogate)
   - Declared as `IS_AIR_VELOCITY_MAPPING_PROVISIONAL = True`.

3. **Model 1 Thermal Comfort Coupling:**
   - Model 1 probabilities are uncalibrated softmax scores from SMOTE-balanced XGBoost.
   - Speed-selection is strictly documented as an illustrative multi-objective simulation scenario.

---

## 5. Proposed Phase 3 Training Protocol (Pending Authorization)

Upon user approval to proceed to Phase 3, the following models, partitions, and metrics will be evaluated:

### 5.1. Evaluated Model Candidates
1. **Analytical Physics Baseline:** Deterministic power lookup based on previous observed state duration.
2. **Persistence Baseline:** Predicts $E_{t+1} = E_t$.
3. **Historical Mean / Median Baseline:** Predicts hour-of-day / room-level historical mean.
4. **Regularized Linear Regressor (Ridge / Lasso):** Ultra-lightweight linear baseline for edge MCU deployment.
5. **Random Forest Regressor:** Non-linear decision tree ensemble capturing complex diurnal-temperature interactions.
6. **Gradient Boosting / XGBoost Regressor:** High-capacity non-linear benchmark.

### 5.2. Evaluation Partitions
1. **Within-Fan Chronological Test:** 70% Train, 15% Validation, 15% Test (strictly chronologically separated).
2. **Leave-One-Fan-Out (LOFO) Spatial Test:** 8 fans for training, 1 held-out fan for zero-shot testing (evaluated across all 9 folds).
3. **Few-Shot Adaptation Test:** 8 base fans + first 7 days of target fan for fine-tuning; remaining days for evaluation.

### 5.3. Quantitative Benchmark Metrics
- **MAE:** Mean Absolute Error (Wh and Joules).
- **RMSE:** Root Mean Squared Error (Wh).
- **$R^2$ Score:** Coefficient of determination.
- **MAPE:** Mean Absolute Percentage Error (on non-zero operating intervals).
- **Model Size:** Serialized artifact footprint (KB).
- **Inference Latency:** Microseconds ($\mu\text{s}$) per single prediction.

---

## 6. Phase 2.5 Conclusion & Execution Gate

- **Phase 2.5 Assessment:** **PASS (WITH DOCUMENTED SCIENTIFIC LIMITATIONS)**
- **Unit Tests:** **20 / 20 Model 3 PASSED, 19 / 19 Model 2 PASSED (39 / 39 Total)**
- **Model 1 & 2 Isolation:** **100% PRESERVED (0 external edits)**
- **Pre-Training Gate:** **HOLDING FOR USER AUTHORIZATION**
