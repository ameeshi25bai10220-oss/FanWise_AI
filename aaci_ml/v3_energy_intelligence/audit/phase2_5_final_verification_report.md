# AACI Model 3 – Phase 2.5: Final Pre-Training Scientific Verification Report

**Author:** AACI Engineering Team  
**Date:** November 4, 2026  
**Status:** COMPLETE & VERIFIED (Pre-Training Gate Active)  
**Target Module:** `aaci_ml/v3_energy_intelligence/`  

---

## Executive Summary

In response to the Phase 2.5 audit directive, an exhaustive scientific and methodological verification of the Model 3 pipeline was conducted. Key findings were uncovered and rectified:

1. **Room Metadata Rectification:** Resolved a discrepancy in room categorization. The Mendeley RDD dataset comprises exactly **three room types** across 9 CSV files (`Bedroom`, `Drawingroom`, and `Lounge`). Erroneous references to "Diningroom, Kitchen, and Hall" in the Phase 2 loader dictionary were eliminated.
2. **Dryad Airflow Mapping Provenance & Formal Designation:** Investigated the source study (Liu et al., 2018, Building and Environment / Dryad DOI: `10.6078/D1V67R`). The airflow mappings are whole-room spatial averages across 180 grid points at 1.1m in a controlled climatic chamber of a 7-speed commercial fan, not direct measurements of Atomberg 5-speed fans. Furthermore, $0.050\text{ m/s}$ is an assumed ASHRAE 55 still-air baseline floor, not a measured speed-zero output (the chamber measured $0.012\text{ m/s}$ with the fan off). The mapping is now formally designated as a **PROVISIONAL AERODYNAMIC SURROGATE FOR SIMULATION ONLY**.
3. **Leakage-Free Causal Imputation:** Engineered a standalone `CausalSensorImputer` ensuring that sensor medians and scalers are estimated strictly from the training partition. For Zero-Shot Leave-One-Fan-Out (LOFO), unobserved values fall back to the training global median, completely isolating the held-out fan. An adversarial future perturbation test confirmed zero leakage.
4. **Target Horizon Definition:** Formally clarified that $E(t+1)$ represents the **energy consumed during the NEXT OBSERVED TELEMETRY PACKET**, with an intrinsically **variable forecast horizon** governed by packet arrivals, rather than a fixed-duration time step.
5. **Model 1 Probability Audit:** Confirmed that Model 1 `predict_proba` outputs are uncalibrated softmax scores from a SMOTE-balanced XGBoost classifier trained on commercial/office ASHRAE Database II data. A 0.50 threshold is a heuristic operating boundary, not a calibrated Bayesian posterior. Speed selection is strictly labelled as an illustrative simulation scenario.

**Overall Assessment:** **PASS (WITH SCIENTIFIC QUALIFICATIONS)**. All 39 tests across Model 2 and Model 3 pass with zero warnings. Model 1 and Model 2 remain 100% frozen.

---

## 1. Findings and Applied Corrections

| Subsystem | Audit Finding | Correction Applied |
|:---|:---|:---|
| **Room Categories** | Phase 2 dictionary contained fictitious "Diningroom, Kitchen, Hall". | Replaced with authoritative 3-room mapping (`Bedroom`, `Drawingroom`, `Lounge`) derived from actual CSV files on disk. |
| **Airflow Mapping** | Mapping was presented as verified speeds, including $0.050\text{ m/s}$ at Speed 0. | Re-classified as `IS_AIR_VELOCITY_MAPPING_PROVISIONAL = True`. Documented $0.050\text{ m/s}$ as an assumed ASHRAE baseline floor, not measured output. |
| **Sensor Imputation** | Global median fallback could theoretically leak across partitions if called globally. | Implemented `CausalSensorImputer` with strict `fit(train_df)` and `transform(test_df)` separation and zero-shot fallback. |
| **Forecasting Horizon** | Informal references suggested fixed 2.5-min intervals. | Explicitly formalized that $E(t+1)$ forecasts the next observed packet with a variable sampling horizon. |
| **Model 1 Integration** | Candidate speed evaluations implied guaranteed thermal comfort. | Formally documented as an illustrative engineering simulation with uncalibrated probabilities. |

---

## 2. Authoritative 9-Fan Room Metadata

Verification of files in `RDD Datset/` confirms exactly nine CSV files representing three room categories:

| S.No. | CSV Filename | Fan Identifier (`fan_id`) | Authoritative Room Category | Raw Row Count |
|:---:|:---|:---|:---|:---:|
| 1 | `Bedroom_Fan_1.csv` | `Bedroom_Fan_1` | **Bedroom** | 100,539 |
| 2 | `Bedroom_Fan_2.csv` | `Bedroom_Fan_2` | **Bedroom** | 179,270 |
| 3 | `Bedroom_Fan_3.csv` | `Bedroom_Fan_3` | **Bedroom** | 129,564 |
| 4 | `Drawingroom_Fan_1.csv` | `Drawingroom_Fan_1` | **Drawingroom** | 71,987 |
| 5 | `Drawingroom_Fan_2.csv` | `Drawingroom_Fan_2` | **Drawingroom** | 46,712 |
| 6 | `Drawingroom_Fan_3.csv` | `Drawingroom_Fan_3` | **Drawingroom** | 80,488 |
| 7 | `Lounge_Fan_1.csv` | `Lounge_Fan_1` | **Lounge** | 19,003 |
| 8 | `Lounge_Fan_2.csv` | `Lounge_Fan_2` | **Lounge** | 50,051 |
| 9 | `Lounge_Fan_3.csv` | `Lounge_Fan_3` | **Lounge** | 47,159 |
| **Total** | **9 Files** | **9 Fans** | **3 Room Categories** | **675,332 Rows** |

---

## 3. Dryad Airflow Mapping – Evidence Audit & Provisional Status

### 3.1. Original Source Investigation
- **Source Study:** Liu, S., Lipczynska, A., Schiavon, S., & Arens, E. (2018). *"Detailed experimental investigation of air speed field induced by ceiling fans."* Building and Environment, 142, 342-360.
- **Dataset:** UC Berkeley Center for the Built Environment (CBE), Dryad Digital Repository (DOI: `10.6078/D1V67R`).
- **Chamber Setup:** Controlled climatic chamber ($5.5\text{m} \times 4.2\text{m} \times 2.6\text{m}$).
- **Spatial Resolution:** High-density 3D grid measuring 15 rows $\times$ 12 columns (180 spatial coordinates) across 4 heights ($0.1\text{m}, 0.6\text{m}, 1.1\text{m}, 1.7\text{m}$).
- **Fan Hardware in Study:** A commercial ceiling fan with **7 discrete speed settings** plus OFF (Speeds 0 to 7).

### 3.2. Quantitative Audit of Mapping Values

| Speed Setting in Mapping | Source File Column Group | Measurement Height | Nature of Measurement | Experimental Fan Speed | Mapped Velocity ($v_a$) | Scientific Status |
|:---:|:---|:---:|:---|:---:|:---:|:---|
| **Speed 0** | N/A (Assumed Floor) | N/A | Assumed Reference | 0 (Fan OFF) | **0.050 m/s** | **Assumed ASHRAE 55 indoor still-air baseline floor.** In the chamber with fan OFF, measured mean background air speed was $0.012\text{ m/s}$ ($SD = 0.007\text{ m/s}$). It is NOT a measured fan output. |
| **Speed 1** | `Speed 1 - 1.1 m` | 1.1m (Seated) | Spatial mean of 180 grid points | Speed 1 of 7 | **0.112 m/s** | **Provisional Surrogate.** Chamber spatial mean (range: $0.038 - 0.281\text{ m/s}$). |
| **Speed 2** | `Speed 2 - 1.1 m` | 1.1m (Seated) | Spatial mean of 180 grid points | Speed 2 of 7 | **0.239 m/s** | **Provisional Surrogate.** Chamber spatial mean (range: $0.051 - 0.542\text{ m/s}$). |
| **Speed 3** | `Speed 3 - 1.1 m` | 1.1m (Seated) | Spatial mean of 180 grid points | Speed 3 of 7 | **0.293 m/s** | **Provisional Surrogate.** Chamber spatial mean (range: $0.062 - 0.684\text{ m/s}$). |
| **Speed 4** | `Speed 4 - 1.1 m` | 1.1m (Seated) | Spatial mean of 180 grid points | Speed 4 of 7 | **0.410 m/s** | **Provisional Surrogate.** Chamber spatial mean (range: $0.081 - 0.892\text{ m/s}$). |
| **Speed 5** | `Speed 5 - 1.1 m` | 1.1m (Seated) | Spatial mean of 180 grid points | Speed 5 of 7 | **0.527 m/s** | **Provisional Surrogate.** Chamber spatial mean (range: $0.098 - 1.124\text{ m/s}$). |

### 3.3. Key Scientific Discrepancies
1. **Speed Level Mismatch:** The CBE test fan has 7 active speed levels; Atomberg BLDC fans have 5 active speed levels. Speeds 1–5 were aligned by ordinal rank, which does not guarantee aerodynamic equivalence (different blade pitch, sweep diameter, and RPM).
2. **Spatial Average vs Local Plume:** The mapped values are spatial means averaged across the entire room. Directly under the fan hub/blade tip, velocities reach $1.12\text{ m/s}$ at Speed 5, while at room corners they drop to $0.10\text{ m/s}$.
3. **Formal Designation:** As required by the audit, `IS_AIR_VELOCITY_MAPPING_PROVISIONAL = True` is declared in code. The mapping must strictly be treated as an illustrative surrogate, not factory calibration.

---

## 4. Preprocessing and Anti-Leakage Verification

### 4.1. Implementation of `CausalSensorImputer`
To prevent information leakage across temporal and cross-fan boundaries:
1. **Train-Set Fitting:**
   $$\mu_{\text{fan}} = \text{median}(X_{\text{train}}[\text{fan\_id}]), \quad \mu_{\text{global}} = \text{median}(X_{\text{train}})$$
   Statistics are calculated strictly on training records.
2. **Causal Forward Fill:**
   Transformations apply chronological `.ffill()` per fan.
3. **Zero-Shot LOFO Fallback:**
   When transforming an unseen held-out test fan, initial unobserved NaNs are filled using $\mu_{\text{global}}$ derived from the 8 training fans. The test fan's own data is never accessed during fitting.
4. **Few-Shot LOFO:**
   Only the first 7 days of the target fan are permitted in the adaptation set to update fan-specific parameters.

### 4.2. Adversarial Future Perturbation Verification
A new automated test (`test_adversarial_future_test_perturbation_isolation`) was added to the test suite:
- **Experiment:** Future test observations were deliberately corrupted with extreme outliers ($T = 999,999.0^\circ\text{C}, RH = -88,888.0\%$).
- **Verification:** Training imputer medians, `StandardScaler.mean_`, `StandardScaler.scale_`, and the transformed training feature matrix $X_{\text{train}}$ were verified to be **100.00% invariant** before and after test corruption.

---

## 5. Next-Interval Target Definition & Horizon Formalization

### 5.1. Construction of $E(t+1)$
In `feature_engineering.py`:
$$\text{target\_next\_eSpent\_wh} = \text{shift}(-1)\left(\frac{\text{eSpent [Joules]}}{3600.0}\right)$$

### 5.2. Formal Operational Properties
1. **Target Meaning:** Represents the electrical energy consumed during the **subsequent observed telemetry packet**.
2. **Variable Forecasting Horizon:** 
   The telemetry packet arrival times in the Mendeley dataset are irregular:
   - 25th percentile: $0 - 60\text{ s}$
   - Median (50th percentile): $60 - 150\text{ s}$
   - 75th percentile: $152 - 360\text{ s}$
   - 90th percentile: $364 - 1260\text{ s}$
   - **Conclusion:** The forecasting horizon is **packet-to-packet and intrinsically variable**. It is NOT a fixed 2.5-minute or 5-minute constant interval.
3. **Zero-Speed Inactive Periods:**
   When $\text{speed} = 0$, $P_{\text{BLDC}} = 0\text{ W}$, and $\text{target\_next\_eSpent\_wh} \equiv 0.0\text{ Wh}$.
4. **Reconnection Backlog Spikes:**
   Packets where $\text{opTime} > 900\text{ s}$ represent multi-hour offline accumulations. An optional parameter `exclude_backlog_targets=True` in `build_supervised_energy_dataset()` allows filtering these abnormal spikes during interval training.
5. **Leakage Safeguards:**
   Future speed $s_{t+1}$, future duration $\text{opTime}_{t+1}$, and future sensor readings $T_{t+1}, RH_{t+1}$ are strictly excluded from the feature matrix $X_t$.

---

## 6. Model 1 Probability & Compatibility Audit

### 6.1. Probability Calibration Status
- **Model Type:** XGBoost (`XGBClassifier`) with multi-class objective (`multi:softprob`).
- **Resampling:** Synthetic Minority Over-sampling Technique (SMOTE) was applied to the 114,375 training rows, artificially forcing class balance to ~33.3% per class (`Cold`, `Neutral`, `Hot`).
- **Calibration Finding:** `CalibratedClassifierCV` was **NOT** used during Model 1 training.
- **Scientific Impact:** The output of `predict_proba()` represents raw decision-tree vote softmax scores, warped by synthetic SMOTE priors. They are **uncalibrated confidence scores**, not true Bayesian posterior probabilities.
- **Decision Threshold:** A threshold of $P(\text{Neutral}) > 0.50$ is a heuristic operating convention, not an experimentally validated threshold of human thermal satisfaction.

### 6.2. Domain Shift & Evaluation Limitations
- Model 1 was trained on the ASHRAE Global Thermal Comfort Database II (predominantly commercial office and educational buildings in North America, Europe, and East Asia, with air conditioning or natural ventilation, clo ~0.6, met ~1.2).
- The Mendeley dataset consists of residential Indian households.
- **Scientific Boundary:** Evaluating candidate fan speeds via Model 1 represents an **illustrative multi-objective simulation scenario**. It does NOT provide a guaranteed thermal comfort prediction for residential users.

---

## 7. Regression Test Suite Execution

All tests were executed using `pytest` across Model 3 and Model 2:

```bash
python -m pytest aaci_ml/v3_energy_intelligence/tests/ aaci_ml/occupancy_intelligence/tests/ -v
```

### Test Output Log
```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: C:\Users\bharg\OneDrive\Desktop\Atomberg
plugins: anyio-4.10.0
collected 39 items

aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_energy_physics_constants PASSED [  2%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_energy_physics_calculations PASSED [  5%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_energy_unit_conversions PASSED [  7%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_speed_to_air_velocity_dryad_mapping PASSED [ 10%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_verify_energy_reading PASSED [ 12%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_authoritative_fan_room_mapping PASSED [ 15%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_timestamp_parsing_and_sorting PASSED [ 17%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_content_deduplication PASSED [ 20%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_sensor_error_tagging_and_filtering PASSED [ 23%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_backlog_tagging PASSED [ 25%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_cyclic_temporal_features PASSED [ 28%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_lag_feature_causality PASSED [ 30%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_supervised_dataset_no_contemporaneous_leakage PASSED [ 33%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_adversarial_future_test_perturbation_isolation PASSED [ 35%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_causal_sensor_imputer_zero_shot_lofo PASSED [ 38%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_temporal_split_no_leakage PASSED [ 41%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_leave_one_fan_out_split PASSED [ 43%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_few_shot_fan_split PASSED [ 46%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_candidate_speed_evaluations PASSED [ 48%]
aaci_ml/v3_energy_intelligence/tests/test_energy_pipeline.py::test_model1_interface_compatibility_smoke_test PASSED [ 51%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestDataLoaderAndPartitions::test_partition_shapes_and_boundaries PASSED [ 53%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestDataLoaderAndPartitions::test_target_values_are_strictly_binary PASSED [ 56%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestDataLoaderAndPartitions::test_no_future_date_leakage PASSED [ 58%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestPsychrometricCalculation::test_calculate_humidity_ratio_accuracy PASSED [ 61%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestPsychrometricCalculation::test_calculate_humidity_ratio_bounds PASSED [ 64%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_a-logistic_regression] PASSED [ 66%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_a-decision_tree] PASSED [ 69%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_a-random_forest] PASSED [ 71%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_a-xgboost] PASSED [ 74%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_b-logistic_regression] PASSED [ 76%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_b-decision_tree] PASSED [ 79%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_b-random_forest] PASSED [ 82%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_b-xgboost] PASSED [ 84%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_c-logistic_regression] PASSED [ 87%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_c-decision_tree] PASSED [ 89%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_c-random_forest] PASSED [ 92%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_all_12_model_configurations_load_and_predict[set_c-xgboost] PASSED [ 94%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_threshold_override_behavior PASSED [ 97%]
aaci_ml/occupancy_intelligence/tests/test_module1_pipeline.py::TestOccupancyPredictor::test_unphysical_sensor_validation_rejection PASSED [100%]

============================= 39 passed in 3.77s ==============================
```

- **Model 3 Tests:** 20 / 20 PASSED (0 warnings).
- **Model 2 Regression Tests:** 19 / 19 PASSED (0 regressions).
- **Model 1 / Model 2 Codebase:** 100% frozen, 0 lines modified.

---

## 8. Remaining Scientific Limitations

1. **Airflow Surrogate Uncertainty:** Without physical anemometer testing on actual Atomberg fans installed in rooms with varied ceiling heights and occupant distances, the Dryad air velocity profile remains an approximate surrogate.
2. **Variable Interval Discretization:** Telemetry packets reflect irregular intervals. Forecasting $E(t+1)$ models energy per packet, which varies with network conditions and fan dwell times.
3. **Model 1 Transfer Domain Gap:** Comfort predictions from Model 1 carry domain shift uncertainty when transferred to residential Indian contexts.

---

## 9. Final Phase 2.5 Assessment & Training Gate Status

### Final Assessment: **PASS (WITH DOCUMENTED SCIENTIFIC LIMITATIONS)**

- **Pipeline Hygiene & Deduplication:** VERIFIED
- **Authoritative Room Mapping:** VERIFIED (9 fans, 3 rooms)
- **Anti-Leakage Imputation & Scaling:** VERIFIED (adversarially tested)
- **Target Construction & Variable Horizon:** FORMALIZED & VERIFIED
- **Provisional Aerodynamic Surrogate:** PROPERLY LABELLED
- **Automated Tests:** 39 / 39 PASSED
- **Model Training Execution Gate:** **ACTIVE & LOCKED.**

**Zero models have been trained.** The pipeline is verified and frozen, awaiting explicit authorization to begin Phase 3 model training.

