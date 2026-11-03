# AACI Model 3: Energy Intelligence – Comprehensive Dataset Audit Report (Phase 1 & Phase 1.5)

## 1. Project Context & Strict Architecture Isolation

**Atomberg Adaptive Comfort Intelligence (AACI)** is structured as a three-model decoupled architecture:
1. **Model 1 – Thermal Comfort Intelligence:** Predicts occupant comfort state (Cold / Neutral / Hot) using ASHRAE Global Thermal Comfort Database II. Completely frozen in `aaci_ml/v1_baseline/`, `aaci_ml/v2_feature_selection/`, and `aaci_ml/v3_personalization/`.
2. **Model 2 – Occupancy Intelligence:** Predicts room presence (Occupied / Vacant) and spatial headcount using UCI IoT sensor data and NUS SDE4 ROBOD data. Completely frozen in `aaci_ml/occupancy_intelligence/`.
3. **Model 3 – Energy Intelligence (NEW DEVELOPMENT):** Estimates ceiling fan energy consumption and enables energy-efficient speed selection under thermal comfort constraints. Resides exclusively in `aaci_ml/v3_energy_intelligence/`.

---

## 2. Dataset Inventory & Provenance

Three distinct dataset sources were audited:

### 2.1 Primary Energy Dataset: Mendeley Smart Ceiling Fan Telemetry
- **Citation:** "Dataset of Usage Pattern and Energy Analysis of an Internet of Things-enabled Ceiling Fan" (Mendeley Data, doi:10.17632/sr7t6xfh8w.3).
- **Location in Workspace:** `RDD Datset/`
- **Contents:** 9 CSV files covering 9 individual smart ceiling fans across 3 room categories in Indian households.
- **Total Raw Records:** **675,332 rows**.

### 2.2 Aerodynamic Research Dataset: Dryad Ceiling Fan Air Speed Distribution
- **Citation:** "Air speed distribution of ceiling fans" (Dryad, doi:10.6078/D1V67R).
- **Location in Workspace:** `doi_10_6078_D1V67R__v20180717/`
- **Contents:** 2 CSV files (`Single_Fan.csv`, `Two_Fan.csv`).
- **Nature of Data:** Spatial fluid velocity measurements ($m/s$) across an indoor test chamber grid at 4 heights and 12 radial positions.
- **Boundary:** Contains **zero electrical energy measurements**. Must not be combined directly with electrical telemetry. Serves as physical aerodynamic transfer function to map fan speed to air velocity ($v_a$).

### 2.3 Commercial Building Dataset: ROBOD (NUS SDE4 Singapore)
- **Location in Workspace:** `SupplementaryData/`
- **Contents:** 5 CSV files (`combined_Room1.csv` through `combined_Room5.csv`).
- **Nature of Data:** Commercial lecture rooms and office spaces in Singapore with CO2, acoustics, and Wi-Fi client counts.
- **Boundary:** Belongs strictly to Model 2 (Occupancy Intelligence); zero residential ceiling fan telemetry.

---

## 3. Comprehensive Mendeley Telemetry Dataset Audit

| Filename | Room Category | File Size | Raw Rows | Content Duplicates (Diff S.No.) | Duplicate % | Unique Timestamps | Clean Records After Dedup | Intra-Minute Distinct Events | Max Single-Packet Repeat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Bedroom_Fan_1.csv` | Bedroom | 3.74 MB | 79,384 | 6,705 | 8.45% | 45,309 | 72,679 | 27,370 | 4 |
| `Bedroom_Fan_2.csv` | Bedroom | 8.49 MB | 179,243 | 129,802 | **72.42%** | 41,434 | 49,441 | 8,007 | 118 |
| `Bedroom_Fan_3.csv` | Bedroom | 4.75 MB | 110,409 | 84 | 0.08% | 95,051 | 110,325 | 15,274 | 2 |
| `Drawingroom_Fan_1.csv` | Drawing Room | 2.79 MB | 60,444 | 46 | 0.08% | 51,551 | 60,398 | 8,847 | 2 |
| `Drawingroom_Fan_2.csv` | Drawing Room | 1.98 MB | 43,529 | 3,038 | 6.98% | 32,274 | 40,491 | 8,217 | 3 |
| `Drawingroom_Fan_3.csv` | Drawing Room | 3.54 MB | 77,764 | 25 | 0.03% | 56,832 | 77,739 | 20,907 | 2 |
| `Lounge_Fan_1.csv` | Lounge | 0.93 MB | 21,305 | 26 | 0.12% | 17,807 | 21,279 | 3,472 | 2 |
| `Lounge_Fan_2.csv` | Lounge | 2.58 MB | 55,488 | 20 | 0.04% | 47,047 | 55,468 | 8,421 | 2 |
| `Lounge_Fan_3.csv` | Lounge | 2.18 MB | 47,766 | 41 | 0.09% | 38,931 | 47,725 | 8,794 | 2 |
| **TOTAL** | **ALL 9 FANS** | **31.0 MB** | **675,332** | **139,787** | **20.70%** | **426,236** | **535,545** | **109,309** | **118** |

### 3.1 Root-Cause Analysis of Duplicate Records
1. **Full-Row Duplicates:** 0 across all 9 files. Every row has a unique `S.No.` (autoincrement primary key).
2. **Database Ingestion Repeats:**
   - In `Bedroom_Fan_2`, individual packets repeat identically up to **118 times** with consecutive `S.No.` values.
   - All 8 content columns (`datetime`, `temperature`, `humidity`, `mode`, `speed`, `opTime`, `eSpent`, `eSaved`) are 100% identical.
   - Root cause: An IoT message broker / database consumer retry loop that wrote repeated rows without a table unique constraint on `(device_id, timestamp)`.
3. **Legitimate Intra-Minute Telemetry:**
   - There are **109,309 intra-minute records** across the 9 files where timestamps match but operational readings differ (e.g. speed adjustment or consecutive packets with different `opTime`).
   - Deduplicating by `datetime` alone would delete valid operational events.
4. **Proposed Deduplication Policy:**
   - Drop rows sharing identical values across all 8 content columns (`subset=['datetime', 'temperature', 'humidity', 'mode', 'speed', 'opTime', 'eSpent', 'eSaved']`), retaining the first appearance.
   - Preserves 100% of distinct physical packets and eliminates database retry artifacts.

---

## 4. Sensor Error Verification & Missing Value Treatment

### 4.1 Sensor Error Sentinel Counts

| Fan Identifier | Raw Records | Temp == 255 | Humidity == 255 | Both == 255 | Temp == 0 | Humidity == 0 | Valid Temp Range (°C) | Valid Hum Range (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Bedroom_Fan_1` | 79,384 | 8 | 8 | 8 | 0 | 0 | 19.0 to 40.0 °C | 17.0% to 95.0% |
| `Bedroom_Fan_2` | 179,243 | 22 | 22 | 22 | 0 | 0 | 21.0 to 41.0 °C | 10.0% to 90.0% |
| `Bedroom_Fan_3` | 110,409 | 2 | 2 | 2 | 0 | 0 | 20.0 to 41.0 °C | 10.0% to 89.0% |
| `Drawingroom_Fan_1` | 60,444 | 29 | 29 | 29 | 22 | 22 | 20.0 to 39.0 °C | 10.0% to 94.0% |
| `Drawingroom_Fan_2` | 43,529 | 2 | 2 | 2 | 0 | 0 | 22.0 to 37.0 °C | 7.0% to 90.0% |
| `Drawingroom_Fan_3` | 77,764 | 2 | 2 | 2 | 0 | 0 | 19.0 to 41.0 °C | 8.0% to 90.0% |
| `Lounge_Fan_1` | 21,305 | 28 | 28 | 28 | 0 | 0 | 19.0 to 37.0 °C | 15.0% to 76.0% |
| `Lounge_Fan_2` | 55,488 | 11 | 11 | 11 | 0 | 0 | 18.0 to 44.0 °C | 10.0% to 82.0% |
| `Lounge_Fan_3` | 47,766 | 558 | 558 | 558 | 0 | 0 | 18.0 to 42.0 °C | 5.0% to 95.0% |
| **TOTAL** | **675,332** | **662** | **662** | **662** | **22** | **22** | **18.0 to 44.0 °C** | **5.0% to 95.0%** |

### 4.2 Findings on Sensor Integrity
1. **Coupled 255 Sentinel:** In **100% of cases**, `Temperature == 255` occurs if and only if `Humidity == 255`. In microcontroller architectures, `0xFF` (255) is the hardware timeout / I2C bus NACK error code when the sensor is uncommunicative.
2. **Boot Sentinel (0°C, 0% RH):** In `Drawingroom_Fan_1`, 22 records report `0` for both temperature and humidity. These represent uninitialized RAM buffer reads on microcontroller reset.
3. **Plausible Physical Range:** Excluding `255` and `0` yields temperature ranges of **18°C to 44°C** and humidity ranges of **5% to 95%**, matching genuine Indian indoor residential climates.
4. **Filtering Treatment:** Filter records where $T \le 0$, $T \ge 60$, $RH \le 0$, or $RH > 100$.

---

## 5. Physical Energy Validation & Exact Mathematical System

Code-level verification over all 675,332 raw records proved with **100.00% precision** that:
$$\text{eSpent} = P_{\text{BLDC}}(\text{speed}) \times \text{opTime} \quad [\text{Joules}]$$
$$\text{eSaved} = P_{\text{Saved}}(\text{speed}) \times \text{opTime} \quad [\text{Joules}]$$
$$P_{\text{Induction}}(\text{speed}) = P_{\text{BLDC}}(\text{speed}) + P_{\text{Saved}}(\text{speed}) \quad [\text{Watts}]$$

$$\text{Energy}_{\text{Wh}} = \frac{\text{Energy}_{\text{Joules}}}{3,600}, \quad \text{Energy}_{\text{kWh}} = \frac{\text{Energy}_{\text{Joules}}}{3,600,000}$$

### 5.1 Power Constants & Physical Residuals

| Speed Setting | Operational State | Research Fan Power ($P_{\text{BLDC}}$) | Modeled Induction Baseline ($P_{\text{Induction}}$) | Modeled Savings ($P_{\text{Saved}}$) | Formula Exceptions | Max Absolute Residual |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | OFF | **0.0 W** | 0.0 W | 0.0 W | **0** | **0.0 Joules** |
| **1** | Ultra Low | **10.0 W** | 24.0 W | 14.0 W | **0** | **0.0 Joules** |
| **2** | Low | **15.0 W** | 34.0 W | 19.0 W | **0** | **0.0 Joules** |
| **3** | Medium | **20.0 W** | 55.0 W | 35.0 W | **0** | **0.0 Joules** |
| **4** | High | **27.0 W** | 64.0 W | 37.0 W | **0** | **0.0 Joules** |
| **5** | Boost / Max | **48.0 W** | 96.0 W | 48.0 W | **0** | **0.0 Joules** |

- **Zero-Speed Records:** 124,021 records have `speed == 0`. In 100% of cases, `eSpent == 0` and `eSaved == 0`.
- **Offline Backlog Records (`opTime > 900s`):** 38,198 records have `opTime > 900s`. In 100% of cases, the power formula holds exactly without residual.
- **Physical Interpretation:** `eSpent` and `eSaved` are calculated in firmware via the lookup formula; they are **not live electrical utility sub-meter measurements**.

---

## 6. Verification of Model 1 Compatibility

### 6.1 Model 1 Feature & Interface Audit
Inspection of `aaci_ml/aaci_comfort_model.joblib` and `aaci_ml/aaci_model_metadata.json` confirmed Model 1's features:
- Numeric: `['Air temperature (C)', 'Relative humidity (%)', 'Air velocity (m/s)', 'Outdoor monthly air temperature (C)', 'Clo', 'Met']`
- Categorical: `['Season', 'Building type', 'Cooling startegy_building level']`

### 6.2 Key Finding: Fan Speed Is NOT an Input to Model 1
- **Model 1 does not take `speed (0..5)` directly.** It operates on physical **`Air velocity (m/s)`**.
- Claiming Model 1 directly predicts comfort from fan speed without an aerodynamic mapping would be technically false.

### 6.3 Technically Honest Integration Strategy
- The fan controller `fan_controller_v3.py` implements an aerodynamic transfer function:
  $$v_a(\text{speed}) = [0: 0.05, 1: 0.15, 2: 0.30, 3: 0.50, 4: 0.75, 5: 1.10]\text{ m/s}$$
- This scale is physically validated by the Dryad laboratory dataset (`Single_Fan.csv`, showing air speeds of 0.1 to 1.2 m/s).
- **Integration Workflow:**
  1. For each candidate speed $s \in \{0, 1, 2, 3, 4, 5\}$, map $s \rightarrow v_a(s)$.
  2. Query Model 1 with $(T, RH, v_a(s))$ to obtain comfort class probabilities.
  3. Select the speed that minimizes motor power $P_{\text{BLDC}}(s)$ while satisfying the comfort threshold.
  4. Model 1 remains completely untouched.

---

## 7. Model 3 Scientific Formulation & Validation Protocol

### 7.1 Three Decoupled Components
1. **Deterministic Physics Calculator:** $E = P(s) \times \Delta t$ (for exact runtime energy accounting).
2. **ML Energy Demand Forecaster ($\hat{E}_{t+1}$):** Forecasts next-period energy from antecedent context ($T, RH, \text{hour}, \text{day}, \text{lag}_s$) without knowing future speed or duration.
3. **Energy-Optimal Speed Selector:** Finds $\arg\min_{s} P(s) \text{ s.t. Comfort}(v_a(s)) \ge \tau$.

### 7.2 Validation Protocols
- **Experiment A (Chronological Forecasting):** 70% Train, 15% Validation, 15% Forward Test per fan. Scalers fit strictly on training fold.
- **Experiment B (Fan Generalization / LOFO):** Train on 8 fans, evaluate on 9th held-out fan. Distinguish zero-shot (no target fan data) from few-shot adaptation (first 7 days available).
