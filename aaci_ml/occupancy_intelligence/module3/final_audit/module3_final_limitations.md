# AACI Module 3: Operational Limitations & Ethical Guardrails

## 1. Scope Correction & Official Designation

> [!CAUTION]
> **Scope Qualification:**
> Module 3 must **never** be described as "Whole-Building Energy Intelligence". The dataset contains telemetry from 9 ceiling fans in residential homes. It contains zero smart-meter utility mains data, zero HVAC compressor telemetry, zero lighting circuits, and zero whole-house load measurements.
> 
> The official, scientifically defensible designation of this module is:
> **AACI Model 2 – Module 3: Fan Telemetry, Usage Pattern & Energy Intelligence.**

---

## 2. Fundamental Physical & Operational Boundaries

### 2.1 Fan State Is NOT Human Occupancy Ground Truth
Under no circumstances should smart fan operation (`speed > 0` or `opTime > 0`) be treated as ground truth for human presence:
1. **False Positives (Fan ON, Room Empty):**
   - Ceiling fans in Indian residences are frequently left running in unoccupied rooms (e.g., forgotten fans during morning departures, fans left on for pets or general room ventilation).
   - In `Bedroom_Fan_1`, the fan ran for 98.2% of the entire monitored 293 days, including continuous 80-hour stretches where the room was undoubtedly vacant for substantial periods.
2. **False Negatives (Fan OFF, Room Occupied):**
   - Occupants frequently occupy rooms without running the ceiling fan (e.g., during winter months when ambient temperature is below 20°C, during air conditioner operation, or during quiet reading).
   - Therefore, fan operating state represents **appliance operational state**, not verified human presence.

### 2.2 Modeled Baseline Savings vs. Utility Bill Savings
1. **Absence of Hardware Sub-Metering:**
   - The dataset contains zero direct power transducer measurements (no CT clamps, no voltage shunts, no revenue-grade power meters).
2. **Theoretical Induction Baseline:**
   - Energy savings `eSaved` are calculated by comparing firmware-indexed BLDC wattages (10W to 48W) against nominal induction motor power ratings (24W to 96W).
   - While based on established BEE (Bureau of Energy Efficiency) 5-star vs non-star ceiling fan benchmarks, these savings represent **synthetic modeled baselines**, not experimentally measured utility reductions.

---

## 3. Telemetry Pipeline & Machine Learning Limits

### 3.1 Network Dropout Fragmentation Artifact
- Smart fans communicate via home Wi-Fi. Packet dropouts exceeding 15 minutes occur routinely (~5% to 10% of transmissions).
- Segmenting sessions by a 15-minute inactivity gap fragments continuous multi-hour overnight operating blocks into multiple short records, artificially depressing the median session duration (to 8–13 minutes).
- Descriptive statistics must distinguish between **raw packet transmission continuity** and **true human operational intention**.

### 3.2 Machine Learning Inefficacy on Next-Interval Energy Prediction
- Rigorous chronological evaluation proved that advanced ML regressors (Random Forest, XGBoost, Ridge, Decision Tree) **do not add value over a simple Historical Mean baseline** for interval energy prediction on raw telemetry:
  - Historical Mean MAE: **5.41 Wh**
  - XGBoost MAE: **5.75 Wh**
  - Decision Tree MAE: **5.76 Wh**
  - Ridge Regression MAE: **5.79 Wh**
  - Random Forest MAE: **6.00 Wh**
- Because fan energy is governed by discrete human user remote-control decisions ($0, 1, 2, 3, 4, 5$) and firmware timers, complex regression models overfit to spurious lag correlations and perform worse than empirical baselines.
- The authoritative method for interval energy accounting is the **deterministic firmware power lookup** ($E = P(\text{speed}) \times \Delta t$), not black-box regression.

---

## 4. Preservation & Stopping Gate
Module 3 artifacts must remain isolated inside `aaci_ml/occupancy_intelligence/module3/`. Cross-module fusion with Model 1 (Thermal Comfort) or Module 2 (Headcount) is strictly prohibited until explicit research freeze and authorization are granted.

