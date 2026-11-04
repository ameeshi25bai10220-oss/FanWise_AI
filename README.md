# FanWise AI – Atomberg Adaptive Comfort Intelligence (AACI)
### AI-Assisted Smart Ceiling Fan Control & Multi-Module Environmental Intelligence

---

## Overview

**FanWise AI** is an intelligent indoor climate and smart ceiling fan control platform developed for the **Atomberg Hackathon 2026**. At its core is **AACI (Atomberg Adaptive Comfort Intelligence)**, an adaptive, uncertainty-aware, and personalized decision engine integrating **three complementary intelligence modules**:

1. **Thermal Comfort Intelligence (Model 1):** Predicts human thermal sensation (Hot, Neutral, Cold) with calibrated Softmax probabilities and individual personalization using the **ASHRAE Global Thermal Comfort Database II**.
2. **Occupancy & Spatial Intelligence (Model 2):** Detects space occupancy (`Occupied` vs. `Unoccupied`), evaluates prediction confidence, estimates continuous headcount, and manages a **configurable occupancy departure timeout** using the **UCI IoT Occupancy Detection** and **NUS SDE4 ROBOD** datasets.
3. **Energy & Fan Telemetry Intelligence (Model 3):** Models physical power consumption across Atomberg BLDC vs. baseline Induction fans and provides advisory next-packet energy forecasting using the **Atomberg Residential Fan Telemetry (RDD)** dataset.

AACI unifies these modules within a deterministic **Four-Stage Decision Pipeline** that actuates Atomberg BLDC ceiling fan motor speeds (0 to 5) while balancing human comfort, occupant presence, and electricity savings.

---

## System Architecture: The Four-Stage Decision Pipeline

```
+---------------------------------------------------------------------------------------------------+
|                            FANWISE AI / AACI INTEGRATED ARCHITECTURE                              |
+---------------------------------------------------------------------------------------------------+
|  [Room Sensors: Ta, RH, Air Velocity] + [Outdoor Weather: Tout] + [Occupancy: Light, CO2, Count]  |
|                                                  |                                                |
|                                                  v                                                |
|  +---------------------------------------------------------------------------------------------+  |
|  |                 STAGE 1: MULTIMODAL INGESTION & COMMON STATE REPRESENTATION (CSR)           |  |
|  |  • Sensor Validation & Staleness Gate (Timestamp delta <= 300s)                             |  |
|  |  • Psychrometric Humidity Ratio W & Vapor Pressure Computation                              |  |
|  |  • Provisional Dryad Speed-to-Airflow Mapping (Speed 0 -> 0.05 m/s ... Speed 5 -> 2.10 m/s) |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                  |                                                |
|                                                  v                                                |
|  +---------------------------------------------------------------------------------------------+  |
|  |                 STAGE 2: TRI-MODULE INTELLIGENCE INFERENCE & GATING                         |  |
|  |                                                                                             |  |
|  |   [MODULE 1: Thermal Comfort]    [MODULE 2: Occupancy & Spatial]    [MODULE 3: Energy]      |  |
|  |   • 9 Biometeorological Feats    • UCI Binary Classifier (Set A/B)  • Deterministic BLDC    |  |
|  |   • Tuned 900-Tree XGBoost       • Occupied / Unoccupied Status       Power Map (0-48W)     |  |
|  |   • Softmax: [P_Cold, P_Neu,     • Prediction Confidence (%)        • Induction Baseline    |  |
|  |              P_Hot]              • ROBOD Headcount Regressor          Map (0-96W)           |  |
|  |   • Individual Personalization   • Configurable Departure Timeout   • LightGBM Advisory     |  |
|  |     Logit Shift (Clo, Met)         Gate (0 to 15 min grace period)    Forecaster (Wh)       |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                  |                                                |
|                                                  v                                                |
|  +---------------------------------------------------------------------------------------------+  |
|  |                 STAGE 3: PRIORITY SAFETY HIERARCHY & DYNAMIC CONFIDENCE GATES               |  |
|  |  • Priority P1: Sensor Staleness Freeze (>300s -> Retain confirmed speed)                  |  |
|  |  • Priority P2: Manual User Override (600s lockout timer -> Commands target speed)         |  |
|  |  • Priority P3: Thermal Safety Floor (Room Temp >= 35°C -> Enforces Speed >= 1)            |  |
|  |  • Priority P4: Occupancy Hard Gate (Vacant & Timeout Expired -> Enforces Speed 0)          |  |
|  |  • Priority Grace: Occupancy Timeout Active (Temporarily vacant -> Delays shutoff)          |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                  |                                                |
|                                                  v                                                |
|  +---------------------------------------------------------------------------------------------+  |
|  |                 STAGE 4: MULTI-OBJECTIVE CONSTRAINED OPTIMIZATION                           |  |
|  |  argmin_{s in {0..5}} [ alpha*Discomfort(s) + beta*Energy(s) + gamma*Delta_s(s)           |  |
|  |                         - delta*OccupancyBonus(s) ]                                         |  |
|  |  • Anti-Hunting Rate Limiting (|Delta_s| <= 2 per cycle)                                    |  |
|  |  • Digital Twin State Update & Structured "Glass Box" Decision Rationale                    |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                  |                                                |
|                                                  v                                                |
|                         Deterministic Atomberg BLDC Motor Speed Command                           |
|                                      (Speed 0 to Speed 5)                                         |
+---------------------------------------------------------------------------------------------------+
```

---

## Detailed Intelligence Modules

### 1. Model 1: Thermal Comfort Intelligence
* **Dataset:** ASHRAE Global Thermal Comfort Database II (13,616 samples across 9 research contributors).
* **Feature Engineering:** 9 physical biometeorological indicators:
  * Vapor pressure ($P_v$ via Magnus-Tetens formula)
  * Dew point temperature ($T_{dp}$)
  * Steadman apparent temperature
  * Humidex
  * Convective cooling effect ($\Delta T_{conv}$)
  * Evaporative cooling capacity
* **Predictor:** 900-tree tuned XGBoost classifier producing continuous Softmax distributions: $P(\text{Cold})$, $P(\text{Neutral})$, $P(\text{Hot})$.
* **Personalization Layer:** Bayesian-inspired logit adjustments tailored to occupant preferences (*Runs Hot / Cooler*, *Neutral*, *Chill Sensitive / Warmer*), clothing insulation (Clo $0.3 – 1.0$), and metabolic rate (Met $0.8 – 2.0$).

### 2. Model 2: Occupancy Intelligence & Configurable Timeout
* **Dataset:** UCI IoT Occupancy Detection (20,560 environmental sensor records) + NUS SDE4 ROBOD (52,128 records across 5 commercial zones).
* **Predictor:** Random Forest & Logistic Regression classifiers evaluating ambient cues (Temperature, Relative Humidity, Psychrometric Humidity Ratio, Light, and $\text{CO}_2$).
* **Outputs:**
  * **Classification:** `Occupied` vs. `Unoccupied`
  * **Confidence:** Probability margin (e.g., $96.4\%$ occupancy confidence)
  * **Headcount:** Continuous density estimation ($N \ge 0$)
* **Configurable Occupancy Timeout (Grace Period):**
  * When an occupant briefly leaves the room (e.g., to fetch water, answer the phone, or step into the hallway), turning off the ceiling fan immediately creates an irritating and jarring user experience.
  * FanWise AI implements a **configurable departure grace period** ($0\text{s}$, $1\text{m}$, $5\text{m}$ default, $10\text{m}$).
  * If the space transitions to unoccupied, the fan enters **Grace Hold State**, retaining its current speed or a quiet idle speed while the countdown runs.
  * Once the vacant duration equals or exceeds the timeout threshold, the **Occupancy Hard Gate** triggers, cutting fan speed to `0` and reducing motor power to `0.0 W`.

#### Illustrative Controller Scenario: Nobody in the Room

| Parameter | State During Brief Departure ($t < \text{Timeout}$) | State After Timeout Expiration ($t \ge \text{Timeout}$) |
|---|---|---|
| **Room Occupancy** | `Unoccupied` (Confidence: 96.0%) | `Unoccupied` (Confidence: 96.0%) |
| **Thermal Comfort** | `Neutral` ($P_{\text{Neutral}} = 0.88$) | `Neutral` ($P_{\text{Neutral}} = 0.88$) |
| **Current Fan Speed** | `3` (11.0 W) | `3` (11.0 W) |
| **Configured Timeout** | `5 minutes` (300 seconds) | `5 minutes` (300 seconds) |
| **Elapsed Vacant Time**| `30 seconds` | `360 seconds` (6 minutes) |
| **Active Controller State** | **P-Occupancy Grace Hold** | **P4 - Occupancy Hard Gate Enforced** |
| **Recommended Speed** | **Speed 3 (Retained)** | **Speed 0 (Shut Off)** |
| **Controller Rationale** | *[Occupancy Grace Timeout Active.] Space became vacant 30s ago. Retaining Speed 3 for remaining 4m 30s of grace period to avoid abrupt shutoff when someone briefly steps out.* | *[Occupancy Hard Gate Active.] Space is unoccupied (360s vacant >= 300s timeout). Fan shut off (Speed 0, 0.0 W) for zero wasted energy.* |

### 3. Model 3: Energy & Fan Telemetry Intelligence
* **Dataset:** Atomberg Residential Telemetry (RDD: 534,857 cleaned operational packets across 9 fans).
* **Deterministic Physics Engine:** Speed-indexed power consumption mapping:
  * **Atomberg BLDC Power Map:** `[0, 5, 10, 16, 22, 28] W` (Speeds 0–5, documented Atomberg Renesa)
  * **Baseline Induction Power Map:** `[0, 30, 40, 50, 65, 75] W` (Speeds 0–5, BEE Baseline)
* **Real-Time ROI:** Computes instantaneous watts saved ($P_{\text{Induction}} - P_{\text{BLDC}}$) and projected annual electricity bill savings in Indian Rupees ($\text{₹}/\text{yr}$) based on standard tariff benchmarks ($\text{₹}7.50/\text{kWh}$).
* **Advisory Forecaster:** LightGBM next-packet energy model ($R^2 \approx 0.1515$) serving strictly as an advisory telemetry forecast without commanding speed.

---

## Energy Optimization: Mathematical Proof & Empirical Comparison

To substantiate energy optimization claims with empirical evidence, FanWise AI implements an **Energy Comparison Engine** (`api/energy_comparison.py`) evaluating three distinct operating strategies over a 24-hour diurnal cycle:

### 1. The Three Compared Operating Strategies
1. **Strategy A (Conventional Fixed-Speed Operation):** Fan runs continuously at a fixed speed (e.g. Speed 4 = 22.0 W), representing traditional Indian households where fans remain running all day.
2. **Strategy B (User Manual Operation):** Models empirical residential behavior derived from the Atomberg Residential Fan Telemetry (RDD dataset), where occupants adjust speeds intermittently (Speed 4–5 during heat peaks, Speed 3 during night/sleep) but frequently forget to turn off the fan when leaving the room.
3. **Strategy C (FanWise AI Adaptive Operation):** Dynamically modulates fan speeds using Model 1 Thermal Comfort Softmax probabilities (throttles to Speeds 1–2 during mild ambient periods) and automatically halts the motor (Speed 0 = 0.0 W) via the Model 2 Occupancy Hard Gate after the departure timeout expires.

### 2. Mathematical Formulation
Energy consumption is calculated as discrete time-integral power:
\[
E = \sum_{t=1}^{n} P(s_t) \cdot \Delta t
\]
Where:
* $P(s_t)$ = Fan power consumption (Watts) at speed $s_t \in \{0, 1, 2, 3, 4, 5\}$.
* $\Delta t$ = Operational timestep duration (1 hour).

Percentage energy savings are calculated against each baseline:
\[
\text{Savings} = \frac{E_{\text{baseline}} - E_{\text{adaptive}}}{E_{\text{baseline}}} \times 100
\]

### 3. Documented Power Curve Specifications
* **Atomberg Renesa 1200mm BLDC Fan (Documented Manufacturer Spec):**
  * Speed 0: `0.0 W` (Standby / Off)
  * Speed 1: `5.0 W` (~130 RPM)
  * Speed 2: `10.0 W` (~180 RPM)
  * Speed 3: `16.0 W` (~240 RPM)
  * Speed 4: `22.0 W` (~300 RPM)
  * Speed 5: `28.0 W` (~360 RPM Boost)
  * *Rated Maximum Power:* `28 W`
* **Baseline Induction Fan (BEE 1200mm Standard):** `[0, 30, 40, 50, 65, 75] W` (Rated Power: 75 W).

### 4. Empirical 24-Hour Benchmark Results

| Scenario | Fixed BLDC (22W) | User Manual BLDC | FanWise Adaptive BLDC | Savings vs. Fixed BLDC | Savings vs. Manual BLDC | Total Savings vs. Induction (75W) | Unoccupied Waste Saved |
|---|---|---|---|---|---|---|---|
| **Workday Routine** (8h absence) | 528.0 Wh | 340.0 Wh | **129.0 Wh** | **75.6%** | **62.1%** | **91.7%** | **198.0 Wh (100% saved over 9h)** |
| **Weekend Routine** (mostly occupied)| 528.0 Wh | 390.0 Wh | **237.0 Wh** | **55.1%** | **39.2%** | **84.8%** | **66.0 Wh (100% saved over 3h)** |
| **Summer Heatwave** (100% occupied) | 528.0 Wh | 594.0 Wh | **492.0 Wh** | **6.8%** | **17.2%** | **73.8%** | **0.0 Wh (occupied continuously)** |

#### Understanding the 25–38% Energy Optimization Range
* **During occupied periods alone**, dynamically throttling motor speeds based on biometeorological comfort votes (e.g. dropping from Speed 4 [22W] to Speed 2 [10W] during cooler morning and nighttime hours) delivers an established **25% to 39.2% electricity reduction** compared to manual/fixed operation.
* **When factoring in unoccupied departure gating** (where the fan automatically shuts off at 0.0 W), overall daily household savings reach **62.1% to 75.6%** compared to conventional BLDC usage, and over **91%** compared to traditional 75W induction fans.

---

## Addition 4: Atomberg BLDC Fan Digital Twin

To validate and demonstrate the control system without requiring physical hardware in hand, FanWise AI includes an electromechanical **Fan Digital Twin** (`api/digital_twin.py`) that models an Atomberg Renesa 1200mm ceiling fan:

* **Tripartite Speed Architecture:** Separates **Target Speed** (controller's desired state), **Commanded Speed** (signal dispatched to motor driver MCU), and **Measured Tachometer Speed** (physical rotor feedback).
* **Ramp-Up / Ramp-Down Physical Dynamics:** Simulates a **15-second physical transition delay** with intermediate RPM and watt interpolation, avoiding instantaneous artificial jumps.
* **Tachometer Response with Aerodynamic Jitter:** Models realistic blade rotational speeds: `[0, 130, 180, 240, 300, 360] RPM` with $\pm 2\text{ RPM}$ aerodynamic flutter.
* **Interactive Anomaly & Fault Injection Suite:**
  * **Motor Stall:** Commanded speed $> 0$, but rotor is locked (Tachometer = 0 RPM, stall current = 4.2 W) $\to$ Triggers emergency fault alert.
  * **Speed Mismatch:** Controller commands Speed 4, but tachometer indicates Speed 1 $\to$ Triggers tachometer mismatch warning.
  * **Sensor Fault:** Ambient sensor disconnected / returning impossible `NaN` $\to$ Triggers Priority P1 fail-safe floor.
  * **Stale Telemetry:** Telemetry stream age $> 300\text{s}$ $\to$ Enforces staleness freeze.
  * **Manual Remote Lockout:** Simulates IR/RF remote command taking precedence for 600 seconds.

> [!NOTE]
> *Software Disclaimer:* This module is an engineering software digital twin simulating BLDC motor electromechanics and tachometer dynamics; it is not physical hardware on a test bench.

---

## Addition 5: Multi-Occupant Comfort Intelligence & Transparent Trade-Offs

Conventional smart fans fail in multi-occupant rooms because they assume a single uniform preference. FanWise AI features a **Multi-Occupant Comfort & Shared Compromise Engine** (`api/multi_occupant.py`):

### 1. Dual-Occupant Divergence Scenario
* **Occupant A (Priya - Runs Hot):** Light clothing (0.5 Clo), active metabolic rate (1.3 Met), setpoint shift $-1.8^\circ\text{C}$ $\to$ Perceived temperature feels $+1.8^\circ\text{C}$ hotter $\to$ Desires **Speed 4/5 (Turbo Cooling)**.
* **Occupant B (Rohan - Chill Sensitive):** Long sleeves (0.8 Clo), sedentary resting (0.9 Met), setpoint shift $+1.8^\circ\text{C}$ $\to$ Perceived temperature feels $-1.8^\circ\text{C}$ cooler $\to$ Desires **Speed 1 (Gentle Breeze)**.

### 2. Conflict Detection & Pareto-Optimal Compromise
* **Conflict Severity:** Evaluates the airflow gap $|s_A - s_B| = |5 - 1| = 4$ steps (Severe Conflict).
* **Compromise Optimization:** Solves a joint utilitarian cost function:
  $$\mathcal{J}(s) = \text{Discomfort}_A(s) + \text{Discomfort}_B(s) + \text{DraftPenalty}_B(s) + \text{Energy}(s)$$
  Recommends **Speed 2 (10.0 W)** as the optimal shared setting.

### 3. Engineering Honesty: Transparent Physical Trade-Off Communication
A single ceiling fan creates a continuous, room-level airflow field and **cannot independently satisfy opposing airflow preferences in different parts of the room**. Rather than pretending AI can bend fluid physics, FanWise AI explicitly communicates this trade-off:
* **Trade-off on Occupant A:** Airflow is reduced to Speed 2 (~0.24 m/s) instead of Speed 4/5. Provides fundamental convective relief while protecting Occupant B from uncomfortable drafts.
* **Trade-off on Occupant B:** Draft velocity is strictly capped safely below draft-nuisance thresholds.
* **Spatial Seating Optimization:** Advises positioning Occupant A directly under the blade center-line (high velocity zone $\approx 0.8\text{ m/s}$) and Occupant B toward the perimeter ($<0.2\text{ m/s}$).

---

## Project Structure

```
FanWise_AI/
├── aaci_ml/                                # Core Machine Learning & Decision Engine
│   ├── 01_inspect_dataset.py               # Dataset inspection & distribution analysis
│   ├── 02_train_model.py                   # Baseline model training pipeline
│   ├── aaci_comfort_model.joblib           # V1 trained XGBoost baseline
│   ├── predict.py                          # V1 prediction interface
│   ├── audit/                              # Technical audit & data leakage checks
│   ├── v2_improved/                        # V2 Biometeorological Feature Engineering
│   ├── v2_verification/                    # Independent V2 verification & CatBoost benchmark
│   ├── v3_personalization/                 # V3 Production Personalization & Reliability
│   ├── occupancy_intelligence/             # Model 2: Occupancy & Spatial Intelligence
│   │   ├── src/                            # UCI IoT Occupancy Detection (Set A, B, C)
│   │   ├── module2/                        # ROBOD Spatial Headcount Regressor
│   │   └── module3/                        # Fan Telemetry & Energy Engine
│   └── system_architecture/                # Full architecture design & integration specs
├── api/                                    # Vercel Serverless Deployment Engine
│   ├── index.py                            # Flask serverless routing & API endpoints
│   ├── lightweight_pipeline.py             # Pure-NumPy tree inference & decision engine (~55MB)
│   └── models_bundle.json                  # Compact exported tree weights for all 3 models (8.6MB)
├── templates/
│   └── index.html                          # Interactive Lovelace/ESPHome Room Simulator Cockpit
├── fanwise_ai/                             # Application backend configuration
├── requirements.txt                        # Production dependencies (flask, numpy)
├── vercel.json                             # Modern Vercel deployment specification
└── README.md                               # Project documentation & architectural guide
```

---

## Interactive Room Simulator Cockpit

The web interface (`templates/index.html`) deployed to Vercel provides a live simulation cockpit where judges can manipulate room conditions and observe FanWise AI respond in real time:

* **Live Environmental Sliders:** Room Temperature ($16^\circ\text{C} – 42^\circ\text{C}$), Relative Humidity ($20\% – 95\%$), Air Velocity ($0.05 – 2.50\text{ m/s}$ with Auto Dryad mapping), Outdoor Temperature ($15^\circ\text{C} – 45^\circ\text{C}$).
* **Animated Ceiling Fan Visualizer:** Blade rotation speed and CSS motion blur dynamically tied to recommended fan speed (Speed 0 stops; Speed 5 spins rapidly).
* **Occupancy & Departure Controls:** Headcount buttons (0, 1, 2, 4, 6+), configurable timeout pills (0m, 1m, 5m, 10m), and elapsed vacant duration slider.
* **Personalization Controls:** Persona selectors (*Runs Hot*, *Neutral*, *Chill Sensitive*), Clothing level (Clo), and Metabolic activity (Met).
* **Glass Box Decision Stepper:** Real-time 4-stage stepper exposing exact internal rationale:
  1. *Sensor Telemetry Ingestion*
  2. *Occupancy Intelligence & Gating* (Pass / Grace Hold / Hard Gate)
  3. *Safety Guard & Priority Hierarchy* (P1 Staleness / P2 Manual / P3 Thermal Floor / P4 Gate)
  4. *Multi-Objective Cost Minimization* (Global argmin across candidates)
* **One-Click Demo Presets:**
  * *Warm Afternoon:* $32^\circ\text{C}$, 2 occupants $\to$ Speed 4
  * *Vacant: Grace Hold:* 0 occupants, vacant 1m < 5m timeout $\to$ Holds Speed 3
  * *Vacant: Gate Enforced:* 0 occupants, vacant 6m > 5m timeout $\to$ Shuts down to Speed 0 (0W)
  * *P3 Thermal Floor:* $37^\circ\text{C}$ extreme heat $\to$ Enforces minimum speed $\ge 1$
  * *Night Sleep Comfort:* $26.5^\circ\text{C}$, 1 occupant resting $\to$ Speed 2

---

## Quickstart & Local Execution

### 1. Run the Interactive Simulator Locally
```powershell
# Clone the repository
git clone https://github.com/bhargav25bai10209-blip/FanWise_AI.git
cd FanWise_AI

# Install lightweight serverless dependencies
pip install flask numpy

# Launch the interactive simulator
python api/index.py
```
*Open your browser at `http://localhost:5000` to interact with the cockpit.*

### 2. Run the Automated Test Suites
```powershell
# Controller V3.1 Reliability Test Suite (21 tests)
python aaci_ml/v3_personalization/test_controller_v3_1.py

# Profile Persistence & Double-Counting Suite (7 tests)
python aaci_ml/v3_personalization/test_personalization_persistence.py

# Occupancy & Spatial Intelligence Unit Suites (39 tests)
python -m pytest aaci_ml/occupancy_intelligence/ -v
```

---

### Author
**Bhargav** (`bhargavraj.s1229@gmail.com`)  
*Atomberg Technologies Hackathon 2026*
