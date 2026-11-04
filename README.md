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
  * **Atomberg BLDC Power Map:** `[0, 10, 15, 20, 27, 48] W` (Speeds 0–5)
  * **Baseline Induction Power Map:** `[0, 24, 34, 55, 64, 96] W` (Speeds 0–5)
* **Real-Time ROI:** Computes instantaneous watts saved ($P_{\text{Induction}} - P_{\text{BLDC}}$) and project annual electricity bill savings in Indian Rupees ($\text{₹}/\text{yr}$) based on standard tariff benchmarks ($\text{₹}7.50/\text{kWh}$).
* **Advisory Forecaster:** LightGBM next-packet energy model ($R^2 \approx 0.1515$) serving strictly as an advisory telemetry forecast without commanding speed.

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
