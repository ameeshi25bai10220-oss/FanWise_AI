# FanWise AI – Atomberg Adaptive Comfort Intelligence (AACI)
### AI-Assisted Smart Ceiling Fan Control & Thermal Comfort Intelligence

---

## Overview

**FanWise AI** is an intelligent indoor climate and smart ceiling fan control platform developed for the **Atomberg Hackathon 2026**. At its core is **AACI (Atomberg Adaptive Comfort Intelligence)**, an adaptive, uncertainty-aware, and personalized thermal comfort decision engine trained and validated on the **ASHRAE Global Thermal Comfort Database II**.

AACI bridges the gap between passive human thermal physiology, physical fluid/biometeorological dynamics, and active **Atomberg BLDC ceiling fan motor actuation**.

```
+---------------------------------------------------------------------------------------------------+
|                                 FANWISE AI / AACI ARCHITECTURE                                    |
+---------------------------------------------------------------------------------------------------+
|  [Room Sensors: Ta, RH, Air Velocity] + [Outdoor Weather: Tout, Season] + [BLDC Motor Telemetry]   |
|                                                  |                                                |
|                                                  v                                                |
|                     +----------------------------------------------------------+                  |
|                     |     Stage 1: Physics-Based Biometeorological Model       |                  |
|                     |     (9 Physical Features + Tuned XGBoost Engine)         |                  |
|                     +----------------------------------------------------------+                  |
|                                                  | [P(Cold), P(Neutral), P(Hot)]                  |
|                                                  v                                                |
|                     +----------------------------------------------------------+                  |
|                     |     Stage 2: Individual Personalization Layer            |                  |
|                     |     (Adaptive Logit Offsets + SQLite Persistence)        |                  |
|                     +----------------------------------------------------------+                  |
|                                                  | [P*(Cold), P*(Neutral), P*(Hot)]               |
|                                                  v                                                |
|                     +----------------------------------------------------------+                  |
|                     |     Stage 3: Class-Specific Dynamic Confidence Gate      |                  |
|                     |     (tau_Hot=0.52, tau_Cold=0.45, Margin >= 0.12)        |                  |
|                     +----------------------------------------------------------+                  |
|                                                  | Confirmed Class / Uncertainty Flag             |
|                                                  v                                                |
|                     +----------------------------------------------------------+                  |
|                     |     Stage 4: AACIFanControllerV3.1 Decision Engine       |                  |
|                     |   - Probability Vector Validation (|sum(p)-1| <= 0.01)   |                  |
|                     |   - Tripartite Speed Tracking (Target, Cmd, Measured)    |                  |
|                     |   - Motor Transition (15s), Stall & Mismatch Detection   |                  |
|                     |   - Sensor Sanity Gate (Missing/Impossible/NaN Temp)     |                  |
|                     |   - Stale Telemetry Timeout (>180s)                      |                  |
|                     |   - Manual Override Priority with Reason Tracking        |                  |
|                     |   - Configurable Thermal Policy Guards (<20C, >=31C)     |                  |
|                     |   - Anti-Hunting Cooldown Hysteresis (300 seconds)       |                  |
|                     +----------------------------------------------------------+                  |
|                                                  |                                                |
|                                                  v                                                |
|                          Deterministic Atomberg BLDC Motor Speed Command                          |
|                                     (Speed 0 to Speed 5)                                          |
+---------------------------------------------------------------------------------------------------+
```

---

## Project Structure

```
FanWise_AI/
├── aaci_ml/                         # Core Machine Learning & Decision Engine
│   ├── 01_inspect_dataset.py        # Dataset inspection & distribution analysis
│   ├── 02_train_model.py            # Baseline model training pipeline
│   ├── aaci_comfort_model.joblib    # V1 trained XGBoost baseline
│   ├── predict.py                   # V1 prediction interface
│   ├── audit/                       # Comprehensive technical audit & leakage checks
│   │   ├── 03_audit_pipeline.py
│   │   └── detailed_classification_report.json
│   ├── v2_improved/                 # V2 Biometeorological Feature Engineering
│   │   ├── 04_comprehensive_experiments.py
│   │   ├── aaci_v2_model.joblib     # V2 Tuned XGBoost model
│   │   ├── predict_v2.py            # V2 prediction & feature computation
│   │   └── fan_controller.py        # V2 fan controller
│   ├── v2_verification/             # Independent V2 verification & CatBoost benchmark
│   │   └── 05_independent_verification.py
│   └── v3_personalization/          # V3 & V3.1 Production Personalization & Reliability
│       ├── personalization_engine.py      # Multi-occupant logit personalization
│       ├── personalization_database.py    # SQLite profile persistence engine
│       ├── fan_controller_v3.py           # Robust uncertainty-aware fan controller
│       ├── streamlit_aaci_component.py    # Interactive web dashboard
│       ├── demo_two_occupants_improved.py # Synthetic two-occupant demonstration
│       ├── test_controller_v3.py          # 16-scenario validation suite
│       ├── test_controller_v3_1.py        # 21-test probability & motor feedback suite
│       ├── test_personalization_persistence.py # 7-test persistence & double-counting suite
│       ├── test_controller_v3_expanded.py # 14-test boundary & stress suite
│       ├── v3_independent_audit_report.md # Comprehensive independent technical audit
│       └── v3_1_change_log.md             # V3.1 reliability release notes
├── fanwise_ai/                      # Application backend & configuration
│   ├── config.py                    # Central system & fan specifications
│   ├── requirements.txt             # Python dependencies
│   ├── models/                      # Multi-model modules
│   └── saved_models/                # Model artifacts
├── archive (2)/                     # Occupancy dataset
├── archive (3)/                     # ASHRAE Global Thermal Comfort Database II
├── doi_10_6078_D1V67R__v20180717/   # Fan velocity distribution datasets
├── RDD Datset/                      # Room-level airflow distributions
└── SupplementaryData/               # Supplementary thermal comfort datasets
```

---

## Quickstart

### 1. Installation
```powershell
# Clone the repository
git clone https://github.com/bhargav25bai10209-blip/FanWise_AI.git
cd FanWise_AI

# Install required dependencies
python -m pip install -r fanwise_ai/requirements.txt
python -m pip install xgboost catboost scikit-learn streamlit
```

### 2. Launch the Interactive Dashboard
```powershell
python -m streamlit run aaci_ml/v3_personalization/streamlit_aaci_component.py
```

### 3. Run the Automated Test Suites
```powershell
# Controller V3.1 Reliability Test Suite (21 tests)
python aaci_ml/v3_personalization/test_controller_v3_1.py

# Profile Persistence & Double-Counting Suite (7 tests)
python aaci_ml/v3_personalization/test_personalization_persistence.py

# 16-Scenario Operational Test Suite (16 tests)
python aaci_ml/v3_personalization/test_controller_v3.py

# Two-Occupant Demonstration
python aaci_ml/v3_personalization/demo_two_occupants_improved.py
```

---

## Key Achievements & Verification

- **Empirical Baseline**: Tuned XGBoost evaluated on 13,616 samples across 9 unseen research contributors achieves Macro F1 = 0.4517 and Balanced Accuracy = 45.48%.
- **Physics-Grounded Features**: Incorporates 9 biometeorological variables (Magnus-Tetens vapor pressure, dew point, Steadman apparent temperature, Humidex, convective cooling effect, and evaporative cooling capacity).
- **Adaptive Personalization**: Bayesian-inspired logit updates adapt to individual occupant votes (`FEELING_HOT`, `FEELING_COLD`, `FEELING_COMFORTABLE`) without retraining the base ML model.
- **SQLite Persistence**: Profiles, logit offsets, setpoint shifts, and acoustic preferences (`WHITE_NOISE`) persist across process and application restarts.
- **Industrial Safety & Fault Handling**: 44/44 automated tests pass (100% pass rate) covering probability validation, motor transition timing, stall alerts, missing sensors, and extreme temperatures.
- **Energy Optimization**: Atomberg BLDC motor hardware delivers up to 83% power reduction compared to induction fans; AACI adaptive software achieves an additional 25–38% energy optimization by throttling speeds during thermally neutral periods.

---

### Author
**Bhargav** (`bhargavraj.s1229@gmail.com`)  
*Atomberg Technologies Hackathon 2026*
