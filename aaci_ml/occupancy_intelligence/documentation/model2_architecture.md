# AACI Model 2 Architecture: Occupancy & Spatial Intelligence

## 1. System Vision & Decoupled Architecture
AACI Model 2 is designed as a modular, decoupled intelligence suite that operates alongside AACI Model 1 (Thermal Comfort Intelligence). Model 1 determines *thermal fan speed preference* (Cold, Neutral, Hot), while Model 2 determines *occupancy state, spatial headcount, and energy conservation gating*.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            AACI MODULAR SYSTEM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   SENSORY TELEMETRY                                                                         │
│   [Temp, RH, Light, CO2, BLDC Tach, Wi-Fi, Acoustics]                                       │
│          │                                                                                  │
│          ▼                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                     MODEL 2: OCCUPANCY & SPATIAL INTELLIGENCE                       │   │
│   │                                                                                     │   │
│   │   ┌────────────────────────┐      ┌────────────────────────┐                        │   │
│   │   │  Module 1: Immediate   │      │  Module 2: Spatial     │                        │   │
│   │   │  Room Occupancy Det.   │      │  Headcount & Activity  │                        │   │
│   │   │  (UCI Archive 2 Data)  │      │  (ROBOD SDE4 Data)     │                        │   │
│   │   │  • Binary Presence     │      │  • Headcount (0 to 38) │                        │   │
│   │   │  • 1-min fast sensing  │      │  • Multi-person load   │                        │   │
│   │   └───────────┬────────────┘      └───────────┬────────────┘                        │   │
│   │               │                               │                                     │   │
│   │               └───────────────┬───────────────┘                                     │   │
│   │                               ▼                                                     │   │
│   │                   ┌────────────────────────┐                                        │   │
│   │                   │  Module 3: Fan/Building│                                        │   │
│   │                   │  Operation & Energy    │                                        │   │
│   │                   │  (ROBOD + RDD Datsets) │                                        │   │
│   │                   │  • Standby Eco Power   │                                        │   │
│   │                   │  • Energy Optimization │                                        │   │
│   │                   └───────────┬────────────┘                                        │   │
│   │                               ▼                                                     │   │
│   │                   ┌────────────────────────┐                                        │   │
│   │                   │  Module 4: Experimental│                                        │   │
│   │                   │  Decision Arbitration  │                                        │   │
│   │                   │  • Hypothesis Policies │                                        │   │
│   │                   │  • Configurable Gating │                                        │   │
│   │                   └───────────┬────────────┘                                        │   │
│   └───────────────────────────────┼─────────────────────────────────────────────────────┘   │
│                                   │                                                         │
│                                   ▼ [Occupancy Status + Headcount Prior]                    │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                     MODEL 1: THERMAL COMFORT INTELLIGENCE (UNTOUCHED)               │   │
│   │                                                                                     │   │
│   │   Occupant Environmental Inputs ──► Generic XGBoost (Cold / Neutral / Hot)          │   │
│   │                                          │                                          │   │
│   │                                          ▼                                          │   │
│   │                               Personalization Layer                                 │   │
│   │                               (Bayesian User Updates)                               │   │
│   │                                          │                                          │   │
│   │                                          ▼                                          │   │
│   │                               V3.1 Fan Decision Engine                              │   │
│   │                               (Target Speed 0 to 5)                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                                 │
│                                           ▼                                                 │
│   FINAL SMART FAN COMMAND                                                                   │
│   • If Unoccupied: Force Fan Speed 0 (or Eco Standby if Departure Delay active)             │
│   • If Occupied: Fan Speed = Target from Model 1 adjusted for Multi-Occupant Headcount      │
│   • Motor Feedback Validation & Tach Discrepancy Trap (V3.1 Reliability)                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module 1: Binary Occupancy Detection Specification

### Candidate Algorithms:
1. **Logistic Regression (L2 Regularized, Calibrated)**: Interpretable baseline; directly demonstrates feature odds ratios.
2. **Decision Tree (CART, Depth-Constrained)**: Interpretable edge rule benchmark capable of direct C-code translation for microcontrollers.
3. **Random Forest (Bagging Ensemble)**: Robust nonlinear ensemble resistant to sensor noise.
4. **XGBoost (Gradient Boosted Decision Trees)**: High-performance gradient-boosted benchmark capturing complex sensor interactions.

### Three Structured Feature Experiments:
- **Experiment A: Full Environmental Feature Set**:
  - Features: `{Temperature, Humidity, Light, CO2, HumidityRatio}`.
  - Purpose: Establishes the theoretical performance upper bound using all available IoT environmental sensors.
- **Experiment B: Reduced Hardware Feature Set**:
  - Features: `{Temperature, Humidity, HumidityRatio, Hour}`.
  - Purpose: Evaluates practical edge feasibility for low-cost smart fans lacking expensive Light or CO2 sensors, relying exclusively on basic ambient sensors (AHT20/SHT40) and internal RTC/NTP time.
- **Experiment C: Psychrometric Ablation Experiment**:
  - Features: `{Temperature, Humidity, Light, CO2}` (without `HumidityRatio`).
  - Purpose: Isolates whether the derived psychrometric `HumidityRatio` ($W = f(T, RH)$) provides genuine incremental discriminatory power over raw temperature and humidity.

---

## 3. Experimental Policy & Arbitration Framework (Hypotheses, Not Hardcoded Rules)

The following control policies are explicitly formulated as **experimental hypotheses** to be evaluated empirically against baseline operational logs:

1. **Hypothesis H1 (Anti-Glitch Confirmation Delay)**: Requiring $N_{\text{confirm}} = 2$ consecutive positive detections (2 minutes) reduces spurious motor activations caused by transient sensor spikes, at the cost of a 120-second entry response latency.
2. **Hypothesis H2 (Vacancy Grace Period)**: Maintaining fan operation for $T_{\text{grace}} \in [5, 10]$ minutes following an occupied-to-unoccupied transition prevents disruptive motor on/off cycling during brief room departures, at a modest energy cost.
3. **Hypothesis H3 (Darkness & Silence Are Not Vacancy)**: Darkness ($0\text{ lux}$) and acoustic silence ($<45\text{ dBA}$) **must never be naively interpreted as proof of vacancy**. In residential environments, sleeping occupants produce zero light and low acoustics while still requiring active fan ventilation.
4. **Hypothesis H4 (Time-Gated Night Schedule Prior)**: Conditioning occupancy prior probabilities on local time (e.g. 22:00–07:00) helps maintain fan operation during nocturnal sleep when optical sensors drop to zero.

---

## 4. Scientific Disclaimer on Domain Transfer
- **Office vs Residential Transfer**: The UCI Occupancy dataset reflects a commercial office environment in Belgium with daytime artificial lighting and closed-room meeting CO2 buildup.
- **Explicit Boundary**: High classification performance on the UCI dataset demonstrates algorithmic validity on environmental telemetry, but **does not prove equivalent performance in residential bedrooms**, where nighttime sleeping in darkness presents fundamentally distinct sensor dynamics. Subsequent modules will evaluate residential fan telemetry (RDD Datset) and multi-room diurnal operations (ROBOD).
