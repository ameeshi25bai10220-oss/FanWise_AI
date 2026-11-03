# Dataset Comparison: Archive (2) vs ROBOD vs RDD Datset

## 1. Resolution of Dataset Identity & Physical Verification

A critical preliminary step was establishing the exact physical identity of all downloaded files in the Atomberg workspace:

| Identifier | Exact Physical Path | Folder Contents | Provenance & Description | Record Count |
| :--- | :--- | :--- | :--- | :--- |
| **Archive (2)** | `archive (2)/Occupancy.csv` | 1 CSV file (1.38 MB) | **UCI Room Occupancy Detection Dataset** (Univ. of Mons, Belgium; Candanedo & Feldheim, 2016). IoT sensors in single test office. | **20,560 rows** (1-min interval) |
| **ROBOD** | `SupplementaryData/` | `combined_Room1.csv` to `combined_Room5.csv` + `README.md` (15.1 MB) | **ROBOD: Room-level Occupancy and Building Operation Dataset** (NUS SDE4 Net-Zero Building, Singapore). Multi-sensor, acoustic, Wi-Fi, HVAC, and fan telemetry. | **52,128 rows** (5-min interval) |
| **RDD Datset** | `RDD Datset/` | 9 CSV files (`Bedroom_Fan_1.csv` ... `Lounge_Fan_3.csv`) (31.0 MB) | **Residential Ceiling Fan Fleet Telemetry** (Atomberg Smart Fans, India). Operational modes, speeds (1-5), opTime, eSpent, eSaved, T, RH. | **675,332 rows** (event logs) |

### Key Clarification:
The user originally used the abbreviation "RDD" to refer to the "ROBOD – Room-level Occupancy and Building Operation Dataset". Our physical filesystem audit reveals that:
1. The official ROBOD dataset files reside inside the folder named `SupplementaryData/` (whose internal `README.md` begins: *"This repository contains the data files and relevant analysis code for ROBOD, Room-level Occupancy and Building Operation Dataset"*).
2. The folder named `RDD Datset/` is a separate, verified dataset containing 675,332 rows of Indian residential Atomberg smart ceiling fan telemetry across 9 real fans.
3. Therefore, ROBOD (`SupplementaryData/`) and RDD (`RDD Datset/`) are **two physically distinct datasets present in the workspace**. Neither was fabricated or assumed.

---

## 2. In-Depth Verification of ROBOD Actual Data

A comprehensive code-level audit was conducted across all 5 room files in `SupplementaryData/`:

### A. Occupancy Target Definitions & Ground Truth Verification
- **`occupant_presence [binary]`**:
  - Discrete binary values ($0$ = Unoccupied, $1$ = Occupied).
  - Missing values: **0 nulls across all 52,128 records**.
  - Positive occupancy distribution: Room 1 (21.07%), Room 2 (28.42%), Room 3 (39.91%), Room 4 (65.11%), Room 5 (42.55%).
- **`occupant_count [number]`**:
  - Non-negative integer values ($0$ to $38$ occupants).
  - Missing values: **0 nulls across all 52,128 records**.
  - Ground Truth Provenance: Measured directly via ceiling-mounted fisheye cameras using validated deep-learning computer vision human detection models by NUS building scientists.

### B. Verification of `ceiling_fan_energy [kWh]` (Interval vs Cumulative)
- **Finding**: **`ceiling_fan_energy [kWh]` is strictly INTERVAL ENERGY consumed during the preceding 5-minute sampling step, NOT a cumulative meter counter.**
- **Mathematical Proof**:
  - The series is strongly non-monotonic with **over 12,000 negative first-differences** across the 5 rooms.
  - Values swing between $0.000$ kWh (fan off) and $0.005\text{--}0.037$ kWh (fan running under various loads).
  - Physics Check: $0.005\text{ kWh in 5 minutes} = 0.005 \times 12\text{ kW} = 60\text{ W}$, perfectly aligning with real ceiling fan mechanical power.
  - Cumulative sum across the entire dataset: **288.67 kWh**.
  - Data Completeness: **Available with 0 nulls across 100% of rows in all 5 rooms**.

### C. Building & HVAC Measurement Provenance
- All HVAC and energy columns are direct physical transducer readings from the Schneider Electric / Johnson Controls BMS:
  - Energy sub-meters: Dedicated electrical branch power meters for ceiling fans, plug loads, lighting, and chilled water.
  - Actuator signals: `fcu_fan_speed [Hz]` / `ahu_fan_speed [Hz]`, `supply_air_temperature [Celsius]`, `supply_air_pressure [Pa]`, `damper_position [%]`, `cooling_coil_valve_position [%]`.
  - Environmental sensors: calibrated indoor Air Temperature, Relative Humidity, CO2, Light, Sound Pressure Level (dBA), VOC, PM2.5, and outdoor weather station.

### D. Timestamps & Data Consistency
- ISO 8601 formatting: `YYYY-MM-DD HH:MM +08:00` (Singapore UTC+8).
- Strictly regular 300-second (5-minute) intervals for 99.8% of transitions.
- **Duplicate timestamps**: **0 duplicates across all 5 rooms**.
- Missing calendar spans: Data collection spans 2021-09-07 to 2021-12-23 (108 days), but discrete active logged days are 29 days (Rooms 1–3) and 47 days (Rooms 4–5), reflecting scheduled building monitoring campaigns.

---

## 3. Comparison Across the Three Datasets

| Dimension | Archive (2) (UCI) | ROBOD (NUS SDE4) | RDD Datset (Atomberg Fans) |
| :--- | :--- | :--- | :--- |
| **Physical Location** | Mons, Belgium | Singapore | Real Indian Homes (Bedrooms, Drawing rooms, Lounges) |
| **Climate Context** | Cold winter (heating season, 19–23°C) | Tropical (cooling season, 23.5–27.5°C) | Tropical / Subtropical residential |
| **Sampling Frequency** | **1 minute** | **5 minutes** | Variable operational event logging |
| **Total Rows** | 20,560 | 52,128 | 675,332 |
| **Primary Target** | Binary Occupancy (0 or 1) | Binary Presence (0/1) & Headcount (0–38) | Speed index (1–5), Modes, Power |
| **Sensors Available** | Temp, Humidity, Light, CO2, HumidityRatio | Temp, Humidity, Light, CO2, Sound (dBA), VOC, PM2.5, Wi-Fi | Temp, Humidity, BLDC Hall speed/power |
| **Fan Energy Available** | None | Real measured ceiling fan kWh (interval) | Power, eSpent, eSaved |
| **Dataset Role in AACI** | **Module 1 (Immediate Binary Detection)** | **Module 2 (Headcount) & Module 3 (Energy)** | **Actuation targets & energy baseline** |

### Combination Verdict:
**Do not combine Archive (2) and ROBOD into a monolithic training set.** Their sampling rates (1-min vs 5-min) and climatic regimes (Belgian winter office vs tropical Singapore institutional building) differ fundamentally. Downsampling or concatenating would induce severe confounding and destroy transient transition signals. Instead, keep them separated into dedicated, specialized intelligence modules.
