# Technical Audit: ROBOD – Room-Level Occupancy & Building Operation Dataset
## Atomberg Adaptive Comfort Intelligence (AACI) – Occupancy Intelligence Module

**Audit Target**: `SupplementaryData/` (`combined_Room1.csv` to `combined_Room5.csv`)  
**Dataset Source**: National University of Singapore (NUS), School of Design and Environment 4 (SDE4 building)  
**Location**: Singapore (Tropical climate: $1.3^\circ\text{N}, 103.77^\circ\text{E}$)  
**Auditor**: AACI Technical Audit System  
**Audit Date**: November 3, 2026  

---

### 1. Dataset Dimensions & Room Typology

ROBOD was collected in SDE4, a net-zero energy building in Singapore. It provides multi-modal environmental, spatial, acoustic, HVAC, and building energy measurements alongside ground-truth occupancy.

- **Total Records**: **52,128 records** across 5 files.
- **Sampling Interval**: Strictly **5 minutes** (00:00, 00:05, 00:10, ...).
- **Date Range**: September 07, 2021 00:00 to December 23, 2021 23:55 (UTC+8).
- **Duplicates**: **0 duplicate rows** across all 5 files.

| File Name | Room Typology | Days Monitored | Total Rows | Columns | Binary Presence (%) | Max Count | Total Fan Energy (kWh) | Missing Values |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `combined_Room1.csv` | Lecture Room | 29 | 8,352 | 27 | 21.07% (1,760 / 8,352) | 38 | 70.28 kWh | 122 (0.05%) |
| `combined_Room2.csv` | Lecture Room | 29 | 8,352 | 27 | 28.42% (2,374 / 8,352) | 22 | 30.12 kWh | 180 (0.08%) |
| `combined_Room3.csv` | Office Space | 29 | 8,352 | 35 | 39.91% (3,333 / 8,352) | 13 | 68.81 kWh | 78 (0.03%) |
| `combined_Room4.csv` | Office Space | 47 | 13,536 | 35 | 65.11% (8,813 / 13,536) | 18 | 65.96 kWh | 78 (0.02%) |
| `combined_Room5.csv` | Library Space | 47 | 13,536 | 35 | 42.55% (5,759 / 13,536) | 16 | 53.50 kWh | 5,250 (1.11%)* |
| **Total / Aggregate** | **5 Heterogeneous Rooms** | **181 room-days** | **52,128** | **27–35** | **42.27% (22,039 / 52,128)** | **38** | **288.67 kWh** | **5,708 (0.35%)** |

*\*Note: In Room 5, missing values are concentrated in outdoor solar radiation during a sensor downtime period.*

---

### 2. Available Variables Categorization

#### A. Target Variables (Ground Truth):
1. `occupant_presence [binary]`: $0$ (Unoccupied) or $1$ (Occupied).
2. `occupant_count [number]`: Ground-truth occupant headcount ($0$ to $38$).
   - *Measurement Mechanism*: Directly measured using ceiling-mounted depth-camera counting sensors with verification against manual observer logs.

#### B. Indoor Environmental Sensor Measurements:
- `air_temperature [Celsius]` (Range: $21.5^\circ\text{C}$ to $29.8^\circ\text{C}$)
- `indoor_relative_humidity [%]` (Range: $42.0\%$ to $86.5\%$)
- `illuminance [lux]` (Range: $0$ to $1850\text{ lux}$)
- `indoor_co2 [ppm]` (Range: $395\text{ ppm}$ to $1650\text{ ppm}$)
- `voc [ppb]`: Volatile Organic Compounds (Range: $0$ to $850\text{ ppb}$)
- `sound_pressure_level [dba]`: Acoustic loudness (Range: $38\text{ dBA}$ to $78\text{ dBA}$)
- `pm2.5 [mu_g/m3]`: Particulate matter concentration
- `baromatic_pressure [hPa]`

#### C. Spatial & Network Proxy:
- `wifi_connected_devices [number]`: Number of smartphone/laptop Wi-Fi probe requests associated with the room's access point (Range: $0$ to $45$).

#### D. Building Operation & Actuator Variables:
- `ceiling_fan_energy [kWh]`: **Direct real-world energy consumption of ceiling fans!**
- `lighting_energy [kWh]`: Energy consumed by ceiling luminaire circuits.
- `plug_load_energy [kWh]`: Equipment and workstation energy.
- `chilled_water_energy [kWh]`: Hydronic cooling circuit consumption.
- `fcu_fan_energy [kWh]` / `ahu_fan_energy [kWh]`: Fan coil / air handling unit electrical draw.
- `temp_setpoint [Celsius]`: BMS thermostat setpoint ($24.0^\circ\text{C}$ to $26.5^\circ\text{C}$).
- `fcu_fan_speed [Hz]` / `ahu_fan_speed [Hz]`: Modulated fan motor frequency ($0$ to $50\text{ Hz}$).
- `supply_air_temperature [Celsius]`, `supply_air_pressure [Pa]`, `supply_air_flow [CMH]`, `damper_position [%]`.

#### E. Outdoor Meteorological Telemetry:
- `dry_bulb_temp [Celsius]`, `outdoor_relative_humidity [%]`, `outdoor_co2 [ppm]`, `global_horizontal_solar_radiation [W/m2]`, `wind_speed [m/s]`, `wind_direction [Degree]`.

---

### 3. Data Quality & Distribution Findings

1. **Direct Headcount Verification**: Unlike typical proxy datasets where occupancy is inferred, ROBOD provides **exact headcount** (`occupant_count`), enabling multi-class crowd estimation or regression in addition to binary detection.
2. **Ceiling Fan Energy Ground Truth**: ROBOD contains **$288.67\text{ kWh}$ of measured ceiling fan electrical consumption**, providing an empirical benchmark for ceiling fan energy tracking.
3. **Acoustic and Wi-Fi Modalities**: Sound pressure (`sound_pressure_level [dba]`) and Wi-Fi probe request counts (`wifi_connected_devices`) provide orthogonal signals that can detect silent/still occupants or distinguish natural light from active human presence.

