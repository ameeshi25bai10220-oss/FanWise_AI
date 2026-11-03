# Comprehensive Data Quality Report: Archive (2) & ROBOD

## 1. Executive Quality Summary

| Metric | Archive (2) (UCI Occupancy) | ROBOD (SupplementaryData) |
| :--- | :--- | :--- |
| **Total Rows** | 20,560 | 52,128 across 5 room CSVs |
| **Total Missing Values (NaN/Null)** | **0 (0.00%)** | **0 in critical columns (minor missing in solar sensor Room 5)** |
| **Total Duplicate Timestamps** | 0 (Strict 1-min progression) | 0 (Strict 5-min progression per room) |
| **Sensor Value Duplicates (Frozen Readings)** | 1,441 rows (7.01%) - primarily deep night | Handled via BMS continuous floating point |
| **Outliers / Range Inconsistencies** | 0 physical violations (all sensors in valid physical ranges) | 1 minor outdoor solar reading spike (filtered) |
| **Class Imbalance** | 23.10% Occupied overall (15,810 '0' vs 4,750 '1') | Room 1: 18.4%, Room 2: 19.1%, Room 3: 31.2%, Room 4: 28.5%, Room 5: 14.2% |
| **Target Ground Truth Reliability** | Digital camera snapshots taken every minute (High) | Ceiling fish-eye camera computer vision counting (High) |

---

## 2. Archive (2) Sensor Drift, Range & Calibration Verification

All sensor features were audited across the 20,560 records:
- **Temperature**: Min: 19.00°C, Max: 23.18°C, Mean: 20.91°C, Std: 1.02°C.
  - Normal distribution with diurnal heating pulses during daytime office hours.
- **Relative Humidity**: Min: 16.74%, Max: 39.12%, Mean: 27.66%, Std: 4.98%.
  - Consistent with European winter heated indoor conditions (dry air).
- **Light**: Min: 0.00 lux, Max: 1697.25 lux, Mean: 130.76 lux, Std: 210.57 lux.
  - Exactly 0.00 lux during 100% of night hours when office lights are turned off.
  - Correlation with binary occupancy: $r = 0.907$.
  - **Risk**: Night-time occupancy with lights off (e.g. night sleeping or movie watching in residential settings) would cause false negatives if relying solely on light.
- **CO2**: Min: 412.75 ppm, Max: 2028.50 ppm, Mean: 684.62 ppm, Std: 314.83 ppm.
  - Baseline outdoor fresh air ambient: ~412 ppm.
  - Peaks at 2028 ppm during occupied meetings with door/window closed.
  - Gradual decay rate after occupant departure due to room air infiltration.
  - Correlation with binary occupancy: $r = 0.712$.
- **HumidityRatio**: Min: 0.002674, Max: 0.006476, Mean: 0.004228 kg water / kg dry air.
  - High collinearity with Relative Humidity ($r = 0.955$) and Temperature ($r = 0.697$).

### Sensor Lag and Autocorrelation Analysis
- Autocorrelation coefficients:
  - Temperature Lag-1: 0.9989
  - Humidity Lag-1: 0.9987
  - Light Lag-1: 0.9934
  - CO2 Lag-1: 0.9986
  - HumidityRatio Lag-1: 0.9988
- **Conclusion**: The temporal dependency is exceptionally high. Shuffling data into random train-test splits causes severe synthetic overperformance (apparent F1 > 0.99) that completely collapses when deployed forward in time.

---

## 3. ROBOD Quality & Physical Plausibility Verification

The 52,128 records across 5 rooms were inspected:
- **Room 1 & Room 2 (Lecture Rooms)**:
  - Characterized by burst occupancy (0 to 35 students during class slots, returning to 0 during breaks).
  - CO2 rises rapidly during lectures (reaching 1400+ ppm).
  - Acoustic sound pressure levels fluctuate between 42 dBA (quiet) and 74 dBA (active discussions).
- **Room 3 & Room 4 (Open-Plan Offices)**:
  - Characterized by gradual arrival (08:30–09:30) and steady headcount (5–12 occupants) with lunch dips.
  - Ceiling fan energy consumption actively logged (`ceiling_fan_energy` correlates strongly with warm periods and occupant counts).
- **Room 5 (Library / Quiet Study Area)**:
  - High occupant presence with very low sound pressure (<48 dBA), demonstrating why acoustics alone cannot be used without thermal/CO2 integration.
  - One minor solar irradiance sensor drop (handled by forward fill).

---

## 4. Key Recommendations for Machine Learning Preprocessing
1. **No Imputation Needed**: Zero missing records in Archive (2).
2. **Feature Standardisation**: StandardScaler or RobustScaler required for logistic regression/linear baselines due to scale discrepancies (Light 0–1700 vs HumidityRatio 0.002–0.006). Tree models (Decision Tree, Random Forest, XGBoost) natively handle scale differences.
3. **Derived Features Required**:
   - Rolling CO2 rate of change ($\Delta \text{CO2}_{5\text{min}} = \text{CO2}_t - \text{CO2}_{t-5}$) to detect entry/exit prior to steady-state accumulation.
   - Rolling Light transition indicators.
   - Cyclical hour encoding: $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$.

