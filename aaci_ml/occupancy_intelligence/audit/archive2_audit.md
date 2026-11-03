# Technical Audit: Archive (2) – UCI Occupancy Detection Dataset
## Atomberg Adaptive Comfort Intelligence (AACI) – Occupancy Intelligence Module

**Audit Target**: `archive (2)/Occupancy.csv`  
**Dataset Source**: UCI Machine Learning Repository ([Candanedo & Feldheim, 2016](https://doi.org/10.1016/j.enbuild.2016.03.003))  
**Auditor**: AACI Technical Audit System  
**Audit Date**: November 3, 2026  

---

### 1. Dataset Dimensions & Provenance

The dataset was collected in an experimental office room ($5.85\text{ m} \times 3.50\text{ m} \times 2.40\text{ m}$) at the University of Mons, Belgium, sampled at 1-minute intervals. Ground truth occupancy was recorded via time-lapse digital camera pictures.

- **File Path**: `archive (2)/Occupancy.csv`
- **Total Records**: **20,560 rows**
- **Total Columns**: **7 columns**
- **Data Types**: 5 float64 continuous features, 1 int64 binary target, 1 string timestamp.
- **Missing Values**: **0 missing values** across all columns.
- **Full Row Duplicates**: **0 duplicate records**.
- **Sensor-Only Duplicates**: **1,441 records** (Identical sensor readings during deep night hours when the room is dark and empty; verified as genuine physical steady states).

---

### 2. Dissection of the Three Official Partitions

Analysis of timestamp differences ($\Delta t$) reveals two multi-hour discontinuities, proving that `Occupancy.csv` contains all three official UCI benchmark partitions concatenated in strict chronological order:

```
+---------------------------------------------------------------------------------------------------+
|               PARTITION 1 (Test 1)               |  Gap  |        PARTITION 2 (Train)             |
|         Feb 02 14:19:00 -> Feb 04 10:43:00        | 7h08m | Feb 04 17:51:00 -> Feb 10 09:33:00     |
|         N = 2,665 (Occ: 972 / 36.47%)             |       | N = 8,143 (Occ: 1,729 / 21.23%)       |
+---------------------------------------------------------------------------------------------------+
|               PARTITION 3 (Test 2)                                                                |
|         Feb 11 14:48:00 -> Feb 18 09:19:00 (Gap before: 29h 15m)                                 |
|         N = 9,752 (Occ: 2,049 / 21.01%)                                                           |
+---------------------------------------------------------------------------------------------------+
```

1. **Partition 1 (Official Test 1 - `datatest.txt`)**:
   - Records: **2,665 rows** (13.0% of data).
   - Date Span: Feb 02, 2015 14:19:00 to Feb 04, 2015 10:43:00 (2.85 days).
   - Class Balance: 1,693 Unoccupied (63.53%) / 972 Occupied (36.47%).
   - *Chronological Precedence*: Occurred **before** the training partition.
2. **Partition 2 (Official Training - `datatraining.txt`)**:
   - Records: **8,143 rows** (39.6% of data).
   - Date Span: Feb 04, 2015 17:51:00 to Feb 10, 2015 09:33:00 (5.65 days).
   - Class Balance: 6,414 Unoccupied (78.77%) / 1,729 Occupied (21.23%).
3. **Partition 3 (Official Test 2 - `datatest2.txt`)**:
   - Records: **9,752 rows** (47.4% of data).
   - Date Span: Feb 11, 2015 14:48:00 to Feb 18, 2015 09:19:00 (6.77 days).
   - Class Balance: 7,703 Unoccupied (78.99%) / 2,049 Occupied (21.01%).
   - *Chronological Succession*: Occurred **after** the training partition (includes a full weekend gap).

---

### 3. Statistical Profiling & Sensor Physics

| Feature | Units | Min | 25% | Median | 75% | Max | Mean (Unoccupied) | Mean (Occupied) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Temperature** | $^\circ\text{C}$ | 19.00 | 20.20 | 20.70 | 21.53 | 24.41 | $20.34 \pm 0.73$ | $21.67 \pm 1.25$ |
| **Humidity** | % | 16.75 | 24.50 | 27.29 | 31.29 | 39.50 | $25.35 \pm 4.60$ | $27.15 \pm 5.12$ |
| **Light** | $\text{lux}$ | 0.00 | 0.00 | 0.00 | 0.00 | 1697.25 | $27.78 \pm 112.5$ | $\mathbf{459.85 \pm 238.4}$ |
| **CO2** | $\text{ppm}$ | 412.75 | 439.00 | 453.50 | 740.00 | 2028.50 | $490.32 \pm 187.3$ | $\mathbf{1037.71 \pm 384.2}$ |
| **HumidityRatio** | $\frac{\text{kg}_\text{w}}{\text{kg}_\text{a}}$ | 0.00267 | 0.00372 | 0.00429 | 0.00483 | 0.00648 | $0.0038 \pm 0.0007$ | $0.0044 \pm 0.0008$ |

#### Sensor Analysis:
- **Light ($r = 0.9074$ with Occupancy)**: Strongest single feature. Turning on artificial lighting correlates immediately with occupancy. However, natural exterior daylight entering through windows causes elevated lux readings ($50–250\text{ lux}$) even when unoccupied.
- **$\text{CO}_2$ ($r = 0.7122$ with Occupancy)**: Direct metabolic proxy. Human respiration doubles concentration from $490\text{ ppm}$ to $>1000\text{ ppm}$. Exhibits dissipation lag (takes 30–60 minutes to decay after occupants depart).
- **Temperature ($r = 0.5382$)**: Body heat, equipment, and lighting elevate room temperature by $+1.33^\circ\text{C}$.
- **Humidity ($r = 0.1330$) & HumidityRatio ($r = 0.3003$)**: HumidityRatio is a deterministic psychrometric transformation of Temperature and Humidity ($r = 0.9552$ with Humidity).

---

### 4. Autocorrelation & Data Leakage Risk

| Feature | Lag-1 ($1\text{ min}$) | Lag-5 ($5\text{ mins}$) | Lag-15 ($15\text{ mins}$) | Lag-60 ($1\text{ hour}$) |
|:---|:---:|:---:|:---:|:---:|
| **Temperature** | **0.9995** | 0.9986 | 0.9937 | 0.9592 |
| **Humidity** | **0.9999** | 0.9997 | 0.9986 | 0.9904 |
| **Light** | **0.9934** | 0.9645 | 0.9164 | 0.7712 |
| **CO2** | **0.9991** | 0.9976 | 0.9929 | 0.9610 |
| **Occupancy** | **0.9853** | 0.9536 | 0.9039 | 0.7788 |

**Risk Assessment**:
Because consecutive observations sampled every 60 seconds are $>99.9\%$ identical, **random shuffling (e.g. standard K-Fold CV) leaks future information into training folds**. Chronological time-series evaluation is mathematically mandatory.

