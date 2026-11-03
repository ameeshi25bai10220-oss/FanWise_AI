# AACI Module 1: Room Occupancy Detection – Comprehensive Evaluation Report

## Executive Summary
This report presents the empirical results of **Phase 4: Module 1 – Occupancy Detection Model Development, Training, and Validation** for Atomberg Adaptive Comfort Intelligence (AACI).

All 12 model configurations (4 algorithms $\times$ 3 feature sets) were developed and evaluated strictly using the **UCI Room Occupancy Detection Dataset** (`archive (2)/Occupancy.csv`, $N = 20,560$). Preprocessing and model training were conducted on the official Training Partition ($N = 8,143$) using rolling-origin temporal validation. True out-of-sample forward generalization was tested on the official Forward-in-Time Benchmark Partition ($N = 9,752$), and historical temporal stability was verified on the Historical Evaluation Partition ($N = 2,665$).

Zero random shuffling was permitted at any stage.

---

## 1. Primary Forward Benchmark Performance (Test 2: `datatest2.txt`, $N = 9,752$)

The primary forward benchmark represents real-world deployment on unseen future calendar days (February 11 to 18, 2015) after a 29-hour downtime gap.

| Algorithm | Feature Set | Optimal $\tau^*$ | Accuracy ($\tau=0.5$) | Occupied F1 ($\tau=0.5$) | Macro F1 ($\tau=0.5$) | Occupied Precision | Occupied Recall | PR-AUC | ROC-AUC | Brier Score | FNR ($\tau=0.5$) | FPR ($\tau=0.5$) | Inference Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **Set A (All 5)** | **0.70** | **99.15%** | **0.9800** | **0.9898** | **96.88%** | **99.12%** | **0.9876** | **0.9969** | **0.00998** | **0.88%** | **0.84%** | **91.1 $\mu$s** | **1,458 KB** |
| Random Forest | Set C (No HumRatio)| 0.77 | 98.66% | 0.9677 | 0.9837 | 97.91% | 95.66% | 0.9872 | 0.9967 | 0.01415 | 4.34% | 0.55% | 87.6 $\mu$s | 1,365 KB |
| Random Forest | Set B (Reduced BOM)| 0.10 | 89.27% | 0.7687 | 0.8471 | 70.36% | 84.82% | 0.7601 | 0.9472 | 0.08089 | 15.18% | 9.54% | 87.1 $\mu$s | 2,530 KB |
| **Logistic Regression** | **Set A (All 5)** | **0.21** | **99.00%** | **0.9763** | **0.9880** | **96.58%** | **98.68%** | **0.9831** | **0.9956** | **0.02301** | **1.32%** | **0.92%** | **1.7 $\mu$s** | **2.8 KB** |
| Logistic Regression | Set C (No HumRatio)| 0.20 | 99.02% | 0.9768 | 0.9882 | 96.63% | 98.78% | 0.9833 | 0.9957 | 0.02243 | 1.22% | 0.92% | 1.4 $\mu$s | 2.7 KB |
| Logistic Regression | Set B (Reduced BOM)| 0.10 | 87.29% | 0.7450 | 0.8308 | 64.50% | 88.34% | 0.5697 | 0.9006 | 0.12680 | 11.66% | 12.98% | 1.5 $\mu$s | 2.7 KB |
| **XGBoost** | **Set A (All 5)** | **0.10** | **95.48%** | **0.8883** | **0.9388** | **92.51%** | **85.55%** | **0.9756** | **0.9941** | **0.02411** | **14.45%** | **1.88%** | **6.8 $\mu$s** | **512 KB** |
| XGBoost | Set C (No HumRatio)| 0.10 | 95.26% | 0.8802 | 0.9351 | 93.89% | 82.87% | 0.9755 | 0.9937 | 0.02524 | 17.13% | 1.44% | 7.5 $\mu$s | 523 KB |
| XGBoost | Set B (Reduced BOM)| 0.10 | 87.44% | 0.7634 | 0.8359 | 63.39% | 96.44% | 0.7484 | 0.9505 | 0.09055 | 3.56% | 14.96% | 7.5 $\mu$s | 553 KB |
| **Decision Tree** | **Set A (All 5)** | **0.11** | **95.14%** | **0.8719** | **0.9333** | **97.89%** | **78.72%** | **0.9798** | **0.9947** | **0.02586** | **21.28%** | **0.49%** | **2.0 $\mu$s** | **10.8 KB** |
| Decision Tree | Set C (No HumRatio)| 0.11 | 95.14% | 0.8719 | 0.9333 | 97.89% | 78.72% | 0.9814 | 0.9949 | 0.02584 | 21.28% | 0.49% | 2.6 $\mu$s | 10.8 KB |
| Decision Tree | Set B (Reduced BOM)| 0.10 | 88.16% | 0.7715 | 0.8465 | 64.91% | 95.17% | 0.6018 | 0.9318 | 0.09601 | 4.83% | 13.71% | 1.4 $\mu$s | 14.1 KB |

---

## 2. Historical Evaluation Performance (Test 1: `datatest.txt`, $N = 2,665$)

Test 1 evaluates historical temporal stability on data collected prior to the training partition (February 2 to 4, 2015; occupancy rate: 36.47%).

| Algorithm | Feature Set | Accuracy ($\tau=0.5$) | Occupied F1 ($\tau=0.5$) | Occupied Precision | Occupied Recall | PR-AUC | ROC-AUC | Brier Score | FNR | FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **Set A** | **95.98%** | **0.9450** | **94.70%** | **94.24%** | **0.9904** | **0.9934** | **0.0298** | **5.76%** | **3.01%** |
| Random Forest | Set C | 93.28% | 0.9039 | 94.62% | 86.52% | 0.9897 | 0.9932 | 0.0468 | 13.48% | 2.83% |
| Random Forest | Set B | 87.32% | 0.7993 | 94.50% | 69.24% | 0.8529 | 0.9090 | 0.0886 | 30.76% | 2.30% |
| **Logistic Regression** | **Set A** | **97.86%** | **0.9714** | **95.42%** | **98.92%** | **0.9959** | **0.9961** | **0.0163** | **1.08%** | **2.72%** |
| Logistic Regression | Set C | 97.86% | 0.9714 | 95.42% | 98.92% | 0.9958 | 0.9961 | 0.0163 | 1.08% | 2.72% |
| Logistic Regression | Set B | 86.90% | 0.8152 | 82.69% | 80.39% | 0.8569 | 0.9237 | 0.0970 | 19.61% | 9.38% |
| **XGBoost** | **Set A** | **94.60%** | **0.9241** | **96.22%** | **88.90%** | **0.9947** | **0.9959** | **0.0381** | **11.10%** | **2.07%** |
| XGBoost | Set C | 92.87% | 0.8974 | 95.73% | 84.47% | 0.9944 | 0.9956 | 0.0496 | 15.53% | 2.18% |
| XGBoost | Set B | 94.37% | 0.9221 | 93.99% | 90.50% | 0.9656 | 0.9822 | 0.0456 | 9.50% | 3.37% |
| **Decision Tree** | **Set A** | **91.71%** | **0.8782** | **96.79%** | **80.39%** | **0.9933** | **0.9949** | **0.0526** | **19.61%** | **1.53%** |
| Decision Tree | Set C | 91.71% | 0.8782 | 96.79% | 80.39% | 0.9933 | 0.9949 | 0.0526 | 19.61% | 1.53% |
| Decision Tree | Set B | 94.48% | 0.9240 | 93.66% | 91.17% | 0.9416 | 0.9754 | 0.0459 | 8.83% | 3.60% |

---

## 3. Transition Dynamics & Detection Delay Analysis

To evaluate real-time responsiveness when occupants enter ($0 \rightarrow 1$) and exit ($1 \rightarrow 0$) a room, detection latency was quantified on the primary test partition:

| Model Configuration | Mean Entry Delay ($0 \rightarrow 1$) | Max Entry Delay | Mean Departure Delay ($1 \rightarrow 0$) | Max Departure Delay |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Set A)** | **0.21 min (12.5 s)** | **1.0 min** | **1.71 min (102 s)** | **3.0 min** |
| **Logistic Regression (Set A)** | **0.25 min (15.0 s)** | **2.0 min** | **1.71 min (102 s)** | **3.0 min** |
| XGBoost (Set A) | 9.13 min | 175.0 min | 1.08 min | 2.0 min |
| Decision Tree (Set A) | 14.13 min | 175.0 min | 1.00 min | 2.0 min |
| **Random Forest (Set B - No Light/CO2)**| **3.63 min** | **16.0 min** | **6.96 min** | **23.0 min** |
| **Logistic Regression (Set B)** | **7.21 min** | **62.0 min** | **24.38 min** | **78.0 min** |

### Key Transition Findings:
1. Under **Feature Set A**, Random Forest and Logistic Regression detect human arrival within **12 to 15 seconds** of physical entry. Departure is registered within **1.7 minutes**, making both models highly responsive.
2. Under **Feature Set B (Reduced BOM)**, without optical (Light) or rapid metabolic (CO2) indicators, entry detection latency increases to **3.6–7.2 minutes**, and departure detection requires **7 to 24 minutes**. This reflects the physical time required for sensible heat and water vapor from occupants to dissipate through room infiltration.

---

## 4. Sensor Noise & Perturbation Sensitivity Analysis

Realistic Gaussian noise was injected into test sensor telemetry to evaluate degradation under physical drift ($\sigma_{\text{Temp}} = 0.3^\circ\text{C}$, $\sigma_{\text{RH}} = 1.5\%$, $\sigma_{\text{Light}} = 10\text{ lux}$, $\sigma_{\text{CO2}} = 30\text{ ppm}$):

| Model Configuration | Clean Test 2 F1 | Noisy Test 2 F1 | $\Delta$ F1 (Robustness) | Clean Brier | Noisy Brier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Set A)** | **0.9800** | **0.9781** | **-0.0019** (Extremely Robust) | **0.00998** | **0.01033** |
| **Logistic Regression (Set A)** | **0.9763** | **0.9605** | **-0.0159** (High Robustness) | **0.02301** | **0.02397** |
| XGBoost (Set A) | 0.8883 | 0.8943 | +0.0060 (Stable) | 0.02411 | 0.02349 |
| Decision Tree (Set A) | 0.8719 | 0.8747 | +0.0028 (Stable) | 0.02586 | 0.02554 |
| **Random Forest (Set B)** | 0.7687 | 0.7311 | **-0.0376** (Moderate Drop) | 0.08089 | 0.08792 |
| **Logistic Regression (Set B)** | 0.7450 | 0.5898 | **-0.1552** (Severe Drop) | 0.12680 | 0.16317 |

---

## 5. Answers to the 9 Required Engineering Questions

### 1. Best-Performing Algorithm:
**Random Forest (Bagging Ensemble)** is the decisive overall performance leader on the primary forward benchmark:
- **Test 2 Accuracy**: **99.15%**
- **Test 2 Occupied F1-Score**: **0.9800** (Precision: 96.88%, Recall: 99.12%)
- **Test 2 PR-AUC**: **0.9876** | **ROC-AUC**: **0.9969**
- **False Negative Rate**: **0.88%** (almost never fails to detect occupants)
- **False Positive Rate**: **0.84%** (almost never spins in an empty room)
- **Noise Degradation**: Less than $0.2\%$ F1 loss under sensor perturbation.

### 2. Best-Performing Feature Set:
**Feature Set A (`Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio`)** delivers the highest performance across all 4 algorithms.
- **Ablation Insight (Set A vs Set C - Omitting `HumidityRatio`)**:
  - `HumidityRatio` provides a marginal performance improvement for Random Forest ($\Delta F_1 = +0.0123$) and XGBoost ($\Delta F_1 = +0.0081$).
  - For Logistic Regression, Set A and Set C are virtually identical ($F_1 = 0.9763$ vs $0.9768$).
  - Raw `Temperature` and `Humidity` capture almost the entire psychrometric signal when combined with tree splits.
- **Reduced BOM Assessment (Set B vs Set A - Omitting Light & CO2)**:
  - Omitting Light and CO2 causes a severe drop in performance across all algorithms: Random Forest F1 drops by **21.1%** (from $0.9800$ to $0.7687$), and false positive rate increases from $0.84\%$ to $9.54\%$.
  - **Verdict**: Set B is **not viable for high-confidence binary fan on/off control on its own**. Without Light or CO2, thermal/humidity accumulation is too slow and vulnerable to diurnal heating false alarms.

### 3. Best Calibrated Model:
**Random Forest Set A** achieved the best calibration score:
- **Brier Score**: **0.00998** on Test 2 (approaching zero).
- Mean predicted probabilities closely align with empirical occupancy fractions along the 45-degree diagonal.
- Logistic Regression Set A is second best with a Brier score of **0.02301**.

### 4. Best Lightweight Deployment Candidate:
**Logistic Regression Set A** is the superior candidate for constrained edge microcontrollers:
- **Model Storage Size**: **2.8 KB** (vs 1,458 KB for Random Forest).
- **Inference Latency**: **1.7 microseconds per prediction** (over $50\times$ faster than Random Forest).
- **Forward Benchmark F1**: **0.9763** (only $0.37\%$ lower than Random Forest).
- **Odds Ratios**: Direct physical interpretability: Light ($OR = 17.65$) and CO2 ($OR = 10.42$) dominate the log-odds of occupancy.

### 5. Forward-Test Performance:
On the primary forward-in-time test partition (Test 2, $N = 9,752$), the top model (Random Forest Set A) achieves **99.15% accuracy**, **0.9800 occupied F1**, **0.9876 PR-AUC**, and **0.9969 ROC-AUC**, with balanced error rates ($\text{FNR} = 0.88\%$, $\text{FPR} = 0.84\%$).

### 6. Historical-Test Performance:
On the historical evaluation partition (Test 1, $N = 2,665$), Logistic Regression Set A leads with **97.86% accuracy**, **0.9714 F1**, and **0.9959 PR-AUC**, while Random Forest Set A achieves **95.98% accuracy** and **0.9450 F1**. This confirms backward temporal stability across separate calendar weeks.

### 7. Important Failure Cases:
1. **Set B False Alarms During Afternoon Diurnal Heating**: In the absence of Light and CO2, rising ambient temperatures between 13:00 and 16:00 cause Set B models to predict occupancy when rooms are empty ($\text{FPR} = 9.5\%\text{--}15.0\%$).
2. **CO2 Venting Lag on Occupant Departure**: When occupants leave a room with windows and doors closed, CO2 decay takes 15–30 minutes, keeping predicted occupancy high during initial vacancy unless optical light drops to zero.
3. **Decision Tree Depth-Constrained Underfitting**: Depth-constrained trees exhibited high false negative rates ($\text{FNR} = 21.28\%$) on Test 2 because rigid axis-aligned cuts failed to separate subtle morning transitions.

### 8. Hardware Limitations:
1. **BOM Cost for True CO2 Sensing**: High-accuracy NDIR photoacoustic CO2 sensors (Sensirion SCD40) add \$5–\$8 to the fan bill-of-materials.
2. **Sensor Placement & Motor Heat Rise**: Mounting temperature sensors directly inside the fan motor casing distorts ambient readings due to motor electrical dissipation; canopy or downrod mounting with calibration offsets is mandatory.
3. **Residential Darkness vs Office Lighting**: In UCI data, `Light` provides the strongest single indicator because office lights are turned off upon vacancy. In residential bedrooms, occupants sleep in darkness ($0\text{ lux}$), which would cause false shut-offs if light is naively used without time/acoustic gating.

### 9. Suitability Verdict for Further Development:
- **Verdict**: **SUITABLE FOR MODULAR EXPANSION, BUT NOT YET DEPLOYABLE IN RESIDENTIAL BEDROOMS ON UCI DATA ALONE.**
- The algorithms and pipelines are fully validated and achieve excellent statistical benchmarks on office IoT telemetry.
- However, benchmark scores on UCI data **must not be conflated with residential bedroom readiness**. The next critical step is to develop **Module 2 (Headcount & Spatial Activity from ROBOD)** and **Module 3 (Fan Operational Actuation from RDD)** to evaluate performance under tropical cooling, variable acoustic levels, and actual residential fan speed logs.

