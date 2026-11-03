# AACI Occupancy Intelligence: Rigorous Evaluation Strategy

## 1. Core Evaluation Protocol & Data Leakage Prevention

Environmental sensor data exhibits severe autocorrelation (lag-1 autocorrelation $> 0.993$ across all features in Archive 2). Consequently, **random train-test splitting or standard random K-fold cross-validation is strictly banned**, as it allows past and future adjacent observations to leak across folds, producing deceptive metrics ($F_1 > 0.99$) that collapse upon real-world deployment.

---

## 2. Partition Structure & Roles

We enforce the exact chronological boundaries defined by the official UCI dataset:

```
TIME ────────────────────────────────────────────────────────────────────────►
[Feb 2 14:19]                      [Feb 4 10:43]   [Feb 4 17:51]                     [Feb 10 09:33]   [Feb 11 14:48]                     [Feb 18 09:19]
      │                                  │               │                                 │                │                                  │
      ▼                                  ▼               ▼                                 ▼                ▼                                  ▼
┌──────────────────────────────────────────────┐   ┌─────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│        Partition 1: Test 1 (Historical)      │   │        Partition 2: Train Set               │    │        Partition 3: Test 2 (Forward-in-Time) │
│        N = 2,665 records (12.96%)            │   │        N = 8,143 records (39.61%)           │    │        N = 9,752 records (47.43%)            │
│        Feb 2 14:19 to Feb 4 10:43            │   │        Feb 4 17:51 to Feb 10 09:33          │    │        Feb 11 14:48 to Feb 18 09:19          │
│        Occupancy Rate: 36.47%                │   │        Occupancy Rate: 21.23%               │    │        Occupancy Rate: 21.01%                │
└──────────────────────────────────────────────┘   └─────────────────────────────────────────────┘    └──────────────────────────────────────────────┘
                    ▲                                                     ▲                                                 ▲
                    │                                                     │                                                 │
          Historical Evaluation Set                             Primary Training Set                         Primary Forward-in-Time Benchmark
     (Precedes Training chronologically)                           (6.6 days data)                            (True Unseen Future Evaluation)
```

### Strict Partition Responsibilities:
1. **Primary Training Set (Partition 2 - `datatraining.txt`, $N = 8,143$)**:
   - Spans 6.6 days (weekday office hours and a full weekend).
   - Exclusively used for model fitting, internal feature selection, and rolling-origin validation.
2. **Primary Forward-in-Time Benchmark (Partition 3 - `datatest2.txt`, $N = 9,752$)**:
   - Primary evaluation set for all benchmark claims.
   - Forward in time after a 29-hour gap. Evaluates true operational generalization under shifted weather and daylight conditions.
3. **Historical Evaluation Set (Partition 1 - `datatest.txt`, $N = 2,665$)**:
   - Chronologically precedes the training partition.
   - Evaluates backward temporal invariance across distinct calendar periods.

---

## 3. Strict Preprocessing & Anti-Leakage Safeguards

To maintain complete methodological integrity:
1. **Isolated Preprocessor Fitting**:
   - All scalers (e.g. `StandardScaler` for Logistic Regression) are fitted **strictly on Partition 2 (Train)** and applied via `.transform()` to Test 1 and Test 2. Under no circumstances may scalers see test observations.
2. **Causal Feature Engineering**:
   - Any rolling or lag features (e.g., $\Delta \text{CO2}_{5\text{min}}$) must be strictly causal ($x_t - x_{t-k}$). No centered or forward-looking windows are allowed.
   - At partition boundaries, lag features do not cross partition gaps.
3. **Strict Validation Isolation for Thresholds & Calibration**:
   - Decision threshold selection $\tau^*$ is optimized **strictly on the internal validation folds within Partition 2**. Neither Test 1 nor Test 2 is ever used for threshold tuning.
   - Probability calibration (Platt scaling / Sigmoid or Isotonic) is fitted on out-of-fold validation predictions to prevent overfitting the calibration curve to the training set.

---

## 4. Rolling-Origin Cross-Validation (Inside Train Partition)

Inside Partition 2 ($N = 8,143$), hyperparameter selection uses expanding-window rolling-origin validation with a 15-minute purge buffer between train and validation splits:

```
Fold 1: [ Train: Day 1-2 (25%) ] ──|Purge|──► [ Val: Day 3 (15%) ]
Fold 2: [   Train: Day 1-3 (40%)   ] ──|Purge|──► [ Val: Day 4 (15%) ]
Fold 3: [     Train: Day 1-4 (55%)     ] ──|Purge|──► [ Val: Day 5 (15%) ]
Fold 4: [       Train: Day 1-5 (70%)       ] ──|Purge|──► [ Val: Day 6 (15%) ]
```

---

## 5. Required Reporting Metrics

Every evaluated model configuration across Experiments A, B, and C will report the following comprehensive metrics on both Test 2 (Primary) and Test 1 (Historical):

1. **Occupied-Class Precision**: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$ (tracks false fan starts / energy waste).
2. **Occupied-Class Recall**: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ (tracks false fan shut-offs / occupant discomfort).
3. **Occupied-Class F1-Score**: Harmonic mean of Occupied Precision and Recall.
4. **False Negative Rate (FNR)**: $\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = 1 - \text{Recall}$ (the primary discomfort risk metric).
5. **False Positive Rate (FPR)**: $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ (the primary energy waste risk metric).
6. **Balanced Accuracy**: Arithmetic mean of sensitivity (Recall) and specificity ($1 - \text{FPR}$).
7. **Precision-Recall AUC (PR-AUC)**: Area under the Precision-Recall curve (the most informative metric under class imbalance).
8. **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
9. **Brier Score**: $\frac{1}{N}\sum (P(\text{occupied}) - y)^2$ measuring probability calibration fidelity.
