# Module 2 Official Research Freeze Addendum

**Document Purpose**: Official Technical Addendum to complete the Research Freeze of AACI Model 2 – Module 2 (Spatial Headcount & Occupancy Pattern Intelligence).  
**Location**: `aaci_ml/occupancy_intelligence/module2/audit_v2/module2_freeze_addendum.md`

---

## 1. Mathematical Basis of the 52.1% Wi-Fi Improvement Figure

In the Independent Verification Audit, adding Wi-Fi client counts (Group 3 vs Group 2) for Random Forest produced a 52.1% error reduction.

### Verification of Calculation Method:
- **Calculation Type**: **Unweighted Arithmetic Mean of Room-Level Forward Test MAEs** (not a sample-weighted pooled aggregate).
- **Exact Room-by-Room Breakdown for Random Forest**:
  - Room 1 (Lecture Room 1): Group 2 MAE = **7.617** $\rightarrow$ Group 3 MAE = **1.007** ($\Delta = -6.610$, **86.8% reduction**)
  - Room 2 (Lecture Room 2): Group 2 MAE = **2.163** $\rightarrow$ Group 3 MAE = **2.056** ($\Delta = -0.107$, **4.9% reduction**)
  - Room 3 (Office Space 3): Group 2 MAE = **1.520** $\rightarrow$ Group 3 MAE = **1.471** ($\Delta = -0.049$, **3.2% reduction**)
  - Room 4 (Office Space 4): Group 2 MAE = **1.280** $\rightarrow$ Group 3 MAE = **1.206** ($\Delta = -0.074$, **5.8% reduction**)
  - Room 5 (Library Space 5): Group 2 MAE = **0.748** $\rightarrow$ Group 3 MAE = **0.638** ($\Delta = -0.110$, **14.7% reduction**)
- **Mean Across Rooms**:
  - Group 2 Mean MAE: $(7.617 + 2.163 + 1.520 + 1.280 + 0.748) / 5 = \mathbf{2.6656\text{ persons}}$
  - Group 3 Mean MAE: $(1.007 + 2.056 + 1.471 + 1.206 + 0.638) / 5 = \mathbf{1.2756\text{ persons}}$
  - Mean Absolute Reduction: $2.6656 - 1.2756 = \mathbf{1.3900\text{ persons}}$
  - Overall Mean Percentage Reduction: $(1.3900 / 2.6656) \times 100\% = \mathbf{52.14\%}$
- **Takeaway**: The 52.1% reduction is heavily driven by Room 1 (where Wi-Fi reduced error by 86.8% during large class sessions). In office rooms (Rooms 3 & 4), the reduction is a more modest 3%–6%.

---

## 2. Complete $5 \times 4$ Matrix: Group 2 vs Group 3 Comparison

The table below provides all 20 room-by-algorithm combinations on the exact same forward test splits:

| Room Identifier | Algorithm | Group 2 MAE (Env + Acoustic) | Group 3 MAE (+ Wi-Fi) | Absolute MAE Reduction | Percentage Reduction | Group 2 RMSE | Group 3 RMSE | Group 2 Within-1 Acc | Group 3 Within-1 Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room 1 (Lecture 1)** | Ridge Regression | 5.763 | 1.336 | -4.427 | 76.8% | 6.657 | 1.686 | 5.03% | 48.52% |
| Room 1 (Lecture 1) | Decision Tree | 5.376 | 0.797 | -4.579 | 85.2% | 8.145 | 1.372 | 43.02% | 89.39% |
| Room 1 (Lecture 1) | Random Forest | 7.617 | 1.007 | -6.610 | 86.8% | 9.414 | 1.399 | 10.77% | 53.79% |
| Room 1 (Lecture 1) | XGBoost | 7.127 | 1.656 | -5.471 | 76.8% | 9.106 | 2.085 | 13.01% | 37.51% |
| **Room 2 (Lecture 2)** | Ridge Regression | 2.290 | 1.868 | -0.422 | 18.4% | 5.279 | 3.543 | 56.42% | 56.74% |
| Room 2 (Lecture 2) | Decision Tree | 2.131 | 1.983 | -0.148 | 6.9% | 5.258 | 4.820 | 68.48% | 70.31% |
| Room 2 (Lecture 2) | Random Forest | 2.163 | 2.056 | -0.107 | 4.9% | 5.279 | 4.706 | 67.04% | 56.58% |
| Room 2 (Lecture 2) | XGBoost | 2.248 | 2.100 | -0.148 | 6.6% | 5.258 | 4.796 | 55.87% | 55.95% |
| **Room 3 (Office 3)** | Ridge Regression | 1.700 | 1.470 | -0.230 | 13.5% | 2.262 | 1.971 | 40.38% | 43.18% |
| Room 3 (Office 3) | Decision Tree | 1.625 | 1.468 | -0.157 | 9.7% | 2.743 | 2.566 | 63.37% | 65.84% |
| Room 3 (Office 3) | Random Forest | 1.520 | 1.471 | -0.049 | 3.2% | 2.576 | 2.503 | 61.05% | 61.37% |
| Room 3 (Office 3) | XGBoost | 1.493 | 1.371 | -0.122 | 8.2% | 2.361 | 2.161 | 59.46% | 59.38% |
| **Room 4 (Office 4)** | Ridge Regression | 1.292 | 1.171 | -0.121 | 9.4% | 1.819 | 1.667 | 50.32% | 54.51% |
| Room 4 (Office 4) | Decision Tree | 1.363 | 1.255 | -0.108 | 7.9% | 2.048 | 1.877 | 55.29% | 59.63% |
| Room 4 (Office 4) | Random Forest | 1.280 | 1.206 | -0.074 | 5.8% | 1.971 | 1.851 | 58.20% | 59.23% |
| Room 4 (Office 4) | XGBoost | 1.227 | 1.105 | -0.122 | 9.9% | 1.860 | 1.661 | 59.13% | 61.74% |
| **Room 5 (Library 5)** | Ridge Regression | 1.200 | 1.049 | -0.151 | 12.6% | 1.715 | 1.517 | 53.42% | 55.34% |
| Room 5 (Library 5) | Decision Tree | 0.682 | 0.591 | -0.091 | 13.3% | 1.333 | 1.102 | 79.07% | 78.53% |
| Room 5 (Library 5) | Random Forest | 0.748 | 0.638 | -0.110 | 14.7% | 1.226 | 1.122 | 66.13% | 74.25% |
| Room 5 (Library 5) | XGBoost | 0.800 | 0.693 | -0.107 | 13.4% | 1.261 | 1.152 | 65.78% | 72.13% |

---

## 3. Code-Level Evidence of Held-Out-Room Isolation Throughout LORO

Code inspection of [`src/robod_loader.py`](file:///c:/Users/bharg/OneDrive/Desktop/Atomberg/aaci_ml/occupancy_intelligence/module2/src/robod_loader.py) and [`src/train_headcount_regressor.py`](file:///c:/Users/bharg/OneDrive/Desktop/Atomberg/aaci_ml/occupancy_intelligence/module2/src/train_headcount_regressor.py) provides positive proof of data isolation:

1. **Partitioning**:
   ```python
   # From robod_loader.py, lines 130-134
   def get_loro_splits(all_rooms: Dict[int, pd.DataFrame], test_room_id: int):
       train_dfs = [df for r_id, df in all_rooms.items() if r_id != test_room_id]
       train_df = pd.concat(train_dfs, ignore_index=True)
       test_df = all_rooms[test_room_id].copy().reset_index(drop=True)
       return train_df, test_df
   ```
   - When evaluating held-out `test_room_id`, its records are explicitly excluded from `train_dfs`.
2. **Preprocessor Fitting**:
   ```python
   # From train_headcount_regressor.py, lines 188-192
   scaler = None
   if algo == "ridge":
       scaler = StandardScaler()
       X_tr_proc = scaler.fit_transform(X_tr)  # Fitted ONLY on the other 4 rooms
       X_te_proc = scaler.transform(X_te)      # Applied out-of-sample
   ```
   - Scalers are fitted exclusively on the 4-room training pool. The held-out room is never observed during standardization.
3. **No Target Leakage**:
   - `occupant_count [number]` is the target variable.
   - `occupant_presence [binary]` is completely excluded from all feature groups.

---

## 4. Official Freeze Declaration

Module 2 is hereby formally frozen as a research baseline.
- **Authoritative Independent Audit**: [`audit_v2/module2_independent_audit_report.md`](file:///c:/Users/bharg/OneDrive/Desktop/Atomberg/aaci_ml/occupancy_intelligence/module2/audit_v2/module2_independent_audit_report.md)
- **Model Role**: Research prototypes demonstrating multi-modal spatial occupancy and headcount estimation. Not approved for standalone physical ceiling fan deployment without edge network integrations.

