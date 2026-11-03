"""
AACI – Step 2: Full ML Training Pipeline
=========================================
Atomberg Adaptive Comfort Intelligence (AACI)
Dataset: ASHRAE Global Thermal Comfort Database II

Classes:
  0 = Cold    (TSV <= -1)
  1 = Neutral (TSV in (-1, 1) → rounded ≈ 0)
  2 = Hot     (TSV >= +1)

Run: python 02_train_model.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Force UTF-8 output on Windows to avoid cp1252 encode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not available – skipping XGBoost model")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("⚠ imbalanced-learn not available – will use class_weight only")

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = r"c:\Users\bharg\OneDrive\Desktop\Atomberg\archive (3)\ashrae_db2.01.csv"
OUTPUT_DIR   = r"c:\Users\bharg\OneDrive\Desktop\Atomberg\aaci_ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
GROUP_COL    = "Data contributor"

# Final feature set for AACI production system
NUMERIC_FEATURES = [
    "Air temperature (C)",
    "Relative humidity (%)",
    "Air velocity (m/s)",
    "Outdoor monthly air temperature (C)",
    "Clo",
    "Met",
]
CATEGORICAL_FEATURES = [
    "Season",
    "Building type",
    "Cooling startegy_building level",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL   = "Thermal sensation"

CLASS_NAMES = ["Cold", "Neutral", "Hot"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: pretty section header
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – LOAD & BASIC SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 1: Load Dataset")

df = pd.read_csv(DATASET_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – TARGET VARIABLE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2: Target Variable Preparation")

# 2a. Drop rows with missing Thermal sensation
n_before = len(df)
df = df.dropna(subset=[TARGET_COL])
print(f"Dropped {n_before - len(df):,} rows with missing TSV → {len(df):,} remaining")

# 2b. Cast to numeric (should already be float64, but confirm)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL])
print(f"After numeric cast: {len(df):,} rows")

# 2c. Round fractional TSV → nearest integer, then clip to [-3, +3]
df["tsv_rounded"] = df[TARGET_COL].round().clip(-3, 3).astype(int)
print("\nRounded TSV distribution:")
print(df["tsv_rounded"].value_counts().sort_index().to_string())

# 2d. Map to 3-class label
#   Cold    = tsv <= -1
#   Neutral = tsv == 0
#   Hot     = tsv >= +1
def map_tsv_to_class(tsv_int):
    if tsv_int <= -1:
        return 0   # Cold
    elif tsv_int >= 1:
        return 2   # Hot
    else:
        return 1   # Neutral

df["comfort_class"] = df["tsv_rounded"].map(map_tsv_to_class)

print("\n3-Class distribution:")
for cls_id, cls_name in enumerate(CLASS_NAMES):
    cnt = (df["comfort_class"] == cls_id).sum()
    pct = cnt / len(df) * 100
    print(f"  {cls_id} – {cls_name:<8s}: {cnt:>7,} ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – FEATURE SELECTION & MISSING AUDIT
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 3: Feature Selection & Missing Value Audit")

# Check features exist
missing_features = [f for f in ALL_FEATURES if f not in df.columns]
if missing_features:
    print(f"WARNING – missing columns: {missing_features}")
    # Remove from lists
    NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f in df.columns]
    CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print("\nFeature missing-value audit on working set:")
for col in ALL_FEATURES:
    n_miss = df[col].isna().sum()
    pct    = n_miss / len(df) * 100
    print(f"  {col:<45s}  missing={n_miss:>7,} ({pct:5.1f}%)")

X = df[ALL_FEATURES].copy()
y = df["comfort_class"].values
groups = df[GROUP_COL].fillna("Unknown").values

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – TRAIN / VALIDATION / TEST SPLIT  (group-aware)
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 4: Group-Aware Train / Val / Test Split")
print(f"Grouping by: '{GROUP_COL}'  ({len(np.unique(groups))} unique groups)")

# First split: 85% trainval, 15% test
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
trainval_idx, test_idx = next(gss_outer.split(X, y, groups=groups))

X_trainval, y_trainval = X.iloc[trainval_idx], y[trainval_idx]
g_trainval             = groups[trainval_idx]
X_test,     y_test     = X.iloc[test_idx], y[test_idx]

# Second split from trainval: val ≈ 15% of full = ~17.6% of trainval
gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=RANDOM_STATE)
train_idx2, val_idx2 = next(gss_inner.split(X_trainval, y_trainval, groups=g_trainval))

X_train = X_trainval.iloc[train_idx2]
y_train = y_trainval[train_idx2]
X_val   = X_trainval.iloc[val_idx2]
y_val   = y_trainval[val_idx2]

print(f"\nTrain set:      {len(X_train):>7,} rows  ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val):>7,} rows  ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set:       {len(X_test):>7,} rows  ({len(X_test)/len(X)*100:.1f}%)")

print("\nClass distribution in splits:")
for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = {CLASS_NAMES[c]: (split_y == c).sum() for c in range(3)}
    print(f"  {split_name}: " + "  ".join(f"{k}={v:,}" for k, v in dist.items()))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 5: Build Preprocessing Pipeline")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer,      NUMERIC_FEATURES),
        ("cat", categorical_transformer,  CATEGORICAL_FEATURES),
    ],
    remainder="drop"
)

print("Preprocessor built:")
print(f"  Numeric features ({len(NUMERIC_FEATURES)}):     {NUMERIC_FEATURES}")
print(f"  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
print("  Numeric  → median impute → StandardScaler")
print("  Categorical → mode impute → OneHotEncoder(handle_unknown='ignore')")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – SMOTE OVERSAMPLING (fit on train ONLY, never val/test)
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 6: Class Imbalance – SMOTE on Training Set")

# Fit preprocessor on training data only, then transform
X_train_pp = preprocessor.fit_transform(X_train)
X_val_pp   = preprocessor.transform(X_val)
X_test_pp  = preprocessor.transform(X_test)

print(f"Preprocessed shapes: train={X_train_pp.shape}, val={X_val_pp.shape}, test={X_test_pp.shape}")

if HAS_IMBALANCED:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_pp, y_train)
    print(f"\nAfter SMOTE: {X_train_res.shape[0]:,} training samples (was {X_train_pp.shape[0]:,})")
    for c, name in enumerate(CLASS_NAMES):
        cnt = (y_train_res == c).sum()
        print(f"  {name}: {cnt:,}")
else:
    X_train_res, y_train_res = X_train_pp, y_train
    print("SMOTE not available; using class_weight='balanced' in models.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 7: Model Training")

models = {}

# 7a. Logistic Regression
print("\n[7a] Training Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(
    C=0.1,
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    solver="lbfgs",
    multi_class="auto",
    n_jobs=-1,
)
lr.fit(X_train_res, y_train_res)
models["Logistic Regression"] = lr
print(f"  Done in {time.time()-t0:.1f}s")

# 7b. Random Forest
print("\n[7b] Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train_res, y_train_res)
models["Random Forest"] = rf
print(f"  Done in {time.time()-t0:.1f}s")

# 7c. XGBoost
if HAS_XGB:
    print("\n[7c] Training XGBoost...")
    t0 = time.time()
    # Compute sample weights to handle class imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_train_res)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train_res, y_train_res, sample_weight=sw)
    models["XGBoost"] = xgb
    print(f"  Done in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – EVALUATION ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 8: Evaluation on Validation Set")

val_results = {}

for model_name, model in models.items():
    y_pred = model.predict(X_val_pp)
    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_val, y_pred, average="macro", zero_division=0)
    f1m  = f1_score(y_val, y_pred, average="macro", zero_division=0)
    f1w  = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    val_results[model_name] = {
        "accuracy":          round(acc,  4),
        "macro_precision":   round(prec, 4),
        "macro_recall":      round(rec,  4),
        "macro_f1":          round(f1m,  4),
        "weighted_f1":       round(f1w,  4),
    }

    print(f"\n{'─'*60}")
    print(f"Model: {model_name}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall   : {rec:.4f}")
    print(f"  Macro F1       : {f1m:.4f}")
    print(f"  Weighted F1    : {f1w:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 – MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 9: Model Selection (best macro F1 on Validation)")

best_model_name = max(val_results, key=lambda m: val_results[m]["macro_f1"])
best_model      = models[best_model_name]

print(f"\nValidation Macro F1 scores:")
for name, res in val_results.items():
    marker = " ◄ SELECTED" if name == best_model_name else ""
    print(f"  {name:<25s}: {res['macro_f1']:.4f}{marker}")

print(f"\n✅ Best model: {best_model_name}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 – RETRAIN BEST MODEL ON TRAIN+VAL, EVALUATE ON TEST
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 10: Retrain Best Model on Train+Val, Evaluate on Test")

# Refit preprocessor on all trainval data
X_trainval_pp = preprocessor.fit_transform(X_trainval)
X_test_pp_final = preprocessor.transform(X_test)

if HAS_IMBALANCED:
    smote2 = SMOTE(random_state=RANDOM_STATE)
    X_tv_res, y_tv_res = smote2.fit_resample(X_trainval_pp, y_trainval)
    print(f"Train+Val after SMOTE: {X_tv_res.shape[0]:,} samples")
else:
    X_tv_res, y_tv_res = X_trainval_pp, y_trainval

# Clone and refit best model
import copy
final_model = copy.deepcopy(best_model)

if best_model_name == "XGBoost":
    sw_tv = compute_sample_weight("balanced", y_tv_res)
    final_model.fit(X_tv_res, y_tv_res, sample_weight=sw_tv)
else:
    final_model.fit(X_tv_res, y_tv_res)

y_test_pred = final_model.predict(X_test_pp_final)

test_acc  = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
test_rec  = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
test_f1m  = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
test_f1w  = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

print(f"\n{'═'*50}")
print(f"FINAL TEST SET RESULTS – {best_model_name}")
print(f"{'═'*50}")
print(f"  Accuracy       : {test_acc:.4f}")
print(f"  Macro Precision: {test_prec:.4f}")
print(f"  Macro Recall   : {test_rec:.4f}")
print(f"  Macro F1       : {test_f1m:.4f}")
print(f"  Weighted F1    : {test_f1w:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, zero_division=0))

test_results = {
    "model":           best_model_name,
    "accuracy":        round(test_acc,  4),
    "macro_precision": round(test_prec, 4),
    "macro_recall":    round(test_rec,  4),
    "macro_f1":        round(test_f1m,  4),
    "weighted_f1":     round(test_f1w,  4),
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 – CONFUSION MATRIX PLOTS
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 11: Confusion Matrices")

fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
if len(models) == 1:
    axes = [axes]

for ax, (model_name, model) in zip(axes, models.items()):
    # Use val predictions (re-predict)
    yp = model.predict(X_val_pp)
    cm = confusion_matrix(y_val, yp, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(f"{model_name}\n(Validation)")

plt.suptitle("AACI Thermal Comfort – Normalized Confusion Matrices (Validation)", fontsize=13, y=1.02)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Confusion matrices saved → {cm_path}")

# Final model test confusion matrix
fig2, ax2 = plt.subplots(figsize=(6, 5))
cm_test = confusion_matrix(y_test, y_test_pred, normalize='true')
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=CLASS_NAMES)
disp2.plot(ax=ax2, colorbar=False, cmap='Greens', values_format='.2f')
ax2.set_title(f"{best_model_name} – Test Set (Final)")
plt.tight_layout()
cm_test_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test_final.png")
plt.savefig(cm_test_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Final test confusion matrix saved → {cm_test_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 – FEATURE IMPORTANCE (for tree-based models)
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 12: Feature Importance")

if best_model_name in ("Random Forest", "XGBoost"):
    importance = final_model.feature_importances_

    # Get feature names from preprocessor
    try:
        ohe_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names = NUMERIC_FEATURES + list(ohe_names)
    except Exception:
        feature_names = [f"f{i}" for i in range(len(importance))]

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False).head(20)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", ax=ax3, color='steelblue')
    ax3.set_title(f"Top 20 Feature Importances – {best_model_name}")
    ax3.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved → {fi_path}")
    print("\nTop 10 features:")
    print(fi_df.head(10).to_string(index=False))
else:
    print("(Logistic Regression selected – feature importance via coefficients)")
    coef = np.abs(final_model.coef_).mean(axis=0)
    try:
        ohe_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names = NUMERIC_FEATURES + list(ohe_names)
    except Exception:
        feature_names = [f"f{i}" for i in range(len(coef))]
    fi_df = pd.DataFrame({"feature": feature_names[:len(coef)], "importance": coef})
    fi_df = fi_df.sort_values("importance", ascending=False).head(20)
    print(fi_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 – SAVE MODEL & PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 13: Save Model & Artifacts")

# Save the final model
model_path = os.path.join(OUTPUT_DIR, "aaci_comfort_model.joblib")
joblib.dump(final_model, model_path)
print(f"Model saved      → {model_path}")

# Save the preprocessor (fitted on trainval)
preproc_path = os.path.join(OUTPUT_DIR, "aaci_preprocessor.joblib")
joblib.dump(preprocessor, preproc_path)
print(f"Preprocessor saved → {preproc_path}")

# Save metadata for inference
metadata = {
    "model_name":           best_model_name,
    "class_names":          CLASS_NAMES,
    "class_mapping":        {"Cold": 0, "Neutral": 1, "Hot": 2},
    "tsv_to_class_rule":    "TSV <= -1 → Cold (0); TSV == 0 → Neutral (1); TSV >= 1 → Hot (2)",
    "numeric_features":     NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features":         ALL_FEATURES,
    "training_rows":        int(len(X_tv_res)),
    "val_results":          val_results,
    "test_results":         test_results,
    "smote_applied":        HAS_IMBALANCED,
    "random_state":         RANDOM_STATE,
    "group_split_key":      GROUP_COL,
}

meta_path = os.path.join(OUTPUT_DIR, "aaci_model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved   → {meta_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 – VALIDATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 14: Final Summary Table")

print("\nModel               Acc    Prec   Rec    MacroF1  WtdF1")
print("-"*65)
for name, res in val_results.items():
    marker = " *" if name == best_model_name else ""
    print(f"  {name:<25s} {res['accuracy']:>6.4f} {res['macro_precision']:>6.4f} {res['macro_recall']:>6.4f} {res['macro_f1']:>8.4f} {res['weighted_f1']:>6.4f}{marker}")

print(f"\n  {'Test (Final)':25s} {test_acc:>6.4f} {test_prec:>6.4f} {test_rec:>6.4f} {test_f1m:>8.4f} {test_f1w:>6.4f}  ← {best_model_name}")

print("\n✅ Training pipeline complete. All artifacts saved to:", OUTPUT_DIR)
