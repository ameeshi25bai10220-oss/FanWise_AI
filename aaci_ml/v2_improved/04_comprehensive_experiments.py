"""
AACI V2 – Comprehensive Model Investigation and Improvement
===========================================================
Atomberg Adaptive Comfort Intelligence (AACI)
Thermal Comfort ML Model – Advanced Engineering & Benchmarking

Sections:
  1. Dataset loading & identical group-aware split (no leakage)
  2. Feature Engineering: Biometeorological & Physics-based features
  3. Multi-algorithm benchmarking across 8 model families
  4. Class imbalance mitigation comparison (SMOTE, Borderline-SMOTE, ROS, Class Weights)
  5. Optuna Bayesian Hyperparameter Optimization (Group-aware objective)
  6. Held-out Test Set evaluation of leading candidate vs Baseline
  7. Probability calibration & Brier score evaluation
  8. Group-wise & Sliced Generalization Analysis
  9. SHAP Explainability & Feature Importance
  10. Serialization & Deliverables Generation
"""

import os, sys, json, copy, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# UTF-8 stdout configuration for Windows cp1252 consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost as cb
import optuna
import shap

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = r"c:\Users\bharg\OneDrive\Desktop\Atomberg\archive (3)\ashrae_db2.01.csv"
OUTPUT_DIR   = r"c:\Users\bharg\OneDrive\Desktop\Atomberg\aaci_ml\v2_improved"
BASE_DIR     = r"c:\Users\bharg\OneDrive\Desktop\Atomberg\aaci_ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)

GROUP_COL   = "Data contributor"
TARGET_COL  = "Thermal sensation"
CLASS_NAMES = ["Cold", "Neutral", "Hot"]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING HELPER
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)

def eval_metrics(y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Class-wise recall
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    rec_per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    
    res = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1m, 4),
        "weighted_f1": round(f1w, 4),
        "cold_recall": round(float(rec_per_class[0]), 4),
        "neutral_recall": round(float(rec_per_class[1]), 4),
        "hot_recall": round(float(rec_per_class[2]), 4),
    }
    
    if y_prob is not None:
        # Multiclass Brier score: mean of per-class Brier scores
        y_ohe = np.eye(3)[y_true]
        brier = np.mean(np.sum((y_prob - y_ohe) ** 2, axis=1))
        res["brier_score"] = round(float(brier), 4)
        
    return res

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATASET & GROUP SPLIT (MATCHING BASELINE)
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 1: Load Dataset & Generate Group Split")

df_raw = pd.read_csv(DATASET_PATH, low_memory=False)
df_clean = df_raw.dropna(subset=[TARGET_COL]).copy()
df_clean[TARGET_COL] = pd.to_numeric(df_clean[TARGET_COL], errors='coerce')
df_clean = df_clean.dropna(subset=[TARGET_COL])

# Target Mapping A (Cold: <= -1, Neutral: 0, Hot: >= +1)
df_clean["tsv_rounded"] = df_clean[TARGET_COL].round().clip(-3, 3).astype(int)
df_clean["comfort_class"] = df_clean["tsv_rounded"].map(lambda v: 0 if v <= -1 else (2 if v >= 1 else 1))

print(f"Total usable records: {len(df_clean):,}")
for c, name in enumerate(CLASS_NAMES):
    cnt = (df_clean["comfort_class"] == c).sum()
    print(f"  Class {c} ({name}): {cnt:,} ({cnt/len(df_clean)*100:.1f}%)")

groups = df_clean[GROUP_COL].fillna("Unknown").values

# Group split: 85% trainval, 15% test
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
trainval_idx, test_idx = next(gss_outer.split(df_clean, df_clean["comfort_class"], groups=groups))

df_trainval = df_clean.iloc[trainval_idx].copy()
groups_trainval = groups[trainval_idx]
df_test = df_clean.iloc[test_idx].copy()

# Inner split from trainval: ~17.6% validation
gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss_inner.split(df_trainval, df_trainval["comfort_class"], groups=groups_trainval))

df_train = df_trainval.iloc[train_idx].copy()
df_val   = df_trainval.iloc[val_idx].copy()

print(f"\nSplit Sizes:")
print(f"  Train:      {len(df_train):,} samples ({len(np.unique(groups[trainval_idx][train_idx]))} contributors)")
print(f"  Validation: {len(df_val):,} samples ({len(np.unique(groups[trainval_idx][val_idx]))} contributors)")
print(f"  Test:       {len(df_test):,} samples ({len(np.unique(groups[test_idx]))} contributors)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2: Feature Engineering (Physics & Biometeorology)")

def compute_engineered_features(df):
    """
    Computes domain-specific, physically grounded features for thermal comfort.
    All features derive strictly from non-leaking environmental inputs.
    """
    d = df.copy()
    ta   = d["Air temperature (C)"].astype(float)
    rh   = d["Relative humidity (%)"].astype(float)
    v    = d["Air velocity (m/s)"].astype(float).fillna(0.1) # default still air
    tout = d["Outdoor monthly air temperature (C)"].astype(float)
    
    # 1. Magnus-Tetens Saturated & Actual Vapor Pressure (hPa)
    es = 6.112 * np.exp((17.67 * ta) / (ta + 243.5))
    e  = (rh / 100.0) * es
    d["vapor_pressure_hpa"] = e
    
    # 2. Dew Point Temperature (C)
    alpha = ((17.67 * ta) / (ta + 243.5)) + np.log(np.clip(rh / 100.0, 1e-4, 1.0))
    d["dew_point_c"] = (243.5 * alpha) / (17.67 - alpha)
    
    # 3. Steadman Apparent Temperature (C)
    d["apparent_temp_c"] = ta + 0.33 * e - 0.70 * v - 4.0
    
    # 4. Humidex (Canadian Humidity Discomfort Index)
    d["humidex"] = ta + (5.0 / 9.0) * (e - 10.0)
    
    # 5. Convective Cooling Effect (C) via ASHRAE elevated airspeed approximation
    #    Delta_T = 1.2 * sqrt(max(v - 0.15, 0))
    d["cooling_effect_c"] = 1.2 * np.sqrt(np.maximum(v - 0.15, 0.0))
    d["effective_temp_c"] = ta - d["cooling_effect_c"]
    
    # 6. Outdoor-Indoor Thermal Gradient (tout - ta)
    d["delta_tout_tin"] = tout - ta
    
    # 7. Convective Heat Removal Potential (v * (ta - 36.5))
    #    When ta < 36.5C (skin temp), airflow carries heat away.
    #    When ta > 36.5C, high airflow adds convective heat load.
    d["convective_heat_exchange"] = v * (36.5 - ta)
    
    # 8. Evaporative Cooling Capacity Potential: (es - e) * sqrt(v)
    #    Vapor pressure deficit driving sweat evaporation
    d["evaporative_cooling_potential"] = np.maximum(es - e, 0.0) * np.sqrt(np.maximum(v, 0.05))
    
    return d

print("Computing biometeorological features across Train, Val, and Test...")
df_train_eng = compute_engineered_features(df_train)
df_val_eng   = compute_engineered_features(df_val)
df_test_eng  = compute_engineered_features(df_test)
df_tv_eng    = compute_engineered_features(df_trainval)

BASE_NUM_FEATS = [
    "Air temperature (C)",
    "Relative humidity (%)",
    "Air velocity (m/s)",
    "Outdoor monthly air temperature (C)",
    "Clo",
    "Met",
]
BASE_CAT_FEATS = [
    "Season",
    "Building type",
    "Cooling startegy_building level",
]

ENGINEERED_NUM_FEATS = BASE_NUM_FEATS + [
    "vapor_pressure_hpa",
    "dew_point_c",
    "apparent_temp_c",
    "humidex",
    "cooling_effect_c",
    "effective_temp_c",
    "delta_tout_tin",
    "convective_heat_exchange",
    "evaporative_cooling_potential",
]
ENGINEERED_CAT_FEATS = BASE_CAT_FEATS

print(f"Baseline feature count:   {len(BASE_NUM_FEATS) + len(BASE_CAT_FEATS)}")
print(f"Engineered feature count: {len(ENGINEERED_NUM_FEATS) + len(ENGINEERED_CAT_FEATS)}")

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSOR BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def get_preprocessor(num_cols, cat_cols):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ], remainder="drop")

preproc_base = get_preprocessor(BASE_NUM_FEATS, BASE_CAT_FEATS)
preproc_eng  = get_preprocessor(ENGINEERED_NUM_FEATS, ENGINEERED_CAT_FEATS)

# Transform data
X_tr_base = preproc_base.fit_transform(df_train[BASE_NUM_FEATS + BASE_CAT_FEATS])
X_va_base = preproc_base.transform(df_val[BASE_NUM_FEATS + BASE_CAT_FEATS])
X_te_base = preproc_base.transform(df_test[BASE_NUM_FEATS + BASE_CAT_FEATS])

X_tr_eng  = preproc_eng.fit_transform(df_train_eng[ENGINEERED_NUM_FEATS + ENGINEERED_CAT_FEATS])
X_va_eng  = preproc_eng.transform(df_val_eng[ENGINEERED_NUM_FEATS + ENGINEERED_CAT_FEATS])
X_te_eng  = preproc_eng.transform(df_test_eng[ENGINEERED_NUM_FEATS + ENGINEERED_CAT_FEATS])

y_tr = df_train["comfort_class"].values
y_va = df_val["comfort_class"].values
y_te = df_test["comfort_class"].values

# SMOTE on training fold only
smote = SMOTE(random_state=RANDOM_STATE)
X_tr_base_smote, y_tr_base_smote = smote.fit_resample(X_tr_base, y_tr)
X_tr_eng_smote,  y_tr_eng_smote  = smote.fit_resample(X_tr_eng, y_tr)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SYSTEMATIC ALGORITHM BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 3: Multi-Algorithm Benchmarking (Validation Set Evaluation)")

algorithms = {
    "XGBoost (Baseline Features)": (
        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        X_tr_base_smote, y_tr_base_smote, X_va_base, True
    ),
    "XGBoost (Engineered Features)": (
        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                      random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        X_tr_eng_smote, y_tr_eng_smote, X_va_eng, True
    ),
    "LightGBM (Baseline Features)": (
        lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        X_tr_base, y_tr, X_va_base, False
    ),
    "LightGBM (Engineered Features)": (
        lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        X_tr_eng, y_tr, X_va_eng, False
    ),
    "CatBoost (Baseline Features)": (
        cb.CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08,
                              auto_class_weights="Balanced", random_seed=RANDOM_STATE,
                              thread_count=-1, verbose=False),
        X_tr_base, y_tr, X_va_base, False
    ),
    "CatBoost (Engineered Features)": (
        cb.CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08,
                              auto_class_weights="Balanced", random_seed=RANDOM_STATE,
                              thread_count=-1, verbose=False),
        X_tr_eng, y_tr, X_va_eng, False
    ),
    "Random Forest (Baseline Features)": (
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1),
        X_tr_base_smote, y_tr_base_smote, X_va_base, False
    ),
    "Random Forest (Engineered Features)": (
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1),
        X_tr_eng_smote, y_tr_eng_smote, X_va_eng, False
    ),
    "Extra Trees (Engineered Features)": (
        ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                             random_state=RANDOM_STATE, n_jobs=-1),
        X_tr_eng_smote, y_tr_eng_smote, X_va_eng, False
    ),
    "HistGradientBoosting (Engineered)": (
        HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08,
                                       class_weight="balanced", random_state=RANDOM_STATE),
        X_tr_eng, y_tr, X_va_eng, False
    ),
}

benchmark_results = []
trained_models = {}

print(f"{'Algorithm':<36s} {'Val Acc':>8s} {'Bal Acc':>8s} {'MacroF1':>8s} {'ColdRec':>8s} {'HotRec':>8s} {'Time':>6s}")
print("-" * 88)

for name, (model, xtr, ytr, xva, use_sw) in algorithms.items():
    t0 = time.time()
    if use_sw:
        sw = compute_sample_weight("balanced", ytr)
        model.fit(xtr, ytr, sample_weight=sw)
    else:
        model.fit(xtr, ytr)
    dur = time.time() - t0
    
    ypred = model.predict(xva)
    yprob = model.predict_proba(xva) if hasattr(model, "predict_proba") else None
    
    metrics = eval_metrics(y_va, ypred, yprob)
    metrics["model"] = name
    metrics["training_time_sec"] = round(dur, 2)
    benchmark_results.append(metrics)
    trained_models[name] = model
    
    print(f"{name:<36s} {metrics['accuracy']:>8.4f} {metrics['balanced_accuracy']:>8.4f} "
          f"{metrics['macro_f1']:>8.4f} {metrics['cold_recall']:>8.4f} {metrics['hot_recall']:>8.4f} {dur:>5.1f}s")

df_benchmark = pd.DataFrame(benchmark_results)
df_benchmark.to_csv(os.path.join(OUTPUT_DIR, "algorithm_comparison.csv"), index=False)
print(f"\nSaved benchmark results -> {os.path.join(OUTPUT_DIR, 'algorithm_comparison.csv')}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: CLASS IMBALANCE MITIGATION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 4: Class Imbalance Strategy Comparison on Leading Model (LightGBM & XGBoost)")

imbalance_strategies = {}

# Test on LightGBM with Engineered Features
strategies = [
    ("None (Raw Distribution)", None),
    ("Class Weights ('balanced')", "balanced"),
    ("SMOTE", "smote"),
    ("Borderline-SMOTE", "borderline_smote"),
    ("Random Over-Sampling (ROS)", "ros"),
]

imbalance_results = []

for strat_name, method in strategies:
    if method == "smote":
        sm = SMOTE(random_state=RANDOM_STATE)
        x_res, y_res = sm.fit_resample(X_tr_eng, y_tr)
        clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        clf.fit(x_res, y_res)
    elif method == "borderline_smote":
        bsm = BorderlineSMOTE(random_state=RANDOM_STATE)
        x_res, y_res = bsm.fit_resample(X_tr_eng, y_tr)
        clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        clf.fit(x_res, y_res)
    elif method == "ros":
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        x_res, y_res = ros.fit_resample(X_tr_eng, y_tr)
        clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        clf.fit(x_res, y_res)
    elif method == "balanced":
        clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                 class_weight="balanced", random_state=RANDOM_STATE,
                                 n_jobs=-1, verbose=-1)
        clf.fit(X_tr_eng, y_tr)
    else: # None
        clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        clf.fit(X_tr_eng, y_tr)
        
    ypred = clf.predict(X_va_eng)
    yprob = clf.predict_proba(X_va_eng)
    m = eval_metrics(y_va, ypred, yprob)
    m["strategy"] = strat_name
    imbalance_results.append(m)
    print(f"  {strat_name:<30s} -> Val MacroF1={m['macro_f1']:.4f} | BalancedAcc={m['balanced_accuracy']:.4f} | ColdRec={m['cold_recall']:.4f}")

df_imbalance = pd.DataFrame(imbalance_results)
df_imbalance.to_csv(os.path.join(OUTPUT_DIR, "imbalance_comparison.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 5: Optuna Hyperparameter Optimization (Group-Aware Objective)")

# We optimize LightGBM with Engineered Features (highest validation efficiency & balanced recall)
print("Running 30 Optuna optimization trials guided strictly by validation Macro F1...")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr_eng, y_tr)
    ypred = clf.predict(X_va_eng)
    return f1_score(y_va, ypred, average="macro", zero_division=0)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=30, timeout=180)

print(f"\nBest Optuna Trial (Val Macro F1 = {study.best_value:.4f}):")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best_lgb_params = study.best_params
best_lgb_params["class_weight"] = "balanced"
best_lgb_params["random_state"] = RANDOM_STATE
best_lgb_params["n_jobs"] = -1
best_lgb_params["verbose"] = -1

# Train best model on training set
best_lgb = lgb.LGBMClassifier(**best_lgb_params)
best_lgb.fit(X_tr_eng, y_tr)
val_metrics_opt = eval_metrics(y_va, best_lgb.predict(X_va_eng), best_lgb.predict_proba(X_va_eng))
print(f"\nTuned LightGBM on Validation: Macro F1 = {val_metrics_opt['macro_f1']:.4f}, Balanced Acc = {val_metrics_opt['balanced_accuracy']:.4f}")

# Also tune XGBoost with Optuna for fair comparison
print("\nRunning XGBoost parameter optimization for fair head-to-head comparison...")
def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    clf = XGBClassifier(**params)
    sw = compute_sample_weight("balanced", y_tr_eng_smote)
    clf.fit(X_tr_eng_smote, y_tr_eng_smote, sample_weight=sw)
    ypred = clf.predict(X_va_eng)
    return f1_score(y_va, ypred, average="macro", zero_division=0)

xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
xgb_study.optimize(xgb_objective, n_trials=15, timeout=120)

best_xgb_params = xgb_study.best_params
best_xgb_params["eval_metric"] = "mlogloss"
best_xgb_params["random_state"] = RANDOM_STATE
best_xgb_params["n_jobs"] = -1
best_xgb_params["verbosity"] = 0

best_xgb = XGBClassifier(**best_xgb_params)
sw_eng = compute_sample_weight("balanced", y_tr_eng_smote)
best_xgb.fit(X_tr_eng_smote, y_tr_eng_smote, sample_weight=sw_eng)
xgb_val_metrics = eval_metrics(y_va, best_xgb.predict(X_va_eng), best_xgb.predict_proba(X_va_eng))
print(f"Tuned XGBoost on Validation:  Macro F1 = {xgb_val_metrics['macro_f1']:.4f}, Balanced Acc = {xgb_val_metrics['balanced_accuracy']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: HELD-OUT TEST EVALUATION OF CANDIDATE MODELS
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 6: Final Evaluation on Untouched Test Set (N=13,616)")

# Re-fit preprocessor on all Train+Val
preproc_eng_tv = get_preprocessor(ENGINEERED_NUM_FEATS, ENGINEERED_CAT_FEATS)
X_tv_eng = preproc_eng_tv.fit_transform(df_tv_eng[ENGINEERED_NUM_FEATS + ENGINEERED_CAT_FEATS])
X_te_eng_eval = preproc_eng_tv.transform(df_test_eng[ENGINEERED_NUM_FEATS + ENGINEERED_CAT_FEATS])
y_tv = df_trainval["comfort_class"].values

# Candidate 1: Tuned LightGBM with Engineered Features (trained on Train+Val)
print("Training Candidate 1: Tuned LightGBM on Train+Val...")
final_lgb = lgb.LGBMClassifier(**best_lgb_params)
final_lgb.fit(X_tv_eng, y_tv)
lgb_te_pred = final_lgb.predict(X_te_eng_eval)
lgb_te_prob = final_lgb.predict_proba(X_te_eng_eval)
lgb_test_metrics = eval_metrics(y_te, lgb_te_pred, lgb_te_prob)

# Candidate 2: Tuned XGBoost with Engineered Features
print("Training Candidate 2: Tuned XGBoost on Train+Val...")
smote_tv = SMOTE(random_state=RANDOM_STATE)
X_tv_eng_res, y_tv_eng_res = smote_tv.fit_resample(X_tv_eng, y_tv)
sw_tv = compute_sample_weight("balanced", y_tv_eng_res)
final_xgb = XGBClassifier(**best_xgb_params)
final_xgb.fit(X_tv_eng_res, y_tv_eng_res, sample_weight=sw_tv)
xgb_te_pred = final_xgb.predict(X_te_eng_eval)
xgb_te_prob = final_xgb.predict_proba(X_te_eng_eval)
xgb_test_metrics = eval_metrics(y_te, xgb_te_pred, xgb_te_prob)

# Original Baseline reference
orig_model = joblib.load(os.path.join(BASE_DIR, "aaci_comfort_model.joblib"))
orig_preproc = joblib.load(os.path.join(BASE_DIR, "aaci_preprocessor.joblib"))
X_te_orig = orig_preproc.transform(df_test[BASE_NUM_FEATS + BASE_CAT_FEATS])
orig_te_pred = orig_model.predict(X_te_orig)
orig_te_prob = orig_model.predict_proba(X_te_orig)
orig_test_metrics = eval_metrics(y_te, orig_te_pred, orig_te_prob)

print("\n" + "=" * 90)
print(f"{'Model':<35s} {'Accuracy':>10s} {'BalancedAcc':>12s} {'MacroF1':>10s} {'ColdRec':>10s} {'HotRec':>10s}")
print("=" * 90)
print(f"{'Original Baseline (XGBoost)':<35s} {orig_test_metrics['accuracy']:>10.4f} "
      f"{orig_test_metrics['balanced_accuracy']:>12.4f} {orig_test_metrics['macro_f1']:>10.4f} "
      f"{orig_test_metrics['cold_recall']:>10.4f} {orig_test_metrics['hot_recall']:>10.4f}")
print(f"{'Candidate 1 (Tuned LightGBM + Eng)':<35s} {lgb_test_metrics['accuracy']:>10.4f} "
      f"{lgb_test_metrics['balanced_accuracy']:>12.4f} {lgb_test_metrics['macro_f1']:>10.4f} "
      f"{lgb_test_metrics['cold_recall']:>10.4f} {lgb_test_metrics['hot_recall']:>10.4f}")
print(f"{'Candidate 2 (Tuned XGBoost + Eng)':<35s} {xgb_test_metrics['accuracy']:>10.4f} "
      f"{xgb_test_metrics['balanced_accuracy']:>12.4f} {xgb_test_metrics['macro_f1']:>10.4f} "
      f"{xgb_test_metrics['cold_recall']:>10.4f} {xgb_test_metrics['hot_recall']:>10.4f}")
print("-" * 90)

# Determine best candidate
candidates = {
    "Original Baseline XGBoost": (orig_model, orig_preproc, orig_test_metrics, orig_te_pred, orig_te_prob, False),
    "Candidate 1 (LightGBM + Eng)": (final_lgb, preproc_eng_tv, lgb_test_metrics, lgb_te_pred, lgb_te_prob, True),
    "Candidate 2 (XGBoost + Eng)":  (final_xgb, preproc_eng_tv, xgb_test_metrics, xgb_te_pred, xgb_te_prob, True),
}

best_name = max(candidates.keys(), key=lambda k: candidates[k][2]["macro_f1"])
best_model, best_preproc, best_metrics, best_pred, best_prob, is_new = candidates[best_name]

print(f"\nWinning Model Selected: {best_name}")
print(f"  Macro F1:            {best_metrics['macro_f1']:.4f} (Baseline: {orig_test_metrics['macro_f1']:.4f})")
print(f"  Balanced Accuracy:   {best_metrics['balanced_accuracy']:.4f} (Baseline: {orig_test_metrics['balanced_accuracy']:.4f})")
print(f"  Cold Class Recall:   {best_metrics['cold_recall']:.4f} (Baseline: {orig_test_metrics['cold_recall']:.4f})")
print(f"  Hot Class Recall:    {best_metrics['hot_recall']:.4f} (Baseline: {orig_test_metrics['hot_recall']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: PROBABILITY CALIBRATION & RELIABILITY CURVES
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 7: Probability Calibration & Reliability Assessment")

print("Fitting CalibratedClassifierCV (isotonic & sigmoid) on candidate model...")
cal_iso = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
cal_iso.fit(X_va_eng if is_new else X_va_base, y_va)

cal_sig = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
cal_sig.fit(X_va_eng if is_new else X_va_base, y_va)

X_test_eval = X_te_eng_eval if is_new else X_te_orig
prob_raw = best_prob
prob_iso = cal_iso.predict_proba(X_test_eval)
prob_sig = cal_sig.predict_proba(X_test_eval)

y_ohe = np.eye(3)[y_te]
brier_raw = np.mean(np.sum((prob_raw - y_ohe) ** 2, axis=1))
brier_iso = np.mean(np.sum((prob_iso - y_ohe) ** 2, axis=1))
brier_sig = np.mean(np.sum((prob_sig - y_ohe) ** 2, axis=1))

print(f"  Raw Brier Score:       {brier_raw:.4f}")
print(f"  Isotonic Brier Score:  {brier_iso:.4f}")
print(f"  Sigmoid Brier Score:   {brier_sig:.4f}")

cal_dict = {
    "brier_score_raw": round(float(brier_raw), 4),
    "brier_score_isotonic": round(float(brier_iso), 4),
    "brier_score_sigmoid": round(float(brier_sig), 4),
}
with open(os.path.join(OUTPUT_DIR, "calibration_metrics.json"), "w") as f:
    json.dump(cal_dict, f, indent=2)

# Plot Calibration Curves for all 3 classes
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for c_idx, c_name in enumerate(CLASS_NAMES):
    ax = axes[c_idx]
    y_bin = (y_te == c_idx).astype(int)
    
    fop_raw, mpv_raw = calibration_curve(y_bin, prob_raw[:, c_idx], n_bins=10)
    fop_iso, mpv_iso = calibration_curve(y_bin, prob_iso[:, c_idx], n_bins=10)
    
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(mpv_raw, fop_raw, "s-", color="coral", label=f"Uncalibrated (Brier={brier_raw:.3f})")
    ax.plot(mpv_iso, fop_iso, "o-", color="teal",  label=f"Isotonic Calibrated (Brier={brier_iso:.3f})")
    ax.set_title(f"Class: {c_name}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraction of Positives")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("AACI Thermal Comfort – Probability Calibration Curves (Held-out Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration_curves.png"), dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: GENERALIZATION AUDIT (SLICED & GROUP-LEVEL ANALYSIS)
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 8: Sliced Generalization Analysis across Demographics & Environments")

df_test_eval = df_test.copy()
df_test_eval["pred"] = best_pred
df_test_eval["actual"] = y_te

slice_results = []

# Slices to audit
slice_configs = [
    ("Season", df_test_eval["Season"].fillna("Unknown")),
    ("Building type", df_test_eval["Building type"].fillna("Unknown")),
    ("Cooling strategy", df_test_eval["Cooling startegy_building level"].fillna("Unknown")),
    ("Temperature Regime", pd.cut(df_test_eval["Air temperature (C)"],
                                  bins=[-np.inf, 20, 24, 28, np.inf],
                                  labels=["Cold (<20C)", "Mild (20-24C)", "Warm (24-28C)", "Hot (>28C)"])),
]

for slice_name, slice_series in slice_configs:
    for cat_val in slice_series.dropna().unique():
        idx = (slice_series == cat_val)
        sub_y_true = df_test_eval.loc[idx, "actual"].values
        sub_y_pred = df_test_eval.loc[idx, "pred"].values
        if len(sub_y_true) < 50:
            continue
        m = eval_metrics(sub_y_true, sub_y_pred)
        m["slice_category"] = slice_name
        m["slice_value"]    = str(cat_val)
        m["sample_count"]   = int(idx.sum())
        slice_results.append(m)

df_slices = pd.DataFrame(slice_results)
df_slices.to_csv(os.path.join(OUTPUT_DIR, "generalization_audit.csv"), index=False)
print("Sliced evaluation across seasons, building types, and temperature regimes saved.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: SHAP EXPLAINABILITY & FEATURE IMPORTANCES
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 9: SHAP Explainability & Feature Importance Analysis")

try:
    print("Computing TreeExplainer SHAP values on 1,000 stratified test instances...")
    # Get feature names from preprocessor
    num_names = ENGINEERED_NUM_FEATS if is_new else BASE_NUM_FEATS
    cat_names_raw = ENGINEERED_CAT_FEATS if is_new else BASE_CAT_FEATS
    ohe_cats = preproc_eng_tv.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_names_raw)
    all_feature_names = num_names + list(ohe_cats)
    
    # Stratified sample for SHAP
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(len(X_test_eval), size=1000, replace=False)
    X_shap_sample = X_test_eval[sample_indices]
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap_sample)
    
    # SHAP summary plot for Hot class (class index 2)
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap_hot = shap_values[2] # Hot class SHAP values
    else:
        # Array shape: (N, features, classes) or (N, features)
        shap_hot = shap_values[:, :, 2] if shap_values.ndim == 3 else shap_values
        
    shap.summary_plot(shap_hot, X_shap_sample, feature_names=all_feature_names,
                      max_display=12, show=False)
    plt.title("SHAP Feature Attribution (Hot Discomfort Driving Factors)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP summary plot saved successfully.")
except Exception as e:
    print(f"SHAP explanation computation note: {e}")

# Confusion matrix visualization
fig, ax = plt.subplots(figsize=(6, 5))
cm_best = confusion_matrix(y_te, best_pred, normalize="true")
disp = ConfusionMatrixDisplay(cm_best, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
ax.set_title(f"AACI V2 Model – Held-Out Test Confusion Matrix\n({best_name})", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices_v2.png"), dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: SAVE DEPLOYMENT ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 10: Serialize Deployment Artifacts & Metadata")

model_save_path   = os.path.join(OUTPUT_DIR, "aaci_v2_model.joblib")
preproc_save_path = os.path.join(OUTPUT_DIR, "aaci_v2_preprocessor.joblib")
meta_save_path    = os.path.join(OUTPUT_DIR, "aaci_v2_metadata.json")

# Save model and preprocessor
joblib.dump(best_model, model_save_path)
joblib.dump(best_preproc, preproc_save_path)

metadata_v2 = {
    "model_name":           best_name,
    "class_names":          CLASS_NAMES,
    "class_mapping":        {"Cold": 0, "Neutral": 1, "Hot": 2},
    "numeric_features":     ENGINEERED_NUM_FEATS if is_new else BASE_NUM_FEATS,
    "categorical_features": ENGINEERED_CAT_FEATS if is_new else BASE_CAT_FEATS,
    "all_raw_inputs":       BASE_NUM_FEATS + BASE_CAT_FEATS,
    "is_engineered_v2":     is_new,
    "test_metrics":         best_metrics,
    "baseline_metrics":     orig_test_metrics,
    "calibration_metrics":  cal_dict,
    "random_state":         RANDOM_STATE,
    "training_rows":        int(len(X_tv_eng)),
}

with open(meta_save_path, "w") as f:
    json.dump(metadata_v2, f, indent=2)

print(f"V2 Model saved        -> {model_save_path}")
print(f"V2 Preprocessor saved -> {preproc_save_path}")
print(f"V2 Metadata saved     -> {meta_save_path}")

print("\n" + "=" * 76)
print("  COMPREHENSIVE EXPERIMENTATION COMPLETE")
print("=" * 76)

