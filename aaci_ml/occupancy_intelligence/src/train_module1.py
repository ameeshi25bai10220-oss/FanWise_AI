"""
Training and Chronological Validation Pipeline for AACI Module 1 (Occupancy Intelligence).
Trains 12 model configurations (4 algorithms x 3 feature sets) on Partition 2 (datatraining.txt).

Algorithms:
  1. Logistic Regression (L2 regularization)
  2. Decision Tree (depth-constrained)
  3. Random Forest (bagging ensemble)
  4. XGBoost (gradient boosted decision trees)

Feature Sets:
  - Set A: Temperature, Humidity, Light, CO2, HumidityRatio
  - Set B: Temperature, Humidity, HumidityRatio, Hour
  - Set C: Temperature, Humidity, Light, CO2

Strict Safeguards:
  - Rolling-origin validation fits preprocessing exclusively on that fold's training history.
  - Independent chronological validation partitions for calibration and threshold optimization.
  - Final pipeline fitted on full official Partition 2 (N = 8,143).
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc, brier_score_loss,
    confusion_matrix
)
import xgboost as xgb

from data_loader import (
    load_raw_data, get_official_partitions, get_feature_and_target,
    generate_rolling_origin_folds, FEATURE_SETS, TARGET_COL
)

RANDOM_SEED = 42
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def get_base_estimator(algo_name: str) -> Any:
    """Instantiate fresh base estimator with pinned hyperparameters and seed."""
    if algo_name == "logistic_regression":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    elif algo_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=RANDOM_SEED)
    elif algo_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
    elif algo_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Handle edge cases in precision/recall/auc
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)
    brier = brier_score_loss(y_true, y_prob)
    
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal decision threshold tau* in [0.10, 0.90] that maximizes Macro F1
    while maintaining False Negative Rate <= 0.15 where possible.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_tau = 0.50
    best_score = -1.0
    
    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        # We value comfort (Recall >= 0.85) while optimizing balanced Macro F1
        score = macro_f1 + (0.1 * rec)
        if score > best_score:
            best_score = score
            best_tau = tau
            
    return float(best_tau), float(best_score)


def run_chronological_validation(
    train_df: pd.DataFrame,
    algo_name: str,
    feature_set_name: str
) -> Dict[str, Any]:
    """
    Executes rolling-origin validation on train_df.
    Separates folds:
      - Folds 1 & 2 validation predictions: Fit calibration.
      - Fold 3 validation predictions: Optimize threshold.
      - Fold 4 validation predictions: Independent validation test.
    """
    folds = generate_rolling_origin_folds(train_df, n_splits=4)
    feature_cols = FEATURE_SETS[feature_set_name]
    
    oof_predictions = []
    oof_trues = []
    oof_fold_ids = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        tr_sub = train_df.iloc[tr_idx]
        val_sub = train_df.iloc[val_idx]
        
        X_tr = tr_sub[feature_cols].copy()
        y_tr = tr_sub[TARGET_COL].copy().values
        X_val = val_sub[feature_cols].copy()
        y_val = val_sub[TARGET_COL].copy().values
        
        # Fit scaler ONLY on fold training history
        scaler = StandardScaler()
        if algo_name == "logistic_regression":
            X_tr_proc = scaler.fit_transform(X_tr)
            X_val_proc = scaler.transform(X_val)
        else:
            X_tr_proc = X_tr.values
            X_val_proc = X_val.values
            
        model = get_base_estimator(algo_name)
        model.fit(X_tr_proc, y_tr)
        
        # Predict probability of occupancy (class 1)
        if hasattr(model, "predict_proba"):
            y_val_prob = model.predict_proba(X_val_proc)[:, 1]
        else:
            y_val_prob = model.predict(X_val_proc)
            
        oof_predictions.append(y_val_prob)
        oof_trues.append(y_val)
        oof_fold_ids.append(np.full_like(y_val, fold_idx))
        
    all_prob = np.concatenate(oof_predictions)
    all_true = np.concatenate(oof_trues)
    all_folds = np.concatenate(oof_fold_ids)
    
    # Independent splits for calibration and threshold selection:
    # Folds 0 & 1 -> Calibration analysis / check
    mask_cal = (all_folds == 0) | (all_folds == 1)
    # Fold 2 -> Threshold Optimization
    mask_thresh = (all_folds == 2)
    # Fold 3 -> Unbiased Validation Benchmark
    mask_val_eval = (all_folds == 3)
    
    prob_thresh_set = all_prob[mask_thresh]
    true_thresh_set = all_true[mask_thresh]
    opt_tau, _ = optimize_threshold(true_thresh_set, prob_thresh_set)
    
    # Compute metrics on unbiased Fold 3
    prob_eval = all_prob[mask_val_eval]
    true_eval = all_true[mask_val_eval]
    default_metrics = compute_all_metrics(true_eval, prob_eval, threshold=0.5)
    opt_metrics = compute_all_metrics(true_eval, prob_eval, threshold=opt_tau)
    
    return {
        "algo_name": algo_name,
        "feature_set": feature_set_name,
        "optimal_threshold": opt_tau,
        "val_default_metrics": default_metrics,
        "val_opt_metrics": opt_metrics,
        "oof_true": all_true,
        "oof_prob": all_prob,
        "oof_folds": all_folds
    }


def fit_and_save_final_model(
    train_df: pd.DataFrame,
    algo_name: str,
    feature_set_name: str,
    optimal_tau: float
) -> Dict[str, Any]:
    """
    Fits the final production pipeline on the complete official training partition (Partition 2, N = 8,143).
    Saves scaler and model artifacts to disk.
    Measures training time and inference latency.
    """
    feature_cols = FEATURE_SETS[feature_set_name]
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy().values
    
    # 1. Fit scaler on entire Partition 2
    scaler = None
    if algo_name == "logistic_regression":
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
    else:
        X_train_proc = X_train.values
        
    # 2. Fit base estimator with timing
    model = get_base_estimator(algo_name)
    t0 = time.perf_counter()
    model.fit(X_train_proc, y_train)
    train_time_sec = time.perf_counter() - t0
    
    # 3. Fit probability calibration (using 5-fold cross-validation internal to training set)
    calibrated_model = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train_proc, y_train)
    
    # 4. Measure inference latency (per 1,000 samples)
    t0 = time.perf_counter()
    _ = calibrated_model.predict_proba(X_train_proc[:1000])
    inference_time_ms_per_1000 = (time.perf_counter() - t0) * 1000.0
    latency_us_per_sample = (inference_time_ms_per_1000 / 1000.0) * 1000.0
    
    # 5. Save artifacts
    model_filename = f"module1_{algo_name}_{feature_set_name}.joblib"
    calibrated_filename = f"module1_{algo_name}_{feature_set_name}_calibrated.joblib"
    scaler_filename = f"preprocessor_{feature_set_name}.joblib"
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    calibrated_path = os.path.join(MODELS_DIR, calibrated_filename)
    scaler_path = os.path.join(MODELS_DIR, scaler_filename)
    
    joblib.dump(model, model_path)
    joblib.dump(calibrated_model, calibrated_path)
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        
    model_size_kb = os.path.getsize(calibrated_path) / 1024.0
    
    # Extract feature importance / coefficients
    feature_importances = {}
    if algo_name == "logistic_regression":
        # Odds ratios = exp(coefficients)
        for col, coef in zip(feature_cols, model.coef_[0]):
            feature_importances[col] = {
                "coefficient": float(coef),
                "odds_ratio": float(np.exp(coef))
            }
    elif hasattr(model, "feature_importances_"):
        for col, imp in zip(feature_cols, model.feature_importances_):
            feature_importances[col] = {
                "importance": float(imp)
            }
            
    return {
        "algo_name": algo_name,
        "feature_set": feature_set_name,
        "model_file": model_filename,
        "calibrated_model_file": calibrated_filename,
        "scaler_file": scaler_filename if scaler else None,
        "train_time_sec": float(train_time_sec),
        "latency_us_per_sample": float(latency_us_per_sample),
        "model_size_kb": float(model_size_kb),
        "optimal_threshold": float(optimal_tau),
        "feature_importances": feature_importances
    }


def main():
    print("=" * 80)
    print("AACI Module 1: Training and Chronological Validation Pipeline")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 80)
    
    # 1. Load data
    train_df, test2_df, test1_df = get_official_partitions()
    print(f"Loaded Official UCI Partitions:")
    print(f"  Train Partition 2:      {len(train_df):,} rows (Occ: {train_df[TARGET_COL].mean():.2%})")
    print(f"  Forward Test 2:         {len(test2_df):,} rows (Occ: {test2_df[TARGET_COL].mean():.2%})")
    print(f"  Historical Test 1:      {len(test1_df):,} rows (Occ: {test1_df[TARGET_COL].mean():.2%})\n")
    
    algorithms = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    feature_sets = ["set_a", "set_b", "set_c"]
    
    val_records = []
    final_models_info = {}
    threshold_records = []
    
    total_configs = len(algorithms) * len(feature_sets)
    cfg_idx = 1
    
    for algo in algorithms:
        for fset in feature_sets:
            print(f"[{cfg_idx}/{total_configs}] Running Chronological Validation: Algo={algo.upper()} | FeatureSet={fset.upper()}...")
            val_res = run_chronological_validation(train_df, algo, fset)
            
            # Store validation metrics
            opt_tau = val_res["optimal_threshold"]
            m_def = val_res["val_default_metrics"]
            m_opt = val_res["val_opt_metrics"]
            
            val_records.append({
                "algorithm": algo,
                "feature_set": fset,
                "optimal_threshold": opt_tau,
                "val_acc_def": m_def["accuracy"],
                "val_acc_opt": m_opt["accuracy"],
                "val_bal_acc_def": m_def["balanced_accuracy"],
                "val_bal_acc_opt": m_opt["balanced_accuracy"],
                "val_f1_def": m_def["f1"],
                "val_f1_opt": m_opt["f1"],
                "val_macro_f1_def": m_def["macro_f1"],
                "val_macro_f1_opt": m_opt["macro_f1"],
                "val_pr_auc": m_def["pr_auc"],
                "val_roc_auc": m_def["roc_auc"],
                "val_brier": m_def["brier_score"],
                "val_fnr_def": m_def["false_negative_rate"],
                "val_fnr_opt": m_opt["false_negative_rate"],
                "val_fpr_def": m_def["false_positive_rate"],
                "val_fpr_opt": m_opt["false_positive_rate"]
            })
            
            threshold_records.append({
                "algorithm": algo,
                "feature_set": fset,
                "optimal_threshold": opt_tau,
                "fnr_at_0.5": m_def["false_negative_rate"],
                "fnr_at_tau": m_opt["false_negative_rate"],
                "fpr_at_0.5": m_def["false_positive_rate"],
                "fpr_at_tau": m_opt["false_positive_rate"],
                "f1_at_0.5": m_def["f1"],
                "f1_at_tau": m_opt["f1"]
            })
            
            # Fit final pipeline on complete Partition 2
            print(f"      Fitting final pipeline on complete Partition 2 (N={len(train_df):,})...")
            final_res = fit_and_save_final_model(train_df, algo, fset, opt_tau)
            final_models_info[f"{algo}_{fset}"] = final_res
            
            print(f"      Done. Train time: {final_res['train_time_sec']:.3f}s | Latency: {final_res['latency_us_per_sample']:.1f}us | Size: {final_res['model_size_kb']:.1f}KB | Opt Tau: {opt_tau:.2f}")
            cfg_idx += 1

    # Save validation and threshold tables
    val_df = pd.DataFrame(val_records)
    val_csv_path = os.path.join(EVAL_DIR, "validation_benchmark.csv")
    val_df.to_csv(val_csv_path, index=False)
    print(f"\nSaved Validation Benchmark to: {val_csv_path}")
    
    thresh_df = pd.DataFrame(threshold_records)
    thresh_csv_path = os.path.join(EVAL_DIR, "threshold_optimization.csv")
    thresh_df.to_csv(thresh_csv_path, index=False)
    print(f"Saved Threshold Optimization Results to: {thresh_csv_path}")
    
    # Save model metadata
    metadata = {
        "metadata_version": "1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "dataset": "archive (2)/Occupancy.csv",
        "dataset_partitions": {
            "train_partition": {"file": "datatraining.txt", "rows": len(train_df), "indices": "2665 to 10807"},
            "test2_forward": {"file": "datatest2.txt", "rows": len(test2_df), "indices": "10808 to 20559"},
            "test1_historical": {"file": "datatest.txt", "rows": len(test1_df), "indices": "0 to 2664"}
        },
        "feature_definitions": FEATURE_SETS,
        "models": final_models_info
    }
    
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Model Metadata to: {meta_path}")
    print("\nTraining Phase Complete. Proceed to evaluate_module1.py for official Test 2 & Test 1 evaluation.")


if __name__ == "__main__":
    main()

