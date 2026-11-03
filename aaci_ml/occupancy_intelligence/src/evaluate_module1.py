"""
Comprehensive Evaluation Suite for AACI Module 1 (Occupancy Intelligence).
Evaluates all 12 trained models against:
  1. Primary Forward Benchmark: Partition 3 (Test 2 - datatest2.txt, N = 9,752)
  2. Historical Evaluation: Partition 1 (Test 1 - datatest.txt, N = 2,665)

Computes:
  - Accuracy, Balanced Accuracy, Occupied Precision, Occupied Recall, Occupied F1, Macro F1
  - PR-AUC, ROC-AUC, Brier Score, FNR, FPR, Confusion Matrix
  - Threshold comparisons (0.5 vs optimal tau)
  - Probability calibration analysis & curves
  - Sensor noise sensitivity tests (Gaussian perturbations)
  - Transition dynamics & detection delay (0->1 and 1->0)
  - Feature importances and ablation insights
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, roc_curve, auc,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

from data_loader import (
    load_raw_data, get_official_partitions, get_feature_and_target,
    FEATURE_SETS, TARGET_COL
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
os.makedirs(EVAL_DIR, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes all required metrics for an evaluation partition."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
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
        "fnr": float(fnr),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def load_model_and_preprocessor(algo_name: str, feature_set_name: str, calibrated: bool = True):
    """Loads calibrated or uncalibrated model and associated scaler."""
    suffix = "_calibrated.joblib" if calibrated else ".joblib"
    model_path = os.path.join(MODELS_DIR, f"module1_{algo_name}_{feature_set_name}{suffix}")
    scaler_path = os.path.join(MODELS_DIR, f"preprocessor_{feature_set_name}.joblib")
    
    model = joblib.load(model_path)
    scaler = None
    if os.path.exists(scaler_path) and algo_name == "logistic_regression":
        scaler = joblib.load(scaler_path)
        
    return model, scaler


def predict_pipeline(model, scaler, X_df: pd.DataFrame) -> np.ndarray:
    """Predicts probabilities using the fitted pipeline."""
    if scaler is not None:
        X_proc = scaler.transform(X_df)
    else:
        X_proc = X_df.values
        
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)[:, 1]
    else:
        probs = model.predict(X_proc)
    return probs


def evaluate_transition_dynamics(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Measures detection delay and transition accuracy during:
      - 0 -> 1 (Occupant Entry)
      - 1 -> 0 (Occupant Departure)
    """
    y_true = df[TARGET_COL].values
    y_pred = (probs >= threshold).astype(int)
    
    # Identify transition indices
    diffs = np.diff(y_true)
    entry_indices = np.where(diffs == 1)[0] + 1     # 0 -> 1
    departure_indices = np.where(diffs == -1)[0] + 1 # 1 -> 0
    
    # Measure entry detection lag (minutes until first positive prediction after entry)
    entry_lags = []
    for idx in entry_indices:
        lag = 0
        while idx + lag < len(y_true) and y_true[idx + lag] == 1:
            if y_pred[idx + lag] == 1:
                break
            lag += 1
        entry_lags.append(lag)
        
    # Measure departure detection lag (minutes until first negative prediction after departure)
    departure_lags = []
    for idx in departure_indices:
        lag = 0
        while idx + lag < len(y_true) and y_true[idx + lag] == 0:
            if y_pred[idx + lag] == 0:
                break
            lag += 1
        departure_lags.append(lag)
        
    mean_entry_lag = float(np.mean(entry_lags)) if entry_lags else 0.0
    mean_departure_lag = float(np.mean(departure_lags)) if departure_lags else 0.0
    
    return {
        "num_entry_events": len(entry_indices),
        "mean_entry_detection_delay_minutes": mean_entry_lag,
        "max_entry_detection_delay_minutes": float(np.max(entry_lags)) if entry_lags else 0.0,
        "num_departure_events": len(departure_indices),
        "mean_departure_detection_delay_minutes": mean_departure_lag,
        "max_departure_detection_delay_minutes": float(np.max(departure_lags)) if departure_lags else 0.0
    }


def evaluate_noise_sensitivity(
    model, scaler, df: pd.DataFrame, feature_set_name: str, threshold: float = 0.5
) -> dict:
    """
    Evaluates model resilience under sensor noise:
      - Temp: Gaussian noise sigma = 0.3 deg C
      - Humidity: Gaussian noise sigma = 1.5 %
      - Light: Gaussian noise sigma = 10 lux
      - CO2: Gaussian noise sigma = 30 ppm
    """
    feature_cols = FEATURE_SETS[feature_set_name]
    X_noisy = df[feature_cols].copy()
    
    noise_specs = {
        "Temperature": 0.3,
        "Humidity": 1.5,
        "Light": 10.0,
        "CO2": 30.0,
        "HumidityRatio": 0.0001
    }
    
    for col in feature_cols:
        if col in noise_specs:
            noise = np.random.normal(0, noise_specs[col], size=len(X_noisy))
            X_noisy[col] = np.maximum(0, X_noisy[col] + noise)
            
    probs_noisy = predict_pipeline(model, scaler, X_noisy)
    y_true = df[TARGET_COL].values
    noisy_metrics = compute_metrics(y_true, probs_noisy, threshold=threshold)
    return noisy_metrics


def main():
    print("=" * 80)
    print("AACI Module 1: Official Test Evaluation Suite")
    print("Primary Benchmark: Partition 3 (Test 2, N = 9,752)")
    print("Historical Test:   Partition 1 (Test 1, N = 2,665)")
    print("=" * 80)
    
    # 1. Load Partitions
    train_df, test2_df, test1_df = get_official_partitions()
    
    # Load optimal thresholds from threshold_optimization.csv
    thresh_csv_path = os.path.join(EVAL_DIR, "threshold_optimization.csv")
    thresh_map = {}
    if os.path.exists(thresh_csv_path):
        tdf = pd.read_csv(thresh_csv_path)
        for _, r in tdf.iterrows():
            thresh_map[(r["algorithm"], r["feature_set"])] = float(r["optimal_threshold"])
            
    algorithms = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    feature_sets = ["set_a", "set_b", "set_c"]
    
    benchmark_records = []
    transition_records = []
    noise_records = []
    
    test2_predictions = {}
    test1_predictions = {}
    
    for algo in algorithms:
        for fset in feature_sets:
            print(f"Evaluating {algo.upper()} on {fset.upper()}...")
            opt_tau = thresh_map.get((algo, fset), 0.50)
            
            # Load calibrated model and scaler
            model, scaler = load_model_and_preprocessor(algo, fset, calibrated=True)
            
            # Predict on Primary Forward Test 2
            X_test2 = test2_df[FEATURE_SETS[fset]].copy()
            y_test2 = test2_df[TARGET_COL].values
            p_test2 = predict_pipeline(model, scaler, X_test2)
            test2_predictions[(algo, fset)] = p_test2
            
            # Predict on Historical Test 1
            X_test1 = test1_df[FEATURE_SETS[fset]].copy()
            y_test1 = test1_df[TARGET_COL].values
            p_test1 = predict_pipeline(model, scaler, X_test1)
            test1_predictions[(algo, fset)] = p_test1
            
            # Metrics for Test 2 at default 0.5 and optimal tau
            m2_def = compute_metrics(y_test2, p_test2, threshold=0.50)
            m2_opt = compute_metrics(y_test2, p_test2, threshold=opt_tau)
            
            # Metrics for Test 1 at default 0.5 and optimal tau
            m1_def = compute_metrics(y_test1, p_test1, threshold=0.50)
            m1_opt = compute_metrics(y_test1, p_test1, threshold=opt_tau)
            
            benchmark_records.append({
                "algorithm": algo,
                "feature_set": fset,
                "optimal_threshold": opt_tau,
                # Test 2 (Forward Benchmark)
                "test2_acc_def": m2_def["accuracy"],
                "test2_acc_opt": m2_opt["accuracy"],
                "test2_bal_acc_def": m2_def["balanced_accuracy"],
                "test2_bal_acc_opt": m2_opt["balanced_accuracy"],
                "test2_f1_def": m2_def["f1"],
                "test2_f1_opt": m2_opt["f1"],
                "test2_macro_f1_def": m2_def["macro_f1"],
                "test2_macro_f1_opt": m2_opt["macro_f1"],
                "test2_precision_def": m2_def["precision"],
                "test2_precision_opt": m2_opt["precision"],
                "test2_recall_def": m2_def["recall"],
                "test2_recall_opt": m2_opt["recall"],
                "test2_pr_auc": m2_def["pr_auc"],
                "test2_roc_auc": m2_def["roc_auc"],
                "test2_brier": m2_def["brier_score"],
                "test2_fnr_def": m2_def["fnr"],
                "test2_fnr_opt": m2_opt["fnr"],
                "test2_fpr_def": m2_def["fpr"],
                "test2_fpr_opt": m2_opt["fpr"],
                "test2_tp": m2_def["tp"],
                "test2_fp": m2_def["fp"],
                "test2_tn": m2_def["tn"],
                "test2_fn": m2_def["fn"],
                # Test 1 (Historical Evaluation)
                "test1_acc_def": m1_def["accuracy"],
                "test1_acc_opt": m1_opt["accuracy"],
                "test1_bal_acc_def": m1_def["balanced_accuracy"],
                "test1_bal_acc_opt": m1_opt["balanced_accuracy"],
                "test1_f1_def": m1_def["f1"],
                "test1_f1_opt": m1_opt["f1"],
                "test1_macro_f1_def": m1_def["macro_f1"],
                "test1_macro_f1_opt": m1_opt["macro_f1"],
                "test1_precision_def": m1_def["precision"],
                "test1_precision_opt": m1_opt["precision"],
                "test1_recall_def": m1_def["recall"],
                "test1_recall_opt": m1_opt["recall"],
                "test1_pr_auc": m1_def["pr_auc"],
                "test1_roc_auc": m1_def["roc_auc"],
                "test1_brier": m1_def["brier_score"],
                "test1_fnr_def": m1_def["fnr"],
                "test1_fnr_opt": m1_opt["fnr"],
                "test1_fpr_def": m1_def["fpr"],
                "test1_fpr_opt": m1_opt["fpr"]
            })
            
            # Transition delay on Test 2
            trans = evaluate_transition_dynamics(test2_df, p_test2, threshold=0.50)
            transition_records.append({
                "algorithm": algo,
                "feature_set": fset,
                "entry_delay_min": trans["mean_entry_detection_delay_minutes"],
                "departure_delay_min": trans["mean_departure_detection_delay_minutes"],
                "max_entry_delay_min": trans["max_entry_detection_delay_minutes"]
            })
            
            # Noise sensitivity on Test 2
            noisy_res = evaluate_noise_sensitivity(model, scaler, test2_df, fset, threshold=0.50)
            noise_records.append({
                "algorithm": algo,
                "feature_set": fset,
                "clean_f1": m2_def["f1"],
                "noisy_f1": noisy_res["f1"],
                "f1_delta": noisy_res["f1"] - m2_def["f1"],
                "clean_acc": m2_def["accuracy"],
                "noisy_acc": noisy_res["accuracy"],
                "clean_brier": m2_def["brier_score"],
                "noisy_brier": noisy_res["brier_score"]
            })

    # Save benchmark CSV
    bench_df = pd.DataFrame(benchmark_records)
    bench_csv_path = os.path.join(EVAL_DIR, "model_comparison_benchmark.csv")
    bench_df.to_csv(bench_csv_path, index=False)
    print(f"\nSaved Comprehensive Benchmark to: {bench_csv_path}")
    
    # Save transitions and noise CSVs
    trans_df = pd.DataFrame(transition_records)
    trans_csv_path = os.path.join(EVAL_DIR, "transition_delay_results.csv")
    trans_df.to_csv(trans_csv_path, index=False)
    
    noise_df = pd.DataFrame(noise_records)
    noise_csv_path = os.path.join(EVAL_DIR, "noise_sensitivity_results.csv")
    noise_df.to_csv(noise_csv_path, index=False)

    # -------------------------------------------------------------------------
    # GENERATE PLOTS
    # -------------------------------------------------------------------------
    print("Generating Evaluation Visualizations...")
    
    # 1. ROC Curves (Test 2)
    plt.figure(figsize=(9, 7))
    for algo in algorithms:
        p = test2_predictions[(algo, "set_a")]
        fpr_arr, tpr_arr, _ = roc_curve(y_test2, p)
        auc_score = roc_auc_score(y_test2, p)
        plt.plot(fpr_arr, tpr_arr, label=f"{algo.replace('_', ' ').title()} Set A (AUC = {auc_score:.4f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    plt.xlabel("False Positive Rate (Empty classified as Occupied)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("AACI Module 1: ROC Curves on Forward-in-Time Benchmark (Test 2)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "roc_curves.png"), dpi=200)
    plt.close()

    # 2. Precision-Recall Curves (Test 2)
    plt.figure(figsize=(9, 7))
    for algo in algorithms:
        p = test2_predictions[(algo, "set_a")]
        p_arr, r_arr, _ = precision_recall_curve(y_test2, p)
        pr_score = auc(r_arr, p_arr)
        plt.plot(r_arr, p_arr, label=f"{algo.replace('_', ' ').title()} Set A (PR-AUC = {pr_score:.4f})", lw=2)
    plt.xlabel("Recall (Occupied Detection Rate)", fontsize=11)
    plt.ylabel("Precision (Occupied Accuracy)", fontsize=11)
    plt.title("AACI Module 1: Precision-Recall Curves on Forward Benchmark (Test 2)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "pr_curves.png"), dpi=200)
    plt.close()

    # 3. Calibration Curves (Test 2)
    plt.figure(figsize=(9, 7))
    for algo in algorithms:
        p = test2_predictions[(algo, "set_a")]
        prob_true, prob_pred = calibration_curve(y_test2, p, n_bins=10)
        brier = brier_score_loss(y_test2, p)
        plt.plot(prob_pred, prob_true, marker="o", label=f"{algo.replace('_', ' ').title()} Set A (Brier = {brier:.4f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (Empirical Occupancy)", fontsize=11)
    plt.title("AACI Module 1: Probability Calibration Curves on Test 2", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "calibration_curves.png"), dpi=200)
    plt.close()

    # 4. Feature Ablation / Comparison Plot (Set A vs Set B vs Set C across algorithms)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(algorithms))
    width = 0.25
    
    f1_set_a = [bench_df[(bench_df['algorithm'] == a) & (bench_df['feature_set'] == 'set_a')]['test2_f1_def'].values[0] for a in algorithms]
    f1_set_b = [bench_df[(bench_df['algorithm'] == a) & (bench_df['feature_set'] == 'set_b')]['test2_f1_def'].values[0] for a in algorithms]
    f1_set_c = [bench_df[(bench_df['algorithm'] == a) & (bench_df['feature_set'] == 'set_c')]['test2_f1_def'].values[0] for a in algorithms]
    
    plt.bar(x - width, f1_set_a, width, label="Set A (All 5 Sensors)", color="#2b5c8f")
    plt.bar(x, f1_set_c, width, label="Set C (Ablation: No HumidityRatio)", color="#4682b4")
    plt.bar(x + width, f1_set_b, width, label="Set B (Reduced BOM: Temp, RH, Hour)", color="#e07a5f")
    
    plt.ylabel("Test 2 Occupied F1-Score", fontsize=11)
    plt.title("AACI Module 1: Feature Set Comparison & Psychrometric Ablation", fontsize=13, fontweight="bold")
    plt.xticks(x, [a.replace("_", " ").title() for a in algorithms], fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "feature_importance.png"), dpi=200)
    plt.close()

    # 5. Threshold Trade-off Plot (XGBoost Set A)
    p_xgb = test2_predictions[("xgboost", "set_a")]
    taus = np.linspace(0.05, 0.95, 50)
    fnrs = []
    fprs = []
    f1s = []
    for tau in taus:
        m = compute_metrics(y_test2, p_xgb, threshold=tau)
        fnrs.append(m["fnr"])
        fprs.append(m["fpr"])
        f1s.append(m["f1"])
        
    plt.figure(figsize=(9, 6))
    plt.plot(taus, fnrs, "r-", lw=2.5, label="False Negative Rate (Occupant Discomfort)")
    plt.plot(taus, fprs, "b-", lw=2.5, label="False Positive Rate (Energy Waste)")
    plt.plot(taus, f1s, "g--", lw=2, label="Occupied F1-Score")
    opt_tau_xgb = thresh_map.get(("xgboost", "set_a"), 0.5)
    plt.axvline(opt_tau_xgb, color="purple", linestyle=":", label=f"Selected Threshold (tau* = {opt_tau_xgb:.2f})")
    plt.xlabel("Decision Threshold (tau)", fontsize=11)
    plt.ylabel("Rate / Score", fontsize=11)
    plt.title("AACI Module 1: Decision Threshold Tradeoff Curve (XGBoost Set A)", fontsize=13, fontweight="bold")
    plt.legend(loc="center right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "threshold_tradeoff.png"), dpi=200)
    plt.close()

    print("All evaluation plots generated successfully.")
    print("Execution complete.")


if __name__ == "__main__":
    main()

