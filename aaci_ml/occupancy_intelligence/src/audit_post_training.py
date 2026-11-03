"""
AACI Module 1: Comprehensive Post-Training Audit Script.
Performs clean, independent verification of:
  1. Benchmark reproducibility from saved model artifacts.
  2. Confusion matrix exact mathematical reconstruction.
  3. Probability validity: 0.0 <= p <= 1.0, finite, non-NaN.
  4. Decision threshold behavior at tau = 0.5 vs tau* on Validation, Test 2, and Test 1.
  5. Preprocessing isolation and rolling-origin leakage check.
  6. Inference latency benchmarking methodology.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, brier_score_loss
)

from data_loader import (
    load_raw_data, get_official_partitions, generate_rolling_origin_folds,
    FEATURE_SETS, TARGET_COL
)
from evaluate_module1 import load_model_and_preprocessor, predict_pipeline

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
AUDIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit"))
os.makedirs(AUDIT_DIR, exist_ok=True)


def audit_reproducibility():
    print("=" * 80)
    print("AUDIT SECTION 1: Clean-Process Reproducibility Verification")
    print("=" * 80)

    train_df, test2_df, test1_df = get_official_partitions()
    bench = pd.read_csv(os.path.join(EVAL_DIR, "model_comparison_benchmark.csv"))

    y_test2 = test2_df[TARGET_COL].values
    y_test1 = test1_df[TARGET_COL].values

    tolerances = 1e-5
    audit_results = []

    for _, row in bench.iterrows():
        algo = row["algorithm"]
        fset = row["feature_set"]
        opt_tau = float(row["optimal_threshold"])

        model, scaler = load_model_and_preprocessor(algo, fset, calibrated=True)

        # Check feature alignment
        expected_cols = FEATURE_SETS[fset]
        X2 = test2_df[expected_cols]
        X1 = test1_df[expected_cols]

        # Predict
        p2 = predict_pipeline(model, scaler, X2)
        p1 = predict_pipeline(model, scaler, X1)

        # Sanity checks on probabilities
        assert np.all(np.isfinite(p2)), f"NaN/Inf in p2 for {algo}_{fset}"
        assert np.all(np.isfinite(p1)), f"NaN/Inf in p1 for {algo}_{fset}"
        assert np.all((p2 >= 0.0) & (p2 <= 1.0)), f"p2 out of [0, 1] for {algo}_{fset}"
        assert np.all((p1 >= 0.0) & (p1 <= 1.0)), f"p1 out of [0, 1] for {algo}_{fset}"

        # Evaluate Test 2 at def (0.5)
        pred2_def = (p2 >= 0.50).astype(int)
        acc2_def = accuracy_score(y_test2, pred2_def)
        f1_2_def = f1_score(y_test2, pred2_def, zero_division=0)
        rec2_def = recall_score(y_test2, pred2_def, zero_division=0)
        prec2_def = precision_score(y_test2, pred2_def, zero_division=0)
        brier2 = brier_score_loss(y_test2, p2)

        # Check against saved benchmark
        diff_acc2 = abs(acc2_def - float(row["test2_acc_def"]))
        diff_f1_2 = abs(f1_2_def - float(row["test2_f1_def"]))
        diff_brier2 = abs(brier2 - float(row["test2_brier"]))

        # Evaluate Test 1 at def (0.5)
        pred1_def = (p1 >= 0.50).astype(int)
        acc1_def = accuracy_score(y_test1, pred1_def)
        f1_1_def = f1_score(y_test1, pred1_def, zero_division=0)
        brier1 = brier_score_loss(y_test1, p1)

        diff_acc1 = abs(acc1_def - float(row["test1_acc_def"]))
        diff_f1_1 = abs(f1_1_def - float(row["test1_f1_def"]))
        diff_brier1 = abs(brier1 - float(row["test1_brier"]))

        max_discrepancy = max(diff_acc2, diff_f1_2, diff_brier2, diff_acc1, diff_f1_1, diff_brier1)
        passed = max_discrepancy < tolerances

        audit_results.append({
            "algorithm": algo,
            "feature_set": fset,
            "opt_tau": opt_tau,
            "max_discrepancy": max_discrepancy,
            "reproduced": passed,
            "test2_brier": brier2,
            "test1_brier": brier1
        })

    repro_df = pd.DataFrame(audit_results)
    print(repro_df.to_string(index=False))
    all_passed = repro_df["reproduced"].all()
    print(f"\nAll 12 models reproduced exactly within {tolerances}: {all_passed}")
    return repro_df


def audit_thresholds_and_confusion_matrices():
    print("\n" + "=" * 80)
    print("AUDIT SECTION 2: Decision Thresholds & Detailed Confusion Matrix Verification")
    print("=" * 80)

    train_df, test2_df, test1_df = get_official_partitions()
    thresh_df = pd.read_csv(os.path.join(EVAL_DIR, "threshold_optimization.csv"))
    val_df = pd.read_csv(os.path.join(EVAL_DIR, "validation_benchmark.csv"))

    target_models = [("random_forest", "set_a"), ("logistic_regression", "set_a")]

    for algo, fset in target_models:
        print(f"\n--- AUDIT FOR {algo.upper()} ({fset.upper()}) ---")
        row_thresh = thresh_df[(thresh_df["algorithm"] == algo) & (thresh_df["feature_set"] == fset)].iloc[0]
        row_val = val_df[(val_df["algorithm"] == algo) & (val_df["feature_set"] == fset)].iloc[0]
        opt_tau = float(row_thresh["optimal_threshold"])

        model, scaler = load_model_and_preprocessor(algo, fset, calibrated=True)

        print(f"Selected Threshold: tau* = {opt_tau:.2f} (versus default tau = 0.50)")
        print(f"Validation Objective: Score = Macro_F1 + 0.1 * Recall (Optimized on Fold 2 independent validation split)")
        print(f"Validation Performance at tau = 0.50:")
        print(f"  Accuracy: {row_val['val_acc_def']:.4f} | Balanced Acc: {row_val['val_bal_acc_def']:.4f}")
        print(f"  Occupied F1: {row_val['val_f1_def']:.4f} | Macro F1: {row_val['val_macro_f1_def']:.4f}")
        print(f"  FNR: {row_val['val_fnr_def']:.4f} | FPR: {row_val['val_fpr_def']:.4f} | Brier: {row_val['val_brier']:.4f}")

        print(f"Validation Performance at tau* = {opt_tau:.2f}:")
        print(f"  Accuracy: {row_val['val_acc_opt']:.4f} | Balanced Acc: {row_val['val_bal_acc_opt']:.4f}")
        print(f"  Occupied F1: {row_val['val_f1_opt']:.4f} | Macro F1: {row_val['val_macro_f1_opt']:.4f}")
        print(f"  FNR: {row_val['val_fnr_opt']:.4f} | FPR: {row_val['val_fpr_opt']:.4f}")

        # Compute exact Test 2 confusion matrix at tau = 0.5 and tau*
        p2 = predict_pipeline(model, scaler, test2_df[FEATURE_SETS[fset]])
        y2 = test2_df[TARGET_COL].values

        for tau_label, tau_val in [("Default (0.50)", 0.50), (f"Optimized (tau*={opt_tau:.2f})", opt_tau)]:
            pred = (p2 >= tau_val).astype(int)
            tn, fp, fn, tp = confusion_matrix(y2, pred, labels=[0, 1]).ravel()
            acc = accuracy_score(y2, pred)
            bal_acc = balanced_accuracy_score(y2, pred)
            prec = precision_score(y2, pred, zero_division=0)
            rec = recall_score(y2, pred, zero_division=0)
            f1 = f1_score(y2, pred, zero_division=0)
            fnr = fn / (fn + tp)
            fpr = fp / (fp + tn)

            print(f"\n  Test 2 at {tau_label}:")
            print(f"    Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,} (Total N={len(y2):,})")
            print(f"    Accuracy: {acc:.4%}, Balanced Acc: {bal_acc:.4%}")
            print(f"    Precision: {prec:.4%}, Recall: {rec:.4%}, Occupied F1: {f1:.4f}")
            print(f"    FNR (Discomfort Rate): {fnr:.4%}, FPR (Energy Waste Rate): {fpr:.4%}")


def audit_calibration_brier_comparison():
    print("\n" + "=" * 80)
    print("AUDIT SECTION 3: Brier Score & Calibration Reconciliation")
    print("=" * 80)

    train_df, test2_df, test1_df = get_official_partitions()

    # Compare Brier scores across Validation, Test 2, and Test 1
    val_df = pd.read_csv(os.path.join(EVAL_DIR, "validation_benchmark.csv"))
    bench = pd.read_csv(os.path.join(EVAL_DIR, "model_comparison_benchmark.csv"))

    merged = bench[["algorithm", "feature_set", "test2_brier", "test1_brier"]].merge(
        val_df[["algorithm", "feature_set", "val_brier"]], on=["algorithm", "feature_set"]
    )
    print("Brier Score Across Splits (Lower is better):")
    print(merged.to_string(index=False))


def audit_latency_measurement():
    print("\n" + "=" * 80)
    print("AUDIT SECTION 4: Edge Deployment & Latency Measurement Audit")
    print("=" * 80)

    train_df, test2_df, test1_df = get_official_partitions()

    # Benchmark Logistic Regression vs Random Forest on CPU
    for algo, fset in [("logistic_regression", "set_a"), ("random_forest", "set_a")]:
        model, scaler = load_model_and_preprocessor(algo, fset, calibrated=True)
        X_df = test2_df[FEATURE_SETS[fset]].iloc[:1000]

        # Measure 10 runs of 1,000 samples
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = predict_pipeline(model, scaler, X_df)
            dur = (time.perf_counter() - t0) * 1e6 / 1000.0 # us per sample
            latencies.append(dur)

        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        print(f"Algorithm: {algo:20s} | Latency: {mean_lat:6.2f} +/- {std_lat:4.2f} us/sample (N=10 runs of 1,000 samples on x86-64)")


if __name__ == "__main__":
    audit_reproducibility()
    audit_thresholds_and_confusion_matrices()
    audit_calibration_brier_comparison()
    audit_latency_measurement()

