"""
AACI V3 – Personalization Evaluation on Contributor-Separated Data
==================================================================
Atomberg Adaptive Comfort Intelligence (AACI)

Evaluates the personalization engine against the held-out test split
of the ASHRAE Global Thermal Comfort Database II (unseen contributors).

Evaluation Methodology:
1. Contributor-separated test split (GroupShuffleSplit, test_size=0.15, seed=42).
2. For each contributor cohort with >= 50 samples:
   - Provide k early observations as feedback: k in [0, 1, 3, 5, 10].
   - Evaluate on subsequent unseen observations (i >= 10) to ensure strictly
     identical test evaluation sets across all k.
3. Compare Generic model (k=0) vs Personalized model (k=1, 3, 5, 10).
4. Save benchmark metrics to 'personalization_benchmark.csv' and plot 'personalization_gain.png'.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import personalization engine from local folder
from personalization_engine import PersonalizationEngine

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2_DIR   = os.path.join(ROOT_DIR, "aaci_ml", "v2_improved")
V3_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "archive (3)", "ashrae_db2.01.csv")

MODEL_PATH    = os.path.join(V2_DIR, "aaci_v2_model.joblib")
PREPROC_PATH  = os.path.join(V2_DIR, "aaci_v2_preprocessor.joblib")
METADATA_PATH = os.path.join(V2_DIR, "aaci_v2_metadata.json")

CLASS_NAMES = ["Cold", "Neutral", "Hot"]

def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physics-grounded biometeorological features for the dataset."""
    d = df.copy()
    ta   = d["Air temperature (C)"].astype(float)
    rh   = d["Relative humidity (%)"].astype(float)
    v    = d["Air velocity (m/s)"].astype(float).fillna(0.1)
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
    
    # 4. Humidex
    d["humidex"] = ta + (5.0 / 9.0) * (e - 10.0)
    
    # 5. Convective Cooling Effect & Effective Temperature
    d["cooling_effect_c"] = 1.2 * np.sqrt(np.maximum(v - 0.15, 0.0))
    d["effective_temp_c"] = ta - d["cooling_effect_c"]
    
    # 6. Outdoor-Indoor Thermal Gradient
    d["delta_tout_tin"] = tout - ta
    
    # 7. Convective Heat Removal Potential
    d["convective_heat_exchange"] = v * (36.5 - ta)
    
    # 8. Evaporative Cooling Capacity Potential
    d["evaporative_cooling_potential"] = np.maximum(es - e, 0.0) * np.sqrt(np.maximum(v, 0.05))
    
    return d

def map_comfort_class(tsv: float) -> int:
    """Maps Thermal Sensation Vote to 0: Cold, 1: Neutral, 2: Hot."""
    if tsv <= -1.0:
        return 0
    elif tsv >= 1.0:
        return 2
    else:
        return 1

def main():
    print("=" * 78)
    print("AACI V3: PERSONALIZATION EVALUATION ON UNSEEN CONTRIBUTORS")
    print("=" * 78)

    # 1. Load trained V2 model and preprocessor
    print("\n[Step 1] Loading trained V2 XGBoost model and preprocessor...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"Loaded model: {metadata.get('model_type', 'XGBoost')} trained with {len(metadata.get('feature_names', []))} features.")

    # 2. Load dataset and reproduce held-out test split
    print("\n[Step 2] Loading ASHRAE DB2 dataset and extracting held-out test split...")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    df_clean = df_raw[df_raw["Thermal sensation"].notnull()].copy()
    df_clean["comfort_class"] = df_clean["Thermal sensation"].apply(map_comfort_class)
    
    groups = df_clean["Data contributor"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    _, test_idx = next(gss.split(df_clean, df_clean["comfort_class"], groups=groups))
    df_test = df_clean.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"Held-out test split: {len(df_test):,} samples across {df_test['Data contributor'].nunique()} unseen contributors.")

    # 3. Compute biometeorological features
    print("\n[Step 3] Computing biometeorological features for test split...")
    df_test_eng = compute_engineered_features(df_test)
    
    # 4. Generate generic model predictions and probability distributions
    print("\n[Step 4] Preprocessing test features and obtaining generic model probabilities...")
    X_test_prep = preprocessor.transform(df_test_eng)
    generic_probs = model.predict_proba(X_test_prep)
    df_test_eng["p_cold_gen"] = generic_probs[:, 0]
    df_test_eng["p_neu_gen"]  = generic_probs[:, 1]
    df_test_eng["p_hot_gen"]  = generic_probs[:, 2]

    # 5. Evaluate personalization across feedback levels k in [0, 1, 3, 5, 10]
    # For fair, strictly leakage-free evaluation, we require cohorts to have >= 50 samples,
    # reserving index i >= 10 as the held-out test evaluation window for ALL k conditions.
    print("\n[Step 5] Simulating online personalization across feedback levels k in [0, 1, 3, 5, 10]...")
    
    contributors = df_test_eng["Data contributor"].value_counts()
    eval_contributors = contributors[contributors >= 50].index.tolist()
    print(f"Qualified contributor cohorts (>= 50 samples): {len(eval_contributors)}")
    for c in eval_contributors:
        print(f"  - {c}: {contributors[c]:,} samples")

    k_values = [0, 1, 3, 5, 10]
    results = []

    # Map sensation class to feedback type
    class_to_feedback = {
        0: "FEELING_COLD",
        1: "FEELING_COMFORTABLE",
        2: "FEELING_HOT",
    }

    # Store predictions for each k
    benchmark_data = []

    for k in k_values:
        y_true_all = []
        y_pred_all = []
        
        for contrib in eval_contributors:
            c_df = df_test_eng[df_test_eng["Data contributor"] == contrib].copy().reset_index(drop=True)
            
            # Initialize a fresh PersonalizationEngine for this contributor cohort
            engine = PersonalizationEngine()
            user_id = f"cohort_{contrib}"
            
            # Step A: Feed k early observations as user feedback
            if k > 0:
                adaptation_samples = c_df.iloc[:k]
                for _, row in adaptation_samples.iterrows():
                    fb_type = class_to_feedback[int(row["comfort_class"])]
                    engine.record_feedback(
                        user_id=user_id,
                        feedback_type=fb_type,
                        air_temp_c=float(row["Air temperature (C)"]) if pd.notnull(row["Air temperature (C)"]) else 24.0,
                        relative_humidity=float(row["Relative humidity (%)"]) if pd.notnull(row["Relative humidity (%)"]) else 50.0,
                        current_speed=2,
                        reason=None,
                    )
            
            # Step B: Evaluate on strictly held-out observations (index >= 10)
            eval_samples = c_df.iloc[10:].copy()
            for _, row in eval_samples.iterrows():
                gen_p = {
                    "Cold": float(row["p_cold_gen"]),
                    "Neutral": float(row["p_neu_gen"]),
                    "Hot": float(row["p_hot_gen"]),
                }
                
                if k == 0:
                    pred_class = int(np.argmax([gen_p["Cold"], gen_p["Neutral"], gen_p["Hot"]]))
                else:
                    pers_res = engine.personalize_probabilities(
                        generic_probs=gen_p,
                        user_id=user_id,
                        air_temp_c=float(row["Air temperature (C)"]) if pd.notnull(row["Air temperature (C)"]) else 24.0,
                    )
                    pred_class = pers_res["class_id"]
                    
                y_true_all.append(int(row["comfort_class"]))
                y_pred_all.append(pred_class)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        acc = accuracy_score(y_true_all, y_pred_all)
        bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
        
        precisions = precision_score(y_true_all, y_pred_all, average=None)
        recalls = recall_score(y_true_all, y_pred_all, average=None)
        f1s = f1_score(y_true_all, y_pred_all, average=None)
        
        cm = confusion_matrix(y_true_all, y_pred_all)
        
        res = {
            "k_feedback": k,
            "eval_samples": len(y_true_all),
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "cold_precision": round(float(precisions[0]), 4),
            "cold_recall": round(float(recalls[0]), 4),
            "cold_f1": round(float(f1s[0]), 4),
            "neutral_precision": round(float(precisions[1]), 4),
            "neutral_recall": round(float(recalls[1]), 4),
            "neutral_f1": round(float(f1s[1]), 4),
            "hot_precision": round(float(precisions[2]), 4),
            "hot_recall": round(float(recalls[2]), 4),
            "hot_f1": round(float(f1s[2]), 4),
        }
        results.append(res)
        
        print(f"\n--- Feedback Level k={k} ---")
        print(f"  Eval Samples:      {len(y_true_all):,}")
        print(f"  Accuracy:          {acc*100:.2f}%")
        print(f"  Balanced Accuracy: {bal_acc*100:.2f}%")
        print(f"  Macro F1:          {macro_f1:.4f}")
        print(f"  Cold Recall / F1:  {recalls[0]*100:.2f}% / {f1s[0]:.4f}")
        print(f"  Neutral Recall/F1: {recalls[1]*100:.2f}% / {f1s[1]:.4f}")
        print(f"  Hot Recall / F1:   {recalls[2]*100:.2f}% / {f1s[2]:.4f}")

    # 6. Save results to CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(V3_DIR, "personalization_benchmark.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved personalization benchmark results to: {csv_path}")

    # 7. Generate visualization plot
    print("\n[Step 7] Generating visualization plot 'personalization_gain.png'...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_results["k_feedback"], df_results["accuracy"] * 100, marker="o", linewidth=2.5, color="#1f77b4", label="Accuracy (%)")
    plt.plot(df_results["k_feedback"], df_results["balanced_accuracy"] * 100, marker="s", linewidth=2.5, color="#2ca02c", label="Balanced Accuracy (%)")
    plt.plot(df_results["k_feedback"], df_results["macro_f1"] * 100, marker="^", linewidth=2.5, color="#ff7f0e", label="Macro F1 (x100)")
    plt.title("AACI V3: Personalization Performance Gain vs Feedback (k)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Initial Feedback Interactions (k)", fontsize=11)
    plt.ylabel("Score (%)", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(df_results["k_feedback"], df_results["cold_recall"] * 100, marker="o", linewidth=2, color="#3498db", label="Cold Recall (%)")
    plt.plot(df_results["k_feedback"], df_results["neutral_recall"] * 100, marker="s", linewidth=2, color="#2ecc71", label="Neutral Recall (%)")
    plt.plot(df_results["k_feedback"], df_results["hot_recall"] * 100, marker="^", linewidth=2, color="#e74c3c", label="Hot Recall (%)")
    plt.title("Per-Class Recall Trajectory with Personalization", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Initial Feedback Interactions (k)", fontsize=11)
    plt.ylabel("Recall (%)", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(V3_DIR, "personalization_gain.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved visualization plot to: {plot_path}")

    # 8. Print formatted summary table
    print("\n" + "=" * 78)
    print("PERSONALIZATION BENCHMARK SUMMARY TABLE")
    print("=" * 78)
    print(f"{'k Feedback':<12} | {'Accuracy':<10} | {'Bal Acc':<10} | {'Macro F1':<10} | {'Cold Rec':<10} | {'Hot Rec':<10}")
    print("-" * 78)
    for _, row in df_results.iterrows():
        print(f"{int(row['k_feedback']):<12} | {row['accuracy']*100:6.2f}%    | {row['balanced_accuracy']*100:6.2f}%    | {row['macro_f1']:8.4f}   | {row['cold_recall']*100:6.2f}%    | {row['hot_recall']*100:6.2f}%")
    print("=" * 78)

if __name__ == "__main__":
    main()

