"""
AACI Module 2A: Spatial Headcount Estimation (ROBOD Dataset).
Trains and evaluates regression algorithms to predict 'occupant_count [number]':
  - Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor

Evaluates across 5 Feature Configurations:
  1. group1_env (Environmental sensors only)
  2. group2_env_acoustic (Environmental + Acoustic)
  3. group3_env_acoustic_wifi (Environmental + Acoustic + Wi-Fi)
  4. group4_multimodal (Full Multimodal)
  5. group5_bms_experimental (Building-Operation / HVAC Leakage Investigation)

Evaluation Modes:
  - Mode 1: Per-Room Chronological Evaluation (70% Train, 15% Val, 15% Forward Test)
  - Mode 2: Pooled Chronological Evaluation across all rooms
  - Mode 3: Leave-One-Room-Out (LORO) Spatial Generalization to completely unseen rooms

Strict Constraints:
  - 'occupant_presence [binary]' is NEVER an input.
  - Zero random shuffling; strict chronological splits.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import xgboost as xgb

from robod_loader import (
    load_all_rooms, load_single_room, get_chronological_room_splits,
    get_loro_splits, FEATURE_GROUPS, TARGET_HEADCOUNT, ROOM_TYPES
)

RANDOM_SEED = 42
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def get_regressor(algo_name: str) -> Any:
    """Instantiate fresh base regressor."""
    if algo_name == "ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_SEED)
    elif algo_name == "decision_tree":
        return DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=RANDOM_SEED)
    elif algo_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1)
    elif algo_name == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown regressor algorithm: {algo_name}")


def compute_headcount_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression and operational headcount metrics."""
    # Clip negative predictions to 0
    y_pred_clipped = np.maximum(0.0, y_pred)
    y_pred_round = np.round(y_pred_clipped)

    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    r2 = r2_score(y_true, y_pred_clipped)
    medae = median_absolute_error(y_true, y_pred_clipped)

    exact_acc = float(np.mean(y_true == y_pred_round))
    within_one_acc = float(np.mean(np.abs(y_true - y_pred_clipped) <= 1.0))

    # Empty room performance (y_true == 0)
    empty_mask = (y_true == 0)
    if np.any(empty_mask):
        empty_mae = float(mean_absolute_error(y_true[empty_mask], y_pred_clipped[empty_mask]))
        empty_exact_acc = float(np.mean(y_pred_round[empty_mask] == 0))
    else:
        empty_mae = 0.0
        empty_exact_acc = 1.0

    # Occupancy levels performance
    # Low (1-3)
    low_mask = (y_true >= 1) & (y_true <= 3)
    low_mae = float(mean_absolute_error(y_true[low_mask], y_pred_clipped[low_mask])) if np.any(low_mask) else 0.0

    # Medium (4-10)
    med_mask = (y_true >= 4) & (y_true <= 10)
    med_mae = float(mean_absolute_error(y_true[med_mask], y_pred_clipped[med_mask])) if np.any(med_mask) else 0.0

    # High (>10)
    high_mask = (y_true > 10)
    high_mae = float(mean_absolute_error(y_true[high_mask], y_pred_clipped[high_mask])) if np.any(high_mask) else 0.0

    # Uncertainty / Residual std
    residuals = y_pred_clipped - y_true
    resid_std = float(np.std(residuals))
    # 90% prediction interval coverage (approx +/- 1.645 * std)
    z90 = 1.645 * resid_std
    coverage_90 = float(np.mean(np.abs(residuals) <= z90))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "medae": float(medae),
        "exact_acc": float(exact_acc),
        "within_one_acc": float(within_one_acc),
        "empty_mae": float(empty_mae),
        "empty_exact_acc": float(empty_exact_acc),
        "low_occ_mae": float(low_mae),
        "med_occ_mae": float(med_mae),
        "high_occ_mae": float(high_mae),
        "resid_std": float(resid_std),
        "coverage_90": float(coverage_90)
    }


def run_per_room_chronological_experiments(rooms: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Evaluates algorithms and feature groups per room using forward chronological splits."""
    algorithms = ["ridge", "decision_tree", "random_forest", "xgboost"]
    feature_groups = ["group1_env", "group2_env_acoustic", "group3_env_acoustic_wifi", "group4_multimodal", "group5_bms_experimental"]

    records = []
    total_runs = len(rooms) * len(algorithms) * len(feature_groups)
    run_cnt = 1

    for r_id, df in rooms.items():
        train_df, val_df, test_df = get_chronological_room_splits(df)

        for algo in algorithms:
            for fgrp in feature_groups:
                feat_cols = FEATURE_GROUPS[fgrp]
                X_tr = train_df[feat_cols].copy()
                y_tr = train_df[TARGET_HEADCOUNT].values
                X_te = test_df[feat_cols].copy()
                y_te = test_df[TARGET_HEADCOUNT].values

                scaler = None
                if algo == "ridge":
                    scaler = StandardScaler()
                    X_tr_proc = scaler.fit_transform(X_tr)
                    X_te_proc = scaler.transform(X_te)
                else:
                    X_tr_proc = X_tr.values
                    X_te_proc = X_te.values

                model = get_regressor(algo)
                t0 = time.perf_counter()
                model.fit(X_tr_proc, y_tr)
                train_time = time.perf_counter() - t0

                y_pred = model.predict(X_te_proc)
                m = compute_headcount_metrics(y_te, y_pred)

                records.append({
                    "experiment": "per_room_forward",
                    "room_id": r_id,
                    "room_type": ROOM_TYPES[r_id],
                    "algorithm": algo,
                    "feature_group": fgrp,
                    "train_time_sec": round(train_time, 4),
                    "test_mae": round(m["mae"], 3),
                    "test_rmse": round(m["rmse"], 3),
                    "test_r2": round(m["r2"], 3),
                    "test_medae": round(m["medae"], 3),
                    "exact_acc": round(m["exact_acc"], 4),
                    "within_one_acc": round(m["within_one_acc"], 4),
                    "empty_mae": round(m["empty_mae"], 3),
                    "empty_exact_acc": round(m["empty_exact_acc"], 4),
                    "low_occ_mae": round(m["low_occ_mae"], 3),
                    "med_occ_mae": round(m["med_occ_mae"], 3),
                    "high_occ_mae": round(m["high_occ_mae"], 3),
                    "resid_std": round(m["resid_std"], 3),
                    "coverage_90": round(m["coverage_90"], 4)
                })

                if run_cnt % 25 == 0 or run_cnt == total_runs:
                    print(f"  [{run_cnt}/{total_runs}] Per-Room: Room {r_id} | {algo} | {fgrp} -> MAE: {m['mae']:.2f}, R2: {m['r2']:.2f}")
                run_cnt += 1

    return pd.DataFrame(records)


def run_loro_generalization_experiments(rooms: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Leave-One-Room-Out (LORO) Generalization:
    Trains on 4 rooms, evaluates on the 5th unseen room.
    Uses the leading model architecture across feature groups.
    """
    algorithms = ["random_forest", "xgboost", "ridge"]
    feature_groups = ["group1_env", "group2_env_acoustic", "group3_env_acoustic_wifi", "group4_multimodal"]

    records = []
    for test_r_id in range(1, 6):
        train_df, test_df = get_loro_splits(rooms, test_r_id)

        for algo in algorithms:
            for fgrp in feature_groups:
                feat_cols = FEATURE_GROUPS[fgrp]
                X_tr = train_df[feat_cols].copy()
                y_tr = train_df[TARGET_HEADCOUNT].values
                X_te = test_df[feat_cols].copy()
                y_te = test_df[TARGET_HEADCOUNT].values

                scaler = None
                if algo == "ridge":
                    scaler = StandardScaler()
                    X_tr_proc = scaler.fit_transform(X_tr)
                    X_te_proc = scaler.transform(X_te)
                else:
                    X_tr_proc = X_tr.values
                    X_te_proc = X_te.values

                model = get_regressor(algo)
                model.fit(X_tr_proc, y_tr)
                y_pred = model.predict(X_te_proc)
                m = compute_headcount_metrics(y_te, y_pred)

                records.append({
                    "test_room_id": test_r_id,
                    "test_room_type": ROOM_TYPES[test_r_id],
                    "algorithm": algo,
                    "feature_group": fgrp,
                    "unseen_room_mae": round(m["mae"], 3),
                    "unseen_room_rmse": round(m["rmse"], 3),
                    "unseen_room_r2": round(m["r2"], 3),
                    "within_one_acc": round(m["within_one_acc"], 4),
                    "empty_mae": round(m["empty_mae"], 3),
                    "high_occ_mae": round(m["high_occ_mae"], 3)
                })
        print(f"  Completed LORO evaluation for held-out Room {test_r_id} ({ROOM_TYPES[test_r_id]})")

    return pd.DataFrame(records)


def train_and_save_final_models(rooms: Dict[int, pd.DataFrame]):
    """
    Fits and saves production candidate models on the full pooled ROBOD dataset:
      - Best high-accuracy model (Random Forest Group 4)
      - Best lightweight/deployable model (Ridge Group 2)
      - Best environmental-only model (Random Forest Group 1)
    """
    # Pooled dataset with room_id encoded as one-hot or features
    dfs = [df for df in rooms.values()]
    pooled_df = pd.concat(dfs, ignore_index=True)
    y_full = pooled_df[TARGET_HEADCOUNT].values

    candidates = [
        ("rf_multimodal", "random_forest", "group4_multimodal"),
        ("rf_acoustic_wifi", "random_forest", "group3_env_acoustic_wifi"),
        ("rf_env_only", "random_forest", "group1_env"),
        ("ridge_acoustic", "ridge", "group2_env_acoustic"),
        ("xgb_multimodal", "xgboost", "group4_multimodal")
    ]

    saved_metadata = {}

    for name, algo, fgrp in candidates:
        feat_cols = FEATURE_GROUPS[fgrp]
        X_df = pooled_df[feat_cols].copy()

        scaler = None
        if algo == "ridge":
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X_df)
            joblib.dump(scaler, os.path.join(MODELS_DIR, f"headcount_scaler_{name}.joblib"))
        else:
            X_proc = X_df.values

        model = get_regressor(algo)
        t0 = time.perf_counter()
        model.fit(X_proc, y_full)
        train_time = time.perf_counter() - t0

        model_path = os.path.join(MODELS_DIR, f"headcount_{name}.joblib")
        joblib.dump(model, model_path)
        model_size_kb = os.path.getsize(model_path) / 1024.0

        # Feature importances
        fi = {}
        if hasattr(model, "feature_importances_"):
            for c, imp in zip(feat_cols, model.feature_importances_):
                fi[c] = round(float(imp), 4)
        elif hasattr(model, "coef_"):
            for c, coef in zip(feat_cols, model.coef_):
                fi[c] = round(float(coef), 4)

        saved_metadata[name] = {
            "algorithm": algo,
            "feature_group": fgrp,
            "features_count": len(feat_cols),
            "features": feat_cols,
            "train_time_sec": round(train_time, 3),
            "model_size_kb": round(model_size_kb, 1),
            "feature_importances": fi
        }
        print(f"Saved candidate model '{name}': {model_path} ({model_size_kb:.1f} KB)")

    meta_path = os.path.join(MODELS_DIR, "headcount_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(saved_metadata, f, indent=2)
    print(f"Saved headcount metadata to: {meta_path}")


def main():
    print("=" * 80)
    print("AACI Module 2A: Spatial Headcount Estimation (ROBOD Dataset)")
    print("=" * 80)

    rooms = load_all_rooms()

    print("\nPhase 1: Per-Room Forward Chronological Experiments...")
    per_room_df = run_per_room_chronological_experiments(rooms)
    bench_csv = os.path.join(EVAL_DIR, "headcount_benchmark.csv")
    per_room_df.to_csv(bench_csv, index=False)
    print(f"Saved per-room benchmark to: {bench_csv}")

    print("\nPhase 2: Leave-One-Room-Out (LORO) Generalization Experiments...")
    loro_df = run_loro_generalization_experiments(rooms)
    loro_csv = os.path.join(EVAL_DIR, "loro_generalization_benchmark.csv")
    loro_df.to_csv(loro_csv, index=False)
    print(f"Saved LORO benchmark to: {loro_csv}")

    print("\nPhase 3: Training & Saving Candidate Production Models on Pooled Dataset...")
    train_and_save_final_models(rooms)

    print("\nModule 2A Headcount Estimation Pipeline Complete.")


if __name__ == "__main__":
    main()

