"""
AACI Model 3 – Master Training & Benchmarking Script
===================================================
Executes Phase 3 training and comprehensive benchmarking for ceiling fan
energy consumption forecasting.

Usage:
    python aaci_ml/v3_energy_intelligence/train_energy_models.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_all_mendeley_data
from src.feature_engineering import build_supervised_energy_dataset
from src.model_trainer import EnergyModelTrainer


def main():
    print("=" * 80)
    print("AACI MODEL 3 – PHASE 3: CONTROLLED MODEL TRAINING & BENCHMARKING")
    print("=" * 80)
    t_start = time.time()

    # Step 1: Load and clean Mendeley dataset
    print("\n[Step 1/4] Loading and cleaning 9-fan Mendeley telemetry...")
    t0 = time.time()
    data_dir = os.path.join(PROJECT_ROOT, "RDD Datset")
    df = load_all_mendeley_data(data_dir, apply_cleaning=True, drop_sensor_errors=True)
    print(f"  Loaded {len(df):,} raw valid records in {time.time() - t0:.2f}s.")

    # Filter out offline reconnection accumulation packets (>900s) for standard interval forecasting
    df_standard = df[~df["is_backlog"]].copy().reset_index(drop=True)
    print(f"  Standard interval records (opTime <= 900s): {len(df_standard):,}")

    # Step 2: Build causal, leak-free supervised dataset
    print("\n[Step 2/4] Constructing causal feature matrix and next-packet target...")
    t0 = time.time()
    dataset_df, feature_cols, target_col = build_supervised_energy_dataset(
        df_standard, target_type="next_energy", drop_burn_in_nans=True
    )
    print(f"  Supervised dataset: {len(dataset_df):,} samples x {len(feature_cols)} features in {time.time() - t0:.2f}s.")
    print(f"  Target: '{target_col}' | Mean = {dataset_df[target_col].mean():.4f} Wh | Std = {dataset_df[target_col].std():.4f} Wh")

    # Step 3: Initialize Trainer
    trainer = EnergyModelTrainer(random_state=42)

    # Step 4: Run Evaluation Protocols
    # Protocol 1: Chronological 70/15/15
    chrono_results, trained_models = trainer.train_chronological_benchmark(
        dataset_df, feature_cols=feature_cols, target_col=target_col
    )

    # Protocol 2: Leave-One-Fan-Out (LOFO) Zero-Shot
    lofo_results = trainer.train_lofo_benchmark(
        dataset_df, feature_cols=feature_cols, target_col=target_col
    )

    # Protocol 3: Few-Shot Adaptation (7-Day Calibration)
    few_shot_results = trainer.train_few_shot_benchmark(
        dataset_df, feature_cols=feature_cols, target_col=target_col, adaptation_days=7
    )

    # Step 5: Save Model Metadata
    metadata = {
        "model_name": "AACI Model 3 – Energy Intelligence (LightGBM Primary Benchmark)",
        "target_variable": target_col,
        "target_units": "Watt-hours (Wh)",
        "target_description": "Electrical energy consumed during next observed telemetry packet",
        "forecasting_horizon": "Variable packet-to-packet (median ~60-150s, bounded <= 900s)",
        "evaluated_models": [
            "Persistence Baseline",
            "Global Historical Mean",
            "Fan-Specific Mean",
            "Ridge Regression",
            "Random Forest",
            "LightGBM Regressor",
            "XGBoost Regressor",
        ],
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "categorical_features": ["room_category"],
        "training_samples_chronological": int(len(dataset_df) * 0.7),
        "test_samples_chronological": int(len(dataset_df) * 0.15),
        "chronological_test_results": chrono_results.to_dict(orient="records"),
        "lofo_mean_summary": {
            "mean_persistence_mae_wh": round(float(lofo_results["mae_persistence_wh"].mean()), 4),
            "mean_historical_mean_mae_wh": round(float(lofo_results["mae_mean_wh"].mean()), 4),
            "mean_ridge_mae_wh": round(float(lofo_results["mae_ridge_wh"].mean()), 4),
            "mean_ridge_r2": round(float(lofo_results["r2_ridge"].mean()), 4),
            "mean_lgbm_mae_wh": round(float(lofo_results["mae_lgbm_wh"].mean()), 4),
            "mean_lgbm_r2": round(float(lofo_results["r2_lgbm"].mean()), 4),
        },
        "few_shot_mean_summary": {
            "mean_zero_shot_mae_wh": round(float(few_shot_results["mae_zero_shot_wh"].mean()), 4),
            "mean_few_shot_mae_wh": round(float(few_shot_results["mae_few_shot_wh"].mean()), 4),
            "mean_adaptation_gain_pct": round(float(few_shot_results["adaptation_improvement_pct"].mean()), 2),
        },
        "total_runtime_seconds": round(time.time() - t_start, 2),
    }

    meta_path = os.path.join(trainer.models_dir, "aaci_energy_model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 3 TRAINING & BENCHMARKING COMPLETE!")
    print(f"Total Elapsed Time: {time.time() - t_start:.2f} seconds")
    print(f"Saved Models:     {trainer.models_dir}")
    print(f"Saved Benchmarks: {trainer.benchmarks_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

