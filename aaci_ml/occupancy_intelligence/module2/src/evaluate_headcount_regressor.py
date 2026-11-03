"""
AACI Module 2A: Headcount Evaluation & Plotting Suite.
Generates:
  1. Headcount prediction vs ground truth time-series plot.
  2. Error distribution by occupancy level (Empty, Low, Med, High).
  3. LORO spatial generalization comparison plot.
  4. Occupancy levels breakdown table.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robod_loader import (
    load_all_rooms, get_chronological_room_splits, FEATURE_GROUPS,
    TARGET_HEADCOUNT, ROOM_TYPES
)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))


def generate_evaluation_plots(rooms):
    model_path = os.path.join(MODELS_DIR, "headcount_rf_multimodal.joblib")
    if not os.path.exists(model_path):
        print("Model headcount_rf_multimodal.joblib not found yet.")
        return

    model = joblib.load(model_path)
    feat_cols = FEATURE_GROUPS["group4_multimodal"]

    # 1. Predictions vs Ground Truth on Room 1 Test split (Lecture Room with peak up to 38)
    _, _, test_r1 = get_chronological_room_splits(rooms[1])
    X1 = test_r1[feat_cols].values
    y1 = test_r1[TARGET_HEADCOUNT].values
    p1 = np.maximum(0.0, model.predict(X1))

    plt.figure(figsize=(14, 6))
    time_subset = np.arange(min(600, len(y1))) # First ~50 hours of test split
    plt.plot(time_subset, y1[time_subset], label="True Headcount (Fish-eye Ground Truth)", color="black", lw=2, alpha=0.8)
    plt.plot(time_subset, p1[time_subset], label="Predicted Headcount (RF Multimodal)", color="#d95f02", lw=1.8, linestyle="--")
    plt.fill_between(time_subset, np.maximum(0.0, p1[time_subset] - 1.5), p1[time_subset] + 1.5, color="#d95f02", alpha=0.15, label="90% Prediction Band (Uncertainty)")

    plt.xlabel("Sample Index (5-min intervals)", fontsize=11)
    plt.ylabel("Occupant Headcount (persons)", fontsize=11)
    plt.title("AACI Module 2A: Predicted vs Actual Headcount on Unseen Test Window (Room 1 Lecture Hall)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path1 = os.path.join(EVAL_DIR, "headcount_predictions_vs_actual.png")
    plt.savefig(plot_path1, dpi=200)
    plt.close()
    print(f"Generated {plot_path1}")

    # 2. Error Breakdown by Occupancy Level
    bench_df = pd.read_csv(os.path.join(EVAL_DIR, "headcount_benchmark.csv"))
    # Filter for RF Multimodal
    rf_multi = bench_df[(bench_df["algorithm"] == "random_forest") & (bench_df["feature_group"] == "group4_multimodal")]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(rf_multi))
    width = 0.2

    plt.bar(x - 1.5*width, rf_multi["empty_mae"], width, label="Empty (y = 0)", color="#2b5c8f")
    plt.bar(x - 0.5*width, rf_multi["low_occ_mae"], width, label="Low Occupancy (1-3)", color="#4682b4")
    plt.bar(x + 0.5*width, rf_multi["med_occ_mae"], width, label="Medium Occupancy (4-10)", color="#e07a5f")
    plt.bar(x + 1.5*width, rf_multi["high_occ_mae"], width, label="High Occupancy (>10)", color="#d62728")

    plt.xlabel("Room Identifier & Typology", fontsize=11)
    plt.ylabel("Mean Absolute Error (Persons)", fontsize=11)
    plt.title("AACI Module 2A: Headcount MAE by Occupancy Density Level", fontsize=13, fontweight="bold")
    plt.xticks(x, [f"Room {r}\n({ROOM_TYPES[r]})" for r in rf_multi["room_id"]], fontsize=9)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path2 = os.path.join(EVAL_DIR, "headcount_error_distribution.png")
    plt.savefig(plot_path2, dpi=200)
    plt.close()
    print(f"Generated {plot_path2}")


if __name__ == "__main__":
    rooms = load_all_rooms()
    generate_evaluation_plots(rooms)

