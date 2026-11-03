"""
Generates confusion matrix plots and exports feature_importance.csv.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data_loader import get_official_partitions, FEATURE_SETS, TARGET_COL
from evaluate_module1 import load_model_and_preprocessor, predict_pipeline

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def export_feature_importance():
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fi_rows = []
    for model_key, info in meta["models"].items():
        algo = info["algo_name"]
        fset = info["feature_set"]
        for feat, vals in info["feature_importances"].items():
            row = {"algorithm": algo, "feature_set": fset, "feature": feat}
            row.update(vals)
            fi_rows.append(row)

    fi_df = pd.DataFrame(fi_rows)
    fi_csv = os.path.join(EVAL_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_csv, index=False)
    print(f"Exported feature_importance.csv to {fi_csv}")


def generate_confusion_matrices():
    train_df, test2_df, test1_df = get_official_partitions()
    y_test2 = test2_df[TARGET_COL].values
    y_test1 = test1_df[TARGET_COL].values

    models = [
        ("random_forest", "set_a"),
        ("logistic_regression", "set_a"),
        ("xgboost", "set_a"),
        ("decision_tree", "set_a")
    ]

    # Test 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (algo, fset) in enumerate(models):
        m, s = load_model_and_preprocessor(algo, fset, calibrated=True)
        probs = predict_pipeline(m, s, test2_df[FEATURE_SETS[fset]])
        cm = confusion_matrix(y_test2, (probs >= 0.5).astype(int), labels=[0, 1])

        ax = axes[idx]
        title_str = algo.replace("_", " ").title() + " (Set A)"
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(title_str, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Unoccupied", "Pred Occupied"])
        ax.set_yticklabels(["True Unoccupied", "True Occupied"])

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices on Primary Forward Benchmark (Test 2, N=9,752)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path2 = os.path.join(EVAL_DIR, "confusion_matrices_test2.png")
    plt.savefig(cm_path2, dpi=200)
    plt.close()
    print(f"Generated {cm_path2}")

    # Test 1
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (algo, fset) in enumerate(models):
        m, s = load_model_and_preprocessor(algo, fset, calibrated=True)
        probs = predict_pipeline(m, s, test1_df[FEATURE_SETS[fset]])
        cm = confusion_matrix(y_test1, (probs >= 0.5).astype(int), labels=[0, 1])

        ax = axes[idx]
        title_str = algo.replace("_", " ").title() + " (Set A)"
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(title_str, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Unoccupied", "Pred Occupied"])
        ax.set_yticklabels(["True Unoccupied", "True Occupied"])

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices on Historical Evaluation (Test 1, N=2,665)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path1 = os.path.join(EVAL_DIR, "confusion_matrices_test1.png")
    plt.savefig(cm_path1, dpi=200)
    plt.close()
    print(f"Generated {cm_path1}")


if __name__ == "__main__":
    export_feature_importance()
    generate_confusion_matrices()

