"""
AACI Model 3 – Model Training & Benchmarking Engine
===================================================
Independent training, hyperparameter configuration, metric calculation,
and systematic benchmarking for ceiling fan energy forecasting.

Evaluated Architectures:
1. Persistence Baseline (most recent valid observed packet energy)
2. Global Historical Mean Baseline
3. Fan-Specific Historical Mean Baseline
4. Ridge Regression (linear regularized edge candidate)
5. Random Forest Regressor (non-linear ensemble benchmark)
6. LightGBM Regressor (fast, gradient-boosted decision trees)
7. XGBoost Regressor (high-capacity gradient boosting benchmark)

Evaluated Protocols:
- Protocol 1: Chronological 70/15/15 Within-Fan Benchmark
- Protocol 2: Leave-One-Fan-Out (LOFO) Zero-Shot Spatial Benchmark (all 9 fans)
- Protocol 3: Few-Shot Fan Adaptation Benchmark (7-day calibration)
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from .preprocessing import EnergyPreprocessor, CausalSensorImputer
from .validation import (
    temporal_train_val_test_split,
    leave_one_fan_out_split,
    few_shot_fan_split,
)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str,
    latency_us: float = 0.0,
    model_size_kb: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate comprehensive regression and domain-specific energy metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Standard metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Joules equivalent (1 Wh = 3600 Joules)
    mae_joules = mae * 3600.0

    # Non-zero operating metrics (packets where actual energy > 0.05 Wh)
    active_mask = y_true > 0.05
    if np.any(active_mask):
        y_true_act = y_true[active_mask]
        y_pred_act = y_pred[active_mask]
        ape = np.abs((y_true_act - y_pred_act) / y_true_act) * 100.0
        mape = float(np.mean(ape))
        mdape = float(np.median(ape))
    else:
        mape = 0.0
        mdape = 0.0

    # Inactive state prediction accuracy (both true and pred <= 0.05 Wh)
    inactive_mask = y_true <= 0.05
    if np.any(inactive_mask):
        inactive_correct = np.mean(y_pred[inactive_mask] <= 0.10) * 100.0
    else:
        inactive_correct = 100.0

    return {
        "model": model_name,
        "split": split_name,
        "mae_wh": round(mae, 4),
        "rmse_wh": round(rmse, 4),
        "r2_score": round(r2, 4),
        "mae_joules": round(mae_joules, 2),
        "mape_percent": round(mape, 2),
        "mdape_percent": round(mdape, 2),
        "inactive_accuracy_percent": round(inactive_correct, 2),
        "latency_us": round(latency_us, 2),
        "model_size_kb": round(model_size_kb, 2),
    }


class EnergyModelTrainer:
    """
    Orchestrates training, cross-validation, and benchmarking for Model 3.
    """

    def __init__(
        self,
        random_state: int = 42,
        models_dir: Optional[str] = None,
        benchmarks_dir: Optional[str] = None,
    ):
        self.random_state = random_state
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        self.benchmarks_dir = benchmarks_dir or os.path.join(base_dir, "benchmarks")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.benchmarks_dir, exist_ok=True)

    def train_chronological_benchmark(
        self,
        dataset_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_next_eSpent_wh",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Train and evaluate all models under Protocol 1: Chronological 70/15/15.
        """
        print("\n" + "=" * 70)
        print("PROTOCOL 1: CHRONOLOGICAL 70/15/15 WITHIN-FAN BENCHMARK")
        print("=" * 70)

        train_df, val_df, test_df = temporal_train_val_test_split(
            dataset_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, per_fan=True
        )

        print(f"Partitions: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

        # Preprocessor fit strictly on train
        preproc = EnergyPreprocessor(
            numeric_features=feature_cols,
            categorical_features=["room_category"] if "room_category" in train_df.columns else [],
        )
        X_train = preproc.fit_transform(train_df)
        X_val = preproc.transform(val_df)
        X_test = preproc.transform(test_df)

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        results = []
        trained_models = {}

        # 1. Baseline: Persistence
        # Uses most recent observed packet energy (eSpent_wh)
        y_pred_persist = test_df["eSpent_wh"].values
        res_persist = calculate_regression_metrics(
            y_test, y_pred_persist, "Persistence Baseline", "Chronological Test"
        )
        results.append(res_persist)
        print(f"  [1/7] Persistence Baseline: MAE={res_persist['mae_wh']} Wh, R2={res_persist['r2_score']}")

        # 2. Baseline: Global Historical Mean
        global_mean = float(y_train.mean())
        y_pred_mean = np.full_like(y_test, global_mean)
        res_mean = calculate_regression_metrics(
            y_test, y_pred_mean, "Global Historical Mean", "Chronological Test"
        )
        results.append(res_mean)
        print(f"  [2/7] Global Mean Baseline: MAE={res_mean['mae_wh']} Wh, R2={res_mean['r2_score']}")

        # 3. Baseline: Fan-Specific Historical Mean
        fan_means = train_df.groupby("fan_id")[target_col].mean().to_dict()
        y_pred_fan_mean = test_df["fan_id"].map(fan_means).fillna(global_mean).values
        res_fan_mean = calculate_regression_metrics(
            y_test, y_pred_fan_mean, "Fan-Specific Mean", "Chronological Test"
        )
        results.append(res_fan_mean)
        print(f"  [3/7] Fan-Specific Mean:    MAE={res_fan_mean['mae_wh']} Wh, R2={res_fan_mean['r2_score']}")

        # 4. Model Candidate: Ridge Regression
        t0 = time.time()
        ridge = Ridge(alpha=10.0, random_state=self.random_state)
        ridge.fit(X_train, y_train)
        fit_time_ridge = time.time() - t0

        t_infer = time.time()
        y_pred_ridge = ridge.predict(X_test)
        latency_ridge = (time.time() - t_infer) / len(X_test) * 1e6

        ridge_path = os.path.join(self.models_dir, "aaci_energy_ridge_model.joblib")
        joblib.dump(ridge, ridge_path)
        size_ridge_kb = os.path.getsize(ridge_path) / 1024.0

        res_ridge = calculate_regression_metrics(
            y_test, y_pred_ridge, "Ridge Regression", "Chronological Test",
            latency_us=latency_ridge, model_size_kb=size_ridge_kb
        )
        results.append(res_ridge)
        trained_models["ridge"] = ridge
        print(f"  [4/7] Ridge Regression:     MAE={res_ridge['mae_wh']} Wh, R2={res_ridge['r2_score']} (fit {fit_time_ridge:.2f}s)")

        # 5. Model Candidate: Random Forest Regressor
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X_train, y_train)
        fit_time_rf = time.time() - t0

        t_infer = time.time()
        y_pred_rf = rf.predict(X_test)
        latency_rf = (time.time() - t_infer) / len(X_test) * 1e6

        rf_path = os.path.join(self.models_dir, "aaci_energy_rf_model.joblib")
        joblib.dump(rf, rf_path)
        size_rf_kb = os.path.getsize(rf_path) / 1024.0

        res_rf = calculate_regression_metrics(
            y_test, y_pred_rf, "Random Forest", "Chronological Test",
            latency_us=latency_rf, model_size_kb=size_rf_kb
        )
        results.append(res_rf)
        trained_models["random_forest"] = rf
        print(f"  [5/7] Random Forest:        MAE={res_rf['mae_wh']} Wh, R2={res_rf['r2_score']} (fit {fit_time_rf:.2f}s)")

        # 6. Model Candidate: LightGBM Regressor (Primary Gradient Boosting Benchmark)
        t0 = time.time()
        lgb = LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        lgb.fit(X_train, y_train)
        fit_time_lgb = time.time() - t0

        t_infer = time.time()
        y_pred_lgb = lgb.predict(X_test)
        latency_lgb = (time.time() - t_infer) / len(X_test) * 1e6

        lgb_path = os.path.join(self.models_dir, "aaci_energy_lightgbm_model.joblib")
        joblib.dump(lgb, lgb_path)
        size_lgb_kb = os.path.getsize(lgb_path) / 1024.0

        res_lgb = calculate_regression_metrics(
            y_test, y_pred_lgb, "LightGBM Regressor", "Chronological Test",
            latency_us=latency_lgb, model_size_kb=size_lgb_kb
        )
        results.append(res_lgb)
        trained_models["lightgbm"] = lgb
        print(f"  [6/7] LightGBM Regressor:   MAE={res_lgb['mae_wh']} Wh, R2={res_lgb['r2_score']} (fit {fit_time_lgb:.2f}s)")

        # 7. Model Candidate: XGBoost Regressor
        t0 = time.time()
        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb.fit(X_train, y_train)
        fit_time_xgb = time.time() - t0

        t_infer = time.time()
        y_pred_xgb = xgb.predict(X_test)
        latency_xgb = (time.time() - t_infer) / len(X_test) * 1e6

        xgb_path = os.path.join(self.models_dir, "aaci_energy_xgboost_model.joblib")
        joblib.dump(xgb, xgb_path)
        size_xgb_kb = os.path.getsize(xgb_path) / 1024.0

        res_xgb = calculate_regression_metrics(
            y_test, y_pred_xgb, "XGBoost Regressor", "Chronological Test",
            latency_us=latency_xgb, model_size_kb=size_xgb_kb
        )
        results.append(res_xgb)
        trained_models["xgboost"] = xgb
        print(f"  [7/7] XGBoost Regressor:    MAE={res_xgb['mae_wh']} Wh, R2={res_xgb['r2_score']} (fit {fit_time_xgb:.2f}s)")

        # Save preprocessor
        preproc_path = os.path.join(self.models_dir, "aaci_energy_preprocessor.joblib")
        preproc.save(preproc_path)

        # Feature Importance Analysis (LightGBM)
        feat_names = preproc.get_feature_names_out()
        importances = lgb.feature_importances_
        feat_imp_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        feat_imp_path = os.path.join(self.benchmarks_dir, "feature_importance_ranking.csv")
        feat_imp_df.to_csv(feat_imp_path, index=False)

        results_df = pd.DataFrame(results)
        bench_path = os.path.join(self.benchmarks_dir, "chronological_benchmark_results.csv")
        results_df.to_csv(bench_path, index=False)

        return results_df, trained_models

    def train_lofo_benchmark(
        self,
        dataset_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_next_eSpent_wh",
    ) -> pd.DataFrame:
        """
        Evaluate zero-shot cross-fan generalization under Protocol 2: LOFO across all 9 fans.
        """
        print("\n" + "=" * 70)
        print("PROTOCOL 2: LEAVE-ONE-FAN-OUT (LOFO) ZERO-SHOT SPATIAL BENCHMARK")
        print("=" * 70)

        unique_fans = sorted(dataset_df["fan_id"].unique())
        lofo_results = []

        for idx, test_fan in enumerate(unique_fans, 1):
            train_df, test_df = leave_one_fan_out_split(dataset_df, test_fan_id=test_fan)
            room_cat = test_df["room_category"].iloc[0]

            # Fit preprocessor on 8 training fans
            preproc = EnergyPreprocessor(
                numeric_features=feature_cols,
                categorical_features=["room_category"],
            )
            X_train = preproc.fit_transform(train_df)
            X_test = preproc.transform(test_df)

            y_train = train_df[target_col].values
            y_test = test_df[target_col].values

            # Baselines
            y_pred_persist = test_df["eSpent_wh"].values
            mae_persist = mean_absolute_error(y_test, y_pred_persist)

            global_mean = float(y_train.mean())
            y_pred_mean = np.full_like(y_test, global_mean)
            mae_mean = mean_absolute_error(y_test, y_pred_mean)

            # Ridge
            ridge = Ridge(alpha=10.0, random_state=self.random_state)
            ridge.fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
            r2_ridge = r2_score(y_test, y_pred_ridge)

            # LightGBM
            lgb = LGBMRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.05,
                random_state=self.random_state, n_jobs=-1, verbose=-1
            )
            lgb.fit(X_train, y_train)
            y_pred_lgb = lgb.predict(X_test)
            mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
            r2_lgb = r2_score(y_test, y_pred_lgb)

            lofo_results.append({
                "test_fan_id": test_fan,
                "room_category": room_cat,
                "test_records": len(test_df),
                "mae_persistence_wh": round(mae_persist, 4),
                "mae_mean_wh": round(mae_mean, 4),
                "mae_ridge_wh": round(mae_ridge, 4),
                "r2_ridge": round(r2_ridge, 4),
                "mae_lgbm_wh": round(mae_lgb, 4),
                "r2_lgbm": round(r2_lgb, 4),
                "lgbm_improvement_over_mean_pct": round((mae_mean - mae_lgb) / mae_mean * 100.0, 2),
            })

            print(f"  Fold [{idx}/9] Fan: {test_fan:<18} ({room_cat:<12}) -> LightGBM MAE={mae_lgb:.4f} Wh (R2={r2_lgb:.4f})")

        lofo_df = pd.DataFrame(lofo_results)
        lofo_path = os.path.join(self.benchmarks_dir, "lofo_spatial_benchmark_results.csv")
        lofo_df.to_csv(lofo_path, index=False)

        print("\nLOFO Mean Summary Across 9 Fans:")
        print(f"  Mean Persistence MAE: {lofo_df['mae_persistence_wh'].mean():.4f} Wh")
        print(f"  Mean Historical Mean MAE: {lofo_df['mae_mean_wh'].mean():.4f} Wh")
        print(f"  Mean Ridge MAE:       {lofo_df['mae_ridge_wh'].mean():.4f} Wh (R2={lofo_df['r2_ridge'].mean():.4f})")
        print(f"  Mean LightGBM MAE:   {lofo_df['mae_lgbm_wh'].mean():.4f} Wh (R2={lofo_df['r2_lgbm'].mean():.4f})")

        return lofo_df

    def train_few_shot_benchmark(
        self,
        dataset_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_next_eSpent_wh",
        adaptation_days: int = 7,
    ) -> pd.DataFrame:
        """
        Evaluate few-shot personalization under Protocol 3: 8 base fans + 7-day calibration.
        """
        print("\n" + "=" * 70)
        print(f"PROTOCOL 3: FEW-SHOT FAN ADAPTATION BENCHMARK ({adaptation_days}-DAY CALIBRATION)")
        print("=" * 70)

        unique_fans = sorted(dataset_df["fan_id"].unique())
        few_shot_results = []

        for idx, target_fan in enumerate(unique_fans, 1):
            train_others, adapt_df, test_df = few_shot_fan_split(
                dataset_df, target_fan_id=target_fan, adaptation_days=adaptation_days
            )
            room_cat = test_df["room_category"].iloc[0]

            # 1. Zero-shot baseline (trained on others only)
            preproc_zs = EnergyPreprocessor(numeric_features=feature_cols, categorical_features=["room_category"])
            X_train_zs = preproc_zs.fit_transform(train_others)
            X_test_zs = preproc_zs.transform(test_df)
            y_train_zs = train_others[target_col].values
            y_test = test_df[target_col].values

            lgb_zs = LGBMRegressor(n_estimators=100, max_depth=8, learning_rate=0.05, random_state=self.random_state, n_jobs=-1, verbose=-1)
            lgb_zs.fit(X_train_zs, y_train_zs)
            y_pred_zs = lgb_zs.predict(X_test_zs)
            mae_zero_shot = mean_absolute_error(y_test, y_pred_zs)

            # 2. Few-shot adapted model (trained on others + first 7 days of target fan)
            train_combined = pd.concat([train_others, adapt_df], ignore_index=True)
            preproc_fs = EnergyPreprocessor(numeric_features=feature_cols, categorical_features=["room_category"])
            X_train_fs = preproc_fs.fit_transform(train_combined)
            X_test_fs = preproc_fs.transform(test_df)
            y_train_fs = train_combined[target_col].values

            lgb_fs = LGBMRegressor(n_estimators=100, max_depth=8, learning_rate=0.05, random_state=self.random_state, n_jobs=-1, verbose=-1)
            lgb_fs.fit(X_train_fs, y_train_fs)
            y_pred_fs = lgb_fs.predict(X_test_fs)
            mae_few_shot = mean_absolute_error(y_test, y_pred_fs)
            r2_few_shot = r2_score(y_test, y_pred_fs)

            improvement_pct = (mae_zero_shot - mae_few_shot) / mae_zero_shot * 100.0

            few_shot_results.append({
                "target_fan_id": target_fan,
                "room_category": room_cat,
                "adaptation_records": len(adapt_df),
                "test_records": len(test_df),
                "mae_zero_shot_wh": round(mae_zero_shot, 4),
                "mae_few_shot_wh": round(mae_few_shot, 4),
                "r2_few_shot": round(r2_few_shot, 4),
                "adaptation_improvement_pct": round(improvement_pct, 2),
            })

            print(f"  Fold [{idx}/9] Fan: {target_fan:<18} -> Zero-Shot MAE={mae_zero_shot:.4f} Wh | Few-Shot MAE={mae_few_shot:.4f} Wh ({improvement_pct:+.2f}%)")

        few_shot_df = pd.DataFrame(few_shot_results)
        fs_path = os.path.join(self.benchmarks_dir, "few_shot_adaptation_benchmark_results.csv")
        few_shot_df.to_csv(fs_path, index=False)

        print("\nFew-Shot Mean Summary Across 9 Fans:")
        print(f"  Mean Zero-Shot MAE: {few_shot_df['mae_zero_shot_wh'].mean():.4f} Wh")
        print(f"  Mean Few-Shot MAE:  {few_shot_df['mae_few_shot_wh'].mean():.4f} Wh")
        print(f"  Mean Adaptation Gain: {few_shot_df['adaptation_improvement_pct'].mean():+.2f}%")

        return few_shot_df

