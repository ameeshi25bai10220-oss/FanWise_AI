"""
AACI Module 3A & 3C: Fan Energy & Duration Predictive Modeling
Strict chronological training and evaluation across residential smart fans.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try importing xgboost, fallback to GradientBoostingRegressor if needed
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    HAS_XGB = False

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rdd_loader import load_fan_telemetry, load_all_fans

MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), "models")
EVAL_DIR = os.path.join(os.path.dirname(SRC_DIR), "evaluation")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct rich temporal, environmental, and lag features for chronological prediction.
    Strictly causal: features at time t only use information up to time t to predict t+1.
    """
    df_feat = df.copy()
    
    # 1. Cyclic time encodings
    df_feat['sin_hour'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['cos_hour'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['sin_day'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7.0)
    df_feat['cos_day'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7.0)
    
    # 2. Autoregressive Lags (strictly causal shift)
    df_feat['lag_speed_1'] = df_feat['speed'].shift(1)
    df_feat['lag_opTime_1'] = df_feat['opTime'].shift(1)
    df_feat['lag_eSpent_Wh_1'] = df_feat['eSpent_Wh'].shift(1)
    df_feat['lag_temp_1'] = df_feat['temperature'].shift(1)
    df_feat['lag_hum_1'] = df_feat['humidity'].shift(1)
    
    # 3. Rolling window features (shift by 1 to prevent leak of current row)
    df_feat['roll_mean_speed_3'] = df_feat['speed'].shift(1).rolling(window=3, min_periods=1).mean()
    df_feat['roll_mean_temp_3'] = df_feat['temperature'].shift(1).rolling(window=3, min_periods=1).mean()
    df_feat['roll_mean_hum_3'] = df_feat['humidity'].shift(1).rolling(window=3, min_periods=1).mean()
    
    # 4. Target variables: NEXT interval energy (Wh) and operating duration (seconds)
    df_feat['target_energy_Wh'] = df_feat['eSpent_Wh'].shift(-1)
    df_feat['target_opTime_sec'] = df_feat['opTime'].shift(-1)
    
    # Drop first and last rows where lags or targets are NaN
    df_feat = df_feat.dropna(subset=['lag_speed_1', 'target_energy_Wh', 'target_opTime_sec']).reset_index(drop=True)
    return df_feat


FEATURE_COLS = [
    'temperature', 'humidity', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'is_weekend', 'mode',
    'lag_speed_1', 'lag_opTime_1', 'lag_eSpent_Wh_1', 'lag_temp_1', 'lag_hum_1',
    'roll_mean_speed_3', 'roll_mean_temp_3', 'roll_mean_hum_3'
]


def train_and_evaluate_fan_models(
    fan_id: str,
    df: pd.DataFrame
) -> Tuple[List[Dict], Dict]:
    """
    Train and evaluate multiple regressors under strict chronological partition (70% train, 15% val, 15% test).
    """
    df_feats = create_feature_matrix(df)
    n = len(df_feats)
    
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    df_train = df_feats.iloc[:train_end].copy()
    df_val = df_feats.iloc[train_end:val_end].copy()
    df_test = df_feats.iloc[val_end:].copy()
    
    X_train = df_train[FEATURE_COLS].values
    X_val = df_val[FEATURE_COLS].values
    X_test = df_test[FEATURE_COLS].values
    
    y_energy_train = df_train['target_energy_Wh'].values
    y_energy_val = df_val['target_energy_Wh'].values
    y_energy_test = df_test['target_energy_Wh'].values
    
    y_op_train = df_train['target_opTime_sec'].values
    y_op_val = df_val['target_opTime_sec'].values
    y_op_test = df_test['target_opTime_sec'].values
    
    # Strictly fit Scaler on Training fold only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Baseline_Persistence": None,
        "Baseline_Mean": None,
        "Ridge_Regression": Ridge(alpha=10.0),
        "Decision_Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random_Forest": RandomForestRegressor(n_estimators=50, max_depth=8, max_samples=0.7, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1) if HAS_XGB else XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.08, random_state=42)
    }
    
    results = []
    saved_artifacts = {}
    
    for name, model in models.items():
        # Predict Energy
        if name == "Baseline_Persistence":
            pred_val_energy = df_val['lag_eSpent_Wh_1'].values
            pred_test_energy = df_test['lag_eSpent_Wh_1'].values
            pred_test_op = df_test['lag_opTime_1'].values
        elif name == "Baseline_Mean":
            mean_e = np.mean(y_energy_train)
            mean_op = np.mean(y_op_train)
            pred_val_energy = np.full_like(y_energy_val, mean_e)
            pred_test_energy = np.full_like(y_energy_test, mean_e)
            pred_test_op = np.full_like(y_op_test, mean_op)
        else:
            # Scaled for Ridge, unscaled for trees
            X_tr = X_train_scaled if "Ridge" in name else X_train
            X_v = X_val_scaled if "Ridge" in name else X_val
            X_ts = X_test_scaled if "Ridge" in name else X_test
            
            # Fit Energy Regressor
            model.fit(X_tr, y_energy_train)
            pred_val_energy = model.predict(X_v)
            pred_test_energy = model.predict(X_ts)
            
            # Also train separate duration model for top tree candidate
            if name == "Random_Forest":
                op_model = RandomForestRegressor(n_estimators=50, max_depth=8, max_samples=0.7, random_state=42, n_jobs=-1)
                op_model.fit(X_tr, y_op_train)
                pred_test_op = op_model.predict(X_ts)
                saved_artifacts["op_model"] = op_model
            else:
                pred_test_op = np.full_like(y_op_test, np.mean(y_op_train))
                
            if name in ["Random_Forest", "Ridge_Regression", "XGBoost"]:
                saved_artifacts[f"energy_model_{name}"] = model

        # Metrics on Forward Test
        mae_energy = mean_absolute_error(y_energy_test, pred_test_energy)
        rmse_energy = np.sqrt(mean_squared_error(y_energy_test, pred_test_energy))
        r2_energy = r2_score(y_energy_test, pred_test_energy)
        
        # Energy saving opportunity estimation:
        # Expected saving = induction_baseline - modeled BLDC energy
        test_baseline_induction = df_test['baseline_induction_W'].values * (y_op_test / 3600.0)
        modeled_savings_mae = mean_absolute_error(test_baseline_induction - y_energy_test, test_baseline_induction - pred_test_energy)
        
        mae_op = mean_absolute_error(y_op_test, pred_test_op)
        rmse_op = np.sqrt(mean_squared_error(y_op_test, pred_test_op))
        
        results.append({
            "fan_id": fan_id,
            "room_type": df['room_type'].iloc[0],
            "model_name": name,
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "mae_energy_Wh": round(float(mae_energy), 4),
            "rmse_energy_Wh": round(float(rmse_energy), 4),
            "r2_energy": round(float(r2_energy), 4),
            "modeled_savings_mae_Wh": round(float(modeled_savings_mae), 4),
            "mae_duration_sec": round(float(mae_op), 2),
            "rmse_duration_sec": round(float(rmse_op), 2)
        })
        
    saved_artifacts["scaler"] = scaler
    saved_artifacts["feature_cols"] = FEATURE_COLS
    return results, saved_artifacts


def run_all_energy_benchmarks():
    """Execute energy prediction benchmarks across all 9 fans."""
    print("Loading all fans for Module 3 energy modeling...")
    fans = load_all_fans()
    
    all_results = []
    best_models = {}
    
    for fan_id, df in fans.items():
        print(f"\n--- Training Energy Models for {fan_id} ({len(df)} rows) ---")
        results, artifacts = train_and_evaluate_fan_models(fan_id, df)
        all_results.extend(results)
        
        # Save best model artifacts (Random Forest & Scaler)
        if "energy_model_Random_Forest" in artifacts:
            model_save_path = os.path.join(MODELS_DIR, f"{fan_id}_rf_energy_pipeline.joblib")
            joblib.dump({
                "fan_id": fan_id,
                "energy_model": artifacts["energy_model_Random_Forest"],
                "duration_model": artifacts.get("op_model"),
                "scaler": artifacts["scaler"],
                "feature_cols": artifacts["feature_cols"]
            }, model_save_path)
            print(f"Saved trained pipeline to: {model_save_path}")

    df_results = pd.DataFrame(all_results)
    benchmark_path = os.path.join(EVAL_DIR, "energy_model_benchmark.csv")
    df_results.to_csv(benchmark_path, index=False)
    print(f"\nSaved benchmark results to: {benchmark_path}")
    
    # Model comparison aggregated across all fans
    agg_summary = df_results.groupby('model_name').agg({
        "mae_energy_Wh": "mean",
        "rmse_energy_Wh": "mean",
        "r2_energy": "mean",
        "modeled_savings_mae_Wh": "mean"
    }).reset_index().sort_values(by="mae_energy_Wh")
    
    print("\n" + "="*80)
    print("AACI MODULE 3A/3C: MODEL PERFORMANCE OVERVIEW (AVERAGED ACROSS 9 FANS):")
    print(agg_summary.round(4).to_string(index=False))
    print("="*80)
    
    # Room category performance breakdown
    room_agg = df_results[df_results['model_name'] == 'Random_Forest'].groupby('room_type').agg({
        "mae_energy_Wh": "mean",
        "rmse_energy_Wh": "mean",
        "r2_energy": "mean"
    }).reset_index()
    print("\nRANDOM FOREST PERFORMANCE BY ROOM CATEGORY:")
    print(room_agg.round(4).to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    run_all_energy_benchmarks()

