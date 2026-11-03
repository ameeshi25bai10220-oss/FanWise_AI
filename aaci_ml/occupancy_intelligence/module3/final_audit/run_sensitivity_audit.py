"""
Comprehensive sensitivity analysis of backlog packet filtering on chronological energy prediction.
Applies cutoff consistently to the entire timeline BEFORE chronological splitting.
"""
import sys
import os
import glob
import pandas as pd
import numpy as np

SRC_DIR = os.path.abspath("aaci_ml/occupancy_intelligence/module3/src")
sys.path.insert(0, SRC_DIR)

from rdd_loader import load_fan_telemetry, load_all_fans
from train_energy_models import create_feature_matrix, FEATURE_COLS

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FINAL_AUDIT_DIR = os.path.abspath("aaci_ml/occupancy_intelligence/module3/final_audit")

CUTOFFS = [None, 1800, 900, 300] # None = unfiltered

sensitivity_records = []

fans = load_all_fans()

for cutoff in CUTOFFS:
    cutoff_label = f"opTime<={cutoff}s" if cutoff else "Unfiltered (All Clean)"
    print(f"\nEvaluating sensitivity: {cutoff_label}...")
    
    for fan_id, df_raw in fans.items():
        room_type = df_raw['room_type'].iloc[0]
        n_initial = len(df_raw)
        
        # Apply filter consistently on historical data before feature creation
        if cutoff is not None:
            df_filtered = df_raw[df_raw['opTime'] <= cutoff].copy()
        else:
            df_filtered = df_raw.copy()
            
        n_filtered = len(df_filtered)
        n_removed = n_initial - n_filtered
        pct_removed = round(n_removed / n_initial * 100.0, 2)
        
        # Create features
        df_feats = create_feature_matrix(df_filtered)
        n = len(df_feats)
        
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        df_train = df_feats.iloc[:train_end]
        df_val = df_feats.iloc[train_end:val_end]
        df_test = df_feats.iloc[val_end:]
        
        X_train = df_train[FEATURE_COLS].values
        X_test = df_test[FEATURE_COLS].values
        
        y_train = df_train['target_energy_Wh'].values
        y_test = df_test['target_energy_Wh'].values
        
        # 1. Historical Mean Baseline
        mean_val = np.mean(y_train)
        pred_mean = np.full_like(y_test, mean_val)
        mae_mean = mean_absolute_error(y_test, pred_mean)
        rmse_mean = np.sqrt(mean_squared_error(y_test, pred_mean))
        r2_mean = r2_score(y_test, pred_mean)
        
        # 2. Persistence Baseline
        pred_persist = df_test['lag_eSpent_Wh_1'].values
        mae_persist = mean_absolute_error(y_test, pred_persist)
        rmse_persist = np.sqrt(mean_squared_error(y_test, pred_persist))
        r2_persist = r2_score(y_test, pred_persist)
        
        # 3. Ridge Regression
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)
        mae_ridge = mean_absolute_error(y_test, pred_ridge)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
        r2_ridge = r2_score(y_test, pred_ridge)
        
        # 4. Random Forest
        rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        mae_rf = mean_absolute_error(y_test, pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
        r2_rf = r2_score(y_test, pred_rf)
        
        sensitivity_records.append({
            "cutoff_filter": cutoff_label,
            "fan_id": fan_id,
            "room_type": room_type,
            "total_records": n_initial,
            "records_retained": n_filtered,
            "records_removed": n_removed,
            "pct_removed": pct_removed,
            "test_samples": len(df_test),
            "mae_historical_mean": round(float(mae_mean), 4),
            "rmse_historical_mean": round(float(rmse_mean), 4),
            "r2_historical_mean": round(float(r2_mean), 4),
            "mae_persistence": round(float(mae_persist), 4),
            "rmse_persistence": round(float(rmse_persist), 4),
            "r2_persistence": round(float(r2_persist), 4),
            "mae_ridge": round(float(mae_ridge), 4),
            "rmse_ridge": round(float(rmse_ridge), 4),
            "r2_ridge": round(float(r2_ridge), 4),
            "mae_rf": round(float(mae_rf), 4),
            "rmse_rf": round(float(rmse_rf), 4),
            "r2_rf": round(float(r2_rf), 4)
        })

df_sens = pd.DataFrame(sensitivity_records)
sens_csv_path = os.path.join(FINAL_AUDIT_DIR, "backlog_filter_sensitivity.csv")
df_sens.to_csv(sens_csv_path, index=False)
print(f"Saved sensitivity results to: {sens_csv_path}")

# Summary aggregated by cutoff
summary = df_sens.groupby("cutoff_filter").agg({
    "pct_removed": "mean",
    "mae_historical_mean": "mean",
    "mae_persistence": "mean",
    "mae_rf": "mean",
    "rmse_historical_mean": "mean",
    "rmse_rf": "mean",
    "r2_rf": "mean"
}).round(4)

print("\n" + "="*80)
print("BACKLOG FILTER SENSITIVITY SUMMARY (ACROSS ALL 9 FANS):")
print(summary.to_string())
print("="*80)

