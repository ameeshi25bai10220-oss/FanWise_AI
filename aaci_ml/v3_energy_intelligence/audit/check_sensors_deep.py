"""
Phase 1.5 Deep Sensor Error & Range Verification across all 9 Mendeley Files
"""
import glob
import os
import pandas as pd
import numpy as np

files = sorted(glob.glob("RDD Datset/*.csv"))

sensor_audit = []

for fpath in files:
    fname = os.path.basename(fpath)
    fan_id = os.path.splitext(fname)[0]
    df = pd.read_csv(fpath)
    n = len(df)
    
    # 255 Sentinel counts
    temp_255 = int((df['temperature'] == 255).sum())
    hum_255 = int((df['humidity'] == 255).sum())
    both_255 = int(((df['temperature'] == 255) & (df['humidity'] == 255)).sum())
    only_temp_255 = int(((df['temperature'] == 255) & (df['humidity'] != 255)).sum())
    only_hum_255 = int(((df['temperature'] != 255) & (df['humidity'] == 255)).sum())
    
    # Check 0 or negative values
    temp_zero = int((df['temperature'] == 0).sum())
    hum_zero = int((df['humidity'] == 0).sum())
    temp_neg = int((df['temperature'] < 0).sum())
    hum_neg = int((df['humidity'] < 0).sum())
    
    # Check extreme high values (< 255 but > 60C, or > 100% RH)
    temp_high = int(((df['temperature'] > 60) & (df['temperature'] < 255)).sum())
    hum_high = int(((df['humidity'] > 100) & (df['humidity'] < 255)).sum())
    
    # Filter valid physical readings
    valid_mask = (df['temperature'] > 0) & (df['temperature'] <= 60) & (df['humidity'] > 0) & (df['humidity'] <= 100)
    df_valid = df[valid_mask]
    
    valid_t_min = float(df_valid['temperature'].min()) if len(df_valid) > 0 else np.nan
    valid_t_max = float(df_valid['temperature'].max()) if len(df_valid) > 0 else np.nan
    valid_t_mean = float(df_valid['temperature'].mean()) if len(df_valid) > 0 else np.nan
    
    valid_h_min = float(df_valid['humidity'].min()) if len(df_valid) > 0 else np.nan
    valid_h_max = float(df_valid['humidity'].max()) if len(df_valid) > 0 else np.nan
    valid_h_mean = float(df_valid['humidity'].mean()) if len(df_valid) > 0 else np.nan
    
    sensor_audit.append({
        "fan_id": fan_id,
        "total_records": n,
        "temp_255_count": temp_255,
        "hum_255_count": hum_255,
        "both_255_count": both_255,
        "only_temp_255": only_temp_255,
        "only_hum_255": only_hum_255,
        "temp_zero_count": temp_zero,
        "hum_zero_count": hum_zero,
        "temp_neg_count": temp_neg,
        "hum_neg_count": hum_neg,
        "temp_high_abnormal": temp_high,
        "hum_high_abnormal": hum_high,
        "valid_temp_min_C": valid_t_min,
        "valid_temp_max_C": valid_t_max,
        "valid_temp_mean_C": round(valid_t_mean, 2),
        "valid_hum_min_pct": valid_h_min,
        "valid_hum_max_pct": valid_h_max,
        "valid_hum_mean_pct": round(valid_h_mean, 2)
    })

df_sensor = pd.DataFrame(sensor_audit)
print(df_sensor[['fan_id', 'total_records', 'temp_255_count', 'hum_255_count', 'both_255_count', 'temp_zero_count', 'hum_zero_count', 'valid_temp_min_C', 'valid_temp_max_C', 'valid_hum_min_pct', 'valid_hum_max_pct']].to_string(index=False))

df_sensor.to_csv("aaci_ml/v3_energy_intelligence/audit/sensor_error_analysis.csv", index=False)

