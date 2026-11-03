"""
AACI Module 3: Comprehensive RDD Dataset Audit & Energy Semantics Verification
Audits all 9 residential smart fan telemetry files in 'RDD Datset/'
"""
import os
import glob
import json
import pandas as pd
import numpy as np

RDD_DIR = "RDD Datset"
AUDIT_DIR = os.path.join("aaci_ml", "occupancy_intelligence", "module3", "audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(RDD_DIR, "*.csv")))
print(f"Auditing {len(csv_files)} files from {RDD_DIR}...")

summary_records = []
speed_mode_records = []
power_verification_records = []

POWER_LOOKUP = {0: 0, 1: 10, 2: 15, 3: 20, 4: 27, 5: 48}
SAVED_LOOKUP = {0: 0, 1: 14, 2: 19, 3: 35, 4: 37, 5: 48}
INDUCTION_LOOKUP = {k: POWER_LOOKUP[k] + SAVED_LOOKUP[k] for k in POWER_LOOKUP}

for fpath in csv_files:
    fname = os.path.basename(fpath)
    room_type = fname.split("_")[0]
    fan_id = fname.split(".")[0]
    
    df = pd.read_csv(fpath)
    total_raw_rows = len(df)
    
    # Check exact duplicates (excluding S.No.)
    content_cols = [c for c in df.columns if c != 'S.No.']
    exact_duplicates = int(df.duplicated(subset=content_cols).sum())
    
    # Parse datetime
    df['dt'] = pd.to_datetime(df['datetime'], format="%d/%m/%Y %H:%M", errors='coerce')
    null_dates = int(df['dt'].isnull().sum())
    unique_timestamps = int(df['dt'].nunique())
    
    # Check lexicographical vs chronological sorting
    is_raw_string_sorted = bool(df['datetime'].is_monotonic_increasing)
    is_raw_chrono_sorted = bool(df['dt'].is_monotonic_increasing)
    
    # Filter valid dates for chronological stats
    df_valid = df.dropna(subset=['dt']).sort_values(by=['dt', 'S.No.']).reset_index(drop=True)
    first_dt = str(df_valid['dt'].min()) if len(df_valid) > 0 else "N/A"
    last_dt = str(df_valid['dt'].max()) if len(df_valid) > 0 else "N/A"
    
    if len(df_valid) > 0:
        total_days = (df_valid['dt'].max() - df_valid['dt'].min()).days + 1
        unique_calendar_dates = int(df_valid['dt'].dt.date.nunique())
    else:
        total_days = 0
        unique_calendar_dates = 0
        
    # Sensor quality checks
    bad_temp_255 = int((df['temperature'] == 255).sum())
    bad_hum_255 = int((df['humidity'] == 255).sum())
    temp_zero = int((df['temperature'] == 0).sum())
    hum_zero = int((df['humidity'] == 0).sum())
    
    # Valid physical subset
    valid_sensors = df[(df['temperature'] > 0) & (df['temperature'] < 60) & 
                       (df['humidity'] > 0) & (df['humidity'] <= 100)]
    valid_sensor_rows = len(valid_sensors)
    
    # opTime, eSpent, eSaved statistics
    op_min = float(df['opTime'].min())
    op_max = float(df['opTime'].max())
    op_median = float(df['opTime'].median())
    op_mean = float(df['opTime'].mean())
    
    spent_min = float(df['eSpent'].min())
    spent_max = float(df['eSpent'].max())
    spent_sum = float(df['eSpent'].sum())
    # Convert Joules to kWh
    spent_kwh = spent_sum / 3_600_000.0
    
    saved_sum = float(df['eSaved'].sum())
    saved_kwh = saved_sum / 3_600_000.0
    
    # Verify exact power formula match
    valid_speed = df['speed'].isin(POWER_LOOKUP.keys()) & df['opTime'].notnull()
    df_pw = df[valid_speed].copy()
    expected_spent = df_pw['speed'].map(POWER_LOOKUP) * df_pw['opTime']
    expected_saved = df_pw['speed'].map(SAVED_LOOKUP) * df_pw['opTime']
    
    spent_match_pct = float((df_pw['eSpent'] == expected_spent).mean() * 100.0)
    saved_match_pct = float((df_pw['eSaved'] == expected_saved).mean() * 100.0)
    
    summary_records.append({
        "fan_id": fan_id,
        "room_type": room_type,
        "file_name": fname,
        "total_raw_rows": total_raw_rows,
        "exact_duplicates": exact_duplicates,
        "exact_duplicate_pct": round(exact_duplicates / total_raw_rows * 100.0, 2),
        "unique_timestamps": unique_timestamps,
        "first_timestamp": first_dt,
        "last_timestamp": last_dt,
        "total_calendar_span_days": total_days,
        "unique_active_days": unique_calendar_dates,
        "raw_string_sorted": is_raw_string_sorted,
        "raw_chrono_sorted": is_raw_chrono_sorted,
        "sensor_err_temp255": bad_temp_255,
        "sensor_err_hum255": bad_hum_255,
        "temp_min": float(valid_sensors['temperature'].min()) if len(valid_sensors) > 0 else np.nan,
        "temp_max": float(valid_sensors['temperature'].max()) if len(valid_sensors) > 0 else np.nan,
        "hum_min": float(valid_sensors['humidity'].min()) if len(valid_sensors) > 0 else np.nan,
        "hum_max": float(valid_sensors['humidity'].max()) if len(valid_sensors) > 0 else np.nan,
        "opTime_median_sec": op_median,
        "opTime_mean_sec": round(op_mean, 2),
        "total_eSpent_raw_Joules": spent_sum,
        "total_eSpent_kWh": round(spent_kwh, 3),
        "total_eSaved_kWh": round(saved_kwh, 3),
        "power_formula_match_pct": round(spent_match_pct, 2),
        "saved_formula_match_pct": round(saved_match_pct, 2)
    })
    
    # Speed and Mode breakdown
    for spd in sorted(df['speed'].dropna().unique()):
        sub_spd = df[df['speed'] == spd]
        count = len(sub_spd)
        pct = round(count / total_raw_rows * 100.0, 2)
        mean_op = round(float(sub_spd['opTime'].mean()), 2)
        median_op = round(float(sub_spd['opTime'].median()), 2)
        rate_spent = round(float((sub_spd['eSpent'] / sub_spd['opTime'].replace(0, np.nan)).median()), 2)
        rate_saved = round(float((sub_spd['eSaved'] / sub_spd['opTime'].replace(0, np.nan)).median()), 2)
        
        speed_mode_records.append({
            "fan_id": fan_id,
            "room_type": room_type,
            "speed": int(spd) if pd.notnull(spd) else -1,
            "sample_count": count,
            "percentage": pct,
            "mean_opTime_sec": mean_op,
            "median_opTime_sec": median_op,
            "empirical_spent_rate_W": rate_spent,
            "empirical_saved_rate_W": rate_saved,
            "theoretical_spent_W": POWER_LOOKUP.get(int(spd), np.nan),
            "theoretical_saved_W": SAVED_LOOKUP.get(int(spd), np.nan),
            "implied_induction_W": INDUCTION_LOOKUP.get(int(spd), np.nan)
        })

df_summary = pd.DataFrame(summary_records)
df_speed_mode = pd.DataFrame(speed_mode_records)

summary_csv_path = os.path.join(AUDIT_DIR, "rdd_dataset_audit_summary.csv")
speed_csv_path = os.path.join(AUDIT_DIR, "rdd_fan_speed_modes.csv")

df_summary.to_csv(summary_csv_path, index=False)
df_speed_mode.to_csv(speed_csv_path, index=False)

print(f"Saved audit summary to: {summary_csv_path}")
print(f"Saved speed breakdown to: {speed_csv_path}")

# Print high-level overview
print("\n" + "="*80)
print("RDD DATASET AUDIT OVERVIEW:")
print(f"Total Fans Audited: {len(df_summary)}")
print(f"Total Raw Records across all fans: {df_summary['total_raw_rows'].sum():,}")
print(f"Total Exact Duplicate Records: {df_summary['exact_duplicates'].sum():,} ({df_summary['exact_duplicates'].sum() / df_summary['total_raw_rows'].sum() * 100:.2f}%)")
print(f"Total Effective Unique Records: {df_summary['total_raw_rows'].sum() - df_summary['exact_duplicates'].sum():,}")
print(f"Total Monitored Energy Spent: {df_summary['total_eSpent_kWh'].sum():.2f} kWh")
print(f"Total Modeled Energy Saved: {df_summary['total_eSaved_kWh'].sum():.2f} kWh")
print(f"Power formula match across all fans: {df_summary['power_formula_match_pct'].min()}% - {df_summary['power_formula_match_pct'].max()}%")
print("="*80)

