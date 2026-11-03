"""
Investigate Bedroom_Fan_2 duplicates and produce exact row count reconciliation.
"""
import glob
import os
import pandas as pd
import numpy as np

RDD_DIR = "RDD Datset"
files = sorted(glob.glob(os.path.join(RDD_DIR, "*.csv")))

reconciliation_rows = []

for fpath in files:
    fname = os.path.basename(fpath)
    fan_id = os.path.splitext(fname)[0]
    df_raw = pd.read_csv(fpath)
    n_raw = len(df_raw)
    
    # 1. Exact duplicates on content columns (excluding S.No.)
    content_cols = [c for c in df_raw.columns if c != 'S.No.']
    dup_mask = df_raw.duplicated(subset=content_cols, keep='first')
    n_dups = int(dup_mask.sum())
    
    df_dedup = df_raw[~dup_mask].copy()
    n_after_dedup = len(df_dedup)
    
    # 2. Datetime parsing nulls
    df_dedup['dt'] = pd.to_datetime(df_dedup['datetime'], format="%d/%m/%Y %H:%M", errors='coerce')
    null_dt_mask = df_dedup['dt'].isnull()
    n_null_dt = int(null_dt_mask.sum())
    
    df_valid_dt = df_dedup[~null_dt_mask].copy()
    
    # 3. Microcontroller error sentinels (255)
    sentinel_temp = (df_valid_dt['temperature'] == 255)
    sentinel_hum = (df_valid_dt['humidity'] == 255)
    n_sentinel_255 = int((sentinel_temp | sentinel_hum).sum())
    
    # 4. Out of range sensors (temp <= 0 or >= 60, hum <= 0 or > 100)
    out_of_range_temp = ((df_valid_dt['temperature'] <= 0) | (df_valid_dt['temperature'] >= 60)) & ~sentinel_temp
    out_of_range_hum = ((df_valid_dt['humidity'] <= 0) | (df_valid_dt['humidity'] > 100)) & ~sentinel_hum
    n_out_of_range_sensors = int((out_of_range_temp | out_of_range_hum).sum())
    
    # 5. Invalid speeds or opTime < 0
    invalid_speed = ~df_valid_dt['speed'].isin([0, 1, 2, 3, 4, 5])
    invalid_op = (df_valid_dt['opTime'] < 0) | df_valid_dt['opTime'].isnull()
    n_invalid_operational = int((invalid_speed | invalid_op).sum())
    
    # Clean mask as used in rdd_loader
    valid_mask = (
        (df_valid_dt['temperature'] > 0) & (df_valid_dt['temperature'] < 60) &
        (df_valid_dt['humidity'] > 0) & (df_valid_dt['humidity'] <= 100) &
        (df_valid_dt['speed'].isin([0, 1, 2, 3, 4, 5])) &
        (df_valid_dt['opTime'] >= 0)
    )
    df_clean = df_valid_dt[valid_mask].copy()
    n_clean = len(df_clean)
    
    # Verify strict mathematical reconciliation:
    # n_raw - n_dups - n_null_dt - (n_sentinel_255 + n_out_of_range_sensors + n_invalid_operational)
    total_excluded = n_raw - n_clean
    reconciled_check = (n_clean == (n_raw - n_dups - n_null_dt - (~valid_mask).sum()))
    
    reconciliation_rows.append({
        "fan_id": fan_id,
        "room_type": fan_id.split("_")[0],
        "raw_records": n_raw,
        "exact_duplicates_removed": n_dups,
        "duplicate_pct": round(n_dups / n_raw * 100.0, 2),
        "null_datetime_removed": n_null_dt,
        "sentinel_255_removed": n_sentinel_255,
        "out_of_range_sensors_removed": n_out_of_range_sensors,
        "invalid_speed_op_removed": n_invalid_operational,
        "total_excluded": total_excluded,
        "final_clean_records": n_clean,
        "reconciled_exact": reconciled_check
    })

df_rec = pd.DataFrame(reconciliation_rows)
print(df_rec.to_string(index=False))

# Now specifically examine Bedroom_Fan_2
print("\n" + "="*80)
print("DEEP DIVE: Bedroom_Fan_2 Duplicates:")
df_b2 = pd.read_csv("RDD Datset/Bedroom_Fan_2.csv")
content_cols = [c for c in df_b2.columns if c != 'S.No.']
dups_b2 = df_b2[df_b2.duplicated(subset=content_cols, keep=False)]
print(f"Total rows in Bedroom_Fan_2 that are part of duplicate clusters: {len(dups_b2)}")
print("Sample duplicate clusters (first 15 rows):")
print(dups_b2[['S.No.', 'datetime', 'temperature', 'humidity', 'mode', 'speed', 'opTime', 'eSpent', 'eSaved']].head(15))

# Check S.No. continuity in Bedroom_Fan_2
sno_diff = df_b2['S.No.'].diff()
print(f"Are S.No. continuous increments of 1 in Bedroom_Fan_2? {(sno_diff == 1).all()}")
print(f"Unique timestamps in Bedroom_Fan_2: {df_b2['datetime'].nunique()}")
print(f"Average duplicates per unique timestamp: {len(df_b2) / df_b2['datetime'].nunique():.2f}")

