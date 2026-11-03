"""
Comprehensive Dataset Audit for Model 3: Energy Intelligence
Audits:
1. Mendeley Smart Fan Telemetry (RDD Datset/ - 9 CSVs)
2. Dryad Airflow Research Dataset (doi_10_6078_D1V67R__v20180717 - 2 CSVs)
3. ROBOD Building Dataset (SupplementaryData/ - 5 CSVs)
"""
import os
import glob
import json
import pandas as pd
import numpy as np

# 1. Mendeley Telemetry Dataset
mendeley_files = sorted(glob.glob("RDD Datset/*.csv"))
print(f"=== MENDELEY DATASET AUDIT ({len(mendeley_files)} files) ===")

mendeley_audit = []
for fpath in mendeley_files:
    fname = os.path.basename(fpath)
    fsize = os.path.getsize(fpath)
    df = pd.read_csv(fpath)
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}
    null_counts = df.isnull().sum().to_dict()
    
    # Content duplicates excluding S.No.
    content_cols = [c for c in cols if c != 'S.No.']
    content_dups = int(df.duplicated(subset=content_cols).sum())
    full_row_dups = int(df.duplicated().sum())
    
    # DateTime
    df['parsed_dt'] = pd.to_datetime(df['datetime'], format="%d/%m/%Y %H:%M", errors='coerce')
    valid_dt = df.dropna(subset=['parsed_dt'])
    first_ts = str(valid_dt['parsed_dt'].min()) if len(valid_dt) > 0 else "N/A"
    last_ts = str(valid_dt['parsed_dt'].max()) if len(valid_dt) > 0 else "N/A"
    unique_ts = int(valid_dt['parsed_dt'].nunique())
    
    # Sampling intervals
    dt_diff_sec = valid_dt['parsed_dt'].diff().dt.total_seconds().dropna()
    dt_median_sec = float(dt_diff_sec.median()) if len(dt_diff_sec) > 0 else 0.0
    dt_mode_sec = float(dt_diff_sec.mode().iloc[0]) if len(dt_diff_sec) > 0 else 0.0
    
    # Speeds and modes
    speeds = sorted([float(x) for x in df['speed'].dropna().unique()])
    modes = sorted([float(x) for x in df['mode'].dropna().unique()])
    
    # Sensor ranges
    temp_min, temp_max = float(df['temperature'].min()), float(df['temperature'].max())
    hum_min, hum_max = float(df['humidity'].min()), float(df['humidity'].max())
    
    # opTime, eSpent, eSaved
    op_min, op_max = float(df['opTime'].min()), float(df['opTime'].max())
    op_median = float(df['opTime'].median())
    spent_min, spent_max = float(df['eSpent'].min()), float(df['eSpent'].max())
    saved_min, saved_max = float(df['eSaved'].min()), float(df['eSaved'].max())
    
    # Power ratio test (eSpent / opTime)
    active = df[(df['speed'] > 0) & (df['opTime'] > 0)]
    implied_power_by_speed = {}
    implied_saved_by_speed = {}
    for s in speeds:
        if s > 0:
            sub = df[(df['speed'] == s) & (df['opTime'] > 0)]
            if len(sub) > 0:
                p_spent = (sub['eSpent'] / sub['opTime']).median()
                p_saved = (sub['eSaved'] / sub['opTime']).median()
                implied_power_by_speed[s] = p_spent
                implied_saved_by_speed[s] = p_saved
                
    mendeley_audit.append({
        "file_name": fname,
        "file_size_bytes": fsize,
        "rows": n_rows,
        "cols": n_cols,
        "column_names": cols,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "content_duplicates": content_dups,
        "full_row_duplicates": full_row_dups,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "unique_timestamps": unique_ts,
        "median_packet_interval_sec": dt_median_sec,
        "modal_packet_interval_sec": dt_mode_sec,
        "fan_id": os.path.splitext(fname)[0],
        "room_category": fname.split("_")[0],
        "speeds_present": speeds,
        "modes_present": modes,
        "temp_range": [temp_min, temp_max],
        "hum_range": [hum_min, hum_max],
        "opTime_range_sec": [op_min, op_max],
        "opTime_median_sec": op_median,
        "eSpent_range": [spent_min, spent_max],
        "eSaved_range": [saved_min, saved_max],
        "implied_power_W": implied_power_by_speed,
        "implied_saved_W": implied_saved_by_speed
    })
    print(f"Audited {fname}: {n_rows} rows, {fsize} bytes")

print("\n=== DRYAD AIRFLOW DATASET AUDIT ===")
dryad_files = sorted(glob.glob("doi_10_6078_D1V67R__v20180717/*.csv"))
dryad_audit = []
for fpath in dryad_files:
    fname = os.path.basename(fpath)
    fsize = os.path.getsize(fpath)
    df = pd.read_csv(fpath)
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    
    dryad_audit.append({
        "file_name": fname,
        "file_size_bytes": fsize,
        "rows": n_rows,
        "cols": n_cols,
        "column_sample": cols[:10],
        "total_columns": len(cols),
        "data_preview": df.iloc[:3, :5].to_dict()
    })
    print(f"Audited Dryad {fname}: {n_rows} rows x {n_cols} cols, {fsize} bytes")

print("\n=== SAVING AUDIT JSON RESULTS ===")
os.makedirs("aaci_ml/v3_energy_intelligence/audit", exist_ok=True)
with open("aaci_ml/v3_energy_intelligence/audit/mendeley_dataset_audit.json", "w") as f:
    json.dump(mendeley_audit, f, indent=2)
with open("aaci_ml/v3_energy_intelligence/audit/dryad_dataset_audit.json", "w") as f:
    json.dump(dryad_audit, f, indent=2)

print("Saved audit JSON files to aaci_ml/v3_energy_intelligence/audit/")
