"""
Phase 1.5 Deep Duplicate Investigation across all 9 Mendeley Files
"""
import glob
import os
import pandas as pd
import numpy as np

files = sorted(glob.glob("RDD Datset/*.csv"))

dup_analysis = []

for fpath in files:
    fname = os.path.basename(fpath)
    fan_id = os.path.splitext(fname)[0]
    df = pd.read_csv(fpath)
    n_rows = len(df)
    
    # 1. Exact full-row duplicates (including S.No.)
    full_dups = int(df.duplicated().sum())
    
    # 2. Content duplicates (excluding S.No.)
    content_cols = [c for c in df.columns if c != 'S.No.']
    content_dups = int(df.duplicated(subset=content_cols, keep='first').sum())
    
    # 3. Duplicate timestamps alone
    ts_dups = int(df.duplicated(subset=['datetime'], keep='first').sum())
    n_unique_ts = int(df['datetime'].nunique())
    
    # 4. Same timestamp, but DIFFERENT content
    # (i.e. legitimate intra-minute multi-packet events like speed change)
    # Total unique timestamp occurrences minus unique (timestamp + content) combinations
    df_unique_content = df.drop_duplicates(subset=content_cols)
    ts_in_clean = int(df_unique_content['datetime'].nunique())
    clean_rows = len(df_unique_content)
    intra_minute_distinct_events = clean_rows - ts_in_clean
    
    # 5. Check repeat pattern for content duplicates:
    # Are duplicate records consecutive?
    dup_mask = df.duplicated(subset=content_cols, keep=False)
    df_dups_only = df[dup_mask]
    
    # Max repeat count of any single packet
    max_repeat = int(df.groupby(content_cols).size().max()) if len(df_dups_only) > 0 else 1
    
    dup_analysis.append({
        "fan_id": fan_id,
        "total_raw_rows": n_rows,
        "full_row_duplicates": full_dups,
        "content_duplicates_diff_sno": content_dups,
        "content_duplicate_pct": round(content_dups / n_rows * 100.0, 2),
        "total_unique_timestamps": n_unique_ts,
        "duplicate_timestamps": ts_dups,
        "clean_rows_after_dedup": clean_rows,
        "intra_minute_events": intra_minute_distinct_events,
        "max_repeat_count": max_repeat
    })

df_res = pd.DataFrame(dup_analysis)
print(df_res.to_string(index=False))

# Save to audit directory
os.makedirs("aaci_ml/v3_energy_intelligence/audit", exist_ok=True)
df_res.to_csv("aaci_ml/v3_energy_intelligence/audit/file_by_file_duplicate_analysis.csv", index=False)

