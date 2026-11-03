"""
AACI – Step 1: Dataset Inspection
ASHRAE Global Thermal Comfort Database II (ashrae_db2.01.csv)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = r"c:\Users\bharg\OneDrive\Desktop\Atomberg\archive (3)\ashrae_db2.01.csv"

print("=" * 70)
print("AACI – ASHRAE DB2 Dataset Inspection")
print("=" * 70)

df = pd.read_csv(DATASET_PATH, low_memory=False)

print(f"\n[1] SHAPE: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\n[2] ALL COLUMN NAMES:")
for i, col in enumerate(df.columns):
    print(f"  [{i:3d}] {col}")

print("\n[3] DTYPES SUMMARY:")
print(df.dtypes.value_counts())

print("\n[4] TARGET VARIABLE – 'Thermal sensation' distribution (raw):")
ts = df["Thermal sensation"]
print(ts.value_counts(dropna=False).to_string())
print(f"\n  dtype: {ts.dtype}")
print(f"  Missing: {ts.isna().sum():,} ({ts.isna().mean()*100:.1f}%)")

print("\n[5] CANDIDATE FEATURE COLUMNS – missing value audit:")
CANDIDATE_FEATURES = [
    "Air temperature (C)",
    "Relative humidity (%)",
    "Air velocity (m/s)",
    "Radiant temperature (C)",
    "Operative temperature (C)",
    "Globe temperature (C)",
    "Outdoor monthly air temperature (C)",
    "PMV",
    "PPD",
    "SET",
    "Clo",
    "Met",
    "Age",
    "Sex",
    "Season",
    "Climate",
    "Building type",
    "Cooling startegy_building level",
]
for col in CANDIDATE_FEATURES:
    if col in df.columns:
        n_miss = df[col].isna().sum()
        pct = n_miss / len(df) * 100
        uq = df[col].nunique()
        print(f"  {col:<45s}  missing={n_miss:>7,} ({pct:5.1f}%)  unique={uq}")
    else:
        print(f"  {col:<45s}  *** NOT IN DATASET ***")

print("\n[6] SAMPLE ROW (first record):")
print(df.iloc[0].to_string())

print("\n[7] NUMERIC SUMMARY (key columns):")
num_cols = ["Air temperature (C)", "Relative humidity (%)", "Air velocity (m/s)",
            "Radiant temperature (C)", "Globe temperature (C)", "Clo", "Met", "PMV", "PPD"]
num_cols = [c for c in num_cols if c in df.columns]
print(df[num_cols].describe().round(3).to_string())

print("\n[8] DATA CONTRIBUTOR / SOURCE distribution (top 15):")
print(df["Data contributor"].value_counts().head(15).to_string())

print("\n[9] BUILDING TYPE distribution:")
print(df["Building type"].value_counts(dropna=False).to_string())

print("\n[10] COUNTRY distribution (top 15):")
print(df["Country"].value_counts().head(15).to_string())

print("\n" + "=" * 70)
print("Inspection complete.")
print("=" * 70)

