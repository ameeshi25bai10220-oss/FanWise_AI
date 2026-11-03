"""
Data Loader and Chronological Partitioning for AACI Module 1 (Occupancy Intelligence).
Loads and partitions archive (2)/Occupancy.csv according to official UCI boundaries:
  - Partition 1 (Test 1 - datatest.txt): indices 0 to 2664 (N = 2,665)
  - Partition 2 (Train - datatraining.txt): indices 2665 to 10807 (N = 8,143)
  - Partition 3 (Test 2 - datatest2.txt): indices 10808 to 20559 (N = 9,752)

Enforces strict chronological integrity with zero random shuffling.
Generates Feature Sets A, B, and C.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..",
    "archive (2)",
    "Occupancy.csv"
)

# Normalize path
DATASET_PATH = os.path.abspath(DATASET_PATH)

# Partition Boundary Constants (UCI official splits)
SPLIT_TEST1_END = 2665   # row indices 0 to 2664
SPLIT_TRAIN_END = 10808  # row indices 2665 to 10807
TOTAL_ROWS = 20560       # row indices 10808 to 20559

FEATURE_SETS = {
    "set_a": ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"],
    "set_b": ["Temperature", "Humidity", "HumidityRatio", "Hour"],
    "set_c": ["Temperature", "Humidity", "Light", "CO2"]
}

TARGET_COL = "Occupancy"


def load_raw_data(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """Load raw dataset and ensure datetime column is parsed."""
    if not os.path.exists(filepath):
        # Fallback to local path relative to current working directory
        filepath = os.path.abspath("archive (2)/Occupancy.csv")
        
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["Hour"] = df["date"].dt.hour
    return df


def get_official_partitions(df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, test2_df, test1_df) strictly maintaining UCI boundaries:
      - train_df: Partition 2 (datatraining.txt), rows 2665 to 10807 (N = 8,143)
      - test2_df: Partition 3 (datatest2.txt), rows 10808 to 20559 (N = 9,752) [Primary forward test]
      - test1_df: Partition 1 (datatest.txt), rows 0 to 2664 (N = 2,665) [Historical evaluation]
    """
    if df is None:
        df = load_raw_data()
        
    assert len(df) == TOTAL_ROWS, f"Expected {TOTAL_ROWS} rows, got {len(df)}"
    
    test1_df = df.iloc[0:SPLIT_TEST1_END].copy().reset_index(drop=True)
    train_df = df.iloc[SPLIT_TEST1_END:SPLIT_TRAIN_END].copy().reset_index(drop=True)
    test2_df = df.iloc[SPLIT_TRAIN_END:TOTAL_ROWS].copy().reset_index(drop=True)
    
    return train_df, test2_df, test1_df


def get_feature_and_target(df: pd.DataFrame, feature_set_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature DataFrame X and target Series y for a given feature set."""
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set_name}'. Valid: {list(FEATURE_SETS.keys())}")
    
    feature_cols = FEATURE_SETS[feature_set_name]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


def generate_rolling_origin_folds(
    train_df: pd.DataFrame,
    n_splits: int = 4,
    min_train_ratio: float = 0.40,
    val_ratio: float = 0.15,
    purge_buffer_minutes: int = 15
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates expanding-window rolling-origin temporal validation folds on train_df.
    Enforces a purge buffer between train and validation folds to avoid leakage.
    Returns list of (train_indices, val_indices).
    """
    n = len(train_df)
    folds = []
    
    val_size = int(n * val_ratio)
    
    for i in range(n_splits):
        train_end = int(n * (min_train_ratio + i * val_ratio))
        val_start = train_end + purge_buffer_minutes
        val_end = min(val_start + val_size, n)
        
        if val_start >= n or val_end <= val_start:
            break
            
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
        
    return folds


if __name__ == "__main__":
    df = load_raw_data()
    train_df, test2_df, test1_df = get_official_partitions(df)
    print(f"Dataset Loaded Successfully:")
    print(f"  Historical Test 1: {len(test1_df):,} rows (Occ: {test1_df['Occupancy'].mean():.2%})")
    print(f"  Training Set:      {len(train_df):,} rows (Occ: {train_df['Occupancy'].mean():.2%})")
    print(f"  Forward Test 2:    {len(test2_df):,} rows (Occ: {test2_df['Occupancy'].mean():.2%})")
    
    folds = generate_rolling_origin_folds(train_df)
    print(f"\nGenerated {len(folds)} Rolling-Origin Folds on Training Partition:")
    for i, (tr, val) in enumerate(folds):
        print(f"  Fold {i+1}: Train={len(tr):,} (0 to {tr[-1]}), Val={len(val):,} ({val[0]} to {val[-1]})")

