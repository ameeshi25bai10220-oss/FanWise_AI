"""
AACI Module 3: RDD Data Loader and Preprocessor
Loads, cleans, validates, and chronologically sorts residential Atomberg fan telemetry data.
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List

DEFAULT_RDD_DIR = "RDD Datset"

POWER_LOOKUP = {0: 0.0, 1: 10.0, 2: 15.0, 3: 20.0, 4: 27.0, 5: 48.0}
SAVED_LOOKUP = {0: 0.0, 1: 14.0, 2: 19.0, 3: 35.0, 4: 37.0, 5: 48.0}
INDUCTION_LOOKUP = {k: POWER_LOOKUP[k] + SAVED_LOOKUP[k] for k in POWER_LOOKUP}

ROOM_CATEGORIES = {
    "Bedroom": ["Bedroom_Fan_1", "Bedroom_Fan_2", "Bedroom_Fan_3"],
    "Drawingroom": ["Drawingroom_Fan_1", "Drawingroom_Fan_2", "Drawingroom_Fan_3"],
    "Lounge": ["Lounge_Fan_1", "Lounge_Fan_2", "Lounge_Fan_3"]
}


def get_rdd_file_path(fan_id_or_path: str, base_dir: str = DEFAULT_RDD_DIR) -> str:
    """Resolve fan identifier or relative path to absolute/valid file path."""
    if os.path.exists(fan_id_or_path):
        return fan_id_or_path
    
    # Check if ends with .csv
    fname = fan_id_or_path if fan_id_or_path.endswith(".csv") else f"{fan_id_or_path}.csv"
    candidate = os.path.join(base_dir, fname)
    if os.path.exists(candidate):
        return candidate
    
    raise FileNotFoundError(f"Could not locate RDD telemetry file for '{fan_id_or_path}' in '{base_dir}'")


def load_fan_telemetry(
    fan_id_or_path: str,
    base_dir: str = DEFAULT_RDD_DIR,
    deduplicate: bool = True,
    clean_sensors: bool = True,
    sort_chrono: bool = True
) -> pd.DataFrame:
    """
    Load a single fan telemetry dataset with robust cleaning and chronological sorting.
    
    Parameters
    ----------
    fan_id_or_path : str
        Fan name (e.g. 'Bedroom_Fan_1') or path to CSV file.
    base_dir : str
        Base directory containing RDD CSV files.
    deduplicate : bool
        Whether to drop exact duplicate rows.
    clean_sensors : bool
        Whether to filter out microcontroller error codes (255) and invalid ranges.
    sort_chrono : bool
        Whether to sort strictly chronologically by parsed datetime.
        
    Returns
    -------
    pd.DataFrame
        Cleaned, feature-enriched fan telemetry dataframe.
    """
    file_path = get_rdd_file_path(fan_id_or_path, base_dir)
    fname = os.path.basename(file_path)
    fan_id = os.path.splitext(fname)[0]
    room_type = fan_id.split("_")[0]
    
    df = pd.read_csv(file_path)
    
    # 1. Deduplication
    if deduplicate:
        content_cols = [c for c in df.columns if c != 'S.No.']
        df = df.drop_duplicates(subset=content_cols).reset_index(drop=True)
        
    # 2. Datetime parsing with Indian format (DD/MM/YYYY HH:MM)
    df['dt'] = pd.to_datetime(df['datetime'], format="%d/%m/%Y %H:%M", errors='coerce')
    df = df.dropna(subset=['dt']).reset_index(drop=True)
    
    # 3. Microcontroller sensor error cleaning (255 / 0xFF errors)
    if clean_sensors:
        valid_mask = (
            (df['temperature'] > 0) & (df['temperature'] < 60) &
            (df['humidity'] > 0) & (df['humidity'] <= 100) &
            (df['speed'].isin([0, 1, 2, 3, 4, 5])) &
            (df['opTime'] >= 0)
        )
        df = df[valid_mask].reset_index(drop=True)
        
    # 4. Strict Chronological Ordering
    if sort_chrono:
        df = df.sort_values(by=['dt', 'S.No.']).reset_index(drop=True)
        
    # 5. Enrichment & Unit conversions
    df['fan_id'] = fan_id
    df['room_type'] = room_type
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['date'] = df['dt'].dt.date
    df['is_on'] = (df['speed'] > 0).astype(int)
    
    # Energy in Joules -> Watt-hours and kWh
    df['eSpent_Wh'] = df['eSpent'] / 3600.0
    df['eSaved_Wh'] = df['eSaved'] / 3600.0
    df['eSpent_kWh'] = df['eSpent'] / 3_600_000.0
    df['eSaved_kWh'] = df['eSaved'] / 3_600_000.0
    
    # Modeled physical power rate in Watts
    df['power_W'] = df['speed'].map(POWER_LOOKUP)
    df['baseline_induction_W'] = df['speed'].map(INDUCTION_LOOKUP)
    df['saved_rate_W'] = df['speed'].map(SAVED_LOOKUP)
    
    return df


def load_all_fans(base_dir: str = DEFAULT_RDD_DIR) -> Dict[str, pd.DataFrame]:
    """Load all 9 fans into a dictionary {fan_id: DataFrame}."""
    fan_files = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    fans = {}
    for f in fan_files:
        fid = os.path.splitext(os.path.basename(f))[0]
        fans[fid] = load_fan_telemetry(f, base_dir=base_dir)
    return fans

