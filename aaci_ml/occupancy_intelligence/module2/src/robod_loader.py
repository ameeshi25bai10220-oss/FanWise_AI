"""
Data Loader and Preprocessing for AACI Module 2 (Spatial Headcount & Pattern Intelligence).
Loads and cleans ROBOD dataset files (SupplementaryData/combined_Room{1..5}.csv).

Features:
  - Parses ISO 8601 timestamps (Singapore UTC+8).
  - Derives cyclical temporal features (Hour sin/cos, DayOfWeek sin/cos, is_weekend).
  - Defines 5 hierarchical feature groups:
      1. Environmental Sensors
      2. Environmental + Acoustic
      3. Environmental + Acoustic + Wi-Fi
      4. Full Multimodal
      5. Building-Operation / HVAC (Experimental Leakage Investigation Group)
  - Enforces strict target exclusion: 'occupant_presence [binary]' is NEVER an input.
  - Implements chronological per-room splits and Leave-One-Room-Out (LORO) splits.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

ROBOD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "SupplementaryData")
)

ROOM_TYPES = {
    1: "Lecture Room 1",
    2: "Lecture Room 2",
    3: "Office Space 3",
    4: "Office Space 4",
    5: "Library Space 5"
}

TARGET_HEADCOUNT = "occupant_count [number]"
TARGET_PRESENCE = "occupant_presence [binary]"

# Feature Groups Definition
FEATURE_GROUPS = {
    "group1_env": [
        "air_temperature [Celsius]",
        "indoor_relative_humidity [%]",
        "indoor_co2 [ppm]",
        "baromatic_pressure [hPa]",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
    ],
    "group2_env_acoustic": [
        "air_temperature [Celsius]",
        "indoor_relative_humidity [%]",
        "indoor_co2 [ppm]",
        "baromatic_pressure [hPa]",
        "sound_pressure_level [dba]",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
    ],
    "group3_env_acoustic_wifi": [
        "air_temperature [Celsius]",
        "indoor_relative_humidity [%]",
        "indoor_co2 [ppm]",
        "baromatic_pressure [hPa]",
        "sound_pressure_level [dba]",
        "wifi_connected_devices [number]",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
    ],
    "group4_multimodal": [
        "air_temperature [Celsius]",
        "indoor_relative_humidity [%]",
        "indoor_co2 [ppm]",
        "baromatic_pressure [hPa]",
        "sound_pressure_level [dba]",
        "wifi_connected_devices [number]",
        "illuminance [lux]",
        "voc [ppb]",
        "pm2.5 [mu_g/m3]",
        "dry_bulb_temp [Celsius]",
        "outdoor_relative_humidity [%]",
        "outdoor_co2 [ppm]",
        "global_horizontal_solar_radiation [W/m2]",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
    ],
    "group5_bms_experimental": [
        "air_temperature [Celsius]",
        "indoor_relative_humidity [%]",
        "indoor_co2 [ppm]",
        "baromatic_pressure [hPa]",
        "sound_pressure_level [dba]",
        "wifi_connected_devices [number]",
        "illuminance [lux]",
        "voc [ppb]",
        "pm2.5 [mu_g/m3]",
        "dry_bulb_temp [Celsius]",
        "outdoor_relative_humidity [%]",
        "outdoor_co2 [ppm]",
        "global_horizontal_solar_radiation [W/m2]",
        "ceiling_fan_energy [kWh]",
        "lighting_energy [kWh]",
        "plug_load_energy [kWh]",
        "chilled_water_energy [kWh]",
        "supply_air_temperature [Celsius]",
        "supply_air_pressure [Pa]",
        "temp_setpoint [Celsius]",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
    ]
}


def load_single_room(room_id: int) -> pd.DataFrame:
    """Loads and preprocesses a single room CSV from SupplementaryData."""
    if room_id not in ROOM_TYPES:
        raise ValueError(f"Invalid room_id {room_id}. Valid: 1 to 5.")
        
    fp = os.path.join(ROBOD_DIR, f"combined_Room{room_id}.csv")
    if not os.path.exists(fp):
        # Fallback relative path
        fp = os.path.abspath(f"SupplementaryData/combined_Room{room_id}.csv")
        
    df = pd.read_csv(fp)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ts").reset_index(drop=True)
    
    # Cyclic temporal features
    hours = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    dows = df["ts"].dt.dayofweek
    
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)
    df["is_weekend"] = (dows >= 5).astype(float)
    df["room_id"] = room_id
    df["room_type"] = ROOM_TYPES[room_id]
    
    # Fill minor missing sensor values (forward fill within room sequence)
    df = df.ffill().bfill()
    return df


def load_all_rooms() -> Dict[int, pd.DataFrame]:
    """Loads all 5 rooms into a dictionary."""
    return {r: load_single_room(r) for r in range(1, 6)}


def get_chronological_room_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a single room DataFrame chronologically:
      - Train: first 70%
      - Validation: next 15%
      - Forward Test: final 15%
    Strictly preserves chronological order without shuffling.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def get_loro_splits(
    all_rooms: Dict[int, pd.DataFrame],
    test_room_id: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-One-Room-Out (LORO) split:
      - Train: Concatenation of the other 4 rooms.
      - Test: The held-out room entirely.
    """
    train_dfs = [df for r_id, df in all_rooms.items() if r_id != test_room_id]
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = all_rooms[test_room_id].copy().reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    rooms = load_all_rooms()
    print("ROBOD Rooms Loaded Successfully:")
    total_records = 0
    for r_id, df in rooms.items():
        total_records += len(df)
        occ = df[TARGET_HEADCOUNT]
        print(f"  Room {r_id} ({ROOM_TYPES[r_id]}): {len(df):,} rows | Max Occupants: {occ.max()} | Mean: {occ.mean():.2f}")
    print(f"Total ROBOD records: {total_records:,}")
    print("\nFeature Groups defined:")
    for g, cols in FEATURE_GROUPS.items():
        print(f"  {g:25s}: {len(cols)} features")

