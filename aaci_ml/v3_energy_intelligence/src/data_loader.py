"""
AACI Model 3 – Data Loader & Ingestion Module
=============================================
Robust ingestion, parsing, deduplication, and sensor validation for the
nine Mendeley ceiling fan IoT telemetry files (675,332 raw records).

AUTHORITATIVE ROOM CATEGORY MAPPING:
The Mendeley RDD dataset contains exactly nine CSV files across THREE room types:
1. Bedroom_Fan_1.csv    -> Bedroom
2. Bedroom_Fan_2.csv    -> Bedroom
3. Bedroom_Fan_3.csv    -> Bedroom
4. Drawingroom_Fan_1.csv -> Drawingroom
5. Drawingroom_Fan_2.csv -> Drawingroom
6. Drawingroom_Fan_3.csv -> Drawingroom
7. Lounge_Fan_1.csv      -> Lounge
8. Lounge_Fan_2.csv      -> Lounge
9. Lounge_Fan_3.csv      -> Lounge

(There are NO Diningroom, Kitchen, or Hall files in the Mendeley dataset).
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .energy_physics import (
    BLDC_POWER_MAP,
    INDUCTION_POWER_MAP,
    SAVED_POWER_MAP,
    joules_to_wh,
    joules_to_kwh,
)

# ─────────────────────────────────────────────────────────────────────────────
# Authoritative Fan and Room Metadata Mapping
# ─────────────────────────────────────────────────────────────────────────────

AUTHORITATIVE_ROOMS: List[str] = ["Bedroom", "Drawingroom", "Lounge"]

FAN_ROOM_MAP: Dict[str, str] = {
    "Bedroom_Fan_1": "Bedroom",
    "Bedroom_Fan_2": "Bedroom",
    "Bedroom_Fan_3": "Bedroom",
    "Drawingroom_Fan_1": "Drawingroom",
    "Drawingroom_Fan_2": "Drawingroom",
    "Drawingroom_Fan_3": "Drawingroom",
    "Lounge_Fan_1": "Lounge",
    "Lounge_Fan_2": "Lounge",
    "Lounge_Fan_3": "Lounge",
}

TELEMETRY_COLUMNS: List[str] = [
    "S.No.",
    "datetime",
    "temperature",
    "humidity",
    "speed",
    "mode",
    "eSpent",
    "eSaved",
    "opTime",
]

CONTENT_COLUMNS: List[str] = [
    "datetime",
    "temperature",
    "humidity",
    "speed",
    "mode",
    "eSpent",
    "eSaved",
    "opTime",
]

# Physical plausible sensor thresholds
TEMP_MIN_PLAUSIBLE: float = 10.0
TEMP_MAX_PLAUSIBLE: float = 50.0
HUMIDITY_MIN_PLAUSIBLE: float = 10.0
HUMIDITY_MAX_PLAUSIBLE: float = 100.0
BACKLOG_THRESHOLD_SEC: float = 900.0  # 15 minutes


def get_room_category_for_fan(fan_id: str) -> str:
    """
    Infer room category from fan identifier using authoritative 9-fan mapping.
    """
    if fan_id in FAN_ROOM_MAP:
        return FAN_ROOM_MAP[fan_id]
    for room in AUTHORITATIVE_ROOMS:
        if fan_id.startswith(room):
            return room
    return "Other"


def parse_and_sort_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps with day-first format and sort chronologically.
    """
    df = df.copy()
    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)

    sort_cols = ["datetime"]
    if "S.No." in df.columns:
        sort_cols.append("S.No.")
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    return df


def apply_content_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply content-based deduplication across the 8 telemetry content columns.
    Eliminates database insertion duplicates while preserving valid intra-minute
    physical updates where speed, mode, or energy readings differ.
    """
    cols_to_check = [c for c in CONTENT_COLUMNS if c in df.columns]
    deduped_df = df.drop_duplicates(subset=cols_to_check, keep="first").copy()
    return deduped_df.reset_index(drop=True)


def tag_sensor_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flags identifying invalid sensor readings:
    - Sentinel 255 (I2C microcontroller bus timeout)
    - Boot zeros (0 deg C or 0% RH)
    - Out-of-physical-range readings
    """
    df = df.copy()

    is_temp_sentinel = df["temperature"] == 255
    is_temp_zero = df["temperature"] <= 0.0
    is_temp_out_of_range = (df["temperature"] < TEMP_MIN_PLAUSIBLE) | (
        df["temperature"] > TEMP_MAX_PLAUSIBLE
    )
    df["is_temp_error"] = is_temp_sentinel | is_temp_zero | is_temp_out_of_range

    is_humidity_sentinel = df["humidity"] == 255
    is_humidity_zero = df["humidity"] <= 0.0
    is_humidity_out_of_range = (df["humidity"] < HUMIDITY_MIN_PLAUSIBLE) | (
        df["humidity"] > HUMIDITY_MAX_PLAUSIBLE
    )
    df["is_humidity_error"] = (
        is_humidity_sentinel | is_humidity_zero | is_humidity_out_of_range
    )

    df["is_sensor_error"] = df["is_temp_error"] | df["is_humidity_error"]
    return df


def filter_sensor_sentinels(
    df: pd.DataFrame, drop_errors: bool = True
) -> pd.DataFrame:
    """
    Filter out or mask sensor sentinel and boot error rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with telemetry.
    drop_errors : bool, default True
        If True, drops rows containing sensor errors.
        If False, replaces invalid temperature/humidity with NaN and preserves rows.
    """
    df = tag_sensor_errors(df)
    if drop_errors:
        clean_df = df[~df["is_sensor_error"]].copy()
    else:
        clean_df = df.copy()
        clean_df.loc[clean_df["is_temp_error"], "temperature"] = np.nan
        clean_df.loc[clean_df["is_humidity_error"], "humidity"] = np.nan

    return clean_df.reset_index(drop=True)


def tag_backlog_records(
    df: pd.DataFrame, threshold_sec: float = BACKLOG_THRESHOLD_SEC
) -> pd.DataFrame:
    """
    Tag offline accumulation / network reconnection packets (opTime > 900s).
    """
    df = df.copy()
    df["is_backlog"] = df["opTime"] > threshold_sec
    return df


def enrich_energy_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add converted energy columns in Watt-hours (Wh) and Kilowatt-hours (kWh),
    along with nominal motor power rating in Watts.
    """
    df = df.copy()
    if "eSpent" in df.columns:
        df["eSpent_wh"] = joules_to_wh(df["eSpent"])
        df["eSpent_kwh"] = joules_to_kwh(df["eSpent"])
    if "eSaved" in df.columns:
        df["eSaved_wh"] = joules_to_wh(df["eSaved"])
        df["eSaved_kwh"] = joules_to_kwh(df["eSaved"])
    if "speed" in df.columns:
        df["bldc_power_w"] = df["speed"].map(BLDC_POWER_MAP)
        df["baseline_power_w"] = df["speed"].map(INDUCTION_POWER_MAP)
        df["saved_power_w"] = df["speed"].map(SAVED_POWER_MAP)
    return df


def load_fan_telemetry(
    file_path: str,
    apply_cleaning: bool = True,
    drop_sensor_errors: bool = False,
    backlog_threshold_sec: float = BACKLOG_THRESHOLD_SEC,
) -> pd.DataFrame:
    """
    Load a single Mendeley fan telemetry CSV with full metadata and cleaning.

    Parameters
    ----------
    file_path : str
        Path to the fan CSV file.
    apply_cleaning : bool, default True
        Whether to sort, deduplicate, and enrich.
    drop_sensor_errors : bool, default False
        Whether to drop rows with 255/boot sensor errors or mask them with NaN.
    backlog_threshold_sec : float, default 900.0
        Threshold in seconds for tagging backlog packets.

    Returns
    -------
    pd.DataFrame
        Structured, cleaned fan telemetry dataframe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    fan_id = os.path.splitext(filename)[0]
    room_category = get_room_category_for_fan(fan_id)

    raw_df = pd.read_csv(file_path)

    # Attach provenance
    raw_df["fan_id"] = fan_id
    raw_df["room_category"] = room_category

    if not apply_cleaning:
        return raw_df

    # 1. Parse and sort chronologically
    sorted_df = parse_and_sort_telemetry(raw_df)

    # 2. Content-based deduplication
    deduped_df = apply_content_deduplication(sorted_df)

    # 3. Sensor validation & error tagging
    validated_df = filter_sensor_sentinels(
        deduped_df, drop_errors=drop_sensor_errors
    )

    # 4. Backlog packet tagging
    tagged_df = tag_backlog_records(
        validated_df, threshold_sec=backlog_threshold_sec
    )

    # 5. Enrich with energy units (Wh, kWh, Watts)
    final_df = enrich_energy_units(tagged_df)

    return final_df


def load_all_mendeley_data(
    data_dir: str,
    apply_cleaning: bool = True,
    drop_sensor_errors: bool = False,
    backlog_threshold_sec: float = BACKLOG_THRESHOLD_SEC,
) -> pd.DataFrame:
    """
    Load all nine Mendeley ceiling fan CSV files from the specified directory.
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Directory not found: {data_dir}")

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    fan_dfs = []
    for fpath in sorted(csv_files):
        fan_df = load_fan_telemetry(
            fpath,
            apply_cleaning=apply_cleaning,
            drop_sensor_errors=drop_sensor_errors,
            backlog_threshold_sec=backlog_threshold_sec,
        )
        fan_dfs.append(fan_df)

    combined_df = pd.concat(fan_dfs, ignore_index=True)
    return combined_df
