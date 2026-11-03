"""
AACI Model 3 – Feature Engineering Module
==========================================
Causal, leakage-free feature engineering for ceiling fan energy intelligence.

EXPLICIT TARGET DEFINITION & FORECASTING HORIZON:
1. Primary Target: target_next_eSpent_wh
   - Definition: Energy consumption in Watt-hours (Wh) during the NEXT OBSERVED
     TELEMETRY PACKET (shift(-1) of eSpent_wh).
   - Forecasting Horizon: The horizon is intrinsically VARIABLE (packet-to-packet)
     because Mendeley IoT sampling intervals are irregular (ranging from ~30s to 300s).
     It must NOT be described as a fixed 2.5-minute or 5-minute interval forecast.
   - Zero-Power / Inactive State: When speed = 0, P_BLDC = 0W, so target = 0.0 Wh.
   - Backlog Handling: Records where opTime > 900s represent offline disconnection
     accumulation. An optional filter (`exclude_backlog_targets=True`) excludes these
     multi-hour disconnection spikes from interval forecasting targets.

2. Prevention of Contemporaneous Tautology:
   - Contemporaneous speed s_t and opTime_t are NEVER used to predict contemporaneous
     energy E_t. E_t = P(s_t) * opTime_t is deterministic physics, not ML.
   - When predicting target_next_eSpent_wh, all features represent information available
     at or before time step t. Future operational parameters (s_{t+1}, opTime_{t+1})
     and future sensor readings (T_{t+1}, RH_{t+1}) are strictly prohibited.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

from .energy_physics import (
    BLDC_POWER_MAP,
    INDUCTION_POWER_MAP,
    SAVED_POWER_MAP,
    SPEED_TO_AIR_VELOCITY_MAP,
    joules_to_wh,
    joules_to_kwh,
)

# Standard forbidden feature patterns when predicting interval energy
FORBIDDEN_ENERGY_TARGET_LEAKAGE = [
    "eSpent",
    "eSaved",
    "opTime",
    "target_",
]


def add_cyclic_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cyclic sinusoidal and calendar features from timestamps.
    """
    df = df.copy()
    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)

    hour = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    day_of_week = df["datetime"].dt.dayofweek
    month = df["datetime"].dt.month

    # 24-hour diurnal cycle
    df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)

    # 12-month seasonal cycle
    df["sin_month"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    df["cos_month"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)

    # Calendar indicators
    df["day_of_week"] = day_of_week
    df["is_weekend"] = (day_of_week >= 5).astype(int)
    df["is_night"] = ((df["datetime"].dt.hour >= 22) | (df["datetime"].dt.hour < 6)).astype(int)

    return df


def add_environmental_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3],
    rolling_windows: List[int] = [3, 6],
) -> pd.DataFrame:
    """
    Generate retrospective environmental lag and rolling statistics per fan.
    """
    df = df.copy()

    for lag in lags:
        df[f"temp_lag_{lag}"] = df.groupby("fan_id")["temperature"].shift(lag)
        df[f"humidity_lag_{lag}"] = df.groupby("fan_id")["humidity"].shift(lag)

    # Environmental rate of change
    if 1 in lags and 2 in lags:
        df["temp_diff_1_2"] = df["temp_lag_1"] - df["temp_lag_2"]
        df["humidity_diff_1_2"] = df["humidity_lag_1"] - df["humidity_lag_2"]

    # Rolling means of strictly retrospective lagged values (shift 1 first!)
    for w in rolling_windows:
        df[f"temp_rolling_mean_{w}"] = (
            df.groupby("fan_id")["temperature"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
        )
        df[f"humidity_rolling_mean_{w}"] = (
            df.groupby("fan_id")["humidity"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
        )

    return df


def add_operational_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2],
    rolling_energy_windows: List[int] = [3, 12],
) -> pd.DataFrame:
    """
    Generate retrospective operational state and historical energy lags per fan.
    """
    df = df.copy()

    # Ensure Wh energy exists
    if "eSpent_wh" not in df.columns and "eSpent" in df.columns:
        df["eSpent_wh"] = joules_to_wh(df["eSpent"])

    # Lagged fan speeds
    for lag in lags:
        df[f"speed_lag_{lag}"] = df.groupby("fan_id")["speed"].shift(lag)

    # State transition flag
    if 1 in lags and 2 in lags:
        df["speed_changed"] = (
            df["speed_lag_1"] != df["speed_lag_2"]
        ).astype(float)

    # Mode lags
    if "mode" in df.columns:
        df["mode_lag_1"] = df.groupby("fan_id")["mode"].shift(1)

    # Lagged energy consumption (Wh)
    if "eSpent_wh" in df.columns:
        for lag in lags:
            df[f"eSpent_wh_lag_{lag}"] = df.groupby("fan_id")["eSpent_wh"].shift(lag)

        for w in rolling_energy_windows:
            df[f"eSpent_wh_rolling_sum_{w}"] = (
                df.groupby("fan_id")["eSpent_wh"]
                .shift(1)
                .rolling(window=w, min_periods=1)
                .sum()
            )

    return df


def build_supervised_energy_dataset(
    df: pd.DataFrame,
    target_type: str = "next_energy",
    drop_burn_in_nans: bool = True,
    exclude_backlog_targets: bool = False,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Build a leak-free supervised dataset for Model 3.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned telemetry DataFrame.
    target_type : str, default 'next_energy'
        - 'next_energy': Predicts eSpent_wh of the NEXT OBSERVED TELEMETRY PACKET (t+1).
        - 'next_speed': Predicts speed of the NEXT OBSERVED TELEMETRY PACKET (t+1).
    drop_burn_in_nans : bool, default True
        Whether to drop rows with NaN resulting from initial lag shifts.
    exclude_backlog_targets : bool, default False
        If True, drops rows where the target packet is a reconnection backlog (opTime > 900s).

    Returns
    -------
    Tuple[pd.DataFrame, List[str], str]
        (dataset_df, feature_column_names, target_column_name)
    """
    df = df.copy()

    # Ensure sorted chronologically per fan
    df = df.sort_values(by=["fan_id", "datetime"]).reset_index(drop=True)

    # 1. Add cyclic temporal signals
    df = add_cyclic_temporal_features(df)

    # 2. Add environmental lag features
    df = add_environmental_lag_features(df, lags=[1, 2, 3], rolling_windows=[3, 6])

    # 3. Add operational lag features
    df = add_operational_lag_features(df, lags=[1, 2], rolling_energy_windows=[3, 12])

    # 4. Formulate strictly forward-looking targets
    if target_type == "next_energy":
        target_col = "target_next_eSpent_wh"
        df[target_col] = df.groupby("fan_id")["eSpent_wh"].shift(-1)
        if exclude_backlog_targets and "is_backlog" in df.columns:
            next_is_backlog = df.groupby("fan_id")["is_backlog"].shift(-1)
            is_backlog_mask = next_is_backlog.fillna(False).astype(bool)
            df = df[~is_backlog_mask].copy()
    elif target_type == "next_speed":
        target_col = "target_next_speed"
        df[target_col] = df.groupby("fan_id")["speed"].shift(-1)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Explicit list of permitted feature columns
    feature_cols = [
        # Current environmental observation
        "temperature",
        "humidity",
        # Temporal cyclic signals
        "sin_hour",
        "cos_hour",
        "sin_month",
        "cos_month",
        "day_of_week",
        "is_weekend",
        "is_night",
        # Environmental lags
        "temp_lag_1",
        "temp_lag_2",
        "humidity_lag_1",
        "humidity_lag_2",
        "temp_diff_1_2",
        "humidity_diff_1_2",
        "temp_rolling_mean_3",
        "humidity_rolling_mean_3",
        "temp_rolling_mean_6",
        "humidity_rolling_mean_6",
        # Retrospective operational lags
        "speed_lag_1",
        "speed_lag_2",
        "speed_changed",
        "eSpent_wh_lag_1",
        "eSpent_wh_lag_2",
        "eSpent_wh_rolling_sum_3",
        "eSpent_wh_rolling_sum_12",
    ]

    # Verify absence of contemporaneous leakage
    prohibited = ["opTime", "eSpent", "eSaved", target_col]
    for p in prohibited:
        if p in feature_cols:
            raise RuntimeError(
                f"LEAKAGE DEFECT DETECTED: Contemporaneous '{p}' is present in feature_cols!"
            )

    # Filter to valid subset of feature columns that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Drop burn-in rows (caused by lags) and final row per fan (caused by shift(-1))
    if drop_burn_in_nans:
        subset_to_check = feature_cols + [target_col]
        df = df.dropna(subset=subset_to_check).reset_index(drop=True)

    return df, feature_cols, target_col


def generate_candidate_speed_evaluations(
    air_temp_c: float,
    relative_humidity: float,
    duration_hours: float = 1.0,
) -> pd.DataFrame:
    """
    Construct a candidate speed evaluation matrix for speeds 0-5 under current
    environmental conditions.

    NOTE: Air velocity values are PROVISIONAL SURROGATES for simulation only.
    """
    rows = []
    for speed in range(6):
        va = SPEED_TO_AIR_VELOCITY_MAP[speed]
        bldc_p = BLDC_POWER_MAP[speed]
        ind_p = INDUCTION_POWER_MAP[speed]
        saved_p = SAVED_POWER_MAP[speed]

        energy_bldc_wh = bldc_p * duration_hours
        energy_ind_wh = ind_p * duration_hours
        energy_saved_wh = saved_p * duration_hours

        rows.append(
            {
                "speed": speed,
                "air_temp_c": air_temp_c,
                "relative_humidity": relative_humidity,
                "air_velocity_ms": va,
                "bldc_power_w": bldc_p,
                "baseline_induction_power_w": ind_p,
                "energy_bldc_wh": energy_bldc_wh,
                "energy_baseline_wh": energy_ind_wh,
                "energy_saved_wh": energy_saved_wh,
                "savings_percentage": (
                    (energy_saved_wh / energy_ind_wh * 100.0)
                    if energy_ind_wh > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)
