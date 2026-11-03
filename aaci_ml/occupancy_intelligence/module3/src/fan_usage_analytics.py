"""
AACI Module 3B: Fan Usage Pattern & Diurnal Behavior Analytics
Analyzes daily run profiles, diurnal speed preferences, continuous session durations, and room-type patterns.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Enable relative import of rdd_loader
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rdd_loader import load_fan_telemetry, load_all_fans, ROOM_CATEGORIES

EVAL_DIR = os.path.join(os.path.dirname(SRC_DIR), "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


def extract_sessions(df: pd.DataFrame, max_gap_seconds: float = 900.0) -> pd.DataFrame:
    """
    Extract contiguous fan operating sessions using vectorized pandas aggregations.
    A session ends when the fan is turned off (speed == 0) or gap between packets > max_gap_seconds (15 min).
    """
    # Identify ON rows
    on_mask = df['speed'] > 0
    df_on = df[on_mask].copy()
    
    if len(df_on) == 0:
        return pd.DataFrame()
        
    df_on['dt_diff'] = df_on['dt'].diff().dt.total_seconds().fillna(0)
    
    # New session starts if time gap > max_gap_seconds or negative jump
    df_on['new_session'] = (df_on['dt_diff'] > max_gap_seconds) | (df_on['dt_diff'] < 0)
    df_on['session_id'] = df_on['new_session'].cumsum()
    
    # Vectorized aggregation
    agg_funcs = {
        'dt': ['min', 'max'],
        'opTime': 'sum',
        'eSpent': 'sum',
        'speed': 'mean'
    }
    grouped = df_on.groupby('session_id').agg(agg_funcs)
    grouped.columns = ['start_time', 'end_time', 'opTime_seconds', 'eSpent_joules', 'mean_speed']
    
    # Calculate wall duration in minutes
    duration_sec = (grouped['end_time'] - grouped['start_time']).dt.total_seconds()
    grouped['duration_minutes'] = np.maximum(2.5, duration_sec / 60.0).round(1)
    grouped['energy_Wh'] = (grouped['eSpent_joules'] / 3600.0).round(2)
    grouped['mean_speed'] = grouped['mean_speed'].round(2)
    grouped['start_hour'] = grouped['start_time'].dt.hour
    grouped['is_night_session'] = ((grouped['start_hour'] >= 22) | (grouped['start_hour'] < 7)).astype(int)
    
    return grouped.reset_index()


def compute_fan_usage_patterns() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute overall fan usage patterns, hourly diurnal profiles, and room comparisons."""
    print("Loading RDD datasets for pattern analytics...")
    fans = load_all_fans()
    
    fan_pattern_records = []
    hourly_records = []
    
    for fan_id, df in fans.items():
        print(f"Processing patterns for {fan_id} ({len(df)} rows)...")
        room_type = df['room_type'].iloc[0]
        total_rows = len(df)
        
        # Date span
        min_date = df['dt'].min()
        max_date = df['dt'].max()
        calendar_days = (max_date - min_date).days + 1
        active_days = df['date'].nunique()
        
        # Operating hours
        total_op_sec = df['opTime'].sum()
        total_run_hours = total_op_sec / 3600.0
        daily_run_hours = total_run_hours / max(1, active_days)
        
        # Duty cycle (% of calendar time fan was running)
        total_monitored_hours = calendar_days * 24.0
        duty_cycle_pct = (total_run_hours / max(1, total_monitored_hours)) * 100.0
        
        # Speed preferences when ON
        df_on = df[df['speed'] > 0]
        pct_on = (len(df_on) / total_rows) * 100.0
        modal_speed = int(df_on['speed'].mode().iloc[0]) if len(df_on) > 0 else 0
        mean_speed_when_on = round(float(df_on['speed'].mean()), 2) if len(df_on) > 0 else 0.0
        mean_power_when_on = round(float(df_on['power_W'].mean()), 2) if len(df_on) > 0 else 0.0
        
        # Energy
        spent_kwh = df['eSpent_kWh'].sum()
        saved_kwh = df['eSaved_kWh'].sum()
        
        # Session metrics
        df_sessions = extract_sessions(df)
        if len(df_sessions) > 0:
            median_session_min = round(float(df_sessions['duration_minutes'].median()), 1)
            mean_session_min = round(float(df_sessions['duration_minutes'].mean()), 1)
            max_session_hours = round(float(df_sessions['duration_minutes'].max() / 60.0), 1)
            night_sessions = df_sessions[df_sessions['is_night_session'] == 1]
            day_sessions = df_sessions[df_sessions['is_night_session'] == 0]
            night_session_median_min = round(float(night_sessions['duration_minutes'].median()), 1) if len(night_sessions) > 0 else 0.0
            day_session_median_min = round(float(day_sessions['duration_minutes'].median()), 1) if len(day_sessions) > 0 else 0.0
            total_sessions = len(df_sessions)
        else:
            median_session_min = mean_session_min = max_session_hours = 0.0
            night_session_median_min = day_session_median_min = 0.0
            total_sessions = 0
            
        fan_pattern_records.append({
            "fan_id": fan_id,
            "room_type": room_type,
            "clean_records": total_rows,
            "calendar_days": calendar_days,
            "active_days": active_days,
            "total_run_hours": round(total_run_hours, 1),
            "daily_run_hours": round(daily_run_hours, 2),
            "duty_cycle_pct": round(duty_cycle_pct, 2),
            "fan_on_packet_pct": round(pct_on, 2),
            "modal_speed": modal_speed,
            "mean_speed_on": mean_speed_when_on,
            "mean_power_on_W": mean_power_when_on,
            "total_eSpent_kWh": round(spent_kwh, 2),
            "total_eSaved_kWh": round(saved_kwh, 2),
            "total_sessions": total_sessions,
            "median_session_min": median_session_min,
            "mean_session_min": mean_session_min,
            "night_session_median_min": night_session_median_min,
            "day_session_median_min": day_session_median_min,
            "max_session_hours": max_session_hours
        })
        
        # Diurnal profile for each hour 0..23
        for h in range(24):
            df_h = df[df['hour'] == h]
            if len(df_h) == 0:
                continue
            prob_on = float((df_h['speed'] > 0).mean())
            mean_spd = float(df_h['speed'].mean())
            mean_spd_on = float(df_h[df_h['speed'] > 0]['speed'].mean()) if prob_on > 0 else 0.0
            mean_pw = float(df_h['power_W'].mean())
            hourly_records.append({
                "fan_id": fan_id,
                "room_type": room_type,
                "hour": h,
                "prob_fan_on": round(prob_on, 4),
                "mean_speed_overall": round(mean_spd, 2),
                "mean_speed_when_on": round(mean_spd_on, 2),
                "mean_power_W": round(mean_pw, 2)
            })

    df_patterns = pd.DataFrame(fan_pattern_records)
    df_hourly = pd.DataFrame(hourly_records)
    
    # Aggregated room comparison
    room_comparison = df_patterns.groupby('room_type').agg({
        "clean_records": "sum",
        "total_run_hours": "sum",
        "daily_run_hours": "mean",
        "duty_cycle_pct": "mean",
        "fan_on_packet_pct": "mean",
        "mean_power_on_W": "mean",
        "total_eSpent_kWh": "sum",
        "total_eSaved_kWh": "sum",
        "median_session_min": "mean",
        "night_session_median_min": "mean",
        "day_session_median_min": "mean"
    }).reset_index().round(2)
    
    return df_patterns, df_hourly, room_comparison


if __name__ == "__main__":
    patterns_df, hourly_df, room_df = compute_fan_usage_patterns()
    
    patterns_path = os.path.join(EVAL_DIR, "fan_usage_patterns.csv")
    hourly_path = os.path.join(EVAL_DIR, "hourly_diurnal_profiles.csv")
    room_path = os.path.join(EVAL_DIR, "room_category_comparison.csv")
    
    patterns_df.to_csv(patterns_path, index=False)
    hourly_df.to_csv(hourly_path, index=False)
    room_df.to_csv(room_path, index=False)
    
    print("\n" + "="*80)
    print("AACI MODULE 3B: FAN USAGE PATTERNS SUMMARY:")
    print(patterns_df[['fan_id', 'room_type', 'daily_run_hours', 'modal_speed', 'median_session_min', 'night_session_median_min', 'total_eSpent_kWh', 'total_eSaved_kWh']].to_string())
    print("\n" + "="*80)
    print("ROOM CATEGORY COMPARISON:")
    print(room_df.to_string())
    print("="*80)
