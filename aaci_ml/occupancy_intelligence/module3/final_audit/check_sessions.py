"""
Audit session durations, overnight continuous operation, and daily runtime in bedrooms.
"""
import sys
import os
import pandas as pd
import numpy as np

SRC_DIR = os.path.abspath("aaci_ml/occupancy_intelligence/module3/src")
sys.path.insert(0, SRC_DIR)

from rdd_loader import load_fan_telemetry

for fan_id in ["Bedroom_Fan_1", "Bedroom_Fan_2", "Bedroom_Fan_3", "Drawingroom_Fan_1"]:
    df = load_fan_telemetry(fan_id)
    
    # 1. Look at packet time gaps when speed > 0
    df_on = df[df['speed'] > 0].copy()
    dt_diff_min = df_on['dt'].diff().dt.total_seconds() / 60.0
    
    print(f"\n==========================================")
    print(f"FAN: {fan_id}")
    print(f"Total rows: {len(df)}, Active (speed > 0) rows: {len(df_on)} ({len(df_on)/len(df)*100:.1f}%)")
    print(f"Time gap between consecutive ON packets (minutes) percentiles:")
    print(dt_diff_min.quantile([0.5, 0.75, 0.9, 0.95, 0.99]))
    
    # 2. Look at night time (22:00 to 07:00) vs day time (07:00 to 22:00)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 7)).astype(int)
    
    night_prob_on = df[df['is_night'] == 1]['is_on'].mean() * 100.0
    day_prob_on = df[df['is_night'] == 0]['is_on'].mean() * 100.0
    
    print(f"Night active probability (22:00 - 07:00): {night_prob_on:.1f}%")
    print(f"Day active probability (07:00 - 22:00): {day_prob_on:.1f}%")
    
    # 3. Look at session duration percentiles
    from fan_usage_analytics import extract_sessions
    sessions = extract_sessions(df, max_gap_seconds=900)
    print(f"Total extracted sessions: {len(sessions)}")
    print(f"Session duration percentiles (minutes):")
    print(sessions['duration_minutes'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))
    
    # Check sessions > 60 minutes
    long_sessions = sessions[sessions['duration_minutes'] >= 60]
    print(f"Sessions >= 1 hour: {len(long_sessions)} ({len(long_sessions)/len(sessions)*100:.1f}%)")
    print(f"Total hours in sessions >= 1 hour: {long_sessions['duration_minutes'].sum()/60.0:.1f} hrs")
    print(f"Total hours in sessions < 1 hour: {sessions[sessions['duration_minutes'] < 60]['duration_minutes'].sum()/60.0:.1f} hrs")

