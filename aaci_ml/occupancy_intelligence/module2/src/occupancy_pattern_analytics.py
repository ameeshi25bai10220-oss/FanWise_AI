"""
AACI Module 2B: Occupancy Pattern Intelligence (Descriptive Spatial & Temporal Analytics).
Computes verified empirical room occupancy patterns across NUS SDE4:
  1. Room utilization rates (% of time occupied).
  2. Average and peak occupancy levels.
  3. Occupancy duration statistics (continuous occupied session lengths).
  4. Diurnal hourly profiles (weekday vs weekend).
  5. Day-of-week profiles (Monday to Sunday).
  6. Peak usage windows.
  7. Cross-typology spatial comparison (Lecture Rooms vs Offices vs Library).
"""

import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robod_loader import load_all_rooms, ROOM_TYPES, TARGET_HEADCOUNT, TARGET_PRESENCE

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
os.makedirs(EVAL_DIR, exist_ok=True)


def compute_occupancy_sessions(presence_series: pd.Series, sampling_interval_minutes: int = 5) -> List[int]:
    """Computes lengths of continuous occupied sessions in minutes."""
    sessions = []
    current_len = 0
    for p in presence_series:
        if p == 1:
            current_len += sampling_interval_minutes
        else:
            if current_len > 0:
                sessions.append(current_len)
                current_len = 0
    if current_len > 0:
        sessions.append(current_len)
    return sessions


def analyze_room_patterns(rooms: Dict[int, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes comprehensive descriptive occupancy analytics per room."""
    summary_records = []
    diurnal_records = []

    for r_id, df in rooms.items():
        r_type = ROOM_TYPES[r_id]
        headcount = df[TARGET_HEADCOUNT]
        presence = df[TARGET_PRESENCE]

        total_obs = len(df)
        total_hours = total_obs * 5.0 / 60.0
        occupied_obs = (presence == 1).sum()
        utilization_rate = occupied_obs / total_obs

        mean_headcount_all = headcount.mean()
        mean_headcount_occupied = headcount[presence == 1].mean() if occupied_obs > 0 else 0.0
        peak_headcount = headcount.max()

        sessions = compute_occupancy_sessions(presence, 5)
        mean_session_min = np.mean(sessions) if sessions else 0.0
        median_session_min = np.median(sessions) if sessions else 0.0
        max_session_min = np.max(sessions) if sessions else 0.0

        # Hourly profile
        df["hour_int"] = df["ts"].dt.hour
        df["day_name"] = df["ts"].dt.day_name()
        df["is_wknd"] = df["ts"].dt.dayofweek >= 5

        # Peak usage window
        hourly_mean = df[df["is_wknd"] == False].groupby("hour_int")[TARGET_HEADCOUNT].mean()
        peak_hour = hourly_mean.idxmax()
        peak_hour_val = hourly_mean.max()

        summary_records.append({
            "room_id": r_id,
            "room_type": r_type,
            "total_records": total_obs,
            "monitored_hours": round(total_hours, 1),
            "utilization_rate_pct": round(utilization_rate * 100, 2),
            "mean_headcount_overall": round(mean_headcount_all, 2),
            "mean_headcount_occupied": round(mean_headcount_occupied, 2),
            "peak_headcount": int(peak_headcount),
            "total_occupied_sessions": len(sessions),
            "mean_session_duration_min": round(mean_session_min, 1),
            "median_session_duration_min": round(median_session_min, 1),
            "max_session_duration_min": round(max_session_min, 1),
            "peak_usage_hour_window": f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00",
            "peak_hour_avg_headcount": round(peak_hour_val, 2)
        })

        for h in range(24):
            wkday_val = df[(df["is_wknd"] == False) & (df["hour_int"] == h)][TARGET_HEADCOUNT].mean()
            wknd_val = df[(df["is_wknd"] == True) & (df["hour_int"] == h)][TARGET_HEADCOUNT].mean()
            diurnal_records.append({
                "room_id": r_id,
                "room_type": r_type,
                "hour": h,
                "weekday_avg_headcount": round(wkday_val, 3) if not np.isnan(wkday_val) else 0.0,
                "weekend_avg_headcount": round(wknd_val, 3) if not np.isnan(wknd_val) else 0.0
            })

    summary_df = pd.DataFrame(summary_records)
    diurnal_df = pd.DataFrame(diurnal_records)
    return summary_df, diurnal_df


def generate_pattern_plots(summary_df: pd.DataFrame, diurnal_df: pd.DataFrame):
    """Generates visualization plots for Module 2B."""
    # 1. Diurnal profiles by room
    plt.figure(figsize=(12, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, r_id in enumerate(range(1, 6)):
        sub = diurnal_df[diurnal_df["room_id"] == r_id]
        r_type = ROOM_TYPES[r_id]
        plt.plot(sub["hour"], sub["weekday_avg_headcount"], marker="o", label=f"Room {r_id} ({r_type}) - Weekday", color=colors[idx], lw=2)
        plt.plot(sub["hour"], sub["weekend_avg_headcount"], linestyle="--", alpha=0.5, color=colors[idx])

    plt.xlabel("Hour of Day (0 to 23, Singapore Time SGT)", fontsize=11)
    plt.ylabel("Mean Occupant Headcount", fontsize=11)
    plt.title("AACI Module 2B: Diurnal Headcount Profiles across NUS SDE4 Rooms", fontsize=13, fontweight="bold")
    plt.xticks(range(0, 24))
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(EVAL_DIR, "diurnal_occupancy_profiles.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Generated {p1}")

    # 2. Spatial typology comparison (Utilization vs Peak Headcount)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.35

    plt.bar(x - width/2, summary_df["utilization_rate_pct"], width, label="Utilization Rate (% of time occupied)", color="#2b5c8f")
    plt.bar(x + width/2, summary_df["peak_headcount"], width, label="Peak Headcount (max persons)", color="#e07a5f")

    plt.xlabel("Room Identifier & Typology", fontsize=11)
    plt.ylabel("Percentage (%) / Person Count", fontsize=11)
    plt.title("AACI Module 2B: Room Utilization vs Peak Headcount by Typology", fontsize=13, fontweight="bold")
    plt.xticks(x, [f"Room {r}\n({t})" for r, t in zip(summary_df["room_id"], summary_df["room_type"])], fontsize=10)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(EVAL_DIR, "spatial_utilization_comparison.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Generated {p2}")


def main():
    print("=" * 80)
    print("AACI Module 2B: Occupancy Pattern Intelligence (ROBOD Dataset)")
    print("=" * 80)

    rooms = load_all_rooms()
    summary_df, diurnal_df = analyze_room_patterns(rooms)

    print("\nDescriptive Pattern Summary across 5 SDE4 Rooms:")
    print(summary_df[["room_id", "room_type", "utilization_rate_pct", "mean_headcount_occupied", "peak_headcount", "mean_session_duration_min", "peak_usage_hour_window"]].to_string(index=False))

    summary_csv = os.path.join(EVAL_DIR, "occupancy_pattern_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved pattern summary to: {summary_csv}")

    generate_pattern_plots(summary_df, diurnal_df)
    print("Module 2B Pattern Analysis Complete.")


if __name__ == "__main__":
    main()
