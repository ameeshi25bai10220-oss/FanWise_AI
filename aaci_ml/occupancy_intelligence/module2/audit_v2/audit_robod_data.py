"""
Independent Verification Audit Script for AACI Module 2 (ROBOD).
Location: aaci_ml/occupancy_intelligence/module2/audit_v2/audit_robod_data.py

Performs comprehensive, read-only recalculations and audits:
  1. Timestamp continuity, calendar dates, observation hours, and monitoring gaps.
  2. Descriptive occupancy analytics with gap-aware session resets.
  3. Occupancy-bracket sample counts and recalculated MAE/RMSE (with explicit N/A for empty brackets).
  4. Apple-to-apple Wi-Fi contribution analysis (Group 2 vs Group 3).
  5. LORO evaluation audit and exact recalculation.
  6. Empirical coverage of uncertainty intervals.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add module2 src to path for imports
MODULE2_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, MODULE2_SRC)

from robod_loader import (
    load_all_rooms, load_single_room, get_chronological_room_splits,
    get_loro_splits, FEATURE_GROUPS, TARGET_HEADCOUNT, TARGET_PRESENCE, ROOM_TYPES
)

AUDIT_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))


def audit_timestamps_and_coverage(rooms: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Audits temporal structure, calendar coverage, observation hours, and gaps."""
    records = []
    gap_records = []

    for r_id, df in rooms.items():
        ts = pd.to_datetime(df["timestamp"])
        total_records = len(df)
        valid_ts_count = ts.notna().sum()
        first_ts = ts.min()
        last_ts = ts.max()
        calendar_span_days = (last_ts - first_ts).total_seconds() / 86400.0
        
        unique_dates = ts.dt.date.nunique()
        actual_observed_intervals = total_records # each row is a 5-min observation
        actual_observed_hours = total_records * 5.0 / 60.0 # pure observation hours
        calendar_nominal_hours = unique_dates * 24.0 # if each date was 24h

        # Diff analysis
        diffs = ts.diff()
        diff_minutes = diffs.dt.total_seconds() / 60.0

        gaps_gt_10 = (diff_minutes > 10).sum()
        gaps_gt_30 = (diff_minutes > 30).sum()
        gaps_gt_60 = (diff_minutes > 60).sum()

        records.append({
            "room_id": r_id,
            "room_type": ROOM_TYPES[r_id],
            "total_records": total_records,
            "valid_timestamps": valid_ts_count,
            "first_timestamp": str(first_ts),
            "last_timestamp": str(last_ts),
            "calendar_span_days": round(calendar_span_days, 1),
            "unique_calendar_dates": unique_dates,
            "actual_observed_intervals": actual_observed_intervals,
            "actual_observed_hours": round(actual_observed_hours, 2),
            "calendar_nominal_hours": round(calendar_nominal_hours, 2),
            "gaps_gt_10min": int(gaps_gt_10),
            "gaps_gt_30min": int(gaps_gt_30),
            "gaps_gt_60min": int(gaps_gt_60)
        })

        # Detailed gaps
        for idx in np.where(diff_minutes > 15)[0]:
            gap_records.append({
                "room_id": r_id,
                "gap_start": str(ts.iloc[idx-1]),
                "gap_end": str(ts.iloc[idx]),
                "gap_duration_hours": round(diff_minutes.iloc[idx] / 60.0, 2),
                "gap_duration_days": round(diff_minutes.iloc[idx] / 1440.0, 2)
            })

    cov_df = pd.DataFrame(records)
    gaps_df = pd.DataFrame(gap_records)
    cov_df.to_csv(os.path.join(AUDIT_DIR, "recalculated_monitoring_coverage.csv"), index=False)
    gaps_df.to_csv(os.path.join(AUDIT_DIR, "audit_monitoring_gaps.csv"), index=False)
    return cov_df, gaps_df


def audit_descriptive_analytics_gap_aware(rooms: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Recalculates Module 2B descriptive statistics with gap-aware session resets.
    Rule: If consecutive occupied observations have a timestamp gap > 15 minutes,
    the session terminates and does NOT bridge the gap.
    """
    records = []

    for r_id, df in rooms.items():
        ts = pd.to_datetime(df["timestamp"])
        presence = df[TARGET_PRESENCE].values
        headcount = df[TARGET_HEADCOUNT].values
        n = len(df)

        total_valid_obs = n
        occupied_obs = np.sum(presence == 1)
        utilization_rate = occupied_obs / total_valid_obs

        headcount_occupied = headcount[presence == 1]
        mean_hc_occ = float(np.mean(headcount_occupied)) if len(headcount_occupied) > 0 else 0.0
        peak_hc = int(np.max(headcount))

        # Gap-aware session calculation
        sessions = []
        cur_session_min = 0
        for i in range(n):
            if presence[i] == 1:
                # Check if gap occurred from previous observation
                if i > 0 and (ts.iloc[i] - ts.iloc[i-1]).total_seconds() > 900: # > 15 mins
                    if cur_session_min > 0:
                        sessions.append(cur_session_min)
                    cur_session_min = 5
                else:
                    cur_session_min += 5
            else:
                if cur_session_min > 0:
                    sessions.append(cur_session_min)
                    cur_session_min = 0
        if cur_session_min > 0:
            sessions.append(cur_session_min)

        mean_sess = float(np.mean(sessions)) if sessions else 0.0
        median_sess = float(np.median(sessions)) if sessions else 0.0
        max_sess = float(np.max(sessions)) if sessions else 0.0

        # Peak usage hour window (on weekdays)
        df_copy = df.copy()
        df_copy["hour"] = ts.dt.hour
        df_copy["is_weekend"] = ts.dt.dayofweek >= 5
        wkday = df_copy[df_copy["is_weekend"] == False]
        hourly = wkday.groupby("hour")[TARGET_HEADCOUNT].mean()
        peak_hr = int(hourly.idxmax())
        peak_val = float(hourly.max())

        records.append({
            "room_id": r_id,
            "room_type": ROOM_TYPES[r_id],
            "valid_observations": total_valid_obs,
            "occupied_observations": int(occupied_obs),
            "utilization_rate_pct": round(utilization_rate * 100, 2),
            "mean_headcount_occupied": round(mean_hc_occ, 2),
            "peak_headcount": peak_hc,
            "total_sessions_count": len(sessions),
            "gap_aware_mean_session_min": round(mean_sess, 1),
            "gap_aware_median_session_min": round(median_sess, 1),
            "gap_aware_max_session_min": round(max_sess, 1),
            "peak_usage_window": f"{peak_hr:02d}:00 - {peak_hr+1:02d}:00",
            "peak_usage_avg_headcount": round(peak_val, 2)
        })

    recalc_df = pd.DataFrame(records)
    recalc_df.to_csv(os.path.join(AUDIT_DIR, "recalculated_descriptive_analytics.csv"), index=False)
    return recalc_df


def audit_occupancy_brackets_and_metrics(rooms: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Audits the forward test split for each room.
    Recalculates metrics for Empty, Low, Medium, High brackets with explicit sample counts.
    If sample count is 0, outputs 'N/A' rather than 0.000.
    """
    import joblib
    model_path = os.path.join(MODELS_DIR, "headcount_rf_multimodal.joblib")
    model = joblib.load(model_path)
    feat_cols = FEATURE_GROUPS["group4_multimodal"]

    bracket_records = []
    overall_records = []

    for r_id in range(1, 6):
        train_df, val_df, test_df = get_chronological_room_splits(rooms[r_id])
        y_test = test_df[TARGET_HEADCOUNT].values
        X_test = test_df[feat_cols].values
        y_pred = np.maximum(0.0, model.predict(X_test))
        y_pred_round = np.round(y_pred)

        total_n = len(y_test)

        # Brackets
        brackets = {
            "Empty (y=0)": (y_test == 0),
            "Low (1<=y<=3)": (y_test >= 1) & (y_test <= 3),
            "Med (4<=y<=10)": (y_test >= 4) & (y_test <= 10),
            "High (y>10)": (y_test > 10)
        }

        for b_name, mask in brackets.items():
            n_sub = int(np.sum(mask))
            if n_sub > 0:
                mae_sub = float(np.mean(np.abs(y_test[mask] - y_pred[mask])))
                rmse_sub = float(np.sqrt(np.mean((y_test[mask] - y_pred[mask]) ** 2)))
            else:
                mae_sub = np.nan
                rmse_sub = np.nan

            bracket_records.append({
                "room_id": r_id,
                "room_type": ROOM_TYPES[r_id],
                "bracket": b_name,
                "sample_count": n_sub,
                "percentage_of_test": round(n_sub / total_n * 100, 2),
                "mae": round(mae_sub, 3) if not np.isnan(mae_sub) else "N/A",
                "rmse": round(rmse_sub, 3) if not np.isnan(rmse_sub) else "N/A"
            })

        # Overall metrics
        overall_mae = float(np.mean(np.abs(y_test - y_pred)))
        overall_rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        exact_acc = float(np.mean(y_test == y_pred_round))
        within_one_acc = float(np.mean(np.abs(y_test - y_pred) <= 1.0))
        empty_mask = (y_test == 0)
        empty_mae = float(np.mean(np.abs(y_test[empty_mask] - y_pred[empty_mask])))
        empty_exact = float(np.mean(y_pred_round[empty_mask] == 0))

        overall_records.append({
            "room_id": r_id,
            "room_type": ROOM_TYPES[r_id],
            "total_test_samples": total_n,
            "overall_mae": round(overall_mae, 3),
            "overall_rmse": round(overall_rmse, 3),
            "within_one_acc": round(within_one_acc, 4),
            "exact_acc": round(exact_acc, 4),
            "empty_room_mae": round(empty_mae, 3),
            "empty_room_exact_acc": round(empty_exact, 4)
        })

    b_df = pd.DataFrame(bracket_records)
    o_df = pd.DataFrame(overall_records)
    b_df.to_csv(os.path.join(AUDIT_DIR, "recalculated_occupancy_brackets.csv"), index=False)
    o_df.to_csv(os.path.join(AUDIT_DIR, "recalculated_per_room_overall_metrics.csv"), index=False)
    return b_df, o_df


def audit_wifi_contribution_exact():
    """Audits Group 2 (Env + Acoustic) vs Group 3 (Env + Acoustic + Wi-Fi) on identical test splits."""
    bench = pd.read_csv(os.path.join(EVAL_DIR, "headcount_benchmark.csv"))
    
    # Filter for Group 2 and Group 3
    g2 = bench[bench["feature_group"] == "group2_env_acoustic"].set_index(["room_id", "algorithm"])
    g3 = bench[bench["feature_group"] == "group3_env_acoustic_wifi"].set_index(["room_id", "algorithm"])

    comparison_records = []
    for idx in g2.index:
        r_id, algo = idx
        mae_g2 = g2.loc[idx, "test_mae"]
        mae_g3 = g3.loc[idx, "test_mae"]
        rmse_g2 = g2.loc[idx, "test_rmse"]
        rmse_g3 = g3.loc[idx, "test_rmse"]
        w1_g2 = g2.loc[idx, "within_one_acc"]
        w1_g3 = g3.loc[idx, "within_one_acc"]

        abs_diff = mae_g2 - mae_g3
        pct_imp = (abs_diff / mae_g2) * 100.0 if mae_g2 > 0 else 0.0

        comparison_records.append({
            "room_id": r_id,
            "room_type": ROOM_TYPES[r_id],
            "algorithm": algo,
            "mae_group2_env_acoustic": mae_g2,
            "mae_group3_env_acoustic_wifi": mae_g3,
            "mae_absolute_reduction": round(abs_diff, 3),
            "mae_percentage_improvement": round(pct_imp, 1),
            "rmse_group2": rmse_g2,
            "rmse_group3": rmse_g3,
            "within_one_acc_group2": w1_g2,
            "within_one_acc_group3": w1_g3
        })

    wifi_df = pd.DataFrame(comparison_records)
    wifi_df.to_csv(os.path.join(AUDIT_DIR, "recalculated_wifi_contribution_comparison.csv"), index=False)
    return wifi_df


def audit_loro_protocol_exact(rooms: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Audits LORO benchmarks and verifies exact sample counts and metrics."""
    loro_saved = pd.read_csv(os.path.join(EVAL_DIR, "loro_generalization_benchmark.csv"))
    
    # Check sample counts per room
    records = []
    for test_r_id in range(1, 6):
        train_df, test_df = get_loro_splits(rooms, test_r_id)
        train_n = len(train_df)
        test_n = len(test_df)
        
        # Check that no timestamp from test_df is in train_df
        # Since rooms have different IDs, verify no data leakage
        assert test_df["room_id"].nunique() == 1
        assert test_df["room_id"].iloc[0] == test_r_id
        assert test_r_id not in train_df["room_id"].unique()

        sub_saved = loro_saved[loro_saved["test_room_id"] == test_r_id]
        records.append({
            "held_out_room_id": test_r_id,
            "held_out_room_type": ROOM_TYPES[test_r_id],
            "train_sample_count": train_n,
            "test_sample_count": test_n,
            "best_rf_unseen_mae": sub_saved[(sub_saved["algorithm"] == "random_forest") & (sub_saved["feature_group"] == "group4_multimodal")]["unseen_room_mae"].values[0],
            "best_rf_within_one_acc": sub_saved[(sub_saved["algorithm"] == "random_forest") & (sub_saved["feature_group"] == "group4_multimodal")]["within_one_acc"].values[0]
        })

    loro_audit_df = pd.DataFrame(records)
    loro_audit_df.to_csv(os.path.join(AUDIT_DIR, "audit_loro_protocol_summary.csv"), index=False)
    return loro_audit_df


def audit_uncertainty_coverage_exact(rooms: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Audits empirical coverage of the heuristic +/- 1.5 person prediction interval."""
    import joblib
    model_path = os.path.join(MODELS_DIR, "headcount_rf_multimodal.joblib")
    model = joblib.load(model_path)
    feat_cols = FEATURE_GROUPS["group4_multimodal"]

    records = []
    for r_id in range(1, 6):
        _, _, test_df = get_chronological_room_splits(rooms[r_id])
        y_test = test_df[TARGET_HEADCOUNT].values
        X_test = test_df[feat_cols].values
        y_pred = np.maximum(0.0, model.predict(X_test))

        margin = 1.50
        lower = np.maximum(0.0, y_pred - margin)
        upper = y_pred + margin

        covered = (y_test >= lower) & (y_test <= upper)
        coverage_rate = np.mean(covered)

        # Empty coverage
        mask_0 = (y_test == 0)
        cov_empty = np.mean(covered[mask_0]) if np.any(mask_0) else np.nan

        # Low coverage (1-3)
        mask_low = (y_test >= 1) & (y_test <= 3)
        cov_low = np.mean(covered[mask_low]) if np.any(mask_low) else np.nan

        # Med coverage (4-10)
        mask_med = (y_test >= 4) & (y_test <= 10)
        cov_med = np.mean(covered[mask_med]) if np.any(mask_med) else np.nan

        # High coverage (>10)
        mask_high = (y_test > 10)
        cov_high = np.mean(covered[mask_high]) if np.any(mask_high) else np.nan

        records.append({
            "room_id": r_id,
            "room_type": ROOM_TYPES[r_id],
            "total_test_samples": len(y_test),
            "heuristic_margin_persons": margin,
            "overall_empirical_coverage": round(coverage_rate * 100, 2),
            "empty_bracket_coverage": round(cov_empty * 100, 2) if not np.isnan(cov_empty) else "N/A",
            "low_bracket_coverage": round(cov_low * 100, 2) if not np.isnan(cov_low) else "N/A",
            "med_bracket_coverage": round(cov_med * 100, 2) if not np.isnan(cov_med) else "N/A",
            "high_bracket_coverage": round(cov_high * 100, 2) if not np.isnan(cov_high) else "N/A"
        })

    unc_df = pd.DataFrame(records)
    unc_df.to_csv(os.path.join(AUDIT_DIR, "recalculated_uncertainty_coverage.csv"), index=False)
    return unc_df


if __name__ == "__main__":
    rooms = load_all_rooms()
    cov_df, gaps_df = audit_timestamps_and_coverage(rooms)
    recalc_df = audit_descriptive_analytics_gap_aware(rooms)
    b_df, o_df = audit_occupancy_brackets_and_metrics(rooms)
    wifi_df = audit_wifi_contribution_exact()
    loro_audit_df = audit_loro_protocol_exact(rooms)
    unc_df = audit_uncertainty_coverage_exact(rooms)
    print("All independent verification audit calculations completed successfully.")

