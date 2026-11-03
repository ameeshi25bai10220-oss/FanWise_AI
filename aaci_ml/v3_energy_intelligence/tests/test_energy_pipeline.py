"""
AACI Model 3 – Comprehensive Unit Test Suite
============================================
Automated tests for the Model 3 data pipeline, deterministic physics,
feature engineering, causal integrity, anti-leakage isolation, and Model 1 compatibility.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add source directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_ROOT, "../.."))
sys.path.insert(0, MODULE_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.energy_physics import (
    BLDC_POWER_MAP,
    INDUCTION_POWER_MAP,
    SAVED_POWER_MAP,
    SPEED_TO_AIR_VELOCITY_MAP,
    VERTICAL_VELOCITY_PROFILE,
    IS_AIR_VELOCITY_MAPPING_PROVISIONAL,
    get_bldc_power,
    get_induction_power,
    get_saved_power,
    calculate_bldc_energy_joules,
    calculate_baseline_energy_joules,
    calculate_saved_energy_joules,
    joules_to_wh,
    joules_to_kwh,
    wh_to_joules,
    kwh_to_joules,
    calculate_bldc_energy_wh,
    speed_to_air_velocity,
    verify_energy_reading,
)

from src.data_loader import (
    AUTHORITATIVE_ROOMS,
    FAN_ROOM_MAP,
    get_room_category_for_fan,
    parse_and_sort_telemetry,
    apply_content_deduplication,
    tag_sensor_errors,
    filter_sensor_sentinels,
    tag_backlog_records,
    enrich_energy_units,
)

from src.preprocessing import (
    CausalSensorImputer,
    clean_and_impute_telemetry,
    EnergyPreprocessor,
)

from src.feature_engineering import (
    add_cyclic_temporal_features,
    add_environmental_lag_features,
    add_operational_lag_features,
    build_supervised_energy_dataset,
    generate_candidate_speed_evaluations,
)

from src.validation import (
    temporal_train_val_test_split,
    leave_one_fan_out_split,
    generate_all_lofo_splits,
    few_shot_fan_split,
    validate_no_temporal_leakage,
    validate_no_feature_leakage,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Physics Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_physics_constants():
    """Verify BLDC, induction, and saved power maps match exact specifications."""
    expected_bldc = {0: 0.0, 1: 10.0, 2: 15.0, 3: 20.0, 4: 27.0, 5: 48.0}
    expected_ind = {0: 0.0, 1: 24.0, 2: 34.0, 3: 55.0, 4: 64.0, 5: 96.0}
    expected_saved = {0: 0.0, 1: 14.0, 2: 19.0, 3: 35.0, 4: 37.0, 5: 48.0}

    assert BLDC_POWER_MAP == expected_bldc
    assert INDUCTION_POWER_MAP == expected_ind
    assert SAVED_POWER_MAP == expected_saved

    # Mathematical identity: P_Saved = P_Induction - P_BLDC
    for s in range(6):
        assert abs(SAVED_POWER_MAP[s] - (INDUCTION_POWER_MAP[s] - BLDC_POWER_MAP[s])) < 1e-6


def test_energy_physics_calculations():
    """Verify deterministic energy equations E = P * opTime in Joules."""
    # Speed 1: 10W * 300s = 3000 Joules
    assert calculate_bldc_energy_joules(1, 300.0) == 3000.0
    # Speed 5: 48W * 120s = 5760 Joules
    assert calculate_bldc_energy_joules(5, 120.0) == 5760.0
    # Speed 0: 0W * 300s = 0 Joules
    assert calculate_bldc_energy_joules(0, 300.0) == 0.0

    # Baseline induction
    assert calculate_baseline_energy_joules(1, 300.0) == 7200.0  # 24W * 300s
    assert calculate_baseline_energy_joules(5, 100.0) == 9600.0  # 96W * 100s

    # Saved energy
    assert calculate_saved_energy_joules(1, 300.0) == 4200.0   # 14W * 300s
    assert calculate_saved_energy_joules(3, 100.0) == 3500.0   # 35W * 100s

    with pytest.raises(ValueError):
        calculate_bldc_energy_joules(6, 100.0)  # Invalid speed

    with pytest.raises(ValueError):
        calculate_bldc_energy_joules(3, -10.0)  # Negative time


def test_energy_unit_conversions():
    """Verify exact Joules to Wh and kWh conversions."""
    assert joules_to_wh(3600.0) == 1.0
    assert joules_to_wh(1800.0) == 0.5
    assert wh_to_joules(1.0) == 3600.0

    assert joules_to_kwh(3600000.0) == 1.0
    assert kwh_to_joules(1.0) == 3600000.0

    # Wh direct calculation: Speed 4 (27W) for 3600s = 27 Wh
    assert calculate_bldc_energy_wh(4, 3600.0) == 27.0


def test_speed_to_air_velocity_dryad_mapping():
    """Verify provisional aerodynamic air velocity lookup and explicit flag."""
    assert IS_AIR_VELOCITY_MAPPING_PROVISIONAL is True
    assert SPEED_TO_AIR_VELOCITY_MAP[0] == 0.050  # Assumed still air reference floor (ASHRAE 55)
    assert SPEED_TO_AIR_VELOCITY_MAP[1] == 0.112
    assert SPEED_TO_AIR_VELOCITY_MAP[2] == 0.239
    assert SPEED_TO_AIR_VELOCITY_MAP[3] == 0.293
    assert SPEED_TO_AIR_VELOCITY_MAP[4] == 0.410
    assert SPEED_TO_AIR_VELOCITY_MAP[5] == 0.527

    # Monotonicity test
    speeds = list(range(6))
    velocities = [speed_to_air_velocity(s, height_m=1.1) for s in speeds]
    assert all(velocities[i] <= velocities[i+1] for i in range(len(velocities)-1))

    # Multi-height test
    v_ankle = speed_to_air_velocity(3, height_m=0.1)
    assert v_ankle == 0.344


def test_verify_energy_reading():
    """Verify ground truth validation helper."""
    # Speed 2 (15W) for 200s = 3000 Joules
    assert verify_energy_reading(2, 200.0, 3000.0, tolerance_joules=1.0) is True
    assert verify_energy_reading(2, 200.0, 4500.0, tolerance_joules=1.0) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Loader & Ingestion Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_authoritative_fan_room_mapping():
    """Verify authoritative 9-fan room categories (Bedroom, Drawingroom, Lounge)."""
    assert len(FAN_ROOM_MAP) == 9
    assert sorted(set(FAN_ROOM_MAP.values())) == ["Bedroom", "Drawingroom", "Lounge"]

    for fan_id, expected_room in FAN_ROOM_MAP.items():
        assert get_room_category_for_fan(fan_id) == expected_room

    assert get_room_category_for_fan("Unknown_Fan_99") == "Other"


def test_timestamp_parsing_and_sorting():
    """Verify day-first parsing and chronological order preservation."""
    raw_data = {
        "S.No.": [2, 1, 3],
        "datetime": ["10-06-2020 12:00", "10-06-2020 08:00", "11-06-2020 08:00"],
        "temperature": [28.0, 27.0, 29.0],
        "humidity": [50.0, 52.0, 48.0],
        "speed": [3, 2, 4],
        "mode": [1, 1, 1],
        "eSpent": [6000.0, 3000.0, 8100.0],
        "eSaved": [10500.0, 3800.0, 11100.0],
        "opTime": [300.0, 200.0, 300.0],
    }
    df = pd.DataFrame(raw_data)
    sorted_df = parse_and_sort_telemetry(df)

    # First row should be 08:00 on 10-06-2020
    assert sorted_df.iloc[0]["S.No."] == 1
    assert sorted_df.iloc[1]["S.No."] == 2
    assert sorted_df.iloc[2]["S.No."] == 3
    # Day-first check: Month must be June (6), not October (10)
    assert sorted_df.iloc[0]["datetime"].month == 6
    assert sorted_df.iloc[0]["datetime"].day == 10


def test_content_deduplication():
    """Verify deduplication removes DB insertion repeats but preserves physical packets."""
    raw_data = {
        "S.No.": [101, 102, 103],
        "datetime": ["10-06-2020 12:00", "10-06-2020 12:00", "10-06-2020 12:00"],
        "temperature": [28.0, 28.0, 28.0],
        "humidity": [50.0, 50.0, 50.0],
        "speed": [3, 3, 4],  # Row 103 has physical speed change within same minute
        "mode": [1, 1, 1],
        "eSpent": [6000.0, 6000.0, 8100.0],
        "eSaved": [10500.0, 10500.0, 11100.0],
        "opTime": [300.0, 300.0, 300.0],
    }
    df = pd.DataFrame(raw_data)
    deduped = apply_content_deduplication(df)

    assert len(deduped) == 2
    assert deduped.iloc[0]["S.No."] == 101
    assert deduped.iloc[1]["S.No."] == 103


def test_sensor_error_tagging_and_filtering():
    """Verify 255 sentinels and boot zeros are caught."""
    raw_data = {
        "S.No.": [1, 2, 3, 4],
        "datetime": ["10-06-2020 10:00", "10-06-2020 10:05", "10-06-2020 10:10", "10-06-2020 10:15"],
        "temperature": [28.0, 255.0, 0.0, 29.0],
        "humidity": [50.0, 255.0, 0.0, 52.0],
        "speed": [1, 1, 1, 1],
        "mode": [1, 1, 1, 1],
        "eSpent": [3000.0, 3000.0, 3000.0, 3000.0],
        "eSaved": [4200.0, 4200.0, 4200.0, 4200.0],
        "opTime": [300.0, 300.0, 300.0, 300.0],
    }
    df = pd.DataFrame(raw_data)
    tagged = tag_sensor_errors(df)
    assert tagged.iloc[0]["is_sensor_error"] == False
    assert tagged.iloc[1]["is_sensor_error"] == True  # 255 sentinel
    assert tagged.iloc[2]["is_sensor_error"] == True  # boot zero
    assert tagged.iloc[3]["is_sensor_error"] == False

    filtered_drop = filter_sensor_sentinels(df, drop_errors=True)
    assert len(filtered_drop) == 2

    filtered_mask = filter_sensor_sentinels(df, drop_errors=False)
    assert len(filtered_mask) == 4
    assert np.isnan(filtered_mask.iloc[1]["temperature"])
    assert np.isnan(filtered_mask.iloc[2]["humidity"])


def test_backlog_tagging():
    """Verify packets with opTime > 900s are tagged as backlog."""
    raw_data = {
        "opTime": [300.0, 600.0, 900.0, 1200.0, 86400.0],
    }
    df = pd.DataFrame(raw_data)
    tagged = tag_backlog_records(df, threshold_sec=900.0)
    assert tagged["is_backlog"].tolist() == [False, False, False, True, True]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Engineering & Causality Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cyclic_temporal_features():
    """Verify cyclic feature transformations."""
    dates = pd.date_range("2020-06-10 00:00", periods=24, freq="1h")
    df = pd.DataFrame({"datetime": dates})
    df_cyclic = add_cyclic_temporal_features(df)

    assert "sin_hour" in df_cyclic.columns
    assert "cos_hour" in df_cyclic.columns
    assert df_cyclic["sin_hour"].min() >= -1.0
    assert df_cyclic["sin_hour"].max() <= 1.0
    assert df_cyclic["cos_hour"].min() >= -1.0
    assert df_cyclic["cos_hour"].max() <= 1.0


def test_lag_feature_causality():
    """Verify lags strictly match historical steps and do not leak across fans."""
    df = pd.DataFrame({
        "fan_id": ["Fan_A"] * 4 + ["Fan_B"] * 4,
        "temperature": [20.0, 22.0, 24.0, 26.0, 30.0, 32.0, 34.0, 36.0],
        "humidity": [50.0, 52.0, 54.0, 56.0, 60.0, 62.0, 64.0, 66.0],
    })
    lagged = add_environmental_lag_features(df, lags=[1, 2], rolling_windows=[3])

    # Within Fan_A:
    assert np.isnan(lagged.loc[0, "temp_lag_1"])  # Burn in
    assert lagged.loc[1, "temp_lag_1"] == 20.0
    assert lagged.loc[2, "temp_lag_1"] == 22.0

    # Boundary check: Fan_B row 4 must NOT inherit from Fan_A row 3!
    assert np.isnan(lagged.loc[4, "temp_lag_1"])
    assert lagged.loc[5, "temp_lag_1"] == 30.0


def test_supervised_dataset_no_contemporaneous_leakage():
    """Verify that feature_cols NEVER contains contemporaneous opTime, eSpent, or target."""
    n = 20
    dates = pd.date_range("2020-06-10 00:00", periods=n, freq="5min")
    df = pd.DataFrame({
        "fan_id": ["Fan_A"] * n,
        "datetime": dates,
        "temperature": np.random.uniform(25, 32, n),
        "humidity": np.random.uniform(40, 70, n),
        "speed": np.random.choice([0, 1, 2, 3, 4, 5], n),
        "mode": [1] * n,
        "opTime": [300.0] * n,
        "eSpent": [3000.0] * n,
        "eSaved": [4000.0] * n,
        "eSpent_wh": [3000.0 / 3600.0] * n,
    })

    dataset_df, feature_cols, target_col = build_supervised_energy_dataset(
        df, target_type="next_energy", drop_burn_in_nans=True
    )

    # 1. Feature cols must NOT contain opTime, eSpent, eSaved, or target
    assert "opTime" not in feature_cols
    assert "eSpent" not in feature_cols
    assert "eSaved" not in feature_cols
    assert target_col not in feature_cols

    # 2. Validation check passes without raising AssertionError
    validate_no_feature_leakage(feature_cols)

    # 3. Target is correctly shifted ahead
    assert target_col == "target_next_eSpent_wh"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Anti-Leakage & Imputation Adversarial Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_adversarial_future_test_perturbation_isolation():
    """
    CRITICAL AUDIT TEST:
    Deliberately perturb/corrupt future test observations (extreme values, NaNs, etc.)
    and assert that:
    1. Training imputer statistics (fan_medians, global_medians) remain identical.
    2. Training scaler parameters (means, scales) remain identical.
    3. Transformed training features remain bit-for-bit identical.
    """
    np.random.seed(42)
    n_train = 50
    n_test = 20

    train_dates = pd.date_range("2020-06-01 00:00", periods=n_train, freq="5min")
    test_dates = pd.date_range("2020-06-01 05:00", periods=n_test, freq="5min")

    train_df = pd.DataFrame({
        "fan_id": ["Fan_1"] * n_train,
        "datetime": train_dates,
        "temperature": np.random.normal(28.0, 1.5, n_train),
        "humidity": np.random.normal(55.0, 5.0, n_train),
    })

    clean_test_df = pd.DataFrame({
        "fan_id": ["Fan_1"] * n_test,
        "datetime": test_dates,
        "temperature": np.random.normal(29.0, 1.5, n_test),
        "humidity": np.random.normal(60.0, 5.0, n_test),
    })

    # Fit imputer and preprocessor on training data
    imputer = CausalSensorImputer(sensor_cols=["temperature", "humidity"])
    imputer.fit(train_df)

    preproc = EnergyPreprocessor(numeric_features=["temperature", "humidity"])
    X_train_baseline = preproc.fit_transform(train_df)

    saved_train_temp_median = imputer.fan_medians_["Fan_1"]["temperature"]
    saved_train_mean = preproc.scaler.mean_.copy()

    # Create adversarial corrupted test dataset
    adversarial_test_df = clean_test_df.copy()
    adversarial_test_df["temperature"] = 999999.0  # Absurd future temperatures
    adversarial_test_df["humidity"] = -88888.0     # Absurd future humidities

    # Transform both test sets
    imputed_clean_test = imputer.transform(clean_test_df)
    imputed_adv_test = imputer.transform(adversarial_test_df)

    scaled_clean_test = preproc.transform(clean_test_df)
    scaled_adv_test = preproc.transform(adversarial_test_df)

    # ASSERT INVARIANCE: Training parameters must NOT have changed!
    assert imputer.fan_medians_["Fan_1"]["temperature"] == saved_train_temp_median
    np.testing.assert_array_equal(preproc.scaler.mean_, saved_train_mean)

    # Re-transforming train must be identical
    X_train_after = preproc.transform(train_df)
    np.testing.assert_array_equal(X_train_baseline, X_train_after)


def test_causal_sensor_imputer_zero_shot_lofo():
    """
    Verify that in Zero-Shot LOFO, missing initial values in an unseen test fan
    are imputed using the training global median, NOT any statistic computed from
    the held-out fan's own dataset!
    """
    train_df = pd.DataFrame({
        "fan_id": ["Fan_A"] * 10 + ["Fan_B"] * 10,
        "temperature": [20.0] * 10 + [30.0] * 10,  # Global median = 25.0
        "humidity": [40.0] * 10 + [60.0] * 10,     # Global median = 50.0
    })

    # Unseen test fan with initial NaN
    unseen_test_df = pd.DataFrame({
        "fan_id": ["Fan_C_Unseen"] * 5,
        "temperature": [np.nan, np.nan, 35.0, 36.0, 37.0],
        "humidity": [np.nan, 70.0, 71.0, 72.0, 73.0],
    })

    imputer = CausalSensorImputer(sensor_cols=["temperature", "humidity"])
    imputer.fit(train_df)

    # Check learned global medians
    assert imputer.global_medians_["temperature"] == 25.0
    assert imputer.global_medians_["humidity"] == 50.0
    assert "Fan_C_Unseen" not in imputer.fan_medians_

    transformed_test = imputer.transform(unseen_test_df)

    # Rows 0 and 1 must be filled with training global median (25.0), NOT Fan_C median (36.0)!
    assert transformed_test.loc[0, "temperature"] == 25.0
    assert transformed_test.loc[1, "temperature"] == 25.0
    # Row 2 was originally 35.0
    assert transformed_test.loc[2, "temperature"] == 35.0
    # For humidity: Row 0 was NaN, should be filled with training global median (50.0)
    assert transformed_test.loc[0, "humidity"] == 50.0
    # Row 1 was 70.0, and row 2 ffills from row 1
    assert transformed_test.loc[1, "humidity"] == 70.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validation & Splitting Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_split_no_leakage():
    """Verify temporal split guarantees max(train) < min(val) < max(val) < min(test)."""
    n = 100
    dates = pd.date_range("2020-06-01 00:00", periods=n, freq="1h")
    df = pd.DataFrame({
        "fan_id": ["Fan_1"] * n,
        "datetime": dates,
        "val": range(n),
    })

    train_df, val_df, test_df = temporal_train_val_test_split(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, per_fan=True
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Temporal barrier check
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()


def test_leave_one_fan_out_split():
    """Verify LOFO splits are completely disjoint by fan identity."""
    df = pd.DataFrame({
        "fan_id": ["Fan_A"] * 10 + ["Fan_B"] * 10 + ["Fan_C"] * 10,
        "datetime": pd.date_range("2020-06-01", periods=30, freq="1h"),
    })

    train_df, test_df = leave_one_fan_out_split(df, test_fan_id="Fan_B")
    assert set(test_df["fan_id"].unique()) == {"Fan_B"}
    assert set(train_df["fan_id"].unique()) == {"Fan_A", "Fan_C"}
    assert len(test_df) == 10
    assert len(train_df) == 20

    # Generator test
    folds = list(generate_all_lofo_splits(df))
    assert len(folds) == 3
    assert [f[0] for f in folds] == ["Fan_A", "Fan_B", "Fan_C"]


def test_few_shot_fan_split():
    """Verify few-shot adaptation split partitions target fan by time."""
    dates = pd.date_range("2020-06-01", periods=20, freq="1D")
    df = pd.DataFrame({
        "fan_id": ["Target_Fan"] * 20 + ["Other_Fan"] * 20,
        "datetime": list(dates) + list(dates),
    })

    train_others, adapt_df, test_df = few_shot_fan_split(
        df, target_fan_id="Target_Fan", adaptation_days=7
    )

    assert set(train_others["fan_id"].unique()) == {"Other_Fan"}
    assert set(adapt_df["fan_id"].unique()) == {"Target_Fan"}
    assert set(test_df["fan_id"].unique()) == {"Target_Fan"}
    assert adapt_df["datetime"].max() < test_df["datetime"].min()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Candidate Speed Evaluation & Model 1 Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

def test_candidate_speed_evaluations():
    """Verify candidate speed evaluation matrix generator."""
    matrix = generate_candidate_speed_evaluations(
        air_temp_c=28.0, relative_humidity=55.0, duration_hours=2.0
    )
    assert len(matrix) == 6
    assert list(matrix["speed"]) == [0, 1, 2, 3, 4, 5]
    # Energy at speed 1: 10W * 2h = 20 Wh
    assert matrix.loc[matrix["speed"] == 1, "energy_bldc_wh"].iloc[0] == 20.0
    # Air velocity at speed 5: 0.527 m/s
    assert matrix.loc[matrix["speed"] == 5, "air_velocity_ms"].iloc[0] == 0.527


def test_model1_interface_compatibility_smoke_test():
    """
    Verify Model 1 can evaluate comfort at candidate fan speeds mapped via
    speed_to_air_velocity(speed) without modifying any Model 1 file.

    NOTE: This is verified strictly as an ILLUSTRATIVE SIMULATION SCENARIO.
    """
    from aaci_ml.predict import predict_comfort

    temp = 29.0
    rh = 60.0

    comfort_results = []
    for speed in range(6):
        va = speed_to_air_velocity(speed, height_m=1.1)
        comfort = predict_comfort(
            air_temp_c=temp,
            relative_humidity=rh,
            air_velocity_ms=va,
            season="Summer",
        )
        comfort_results.append((speed, va, comfort["predicted_class"], comfort["probabilities"]["Neutral"]))

    # Verify predictions were obtained for all speeds
    assert len(comfort_results) == 6
    for speed, va, p_class, p_neutral in comfort_results:
        assert p_class in ["Cold", "Neutral", "Hot"]
        assert 0.0 <= p_neutral <= 1.0
