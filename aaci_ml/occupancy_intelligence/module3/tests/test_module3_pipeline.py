"""
Automated unit tests for AACI Module 3: Fan & Building Energy Intelligence.
Verifies data loading, cleaning, chronological order, power semantics, predictive models, and inference API.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Insert Module 3 src to sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE3_DIR = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(MODULE3_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rdd_loader import (
    load_fan_telemetry,
    get_rdd_file_path,
    POWER_LOOKUP,
    SAVED_LOOKUP,
    INDUCTION_LOOKUP,
    DEFAULT_RDD_DIR
)
from fan_usage_analytics import extract_sessions
from train_energy_models import create_feature_matrix, FEATURE_COLS
from energy_predictor import EnergyIntelligencePredictor


def test_rdd_file_resolution():
    """Verify that fan names and paths resolve properly."""
    path1 = get_rdd_file_path("Bedroom_Fan_1")
    assert os.path.exists(path1), f"Path does not exist: {path1}"
    assert "Bedroom_Fan_1.csv" in path1


def test_rdd_loader_cleaning_and_columns():
    """Verify data loading, column generation, and sensor cleaning."""
    df = load_fan_telemetry("Lounge_Fan_1")
    assert len(df) > 0
    expected_cols = [
        'dt', 'temperature', 'humidity', 'speed', 'mode', 'opTime',
        'eSpent', 'eSaved', 'fan_id', 'room_type', 'hour', 'eSpent_Wh',
        'power_W', 'baseline_induction_W'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
        
    # Check that 255 error codes were removed
    assert (df['temperature'] == 255).sum() == 0
    assert (df['humidity'] == 255).sum() == 0


def test_chronological_ordering():
    """Verify that returned data is strictly monotonically increasing by datetime."""
    df = load_fan_telemetry("Bedroom_Fan_1")
    assert df['dt'].is_monotonic_increasing, "Dataset is not sorted chronologically!"


def test_power_formula_exact_match():
    """Verify that eSpent == power_W * opTime on cleaned data."""
    df = load_fan_telemetry("Drawingroom_Fan_2")
    expected_spent = df['speed'].map(POWER_LOOKUP) * df['opTime']
    diff = (df['eSpent'] - expected_spent).abs()
    assert (diff == 0).all(), "eSpent deviates from deterministic power formula!"


def test_induction_savings_exact_match():
    """Verify that eSaved == saved_rate_W * opTime on cleaned data."""
    df = load_fan_telemetry("Drawingroom_Fan_3")
    expected_saved = df['speed'].map(SAVED_LOOKUP) * df['opTime']
    diff = (df['eSaved'] - expected_saved).abs()
    assert (diff == 0).all(), "eSaved deviates from deterministic savings formula!"


def test_power_ratings_spec():
    """Verify physical power ratings match Atomberg BLDC specifications."""
    expected_bldc = {0: 0.0, 1: 10.0, 2: 15.0, 3: 20.0, 4: 27.0, 5: 48.0}
    expected_induction = {0: 0.0, 1: 24.0, 2: 34.0, 3: 55.0, 4: 64.0, 5: 96.0}
    
    for speed, expected_w in expected_bldc.items():
        ratings = EnergyIntelligencePredictor.get_power_ratings(speed)
        assert ratings["bldc_power_W"] == expected_w
        assert ratings["induction_baseline_power_W"] == expected_induction[speed]


def test_exact_energy_calculation():
    """Verify deterministic energy calculation for specific speeds and durations."""
    # Speed 1 (10W BLDC, 24W Induction, 14W Saved) for 3600 seconds (1 hour)
    calc = EnergyIntelligencePredictor.calculate_exact_energy_and_savings(speed=1, duration_seconds=3600)
    assert calc["bldc_energy_spent_joules"] == 36000.0
    assert calc["bldc_energy_spent_Wh"] == 10.0
    assert calc["induction_baseline_Wh"] == 24.0
    assert calc["energy_saved_Wh"] == 14.0
    assert calc["energy_saved_pct"] == pytest.approx(58.3, rel=1e-2)
    
    # Speed 0 (OFF)
    calc_off = EnergyIntelligencePredictor.calculate_exact_energy_and_savings(speed=0, duration_seconds=3600)
    assert calc_off["bldc_energy_spent_Wh"] == 0.0
    assert calc_off["energy_saved_Wh"] == 0.0


def test_predictor_inference_api():
    """Verify runtime prediction API returns valid structure and values."""
    predictor = EnergyIntelligencePredictor("Bedroom_Fan_1")
    res = predictor.predict_next_interval_energy(
        temperature=28.5,
        humidity=65.0,
        current_speed=3,
        current_op_time=150.0,
        timestamp=datetime(2021, 6, 15, 14, 30)
    )
    
    assert "predicted_energy_Wh" in res
    assert "predicted_energy_Joules" in res
    assert "predicted_duration_seconds" in res
    assert "modeled_energy_saved_Wh" in res
    assert res["predicted_energy_Wh"] >= 0.0
    assert res["bldc_rated_power_W"] == 20.0
    assert res["is_fan_running"] is True


def test_feature_matrix_causality():
    """Verify that lag features strictly shift backwards in time and targets shift forward."""
    df_sample = pd.DataFrame({
        'dt': pd.date_range('2021-01-01', periods=10, freq='150s'),
        'hour': [10] * 10,
        'day_of_week': [4] * 10,
        'temperature': [28.0 + i for i in range(10)],
        'humidity': [60.0] * 10,
        'mode': [0] * 10,
        'speed': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'opTime': [150] * 10,
        'eSpent_Wh': [0.42] * 10
    })
    
    df_feat = create_feature_matrix(df_sample)
    # Target energy should equal original index 1's eSpent_Wh
    assert df_feat['target_energy_Wh'].iloc[0] == df_sample['eSpent_Wh'].iloc[1]
    # Lag speed should equal original index 0's speed
    assert df_feat['lag_speed_1'].iloc[0] == df_sample['speed'].iloc[0]


def test_extract_sessions():
    """Verify session extraction identifies continuous operational blocks."""
    df = load_fan_telemetry("Lounge_Fan_1")
    sessions = extract_sessions(df)
    assert len(sessions) > 0
    assert 'duration_minutes' in sessions.columns
    assert 'mean_speed' in sessions.columns
    assert 'is_night_session' in sessions.columns
    assert (sessions['duration_minutes'] >= 2.5).all()

