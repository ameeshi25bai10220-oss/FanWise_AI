"""
Automated Unit and Integration Tests for AACI Module 2 (Spatial Headcount & Pattern Intelligence).
Verifies:
  1. ROBOD data loading and schema integrity across all 5 rooms.
  2. Strict feature isolation: 'occupant_presence [binary]' is NEVER an input feature.
  3. Preprocessing and feature groups alignment.
  4. Module 2A headcount regression model loading and artifact persistence.
  5. Predictor output ranges: estimated_headcount >= 0.0, valid density categories.
  6. 90% uncertainty prediction interval bounds (lower <= upper).
  7. Module 2B pattern summary table validity.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from robod_loader import (
    load_all_rooms, load_single_room, get_chronological_room_splits,
    get_loro_splits, FEATURE_GROUPS, TARGET_HEADCOUNT, TARGET_PRESENCE, ROOM_TYPES
)
from module2_predictor import Module2SpatialPredictor


class TestRobodDataLoader:
    def test_all_five_rooms_load_with_correct_shapes(self):
        rooms = load_all_rooms()
        assert len(rooms) == 5
        assert len(rooms[1]) == 8352
        assert len(rooms[2]) == 8352
        assert len(rooms[3]) == 8352
        assert len(rooms[4]) == 13536
        assert len(rooms[5]) == 13536

    def test_strict_exclusion_of_presence_from_features(self):
        for grp_name, feat_cols in FEATURE_GROUPS.items():
            assert TARGET_PRESENCE not in feat_cols, f"Target presence leaked into {grp_name}!"
            assert TARGET_HEADCOUNT not in feat_cols, f"Target headcount leaked into {grp_name}!"

    def test_chronological_splits_preserve_order(self):
        df = load_single_room(1)
        tr, val, te = get_chronological_room_splits(df)
        assert len(tr) + len(val) + len(te) == len(df)
        assert tr["ts"].max() <= val["ts"].min()
        assert val["ts"].max() <= te["ts"].min()


class TestModule2Predictor:
    @pytest.mark.parametrize("model_name", ["rf_multimodal", "rf_acoustic_wifi", "rf_env_only", "ridge_acoustic", "xgb_multimodal"])
    def test_candidate_models_load_and_predict(self, model_name):
        predictor = Module2SpatialPredictor(candidate_model_name=model_name, room_id=1)
        sample = {
            "air_temperature [Celsius]": 24.0,
            "indoor_relative_humidity [%]": 65.0,
            "indoor_co2 [ppm]": 600.0,
            "sound_pressure_level [dba]": 52.0,
            "wifi_connected_devices [number]": 4.0,
            "hour": 14
        }
        res = predictor.predict(sample)

        assert "estimated_headcount" in res
        assert res["estimated_headcount"] >= 0.0
        assert res["headcount_rounded"] >= 0
        assert res["occupancy_state"] in ("unoccupied", "occupied")
        assert res["density_category"] in ("empty", "low (1-3 persons)", "medium (4-10 persons)", "high (>10 persons)")
        assert len(res["uncertainty_90pct_interval"]) == 2
        assert res["uncertainty_90pct_interval"][0] <= res["uncertainty_90pct_interval"][1]

    def test_empty_room_prediction_consistency(self):
        predictor = Module2SpatialPredictor(candidate_model_name="rf_multimodal", room_id=1)
        # Deep night empty conditions (Singapore ambient)
        sample_empty = {
            "air_temperature [Celsius]": 26.2,
            "indoor_relative_humidity [%]": 72.5,
            "indoor_co2 [ppm]": 455.0,
            "sound_pressure_level [dba]": 50.0,
            "wifi_connected_devices [number]": 0.0,
            "illuminance [lux]": 20.0,
            "hour": 3,
            "is_weekend": 0.0
        }
        res = predictor.predict(sample_empty)
        assert res["headcount_rounded"] == 0
        assert res["occupancy_state"] == "unoccupied"
        assert res["density_category"] == "empty"


class TestPatternAnalyticsOutput:
    def test_summary_csv_exists_and_valid(self):
        summary_path = os.path.join(
            os.path.dirname(__file__), "..", "evaluation", "occupancy_pattern_summary.csv"
        )
        assert os.path.exists(summary_path)
        df = pd.read_csv(summary_path)
        assert len(df) == 5
        assert set(df["room_id"]) == {1, 2, 3, 4, 5}
        assert (df["utilization_rate_pct"] >= 0.0).all()
        assert (df["peak_headcount"] > 0).all()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
