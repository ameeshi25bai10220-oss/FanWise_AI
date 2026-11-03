"""
Unit and Integration Tests for AACI Module 1 (Occupancy Intelligence).
Verifies:
  1. Data partition boundaries and zero random shuffling.
  2. Psychrometric derivation of HumidityRatio.
  3. Preprocessing, model loading, and artifact persistence for all 4 algorithms and 3 feature sets.
  4. Probability calibration limits: 0.0 <= P(occupied) <= 1.0.
  5. Calibrated threshold application.
  6. Physical range validation and error handling.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add src to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from data_loader import (
    load_raw_data, get_official_partitions, FEATURE_SETS, TARGET_COL,
    SPLIT_TEST1_END, SPLIT_TRAIN_END, TOTAL_ROWS
)
from predictor import OccupancyPredictor, calculate_humidity_ratio


class TestDataLoaderAndPartitions:
    def test_partition_shapes_and_boundaries(self):
        train_df, test2_df, test1_df = get_official_partitions()
        assert len(test1_df) == 2665, f"Expected Test 1 length 2,665, got {len(test1_df)}"
        assert len(train_df) == 8143, f"Expected Train length 8,143, got {len(train_df)}"
        assert len(test2_df) == 9752, f"Expected Test 2 length 9,752, got {len(test2_df)}"
        assert len(test1_df) + len(train_df) + len(test2_df) == TOTAL_ROWS

    def test_target_values_are_strictly_binary(self):
        train_df, test2_df, test1_df = get_official_partitions()
        for df in [train_df, test2_df, test1_df]:
            unique_vals = set(df[TARGET_COL].unique())
            assert unique_vals.issubset({0, 1}), f"Non-binary values in target: {unique_vals}"

    def test_no_future_date_leakage(self):
        train_df, test2_df, test1_df = get_official_partitions()
        # Test 1 strictly precedes Train
        assert test1_df["date"].max() < train_df["date"].min(), "Test 1 does not precede Train chronologically"
        # Train strictly precedes Test 2
        assert train_df["date"].max() < test2_df["date"].min(), "Train does not precede Test 2 chronologically"


class TestPsychrometricCalculation:
    def test_calculate_humidity_ratio_accuracy(self):
        # At 20 deg C, 50% RH at sea level, W is approx 0.0073 kg/kg
        w = calculate_humidity_ratio(20.0, 50.0)
        assert 0.0065 < w < 0.0080, f"Humidity ratio calculation out of physical bounds: {w}"

    def test_calculate_humidity_ratio_bounds(self):
        w_low = calculate_humidity_ratio(10.0, 10.0)
        w_high = calculate_humidity_ratio(35.0, 90.0)
        assert 0.0 < w_low < w_high < 0.05


class TestOccupancyPredictor:
    @pytest.mark.parametrize("algo", ["logistic_regression", "decision_tree", "random_forest", "xgboost"])
    @pytest.mark.parametrize("fset", ["set_a", "set_b", "set_c"])
    def test_all_12_model_configurations_load_and_predict(self, algo, fset):
        predictor = OccupancyPredictor(algorithm=algo, feature_set=fset)
        
        sample_input = {
            "Temperature": 22.0,
            "Humidity": 30.0,
            "Light": 200.0,
            "CO2": 600.0,
            "Hour": 14
        }
        res = predictor.predict(sample_input)
        
        assert "occupied" in res
        assert res["occupied"] in (0, 1)
        assert 0.0 <= res["probability"] <= 1.0
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["algorithm"] == algo
        assert res["feature_set"] == fset

    def test_threshold_override_behavior(self):
        predictor_low = OccupancyPredictor(algorithm="random_forest", feature_set="set_a", threshold_override=0.10)
        predictor_high = OccupancyPredictor(algorithm="random_forest", feature_set="set_a", threshold_override=0.90)
        
        marginal_input = {
            "Temperature": 21.0,
            "Humidity": 25.0,
            "Light": 50.0,
            "CO2": 500.0
        }
        res_low = predictor_low.predict(marginal_input)
        res_high = predictor_high.predict(marginal_input)
        
        # Lower threshold should be at least as likely to flag occupied
        assert res_low["occupied"] >= res_high["occupied"]

    def test_unphysical_sensor_validation_rejection(self):
        predictor = OccupancyPredictor(algorithm="random_forest", feature_set="set_a")
        
        # Test negative light
        with pytest.raises(ValueError, match="Negative Light"):
            predictor.predict({"Temperature": 21.0, "Humidity": 30.0, "Light": -5.0, "CO2": 450.0})
            
        # Test extreme temperature
        with pytest.raises(ValueError, match="Unphysical Temperature"):
            predictor.predict({"Temperature": 85.0, "Humidity": 30.0, "Light": 100.0, "CO2": 450.0})
            
        # Test extreme CO2
        with pytest.raises(ValueError, match="Unphysical CO2"):
            predictor.predict({"Temperature": 21.0, "Humidity": 30.0, "Light": 100.0, "CO2": 50.0})


if __name__ == "__main__":
    pytest.main(["-v", __file__])

