"""
Unit and Contract Tests for OccupancyAdapter (Model 2)
======================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Verifies:
  1. Valid input inference under Fan IoT context (No Light, No CO2 -> Set B).
  2. Valid input inference under Extended context (Light & CO2 present -> Set A).
  3. Missing required inputs (Temperature, Humidity).
  4. Invalid data types and malformed inputs.
  5. Out-of-range values (Temperature, Humidity, Light, CO2).
  6. Incompatible feature sets and zero sensor fabrication (DEFECT-02 verification).
  7. Sentinels (255) and zero boot errors.
  8. Model artifact loading & exception handling for Module 1 and Module 2.
  9. Headcount regression & integer room_id mapping (MINOR-02 verification).
 10. Repeated inference consistency and output schema validation.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aaci_ml.system_architecture.adapters.occupancy_adapter import OccupancyAdapter
from aaci_ml.system_architecture.adapters.contract import AdapterOutputEnvelope


@pytest.fixture
def adapter():
    return OccupancyAdapter()


class TestOccupancyAdapterFeatureSetHierarchy:
    """1 & 6: Feature set selection and zero sensor fabrication (DEFECT-02)."""

    def test_fan_iot_context_selects_set_b_only(self, adapter):
        """When Light and CO2 are absent, set_b MUST be selected. No fabrication allowed."""
        inputs = {
            "temperature": 23.5,
            "humidity": 50.0,
            "timestamp": "2026-03-20 15:30:00",
        }
        res = adapter.predict(inputs)
        assert isinstance(res, AdapterOutputEnvelope)
        assert res.model_id == "AACI_Model_2_Occupancy_Module1_RF_Module2_Spatial"
        assert res.prediction in ["Occupied", "Unoccupied"]
        assert res.prediction_unit == "binary_occupancy_state"
        assert res.input_validity.is_valid is True

        p = res.normalized_payload
        assert p["feature_set_used"] == "set_b"
        assert p["sensor_gap_active"] is True
        assert p["light_available"] is False
        assert p["co2_available"] is False
        assert p["fallback_strategy"] == "feature_set_downgrade"
        assert p["is_uncertain"] is True  # sensor gap active triggers uncertainty
        assert "light_lux" in res.missing_inputs
        assert "co2_ppm" in res.missing_inputs

        # Ensure no fabricated light or co2 in measured provenance
        assert "light_lux" not in res.value_provenance.measured
        assert "co2_ppm" not in res.value_provenance.measured

    def test_full_sensor_context_selects_set_a(self, adapter):
        """When Light and CO2 are both provided and valid, set_a is selected."""
        inputs = {
            "temperature": 24.0,
            "humidity": 45.0,
            "light_lux": 420.0,
            "co2_ppm": 750.0,
            "timestamp": "2026-03-20 15:30:00",
        }
        res = adapter.predict(inputs)
        assert res.input_validity.is_valid is True
        p = res.normalized_payload
        assert p["feature_set_used"] == "set_a"
        assert p["sensor_gap_active"] is False
        assert p["light_available"] is True
        assert p["co2_available"] is True
        assert p["fallback_strategy"] == "full_sensors_primary"
        assert res.value_provenance.measured["light_lux"] == 420.0
        assert res.value_provenance.measured["co2_ppm"] == 750.0

    def test_partial_sensor_gap_selects_set_b(self, adapter):
        """Light present but CO2 missing -> set_a and set_c incompatible -> set_b selected."""
        inputs = {
            "temperature": 24.0,
            "humidity": 45.0,
            "light_lux": 420.0,
            # co2 missing
        }
        res = adapter.predict(inputs)
        assert res.normalized_payload["feature_set_used"] == "set_b"
        assert res.normalized_payload["sensor_gap_active"] is True
        assert res.normalized_payload["co2_available"] is False


class TestOccupancyAdapterMissingInputs:
    """2. Missing required inputs (Temperature, Humidity)."""

    def test_missing_temperature(self, adapter):
        res = adapter.predict({"humidity": 50.0})
        assert res.input_validity.is_valid is False
        assert res.model_availability.is_available is False
        assert "temperature" in res.missing_inputs
        assert res.prediction == "Unknown"
        assert res.normalized_payload["occupancy_state"] == "Unknown"

    def test_missing_humidity(self, adapter):
        res = adapter.predict({"temperature": 25.0})
        assert res.input_validity.is_valid is False
        assert res.model_availability.is_available is False
        assert "humidity" in res.missing_inputs
        assert res.prediction == "Unknown"


class TestOccupancyAdapterValidationAndRanges:
    """3, 4, 5: Types, ranges, sentinels, boot errors."""

    def test_non_numeric_temperature(self, adapter):
        res = adapter.predict({"temperature": "hot", "humidity": 50.0})
        assert res.input_validity.is_valid is False

    def test_temperature_out_of_range(self, adapter):
        res = adapter.predict({"temperature": 75.0, "humidity": 50.0})
        assert res.input_validity.is_valid is False

    def test_humidity_out_of_range(self, adapter):
        res = adapter.predict({"temperature": 25.0, "humidity": -10.0})
        assert res.input_validity.is_valid is False

    def test_temperature_sentinel_255(self, adapter):
        res = adapter.predict({"temperature": 255.0, "humidity": 50.0})
        assert res.input_validity.is_valid is False
        assert any("sentinel" in err for err in res.input_validity.validation_errors)

    def test_humidity_sentinel_255(self, adapter):
        res = adapter.predict({"temperature": 25.0, "humidity": 255.0})
        assert res.input_validity.is_valid is False
        assert any("sentinel" in err for err in res.input_validity.validation_errors)

    def test_zero_boot_error(self, adapter):
        res = adapter.predict({"temperature": 0.0, "humidity": 0.0})
        assert res.input_validity.is_valid is False
        assert any("Zero boot error" in err for err in res.input_validity.validation_errors)

    def test_stale_sensor_reading(self, adapter):
        res = adapter.predict({"temperature": 25.0, "humidity": 50.0, "sensor_age_seconds": 400.0})
        assert res.input_validity.is_valid is True
        assert res.input_freshness.is_stale is True
        assert res.normalized_payload["is_uncertain"] is True


class TestOccupancyAdapterModule2Spatial:
    """Headcount estimation & integer room_id mapping (MINOR-02)."""

    def test_module2_headcount_estimation_fan_mapping(self, adapter):
        inputs = {
            "temperature": 24.5,
            "humidity": 55.0,
            "fan_id": "Drawingroom_Fan_1",
            "timestamp": "2026-03-20 14:00:00",
        }
        res = adapter.predict(inputs)
        assert res.normalized_payload["spatial_module_available"] is True
        assert res.normalized_payload["headcount_estimate"] is not None
        assert res.normalized_payload["headcount_rounded"] is not None
        assert res.normalized_payload["density_category"] in [
            "empty", "low (1-3 persons)", "medium (4-10 persons)", "high (>10 persons)"
        ]
        assert res.value_provenance.assumed["module2_room_id"] == 3

    def test_module2_failure_does_not_break_module1(self, adapter):
        """If Module 2 throws an exception, Module 1's binary prediction still succeeds."""
        with patch.object(adapter, "_get_module2_predictor", side_effect=RuntimeError("Module 2 failure")):
            res = adapter.predict({"temperature": 24.5, "humidity": 55.0})
            assert res.model_availability.is_available is True
            assert res.normalized_payload["model_available"] is True
            assert res.prediction in ["Occupied", "Unoccupied"]
            assert res.normalized_payload["spatial_module_available"] is False
            assert any("Module 2" in warn for warn in res.input_validity.validation_warnings)


class TestOccupancyAdapterExceptionHandlingAndSchema:
    """7, 8, 9, 10: Artifact failures, schema validation, determinism."""

    def test_module1_exception_triggers_fallback(self, adapter):
        with patch.object(adapter, "_get_module1_predictor", side_effect=RuntimeError("Model 1 artifact missing")):
            res = adapter.predict({"temperature": 24.5, "humidity": 55.0})
            assert res.model_availability.is_available is False
            assert res.degraded_state.fallback_triggered is True
            assert res.prediction == "Unknown"
            assert res.normalized_payload["is_uncertain"] is True

    def test_output_schema_structure(self, adapter):
        res = adapter.predict({"temperature": 24.5, "humidity": 55.0})
        d = res.to_dict()
        required_top_keys = [
            "model_id", "adapter_name", "prediction", "prediction_unit", "timestamp",
            "input_validity", "input_freshness", "model_availability", "missing_inputs",
            "value_provenance", "limitations", "degraded_state", "raw_model_output",
            "normalized_payload"
        ]
        for k in required_top_keys:
            assert k in d

    def test_repeated_inference_consistency(self, adapter):
        payload = {"temperature": 26.2, "humidity": 52.0, "timestamp": "2026-03-25 10:15:00"}
        res1 = adapter.predict(payload)
        res2 = adapter.predict(payload)
        assert res1.prediction == res2.prediction
        assert res1.normalized_payload["occupancy_probability"] == res2.normalized_payload["occupancy_probability"]
        assert res1.normalized_payload["feature_set_used"] == res2.normalized_payload["feature_set_used"]

