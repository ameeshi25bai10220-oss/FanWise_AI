"""
Unit and Contract Tests for ThermalComfortAdapter (Model 1)
===========================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Verifies:
  1. Valid input inference (both flat and CSR structures).
  2. Missing required inputs.
  3. Invalid data types.
  4. Out-of-range values.
  5. Missing sensor / sentinel scenarios (sentinel=255, zero boot error).
  6. Provisional Dryad airflow mapping transparency and disclosure.
  7. Model artifact loading / inference failure handling.
  8. Malformed inputs.
  9. Output schema validation against common contract envelope.
 10. Repeated inference consistency.
"""

import os
import sys
import pytest
import datetime
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aaci_ml.system_architecture.adapters.thermal_comfort_adapter import ThermalComfortAdapter
from aaci_ml.system_architecture.adapters.contract import AdapterOutputEnvelope
from aaci_ml.system_architecture.adapters.constants import PROVISIONAL_SPEED_VELOCITY_MAP


@pytest.fixture
def adapter():
    return ThermalComfortAdapter()


class TestThermalComfortAdapterValidInference:
    """1. Valid-input inference tests."""

    def test_valid_inference_with_fan_speed(self, adapter):
        inputs = {
            "temperature": 27.5,
            "humidity": 55.0,
            "fan_speed": 3,
            "timestamp": "2026-04-15 14:30:00",
        }
        res = adapter.predict(inputs)
        assert isinstance(res, AdapterOutputEnvelope)
        assert res.model_id == "AACI_Model_1_ThermalComfort_XGBoost_v3"
        assert res.adapter_name == "ThermalComfortAdapter"
        assert res.prediction in ["Cold", "Neutral", "Hot"]
        assert res.prediction_unit == "thermal_comfort_class"
        assert res.input_validity.is_valid is True
        assert res.model_availability.is_available is True
        assert res.raw_model_output is not None

        # Normalised payload checks
        p = res.normalized_payload
        assert p["comfort_label"] in ["Cold", "Neutral", "Hot"]
        assert p["comfort_class_id"] in [0, 1, 2]
        assert set(p["comfort_probabilities"].keys()) == {"Cold", "Neutral", "Hot"}
        assert abs(sum(p["comfort_probabilities"].values()) - 1.0) < 0.05
        assert p["air_velocity_used_ms"] == PROVISIONAL_SPEED_VELOCITY_MAP[3]
        assert p["air_velocity_mapping"] == "PROVISIONAL_DRYAD"
        assert p["season_used"] == "Summer"
        assert p["domain_mismatch_logged"] is True

    def test_valid_inference_with_direct_air_velocity(self, adapter):
        inputs = {
            "air_temp_c": 26.0,
            "relative_humidity": 65.0,
            "air_velocity_ms": 0.45,
            "season": "Monsoon",
        }
        res = adapter.predict(inputs)
        assert res.input_validity.is_valid is True
        assert res.normalized_payload["air_velocity_used_ms"] == 0.45
        assert res.normalized_payload["air_velocity_mapping"] == "DIRECT_MEASUREMENT"
        assert res.value_provenance.measured["air_velocity_ms"] == 0.45

    def test_csr_hierarchical_structure(self, adapter):
        csr = {
            "environmental": {
                "temperature_c": 28.4,
                "relative_humidity_pct": 62.0,
            },
            "fan_state": {
                "current_speed": 4,
            },
            "metadata": {
                "timestamp": "2026-03-15T14:32:00",
                "sensor_age_seconds": 15.0,
            },
        }
        res = adapter.predict(csr)
        assert res.input_validity.is_valid is True
        assert res.input_freshness.is_stale is False
        assert res.normalized_payload["comfort_label"] in ["Cold", "Neutral", "Hot"]


class TestThermalComfortAdapterMissingInputs:
    """2. Missing required inputs."""

    def test_missing_temperature(self, adapter):
        res = adapter.predict({"humidity": 55.0, "fan_speed": 3})
        assert res.input_validity.is_valid is False
        assert res.model_availability.is_available is False
        assert any("air_temp_c" in err for err in res.input_validity.validation_errors)
        assert "air_temp_c" in res.missing_inputs
        assert res.prediction == "Unknown"
        assert res.normalized_payload["fallback_triggered"] is True

    def test_missing_humidity(self, adapter):
        res = adapter.predict({"temperature": 27.5, "fan_speed": 3})
        assert res.input_validity.is_valid is False
        assert res.model_availability.is_available is False
        assert "relative_humidity" in res.missing_inputs
        assert res.prediction == "Unknown"

    def test_missing_air_velocity_and_speed(self, adapter):
        res = adapter.predict({"temperature": 27.5, "humidity": 55.0})
        assert res.input_validity.is_valid is False
        assert "air_velocity_ms" in res.missing_inputs
        assert any("air velocity" in err for err in res.input_validity.validation_errors)


class TestThermalComfortAdapterInvalidDataTypes:
    """3. Invalid data types."""

    def test_string_temperature_non_numeric(self, adapter):
        res = adapter.predict({"temperature": "invalid_temp", "humidity": 55.0, "fan_speed": 3})
        assert res.input_validity.is_valid is False
        assert any("temperature" in err for err in res.input_validity.validation_errors)

    def test_none_inputs_dict(self, adapter):
        res = adapter.predict({})
        assert res.input_validity.is_valid is False
        assert len(res.input_validity.validation_errors) >= 3


class TestThermalComfortAdapterOutOfRangeValues:
    """4. Out-of-range values."""

    def test_temperature_above_60(self, adapter):
        res = adapter.predict({"temperature": 65.0, "humidity": 55.0, "fan_speed": 3})
        assert res.input_validity.is_valid is False
        assert any("out of valid range" in err for err in res.input_validity.validation_errors)

    def test_temperature_below_minus_10(self, adapter):
        res = adapter.predict({"temperature": -15.0, "humidity": 55.0, "fan_speed": 3})
        assert res.input_validity.is_valid is False

    def test_humidity_above_100(self, adapter):
        res = adapter.predict({"temperature": 27.5, "humidity": 105.0, "fan_speed": 3})
        assert res.input_validity.is_valid is False

    def test_invalid_speed(self, adapter):
        res = adapter.predict({"temperature": 27.5, "humidity": 55.0, "fan_speed": 7})
        assert res.input_validity.is_valid is False


class TestThermalComfortAdapterMissingSensorScenarios:
    """5. Missing sensor scenarios & sentinels."""

    def test_sentinel_temperature_255(self, adapter):
        res = adapter.predict({"temperature": 255.0, "humidity": 60.0, "fan_speed": 2})
        assert res.input_validity.is_valid is False
        assert any("sentinel" in err for err in res.input_validity.validation_errors)

    def test_sentinel_humidity_255(self, adapter):
        res = adapter.predict({"temperature": 27.5, "humidity": 255.0, "fan_speed": 2})
        assert res.input_validity.is_valid is False
        assert any("sentinel" in err for err in res.input_validity.validation_errors)

    def test_zero_boot_error(self, adapter):
        res = adapter.predict({"temperature": 0.0, "humidity": 0.0, "fan_speed": 2})
        assert res.input_validity.is_valid is False
        assert any("Zero boot error" in err for err in res.input_validity.validation_errors)

    def test_stale_telemetry(self, adapter):
        res = adapter.predict({
            "temperature": 27.5,
            "humidity": 60.0,
            "fan_speed": 2,
            "sensor_age_seconds": 450.0,
        })
        assert res.input_validity.is_valid is True  # inference allowed but flagged stale
        assert res.input_freshness.is_stale is True
        assert res.normalized_payload["is_uncertain"] is True


class TestThermalComfortAdapterProvisionalMapping:
    """6. Provisional airflow mapping requirements."""

    def test_provisional_mapping_disclosure(self, adapter):
        res = adapter.predict({"temperature": 27.5, "humidity": 60.0, "fan_speed": 4})
        assert res.normalized_payload["air_velocity_mapping"] == "PROVISIONAL_DRYAD"
        assert res.normalized_payload["air_velocity_used_ms"] == 0.410
        assert any("PROVISIONAL" in lim for lim in res.limitations)
        assert res.value_provenance.estimated["air_velocity_ms"] == 0.410
        assert res.value_provenance.measured["fan_speed"] == 4


class TestThermalComfortAdapterExceptionHandling:
    """7. Model artifact / inference failure handling."""

    def test_predict_comfort_exception_graceful_fallback(self, adapter):
        with patch("aaci_ml.system_architecture.adapters.thermal_comfort_adapter.predict_comfort", side_effect=RuntimeError("Simulated XGBoost OOM")):
            res = adapter.predict({"temperature": 27.5, "humidity": 60.0, "fan_speed": 2})
            assert res.model_availability.is_available is False
            assert res.degraded_state.fallback_triggered is True
            assert res.prediction == "Unknown"
            assert res.normalized_payload["comfort_label"] == "Unknown"
            assert res.normalized_payload["is_uncertain"] is True


class TestThermalComfortAdapterSchemaAndDeterminism:
    """8, 9, 10: Output schema validation and repeated inference consistency."""

    def test_output_schema_structure(self, adapter):
        res = adapter.predict({"temperature": 28.0, "humidity": 60.0, "fan_speed": 3})
        d = res.to_dict()
        required_top_keys = [
            "model_id", "adapter_name", "prediction", "prediction_unit", "timestamp",
            "input_validity", "input_freshness", "model_availability", "missing_inputs",
            "value_provenance", "limitations", "degraded_state", "raw_model_output",
            "normalized_payload"
        ]
        for k in required_top_keys:
            assert k in d, f"Missing key in envelope: {k}"

    def test_repeated_inference_consistency(self, adapter):
        payload = {"temperature": 29.1, "humidity": 58.0, "fan_speed": 2, "timestamp": "2026-05-10 12:00:00"}
        res1 = adapter.predict(payload)
        res2 = adapter.predict(payload)
        assert res1.prediction == res2.prediction
        assert res1.normalized_payload["comfort_probabilities"] == res2.normalized_payload["comfort_probabilities"]
        assert res1.normalized_payload["model_confidence"] == res2.normalized_payload["model_confidence"]

