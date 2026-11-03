"""
Unit Tests for CentralizedSensorValidator
=========================================
Phase 5 – Stage 2: Common State Representation (CSR) Implementation

Verifies:
  1. Missing required fields (temperature, humidity, fan_id).
  2. Invalid data types and malformed inputs.
  3. Physically implausible sensor ranges.
  4. Hardware sensor sentinels (255.0).
  5. Zero boot errors (temp=0.0 and humidity=0.0 simultaneously).
  6. Negative operating durations (explicitly rejected, zero silent positive clamping).
  7. Duplicate telemetry timestamps (explicit rejection policy).
  8. Out-of-order telemetry detection.
  9. Configurable staleness threshold and age validation.
 10. Timestamp parsing across supported formats.
"""

import os
import sys
import pytest
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aaci_ml.system_architecture.common_state.sensor_validator import (
    CentralizedSensorValidator,
    ValidationResult,
)


@pytest.fixture
def validator():
    v = CentralizedSensorValidator(staleness_threshold_s=300.0)
    v.reset_fan_tracking()
    return v


class TestSensorValidatorBasics:
    """1, 2: Required fields and data types."""

    def test_valid_telemetry_passes(self, validator):
        payload = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 27.5,
            "humidity": 55.0,
            "current_speed": 3,
            "op_time_s": 14.4,
            "espent_wh": 0.08,
            "timestamp": "2026-04-12T10:00:00",
            "sensor_age_s": 10.0,
        }
        res = validator.validate_raw_telemetry(payload)
        assert res.is_valid is True
        assert len(res.errors) == 0
        assert res.parsed_dt is not None

    def test_missing_fan_id_rejected(self, validator):
        payload = {"temperature": 27.5, "humidity": 55.0}
        res = validator.validate_raw_telemetry(payload)
        assert res.is_valid is False
        assert "fan_id" in res.missing_fields
        assert any("fan identifier" in err.lower() for err in res.errors)

    def test_missing_temperature_rejected(self, validator):
        payload = {"fan_id": "Fan_1", "humidity": 55.0}
        res = validator.validate_raw_telemetry(payload)
        assert res.is_valid is False
        assert "temperature" in res.missing_fields

    def test_missing_humidity_rejected(self, validator):
        payload = {"fan_id": "Fan_1", "temperature": 27.5}
        res = validator.validate_raw_telemetry(payload)
        assert res.is_valid is False
        assert "relative_humidity" in res.missing_fields

    def test_invalid_types(self, validator):
        payload = {
            "fan_id": "Fan_1",
            "temperature": "not_a_number",
            "humidity": [50.0],
            "current_speed": "high",
        }
        res = validator.validate_raw_telemetry(payload)
        assert res.is_valid is False
        assert len(res.errors) >= 3


class TestSensorValidatorPhysicalLimitsAndSentinels:
    """3, 4, 5: Plausible limits, sentinels, and boot errors."""

    def test_temperature_out_of_physical_bounds(self, validator):
        res_hot = validator.validate_raw_telemetry({"fan_id": "F1", "temperature": 65.0, "humidity": 50.0})
        assert res_hot.is_valid is False
        assert any("temperature" in err.lower() for err in res_hot.errors)

        res_cold = validator.validate_raw_telemetry({"fan_id": "F1", "temperature": -15.0, "humidity": 50.0})
        assert res_cold.is_valid is False

    def test_humidity_out_of_physical_bounds(self, validator):
        res_high = validator.validate_raw_telemetry({"fan_id": "F1", "temperature": 25.0, "humidity": 105.0})
        assert res_high.is_valid is False

        res_low = validator.validate_raw_telemetry({"fan_id": "F1", "temperature": 25.0, "humidity": -5.0})
        assert res_low.is_valid is False

    def test_sentinel_temperature_255(self, validator):
        res = validator.validate_raw_telemetry({"fan_id": "F1", "temperature": 255.0, "humidity": 50.0})
        assert res.is_valid is False
        assert any("sentinel" in err.lower() for err in res.errors)

    def test_sentinel_humidity_255(self, validator):
        res = validator.validate_raw_telemetry({"fan_id": "F1", "temperature": 25.0, "humidity": 255.0})
        assert res.is_valid is False
        assert any("sentinel" in err.lower() for err in res.errors)

    def test_zero_boot_error_detection(self, validator):
        """Simultaneous T=0.0 and RH=0.0 indicates uninitialized boot fault."""
        res = validator.validate_raw_telemetry({"fan_id": "F1", "temperature": 0.0, "humidity": 0.0})
        assert res.is_valid is False
        assert any("zero boot error" in err.lower() for err in res.errors)


class TestSensorValidatorOperatingDurationRules:
    """6: Strict negative operating duration rule."""

    def test_negative_op_time_explicitly_rejected(self, validator):
        """Negative operating duration MUST be rejected, never silently clamped."""
        payload = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 26.0,
            "humidity": 60.0,
            "current_speed": 3,
            "op_time_s": -15.0,  # Negative duration
        }
        res = validator.validate_raw_telemetry(payload)
        assert res.is_valid is False
        assert "op_time_s" in res.invalid_fields
        assert any("negative operating duration" in err.lower() for err in res.errors)
        assert any("no silent positive clamping" in err.lower() for err in res.errors)

    def test_positive_and_zero_op_time_accepted(self, validator):
        payload_zero = {"fan_id": "F1", "temperature": 26.0, "humidity": 60.0, "op_time_s": 0.0}
        assert validator.validate_raw_telemetry(payload_zero).is_valid is True

        payload_pos = {"fan_id": "F1", "temperature": 26.0, "humidity": 60.0, "op_time_s": 14.4}
        assert validator.validate_raw_telemetry(payload_pos).is_valid is True


class TestSensorValidatorChronologicalFlowAndDeduplication:
    """7, 8: Duplicate timestamp rejection and out-of-order detection."""

    def test_duplicate_timestamp_explicitly_rejected(self, validator):
        fan = "Bedroom_Fan_1"
        ts = "2026-04-12T12:00:00"

        # Packet 1 succeeds
        p1 = {"fan_id": fan, "temperature": 28.0, "humidity": 55.0, "timestamp": ts}
        res1 = validator.validate_raw_telemetry(p1)
        assert res1.is_valid is True
        assert res1.is_duplicate is False

        # Exact duplicate packet arriving next
        p2 = {"fan_id": fan, "temperature": 28.0, "humidity": 55.0, "timestamp": ts}
        res2 = validator.validate_raw_telemetry(p2)
        assert res2.is_valid is False
        assert res2.is_duplicate is True
        assert any("duplicate telemetry packet" in err.lower() for err in res2.errors)
        assert any("arbitrary record selection forbidden" in err.lower() for err in res2.errors)

    def test_out_of_order_telemetry_detected(self, validator):
        fan = "Drawingroom_Fan_1"
        # Packet 1 at 12:05
        validator.validate_raw_telemetry({
            "fan_id": fan, "temperature": 28.0, "humidity": 55.0, "timestamp": "2026-04-12T12:05:00"
        })

        # Packet 2 with earlier timestamp 12:00
        res = validator.validate_raw_telemetry({
            "fan_id": fan, "temperature": 28.0, "humidity": 55.0, "timestamp": "2026-04-12T12:00:00"
        })
        assert res.is_out_of_order is True
        assert any("precedes previously observed timestamp" in w for w in res.warnings)

    def test_independent_fan_chronological_tracking(self, validator):
        """Fan A and Fan B maintain separate timestamp tracking without false duplicate collisions."""
        ts = "2026-04-12T12:00:00"
        res_a = validator.validate_raw_telemetry({"fan_id": "Fan_A", "temperature": 27.0, "humidity": 50.0, "timestamp": ts})
        res_b = validator.validate_raw_telemetry({"fan_id": "Fan_B", "temperature": 27.0, "humidity": 50.0, "timestamp": ts})

        assert res_a.is_valid is True
        assert res_b.is_valid is True
        assert res_a.is_duplicate is False
        assert res_b.is_duplicate is False


class TestSensorValidatorStalenessAndParsing:
    """9, 10: Configurable staleness thresholds and parsing."""

    def test_stale_sensor_age_flagged(self, validator):
        payload = {
            "fan_id": "F1",
            "temperature": 27.0,
            "humidity": 55.0,
            "sensor_age_s": 450.0,  # > default 300s
        }
        res = validator.validate_raw_telemetry(payload)
        assert res.is_valid is True  # valid structure
        assert res.is_stale is True
        assert any("stale" in w.lower() for w in res.warnings)

    def test_custom_configurable_staleness_threshold(self):
        strict_validator = CentralizedSensorValidator(staleness_threshold_s=60.0)
        # 90s is stale under 60s threshold
        res = strict_validator.validate_raw_telemetry({
            "fan_id": "F1", "temperature": 27.0, "humidity": 55.0, "sensor_age_s": 90.0
        })
        assert res.is_stale is True

    def test_various_timestamp_formats(self, validator):
        formats = [
            "2026-04-12T15:30:00Z",
            "2026-04-12 15:30:00",
            "12-04-2026 15:30:00",
            "12/04/2026 15:30:00",
        ]
        for f in formats:
            dt, iso_str = validator.parse_timestamp(f)
            assert dt is not None
            assert iso_str is not None

