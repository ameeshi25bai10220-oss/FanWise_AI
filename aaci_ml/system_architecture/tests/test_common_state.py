"""
Unit and Integration Tests for Common State Representation (CSR)
================================================================
Phase 5 – Stage 2: Common State Representation (CSR) Implementation

Verifies:
  1. Valid CommonState construction from raw telemetry dict.
  2. Construction from realistic Mendeley ceiling fan telemetry records.
  3. Value provenance preservation (measured, assumed, estimated, predicted).
  4. Strict segregation of the four energy fields (measured, physics, ML, calibrated).
  5. ROBOD room ID preservation (1..5) and non-conflation with residential categories.
  6. Rejection of negative operating durations and duplicate timestamps.
  7. Stage 1 adapter direct interoperability (CSR passed to all 3 adapters).
  8. Bidirectional CSR enrichment via `update_from_adapter_envelope()`.
  9. System health diagnostics and degraded state tracking.
 10. Multi-device independent CSR instances.
"""

import os
import sys
import pytest
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aaci_ml.system_architecture.common_state import (
    CommonState,
    EnvironmentalState,
    OccupancyState,
    FanOperatingState,
    EnergyState,
    SystemHealthState,
    CentralizedSensorValidator,
)
from aaci_ml.system_architecture.adapters import (
    ThermalComfortAdapter,
    OccupancyAdapter,
    EnergyIntelligenceAdapter,
    AdapterOutputEnvelope,
)


@pytest.fixture
def clean_validator():
    v = CentralizedSensorValidator()
    v.reset_fan_tracking()
    return v


class TestCommonStateConstruction:
    """1, 2: Construction from raw and realistic Mendeley telemetry."""

    def test_construction_from_valid_raw_telemetry(self, clean_validator):
        raw = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 28.4,
            "humidity": 62.0,
            "current_speed": 3,
            "mode": "BLDC",
            "op_time_s": 14.4,
            "espent_wh": 0.072,
            "timestamp": "2026-04-15T14:32:00",
            "sensor_age_s": 15.0,
            "room_category": "Bedroom",
        }
        csr = CommonState.from_telemetry(raw, validator=clean_validator)

        assert isinstance(csr, CommonState)
        assert csr.health.overall_state_valid is True
        assert csr.environmental.air_temp_c == 28.4
        assert csr.environmental.relative_humidity_pct == 62.0
        assert csr.fan_state.fan_id == "Bedroom_Fan_1"
        assert csr.fan_state.current_speed == 3
        assert csr.fan_state.op_time_s == 14.4
        assert csr.energy.measured_espent_wh == 0.072
        assert csr.metadata["room_category"] == "Bedroom"

    def test_construction_from_mendeley_row_format(self, clean_validator):
        """Simulate direct row formatting from Mendeley CSV."""
        mendeley_row = {
            "room_id": "Drawingroom_Fan_2",
            "temperature_c": 29.8,
            "relative_humidity_pct": 68.0,
            "fan_speed": 4,
            "fan_mode": "BLDC",
            "opTime": 120.0,
            "eSpent": 0.90,  # Wh
            "datetime": "2026-04-15 16:45:00",
            "sensor_age_seconds": 25.0,
        }
        csr = CommonState.from_telemetry(mendeley_row, validator=clean_validator)
        assert csr.health.overall_state_valid is True
        assert csr.environmental.air_temp_c == 29.8
        assert csr.fan_state.current_speed == 4
        assert csr.fan_state.op_time_s == 120.0
        assert csr.energy.measured_espent_wh == 0.90


class TestCommonStateProvenanceAndSeparation:
    """3, 4: Value provenance and strict 4-way energy separation."""

    def test_value_provenance_preservation(self, clean_validator):
        raw = {
            "fan_id": "Lounge_Fan_1",
            "temperature": 27.0,
            "humidity": 58.0,
            "current_speed": 2,
            "op_time_s": 14.4,
            "air_velocity_ms": 0.25,
            "outdoor_monthly_temp_c": 31.0,
        }
        csr = CommonState.from_telemetry(raw, validator=clean_validator)

        # Environmental provenance
        assert csr.environmental.provenance["air_temp_c"] == "measured"
        assert csr.environmental.provenance["relative_humidity_pct"] == "measured"
        assert csr.environmental.provenance["air_velocity_ms"] == "measured"
        assert csr.environmental.provenance["outdoor_monthly_temp_c"] == "measured"
        assert csr.environmental.air_velocity_source == "measured_direct"

        # Fan state provenance
        assert csr.fan_state.provenance["fan_id"] == "measured"
        assert csr.fan_state.provenance["current_speed"] == "measured"
        assert csr.fan_state.provenance["op_time_s"] == "measured"

        # Energy physics table is estimated
        assert csr.energy.provenance["physics_candidate_table"] == "estimated"

    def test_strict_energy_state_segregation(self, clean_validator):
        """Verify the 4 energy fields remain strictly independent and never merged."""
        raw = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 28.0,
            "humidity": 60.0,
            "current_speed": 3,
            "op_time_s": 14.4,
            "espent_wh": 0.08,
        }
        csr = CommonState.from_telemetry(raw, validator=clean_validator)

        # 1. Measured telemetry energy
        assert csr.energy.measured_espent_wh == 0.08
        assert csr.energy.measured_espent_source == "firmware_telemetry_packet"

        # 2. Deterministic research-dataset physics candidate table
        assert len(csr.energy.physics_candidate_table) == 6
        assert "NOT validated Atomberg hardware measurement" in csr.energy.physics_power_map_source

        # 3. Machine learning predicted next-packet energy
        assert csr.energy.ml_predicted_next_energy_wh is None  # initially unset until adapter runs
        assert csr.energy.ml_forecast_is_advisory is True
        assert csr.energy.ml_forecast_cannot_command_speed is True
        assert csr.energy.ml_forecast_r2_benchmark == 0.1515

        # 4. Hardware-calibrated energy profile
        assert csr.energy.hardware_calibrated_energy_wh is None
        assert csr.energy.hardware_calibration_source is None


class TestCommonStateROBODTrainingContext:
    """5: ROBOD training room context preservation."""

    def test_robod_room_context_not_equated_with_residential_category(self, clean_validator):
        raw = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 28.0,
            "humidity": 60.0,
            "room_category": "Bedroom",  # Residential category
        }
        csr = CommonState.from_telemetry(raw, validator=clean_validator)

        # Before adapter enrichment
        assert csr.metadata["room_category"] == "Bedroom"
        assert csr.occupancy.robod_room_id is None
        assert "NOT be equated with residential Bedroom" in csr.occupancy.robod_training_context_note

        # Enrich via OccupancyAdapter
        o_adapter = OccupancyAdapter()
        o_res = o_adapter.predict(csr)
        csr.update_from_adapter_envelope(o_res)

        # ROBOD room ID is preserved in its academic context (room 1: Lecture Room 1)
        assert csr.occupancy.robod_room_id == 1
        assert csr.occupancy.robod_room_type == "Lecture Room 1"
        # Residential category in metadata remains unmodified
        assert csr.metadata["room_category"] == "Bedroom"


class TestCommonStateValidationRejections:
    """6: Strict rejection of negative durations and duplicate timestamps."""

    def test_negative_op_time_rejects_csr_validity(self, clean_validator):
        raw = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 28.0,
            "humidity": 60.0,
            "op_time_s": -25.0,  # Negative duration
        }
        csr = CommonState.from_telemetry(raw, validator=clean_validator)
        assert csr.health.overall_state_valid is False
        assert "op_time_s" in csr.health.invalid_inputs
        assert any("negative operating duration" in err.lower() for err in csr.health.validation_errors)

    def test_duplicate_timestamp_rejects_csr_validity(self, clean_validator):
        raw = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 28.0,
            "humidity": 60.0,
            "timestamp": "2026-04-15T12:00:00",
        }
        csr1 = CommonState.from_telemetry(raw, validator=clean_validator)
        assert csr1.health.overall_state_valid is True

        # Second arrival of exact same timestamp
        csr2 = CommonState.from_telemetry(raw, validator=clean_validator)
        assert csr2.health.overall_state_valid is False
        assert any("duplicate telemetry packet" in err.lower() for err in csr2.health.validation_errors)


class TestCommonStateAdapterInteroperability:
    """7, 8, 9: Stage 1 adapter direct interoperability and enrichment."""

    def test_csr_consumed_directly_by_all_three_adapters(self, clean_validator):
        """Stage 1 adapters accept CSR object directly via duck typing."""
        raw = {
            "fan_id": "Drawingroom_Fan_1",
            "temperature": 28.0,
            "humidity": 60.0,
            "current_speed": 3,
            "op_time_s": 14.4,
            "espent_wh": 0.08,
            "timestamp": "2026-04-15T14:00:00",
        }
        csr = CommonState.from_telemetry(raw, validator=clean_validator)

        c_adapter = ThermalComfortAdapter()
        o_adapter = OccupancyAdapter()
        e_adapter = EnergyIntelligenceAdapter()

        # Pass CSR directly (without .to_dict())
        c_res = c_adapter.predict(csr)
        o_res = o_adapter.predict(csr)
        e_res = e_adapter.predict(csr)

        assert isinstance(c_res, AdapterOutputEnvelope)
        assert isinstance(o_res, AdapterOutputEnvelope)
        assert isinstance(e_res, AdapterOutputEnvelope)
        assert c_res.prediction in ["Cold", "Neutral", "Hot"]
        assert o_res.prediction in ["Occupied", "Unoccupied"]
        assert len(e_res.normalized_payload["candidate_evaluation_table"]) == 6

    def test_bidirectional_enrichment_via_adapter_envelopes(self, clean_validator):
        raw = {
            "fan_id": "Bedroom_Fan_1",
            "temperature": 28.0,
            "humidity": 60.0,
            "current_speed": 3,
            "op_time_s": 14.4,
            "espent_wh": 0.08,
            "timestamp": "2026-04-15T14:00:00",
        }
        csr = CommonState.from_telemetry(raw, validator=clean_validator)

        c_adapter = ThermalComfortAdapter()
        o_adapter = OccupancyAdapter()
        e_adapter = EnergyIntelligenceAdapter()

        # Update from Comfort Adapter
        c_res = c_adapter.predict(csr)
        csr.update_from_adapter_envelope(c_res)
        assert csr.health.thermal_adapter_available is True
        assert csr.environmental.air_velocity_source == "PROVISIONAL_DRYAD"
        assert csr.metadata["comfort_label"] in ["Cold", "Neutral", "Hot"]

        # Update from Occupancy Adapter
        o_res = o_adapter.predict(csr)
        csr.update_from_adapter_envelope(o_res)
        assert csr.health.occupancy_adapter_available is True
        assert csr.occupancy.occupancy_classification in ["Occupied", "Unoccupied"]
        assert csr.occupancy.feature_set_used == "set_b"
        assert csr.occupancy.robod_room_id == 1

        # Update from Energy Adapter
        e_res = e_adapter.predict(csr)
        csr.update_from_adapter_envelope(e_res)
        assert csr.health.energy_adapter_available is True
        assert len(csr.energy.physics_candidate_table) == 6
        assert csr.energy.ml_forecast_is_advisory is True

    def test_multi_device_csr_isolation(self, clean_validator):
        raw1 = {"fan_id": "Fan_1", "temperature": 26.0, "humidity": 55.0}
        raw2 = {"fan_id": "Fan_2", "temperature": 30.0, "humidity": 65.0}

        csr1 = CommonState.from_telemetry(raw1, validator=clean_validator)
        csr2 = CommonState.from_telemetry(raw2, validator=clean_validator)

        assert csr1.fan_state.fan_id == "Fan_1"
        assert csr2.fan_state.fan_id == "Fan_2"
        assert csr1.environmental.air_temp_c == 26.0
        assert csr2.environmental.air_temp_c == 30.0

