"""
AACI Fan Digital Twin – Unit & Integration Test Suite
=====================================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Tests the Fan Digital Twin runtime against requirements 1-14, 17, 18:
1. Initial Digital Twin state
2. Valid CSR ingestion
3. Fan ON/OFF transitions
4. Fan speed transitions
5. Room condition updates
6. Occupancy updates
7. Strict 4-way energy history separation
8. Multiple fan isolation
9. Stale sensor events
10. Missing sensor events
11. Adapter failure handling
12. Duplicate event handling
13. Out-of-order event handling
14. Historical snapshot consistency
17. State recovery after invalid updates
18. Preservation of value provenance
"""

import copy
import time
import pytest

from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.digital_twin import (
    FanDigitalTwin,
    TwinSupervisor,
    StateTransitionRecord,
    BLDC_POWER_TABLE_WATTS,
    INDUCTION_POWER_TABLE_WATTS,
)


class TestDigitalTwinRuntime:
    """Comprehensive verification suite for FanDigitalTwin."""

    def test_01_initial_digital_twin_state(self):
        """Verify initial default state matches fan_digital_twin_design.md schema."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1", fan_mode="BLDC", initial_speed=0)
        state = twin.inspect_state()

        assert state["fan_id"] == "Bedroom_Fan_1"
        assert state["room_category"] == "Bedroom"
        assert state["fan_mode"] == "BLDC"
        assert state["current_speed"] == 0
        assert state["proposed_speed"] == 0
        assert state["is_on"] is False
        assert state["speed_change_pending"] is False
        assert state["estimated_power_w"] == 0.0
        assert state["predicted_next_energy_wh"] is None
        assert state["cumulative_energy_session_wh"] == 0.0
        assert state["cumulative_energy_lifetime_wh"] == 0.0
        assert len(state["state_transition_history"]) == 1
        assert state["state_transition_history"][0]["trigger"] == "INITIALIZATION"
        assert state["provisional_flags"]["hardware_validation_performed"] is False
        assert state["provisional_flags"]["airflow_mapping_provisional"] is True
        assert state["data_quality"]["ml_cold_start"] is True

    def test_02_valid_csr_ingestion(self):
        """Verify ingestion of valid CommonState updates twin fields accurately."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")
        csr = CommonState.from_telemetry(
            raw_telemetry={
                "fan_id": "Bedroom_Fan_1",
                "temperature": 27.5,
                "relative_humidity": 55.0,
                "air_velocity_ms": 0.35,
                "fan_speed": 3,
                "espent_wh": 14.2,
                "light_lux": 280.0,
                "co2_ppm": 620.0,
                "provenance": "MEASURED_TELEMETRY",
            }
        )

        twin.update_from_csr(csr)
        state = twin.inspect_state()

        assert state["current_speed"] == 3
        assert state["is_on"] is True
        assert state["room_state"]["temperature_c"] == 27.5
        assert state["room_state"]["relative_humidity_pct"] == 55.0
        assert state["room_state"]["air_velocity_ms"] == 0.35
        assert state["data_quality"]["light_available"] is True
        assert state["data_quality"]["co2_available"] is True

    def test_03_fan_on_off_transitions(self):
        """Verify distinct transitions for switching fan ON and OFF."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1", initial_speed=0)

        # Switch ON (0 -> 2)
        twin.confirm_speed(2)
        assert twin.fan_state.is_on is True
        trans = twin.history_manager.get_transitions()
        on_trans = [t for t in trans if t.triggering_event == "FAN_SWITCHED_ON"]
        assert len(on_trans) == 1
        assert on_trans[0].previous_state["current_speed"] == 0
        assert on_trans[0].new_state["current_speed"] == 2

        # Switch OFF (2 -> 0)
        twin.confirm_speed(0)
        assert twin.fan_state.is_on is False
        trans = twin.history_manager.get_transitions()
        off_trans = [t for t in trans if t.triggering_event == "FAN_SWITCHED_OFF"]
        assert len(off_trans) == 1
        assert off_trans[0].previous_state["current_speed"] == 2
        assert off_trans[0].new_state["current_speed"] == 0

    def test_04_fan_speed_transitions(self):
        """Verify SPEED_CHANGED transition between non-zero speeds."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1", initial_speed=1)
        twin.confirm_speed(4)

        trans = twin.history_manager.get_transitions()
        speed_trans = [t for t in trans if t.triggering_event == "SPEED_CHANGED"]
        assert len(speed_trans) == 1
        assert speed_trans[0].previous_state["current_speed"] == 1
        assert speed_trans[0].new_state["current_speed"] == 4

    def test_05_room_condition_updates(self):
        """Verify room state updates temperature and humidity while preserving flags."""
        twin = FanDigitalTwin(fan_id="Drawingroom_Fan_1")
        csr = CommonState.from_telemetry(
            raw_telemetry={"fan_id": "Drawingroom_Fan_1", "temperature": 31.0, "relative_humidity": 70.0}
        )
        twin.update_from_csr(csr)
        state = twin.inspect_state()

        assert state["room_category"] == "Drawingroom"
        assert state["room_state"]["temperature_c"] == 31.0
        assert state["room_state"]["relative_humidity_pct"] == 70.0

    def test_06_occupancy_updates(self):
        """Verify occupancy state updates from adapter envelope."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")
        envelope = {
            "occupancy_state": "Occupied",
            "estimated_headcount": 2.0,
            "feature_set_used": "set_b",
            "data_provenance": "ROBOD_MODEL_2",
        }
        twin.update_from_adapter_envelope("occupancy", envelope)
        state = twin.inspect_state()

        assert state["room_state"]["occupancy_classification"] == "Occupied"
        assert state["room_state"]["estimated_headcount"] == 2.0
        assert state["room_state"]["occupancy_feature_set"] == "set_b"
        trans = [t for t in twin.history_manager.get_transitions() if t.triggering_event == "OCCUPANCY_STATE_CHANGED"]
        assert len(trans) == 1

    def test_07_energy_history_4_way_segregation(self):
        """Verify strict 4-way separation of energy accounting."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1", fan_mode="BLDC", initial_speed=3)

        # 1. Physics accumulator (Speed 3 BLDC = 20.0W for 360s = 2.0 Wh)
        twin.energy_state.update_physics_energy(3, "BLDC", elapsed_seconds=360.0)

        # 2. Measured telemetry (ground truth 1.95 Wh)
        twin.energy_state.record_measured_telemetry(espent_wh=1.95, power_w=19.5)

        # 3. ML advisory forecast
        ml_envelope = {
            "predicted_energy_wh": 0.082,
            "is_advisory": True,
            "cannot_command_speed": True,
        }
        twin.update_from_adapter_envelope("energy", ml_envelope)

        # Inspect state
        state = twin.inspect_state()
        energy_sec = state["energy_accounting_segregated"]

        cat1 = energy_sec["category_1_measured_telemetry"]
        cat2 = energy_sec["category_2_physics_estimates"]
        cat3 = energy_sec["category_3_ml_advisory_forecast"]
        cat4 = energy_sec["category_4_hardware_calibrated"]

        # Assert no blending occurred
        assert cat1["measured_espent_wh"] == 1.95
        assert cat1["measured_power_w"] == 19.5

        assert cat2["estimated_power_w"] == 20.0
        assert abs(cat2["cumulative_energy_session_wh"] - 2.0) < 1e-4

        assert cat3["predicted_next_energy_wh"] == 0.082
        assert cat3["is_advisory"] is True
        assert cat3["cannot_command_speed"] is True
        assert cat3["predicted_next_energy_r2_benchmark"] == 0.1515

        assert cat4["calibrated_measurement"] is None
        assert cat4["is_hardware_validated"] is False

    def test_08_multiple_fan_isolation(self):
        """Verify twin supervisor maintains complete state isolation across fans."""
        supervisor = TwinSupervisor()
        twin1 = supervisor.get_or_create_twin("Bedroom_Fan_1")
        twin2 = supervisor.get_or_create_twin("Lounge_Fan_1")

        twin1.confirm_speed(4)
        twin2.confirm_speed(1)

        assert twin1.fan_state.current_speed == 4
        assert twin2.fan_state.current_speed == 1
        assert twin1.room_category == "Bedroom"
        assert twin2.room_category == "Lounge"

        # Routing to fan 1 does not alter fan 2
        supervisor.route_telemetry({"fan_id": "Bedroom_Fan_1", "temperature": 29.0, "fan_speed": 5})
        assert supervisor.get_twin("Bedroom_Fan_1").fan_state.current_speed == 5
        assert supervisor.get_twin("Lounge_Fan_1").fan_state.current_speed == 1

    def test_09_stale_sensor_events(self):
        """Verify SENSOR_STALE transition when sensor age exceeds limit."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")

        # Ingest fresh
        csr_fresh = CommonState.from_telemetry(
            raw_telemetry={"fan_id": "Bedroom_Fan_1", "temperature": 27.0, "sensor_age_s": 10.0, "is_stale": False}
        )
        twin.update_from_csr(csr_fresh)
        assert twin.data_quality["sensor_stale"] is False

        # Ingest stale
        csr_stale = CommonState.from_telemetry(
            raw_telemetry={"fan_id": "Bedroom_Fan_1", "temperature": 27.0, "sensor_age_s": 350.0, "is_stale": True}
        )
        twin.update_from_csr(csr_stale)
        assert twin.data_quality["sensor_stale"] is True

        stale_trans = [t for t in twin.history_manager.get_transitions() if t.triggering_event == "SENSOR_STALE"]
        assert len(stale_trans) == 1
        assert stale_trans[0].validity_status == "STALE"

    def test_10_missing_sensor_events(self):
        """Verify sensor gaps (missing light, CO2) are accurately tracked."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")
        csr = CommonState.from_telemetry(
            raw_telemetry={"fan_id": "Bedroom_Fan_1", "temperature": 26.0, "relative_humidity": 50.0}  # light and co2 omitted
        )
        twin.update_from_csr(csr)
        state = twin.inspect_state()

        assert state["data_quality"]["sensor_gap_active"] is True
        assert state["data_quality"]["light_available"] is False
        assert state["data_quality"]["co2_available"] is False
        assert "light" in state["room_state"]["missing_sensors"]
        assert "co2" in state["room_state"]["missing_sensors"]

        assert state["data_quality"]["sensor_gap_active"] is True
        assert state["data_quality"]["light_available"] is False
        assert state["data_quality"]["co2_available"] is False
        assert "light" in state["room_state"]["missing_sensors"]
        assert "co2" in state["room_state"]["missing_sensors"]

    def test_11_adapter_failure_handling(self):
        """Verify digital twin gracefully records unavailable / degraded adapter envelopes."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")
        degraded_envelope = {
            "status": "UNAVAILABLE",
            "error": "Sensor timeout in Model 1 preprocessor",
            "fallback_used": True,
        }
        transitions = twin.update_from_adapter_envelope("comfort", degraded_envelope)
        state = twin.inspect_state()

        # Twin should remain stable and valid
        assert state["fan_state"] if "fan_state" in state else state["fan_id"] == "Bedroom_Fan_1"
        assert state["room_state"]["comfort_assessment"] is None

    def test_12_duplicate_event_handling(self):
        """Verify historical state manager rejects identical duplicate transition events."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")
        fixed_ts = "2024-03-15T12:00:00+00:00"
        record = StateTransitionRecord(
            timestamp=fixed_ts,
            fan_id="Bedroom_Fan_1",
            triggering_event="SPEED_CHANGED",
            previous_state={"current_speed": 1},
            new_state={"current_speed": 2},
            data_provenance="TEST",
        )

        appended_1 = twin.history_manager.append_transition(record)
        appended_2 = twin.history_manager.append_transition(record)

        assert appended_1 is True
        assert appended_2 is False  # Duplicate rejected

    def test_13_out_of_order_event_handling(self):
        """Verify historical state manager correctly orders out-of-order records by timestamp."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")
        twin.history_manager.clear()

        r1 = StateTransitionRecord("2024-03-15T12:00:10", "Bedroom_Fan_1", "SPEED_CHANGED", {}, {}, "TEST")
        r2 = StateTransitionRecord("2024-03-15T12:00:05", "Bedroom_Fan_1", "FAN_SWITCHED_ON", {}, {}, "TEST")
        r3 = StateTransitionRecord("2024-03-15T12:00:15", "Bedroom_Fan_1", "OCCUPANCY_STATE_CHANGED", {}, {}, "TEST")

        twin.history_manager.append_transition(r1)
        twin.history_manager.append_transition(r2)  # Inserted out-of-order
        twin.history_manager.append_transition(r3)

        transitions = twin.history_manager.get_transitions()
        timestamps = [t.timestamp for t in transitions]
        assert timestamps == ["2024-03-15T12:00:05", "2024-03-15T12:00:10", "2024-03-15T12:00:15"]

    def test_14_historical_snapshot_consistency(self):
        """Verify snapshot history respects circular buffer capacity (N=50)."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1", snapshot_capacity=5)
        twin.history_manager.clear()

        for i in range(10):
            twin.history_manager.record_snapshot({"step": i}, timestamp=f"2024-03-15T12:00:{i:02d}")

        snaps = twin.history_manager.get_snapshots()
        assert len(snaps) == 5
        assert snaps[0]["step"] == 5
        assert snaps[-1]["step"] == 9

    def test_17_state_recovery_after_invalid_updates(self):
        """Verify twin raises ValueError on invalid speeds and preserves valid state."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1", initial_speed=2)

        with pytest.raises(ValueError):
            twin.fan_state.set_speed(99)  # Invalid speed > 5

        with pytest.raises(ValueError):
            twin.fan_state.propose_speed(-1)  # Invalid speed < 0

        # Verify state integrity remained intact
        state = twin.inspect_state()
        assert state["current_speed"] == 2
        assert state["proposed_speed"] == 2

    def test_18_preservation_of_value_provenance(self):
        """Verify provenance tags are maintained across room and energy state."""
        twin = FanDigitalTwin(fan_id="Bedroom_Fan_1")
        csr = CommonState.from_telemetry(
            raw_telemetry={
                "fan_id": "Bedroom_Fan_1",
                "temperature": 28.2,
                "relative_humidity": 62.0,
                "provenance": "BENCHMARK_GROUND_TRUTH",
            }
        )
        twin.update_from_csr(csr, data_provenance="BENCHMARK_GROUND_TRUTH")
        state = twin.inspect_state()

        assert state["state_transition_history"][-1]["provenance"] == "BENCHMARK_GROUND_TRUTH"
