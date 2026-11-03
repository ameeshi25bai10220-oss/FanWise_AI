"""
AACI Fan Digital Twin – Simulation & Replay Test Suite
======================================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Tests the DigitalTwinSimulator against requirements 15 and 16:
15. Deterministic replay of operational scenarios
16. Strict synthetic simulation labelling (SYNTHETIC_SIMULATION tag)
Plus simulation scenario runs:
- Staleness scenario
- Sensor dropout scenario
- Operational speed ramp scenario
"""

import pytest

from aaci_ml.system_architecture.digital_twin import (
    DigitalTwinSimulator,
    FanDigitalTwin,
    SYNTHETIC_SIMULATION_PROVENANCE_TAG,
)


class TestDigitalTwinSimulation:
    """Verifies simulator functionality, deterministic replay, and provenance labelling."""

    def test_15_deterministic_replay(self):
        """Verify identical input sequence yields identical transition and state histories."""
        sim = DigitalTwinSimulator(fan_id="Simulated_Fan_1")
        seq = sim.create_speed_transition_scenario()

        # Run 1
        twin1 = FanDigitalTwin(fan_id="Simulated_Fan_1", initial_speed=0)
        transitions1 = sim.run_replay(twin1, seq)
        state1 = twin1.inspect_state()

        # Run 2
        twin2 = FanDigitalTwin(fan_id="Simulated_Fan_1", initial_speed=0)
        transitions2 = sim.run_replay(twin2, seq)
        state2 = twin2.inspect_state()

        # Compare outputs
        assert len(transitions1) == len(transitions2)
        assert [t.triggering_event for t in transitions1] == [t.triggering_event for t in transitions2]
        assert state1["current_speed"] == state2["current_speed"]
        assert state1["cumulative_energy_session_wh"] == state2["cumulative_energy_session_wh"]

    def test_16_synthetic_simulation_labelling(self):
        """Verify that all synthetic packets and transitions carry the SYNTHETIC_SIMULATION tag."""
        sim = DigitalTwinSimulator(fan_id="Simulated_Fan_1")
        pkt = sim.generate_synthetic_telemetry(temperature=28.0, humidity=50.0)

        # Check raw packet
        assert pkt["provenance"] == SYNTHETIC_SIMULATION_PROVENANCE_TAG
        assert "disclaimer" in pkt
        assert "NOT REAL MEASUREMENTS" in pkt["disclaimer"]

        # Ingest into twin
        twin = FanDigitalTwin(fan_id="Simulated_Fan_1")
        transitions = twin.update_from_telemetry(pkt, data_provenance=pkt["provenance"])

        # Check recorded transitions
        for t in transitions:
            assert t.data_provenance == SYNTHETIC_SIMULATION_PROVENANCE_TAG
            assert t.validity_status == "SIMULATION"

    def test_staleness_scenario_replay(self):
        """Verify simulator staleness scenario triggers SENSOR_STALE followed by recovery."""
        sim = DigitalTwinSimulator(fan_id="Stale_Test_Fan")
        scenario = sim.create_staleness_scenario(initial_speed=2)

        twin = FanDigitalTwin(fan_id="Stale_Test_Fan")
        transitions = sim.run_replay(twin, scenario)

        events = [t.triggering_event for t in transitions]
        assert "SENSOR_STALE" in events
        assert "SENSOR_RECOVERED" in events

    def test_sensor_dropout_scenario_replay(self):
        """Verify simulator dropout scenario correctly reflects sensor gaps."""
        sim = DigitalTwinSimulator(fan_id="Dropout_Fan")
        scenario = sim.create_sensor_dropout_scenario()

        twin = FanDigitalTwin(fan_id="Dropout_Fan")
        sim.run_replay(twin, scenario)

        state = twin.inspect_state()
        assert state["data_quality"]["sensor_gap_active"] is True
        assert state["data_quality"]["light_available"] is False
        assert state["data_quality"]["co2_available"] is False

    def test_speed_ramp_scenario_replay(self):
        """Verify speed ramp scenario records correct speed changes and energy accumulation."""
        sim = DigitalTwinSimulator(fan_id="Ramp_Fan")
        scenario = sim.create_speed_transition_scenario()

        twin = FanDigitalTwin(fan_id="Ramp_Fan", fan_mode="BLDC")
        transitions = sim.run_replay(twin, scenario)

        events = [t.triggering_event for t in transitions]
        assert "FAN_SWITCHED_ON" in events
        assert "SPEED_CHANGED" in events
        assert "FAN_SWITCHED_OFF" in events

        state = twin.inspect_state()
        assert state["current_speed"] == 0
        assert state["is_on"] is False
        assert state["cumulative_energy_session_wh"] > 0.0

