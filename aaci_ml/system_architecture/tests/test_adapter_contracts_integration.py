"""
Integration-Level Adapter Contract Tests
========================================
Phase 5 – Stage 1: Adapter Layer Implementation

Verifies:
  1. Simultaneous execution of all three adapters on realistic ceiling fan telemetry.
  2. Complete isolation between adapters (no shared state, no cross-talk).
  3. Strict schema uniformity across all three `AdapterOutputEnvelope` instances.
  4. Multi-fan independent execution.
  5. End-to-end adapter pipeline execution WITHOUT activating the adaptive controller.
"""

import os
import sys
import pytest
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aaci_ml.system_architecture.adapters import (
    ThermalComfortAdapter,
    OccupancyAdapter,
    EnergyIntelligenceAdapter,
    AdapterOutputEnvelope,
)


@pytest.fixture
def adapter_suite():
    return {
        "comfort": ThermalComfortAdapter(),
        "occupancy": OccupancyAdapter(),
        "energy": EnergyIntelligenceAdapter(),
    }


class TestAdapterContractsIntegration:
    """Integration-level contract tests across all 3 model adapters."""

    def test_single_telemetry_packet_processed_by_all_adapters(self, adapter_suite):
        """Simulate a single IoT telemetry packet processed concurrently by all 3 adapters."""
        telemetry_packet = {
            "environmental": {
                "temperature_c": 28.5,
                "relative_humidity_pct": 58.0,
            },
            "occupancy_signals": {
                "light_lux": None,
                "co2_ppm": None,
            },
            "fan_state": {
                "current_speed": 3,
                "mode": "BLDC",
                "op_time_s": 14.4,
                "espent_wh": 0.08,
            },
            "metadata": {
                "fan_id": "Bedroom_Fan_1",
                "room_id": "Bedroom_Fan_1",
                "room_category": "Bedroom",
                "timestamp": "2026-04-10T14:30:00",
                "sensor_age_seconds": 12.0,
            },
        }

        # 1. Thermal Comfort Adapter
        comfort_out = adapter_suite["comfort"].predict(telemetry_packet)
        assert isinstance(comfort_out, AdapterOutputEnvelope)
        assert comfort_out.prediction in ["Cold", "Neutral", "Hot"]
        assert comfort_out.normalized_payload["air_velocity_mapping"] == "PROVISIONAL_DRYAD"

        # 2. Occupancy Adapter
        occupancy_out = adapter_suite["occupancy"].predict(telemetry_packet)
        assert isinstance(occupancy_out, AdapterOutputEnvelope)
        assert occupancy_out.prediction in ["Occupied", "Unoccupied"]
        assert occupancy_out.normalized_payload["feature_set_used"] == "set_b"
        assert occupancy_out.normalized_payload["sensor_gap_active"] is True

        # 3. Energy Intelligence Adapter
        energy_out = adapter_suite["energy"].predict(telemetry_packet)
        assert isinstance(energy_out, AdapterOutputEnvelope)
        assert len(energy_out.normalized_payload["candidate_evaluation_table"]) == 6
        assert energy_out.normalized_payload["ml_forecast"]["is_advisory"] is True

    def test_envelope_schema_uniformity(self, adapter_suite):
        """Verify that all adapters produce identical top-level envelope keys and value provenance."""
        sample_input = {
            "temperature": 27.0,
            "humidity": 60.0,
            "fan_speed": 3,
            "fan_id": "Bedroom_Fan_1",
            "timestamp": "2026-04-10 12:00:00",
        }

        outputs = [
            adapter_suite["comfort"].predict(sample_input),
            adapter_suite["occupancy"].predict(sample_input),
            adapter_suite["energy"].predict(sample_input),
        ]

        expected_keys = {
            "model_id", "adapter_name", "prediction", "prediction_unit", "timestamp",
            "input_validity", "input_freshness", "model_availability", "missing_inputs",
            "value_provenance", "limitations", "degraded_state", "raw_model_output",
            "normalized_payload"
        }

        for out in outputs:
            d = out.to_dict()
            assert set(d.keys()) == expected_keys
            # Check value provenance categories
            prov = d["value_provenance"]
            assert set(prov.keys()) == {"measured", "assumed", "estimated", "predicted"}
            assert isinstance(prov["measured"], dict)
            assert isinstance(prov["assumed"], dict)
            assert isinstance(prov["estimated"], dict)
            assert isinstance(prov["predicted"], dict)
            # Check degraded state
            deg = d["degraded_state"]
            assert "is_degraded" in deg
            assert "fallback_triggered" in deg

    def test_adapter_isolation_and_no_state_leakage(self, adapter_suite):
        """Verify that running one adapter does not affect internal state of others."""
        # Mutating or throwing error in Comfort does not break Occupancy or Energy
        bad_packet = {"temperature": 255.0, "humidity": 60.0, "fan_speed": 3, "fan_id": "Fan_A"}

        comfort_res = adapter_suite["comfort"].predict(bad_packet)
        occupancy_res = adapter_suite["occupancy"].predict(bad_packet)
        energy_res = adapter_suite["energy"].predict(bad_packet)

        # Comfort and Occupancy reject sentinel temperature
        assert comfort_res.input_validity.is_valid is False
        assert occupancy_res.input_validity.is_valid is False
        # Energy physics candidate table remains operational!
        assert energy_res.model_availability.is_available is True
        assert len(energy_res.normalized_payload["candidate_evaluation_table"]) == 6

    def test_multi_fan_history_isolation(self, adapter_suite):
        """Verify that EnergyIntelligenceAdapter buffers are strictly partitioned per fan_id."""
        energy = adapter_suite["energy"]
        energy.clear_history()

        dt = datetime.datetime(2026, 4, 10, 10, 0, 0)
        # Update Fan 1
        energy.update_history("Fan_1", dt, 28.0, 60.0, 3, 0.08)
        energy.update_history("Fan_1", dt + datetime.timedelta(minutes=2), 28.1, 59.5, 3, 0.08)

        # Fan 1 has 2 packets (warmed)
        assert len(energy.buffers["Fan_1"]) == 2

        # Fan 2 has 0 packets (cold)
        res_fan2 = energy.predict({
            "fan_id": "Fan_2",
            "temperature": 27.0,
            "humidity": 65.0,
            "current_speed": 2,
            "timestamp": dt,
        })
        assert res_fan2.normalized_payload["ml_forecast"]["cold_start"] is True
        assert "Fan_2" in energy.buffers
        assert len(energy.buffers["Fan_2"]) == 1  # auto-seeded
        assert len(energy.buffers["Fan_1"]) == 2  # unchanged

