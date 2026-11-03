"""
Unit and Contract Tests for EnergyIntelligenceAdapter (Model 3)
==============================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Verifies:
  1. Authoritative physics candidate speed evaluation table for speeds 0..5 (BLDC & Induction).
  2. Cold start behavior when causal history < 2 packets.
  3. ML next-packet energy forecasting after history buffer warmup.
  4. Advisory nature of ML forecaster (is_advisory=True, cannot_command_speed=True).
  5. Strict independence between physics engine calculations and ML forecast.
  6. Missing sensor / sentinel handling for ML path.
  7. Artifact loading failure robustness (physics engine remains operational).
  8. Buffer management and per-fan isolation.
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

from aaci_ml.system_architecture.adapters.energy_adapter import EnergyIntelligenceAdapter
from aaci_ml.system_architecture.adapters.contract import AdapterOutputEnvelope
from aaci_ml.system_architecture.adapters.constants import BLDC_POWER_MAP, INDUCTION_POWER_MAP


@pytest.fixture
def adapter():
    a = EnergyIntelligenceAdapter()
    a.clear_history()
    return a


class TestEnergyIntelligenceAdapterPhysicsEngine:
    """1. Physics engine candidate speed evaluations."""

    def test_bldc_candidate_evaluation_table(self, adapter):
        inputs = {
            "fan_id": "Bedroom_Fan_1",
            "current_speed": 3,
            "fan_mode": "BLDC",
            "op_time_s": 14.4,
        }
        res = adapter.predict(inputs)
        assert isinstance(res, AdapterOutputEnvelope)
        assert res.model_id == "AACI_Model_3_Energy_LightGBM_PhysicsEngine"
        assert res.model_availability.is_available is True

        table = res.normalized_payload["candidate_evaluation_table"]
        assert len(table) == 6
        for entry in table:
            s = entry["speed"]
            expected_power = BLDC_POWER_MAP[s]
            expected_energy = round(expected_power * 14.4 / 3600.0, 4)
            assert entry["power_w"] == expected_power
            assert entry["energy_wh_physics"] == expected_energy
            assert entry["source"] == "physics_engine_deterministic"

    def test_induction_candidate_evaluation_table(self, adapter):
        inputs = {
            "fan_id": "Bedroom_Fan_2",
            "current_speed": 4,
            "fan_mode": "Induction",
            "op_time_s": 3600.0,  # 1 hour
        }
        res = adapter.predict(inputs)
        table = res.normalized_payload["candidate_evaluation_table"]
        assert len(table) == 6
        for entry in table:
            s = entry["speed"]
            expected_power = INDUCTION_POWER_MAP[s]
            expected_energy = round(expected_power * 3600.0 / 3600.0, 4)
            assert entry["power_w"] == expected_power
            assert entry["energy_wh_physics"] == expected_energy

    def test_speed_zero_power_is_strictly_zero(self, adapter):
        inputs = {"current_speed": 0, "op_time_s": 1000.0}
        res = adapter.predict(inputs)
        s0_entry = res.normalized_payload["candidate_evaluation_table"][0]
        assert s0_entry["power_w"] == 0.0
        assert s0_entry["energy_wh_physics"] == 0.0


class TestEnergyIntelligenceAdapterMLForecaster:
    """2, 3, 4, 5: Cold start, ML warmup, advisory status, and causal lag buffer."""

    def test_cold_start_on_fresh_device(self, adapter):
        """First packet has no lag history -> cold start."""
        inputs = {
            "fan_id": "Lounge_Fan_1",
            "temperature": 29.0,
            "humidity": 62.0,
            "current_speed": 2,
            "timestamp": "2026-04-01 10:00:00",
        }
        res = adapter.predict(inputs)
        ml = res.normalized_payload["ml_forecast"]
        assert ml["cold_start"] is True
        assert ml["ml_available"] is False
        assert ml["is_advisory"] is True
        assert ml["cannot_command_speed"] is True
        # Falls back to physics engine energy for current speed
        current_physics_energy = res.normalized_payload["candidate_evaluation_table"][2]["energy_wh_physics"]
        assert ml["predicted_next_energy_wh"] == current_physics_energy

    def test_ml_forecast_after_warmup(self, adapter):
        """Feeding at least 2 historical packets enables LightGBM inference."""
        fan = "Bedroom_Fan_1"
        t0 = datetime.datetime(2026, 4, 1, 12, 0, 0)

        # Seed packet 1
        adapter.predict({
            "fan_id": fan,
            "temperature": 28.0,
            "humidity": 60.0,
            "current_speed": 3,
            "espent_wh": 0.075,
            "timestamp": t0,
        })

        # Seed packet 2
        adapter.predict({
            "fan_id": fan,
            "temperature": 28.1,
            "humidity": 59.8,
            "current_speed": 3,
            "espent_wh": 0.080,
            "timestamp": t0 + datetime.timedelta(minutes=2),
        })

        # Packet 3: history >= 2 -> ML forecaster generates forecast
        res = adapter.predict({
            "fan_id": fan,
            "temperature": 28.2,
            "humidity": 59.5,
            "current_speed": 3,
            "espent_wh": 0.082,
            "timestamp": t0 + datetime.timedelta(minutes=4),
        })

        ml = res.normalized_payload["ml_forecast"]
        assert ml["cold_start"] is False
        assert ml["ml_available"] is True
        assert ml["predicted_next_energy_wh"] >= 0.0
        assert ml["model_used"] == "LightGBM"
        assert ml["is_advisory"] is True
        assert ml["cannot_command_speed"] is True
        assert "advisory" in res.limitations[0].lower()


class TestEnergyIntelligenceAdapterMissingSensorsAndFailures:
    """6, 7, 8: Missing sensors, sentinels, artifact failure handling."""

    def test_missing_environmental_sensors_ml_fails_gracefully(self, adapter):
        """If temperature/humidity missing, physics engine still works, ML flags unavailable."""
        res = adapter.predict({
            "fan_id": "Bedroom_Fan_1",
            "current_speed": 3,
            # no temperature, no humidity
        })
        assert res.model_availability.is_available is True
        assert res.normalized_payload["physics_engine_available"] is True
        assert res.normalized_payload["ml_forecast"]["ml_available"] is False
        assert "temperature" in res.missing_inputs

    def test_temperature_sentinel_handling(self, adapter):
        res = adapter.predict({
            "fan_id": "Bedroom_Fan_1",
            "temperature": 255.0,
            "humidity": 60.0,
            "current_speed": 3,
        })
        # Physics engine still evaluates candidates
        assert len(res.normalized_payload["candidate_evaluation_table"]) == 6
        # ML cannot run with sentinel temperature
        assert res.normalized_payload["ml_forecast"]["ml_available"] is False

    def test_artifact_loading_failure_preserves_physics_engine(self, adapter):
        """Simulating missing LightGBM artifact does not crash physics engine."""
        with patch.object(adapter, "_ensure_artifacts_loaded", return_value=False):
            res = adapter.predict({
                "fan_id": "Bedroom_Fan_1",
                "temperature": 28.0,
                "humidity": 60.0,
                "current_speed": 2,
            })
            assert res.model_availability.is_available is True
            assert res.normalized_payload["physics_engine_available"] is True
            assert res.normalized_payload["ml_forecast"]["ml_available"] is False


class TestEnergyIntelligenceAdapterSchemaAndDeterminism:
    """9, 10: Schema validation and determinism."""

    def test_output_schema_structure(self, adapter):
        res = adapter.predict({
            "fan_id": "Bedroom_Fan_1",
            "temperature": 28.0,
            "humidity": 60.0,
            "current_speed": 3,
        })
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
        fan = "Bedroom_Fan_1"
        dt = datetime.datetime(2026, 4, 1, 14, 0, 0)
        # Seed history
        adapter.update_history(fan, dt - datetime.timedelta(minutes=4), 28.0, 60.0, 3, 0.08)
        adapter.update_history(fan, dt - datetime.timedelta(minutes=2), 28.1, 59.8, 3, 0.08)

        payload = {
            "fan_id": fan,
            "temperature": 28.2,
            "humidity": 59.5,
            "current_speed": 3,
            "timestamp": dt,
        }
        res1 = adapter.predict(payload)
        # Reset history to same state and re-predict
        adapter.clear_history(fan)
        adapter.update_history(fan, dt - datetime.timedelta(minutes=4), 28.0, 60.0, 3, 0.08)
        adapter.update_history(fan, dt - datetime.timedelta(minutes=2), 28.1, 59.8, 3, 0.08)
        res2 = adapter.predict(payload)

        assert res1.prediction == res2.prediction
        assert (
            res1.normalized_payload["ml_forecast"]["predicted_next_energy_wh"]
            == res2.normalized_payload["ml_forecast"]["predicted_next_energy_wh"]
        )

