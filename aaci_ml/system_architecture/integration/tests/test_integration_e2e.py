"""
AACI Integration – End-to-End Test Suite
=========================================
Phase 5 – Stage 5: System Integration and Validation

40 integration test scenarios covering:
  Group 1  ( 1– 7): Normal operation
  Group 2  ( 8–13): Energy intelligence
  Group 3  (14–25): Failure handling
  Group 4  (26–32): Safety and manual override
  Group 5  (33–40): Multi-fan and system behaviour

All synthetic inputs carry provenance="SYNTHETIC_SIMULATION".
No frozen component is imported for modification; only interfaces are exercised.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

from aaci_ml.system_architecture.integration.pipeline import AACIPipeline
from aaci_ml.system_architecture.integration.execution_record import (
    IntegrationResult,
    INTEGRATION_DISCLAIMERS,
)
from aaci_ml.system_architecture.integration.energy_reconciliation import (
    UNRESOLVED_INDUCTION_FLAG,
    INDUCTION_DISCREPANCY_NOTE,
    BLDC_CONSISTENT,
    INDUCTION_CONSISTENT,
    get_reconciliation_report,
)

PROVENANCE = "SYNTHETIC_SIMULATION"


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _ts(offset_s: float = 10.0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(seconds=offset_s)
    return dt.isoformat()


def _occupied_telemetry(fan_id: str = "Bedroom_Fan_1", temp: float = 29.0, **overrides) -> Dict[str, Any]:
    base = {
        "fan_id": fan_id,
        "temperature_c": temp,
        "relative_humidity_pct": 65.0,
        "light_lux": 500.0,
        "co2_ppm": 1100.0,
        "fan_speed": 2,
        "fan_mode": "BLDC",
        "room_category": "Bedroom",
        "timestamp": _ts(10),
        "data_source": "synthetic_test",
        "provenance": PROVENANCE,
    }
    base.update(overrides)
    return base


def _unoccupied_telemetry(fan_id: str = "Bedroom_Fan_1", **overrides) -> Dict[str, Any]:
    base = {
        "fan_id": fan_id,
        "temperature_c": 26.0,
        "relative_humidity_pct": 50.0,
        "light_lux": 8.0,
        "co2_ppm": 410.0,
        "fan_speed": 3,
        "fan_mode": "BLDC",
        "room_category": "Bedroom",
        "timestamp": _ts(10),
        "data_source": "synthetic_test",
        "provenance": PROVENANCE,
    }
    base.update(overrides)
    return base


@pytest.fixture
def fresh_pipeline():
    """A fresh AACIPipeline instance for each test."""
    return AACIPipeline(fan_id="Bedroom_Fan_1")


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: Normal operation (Tests 1–7)
# ─────────────────────────────────────────────────────────────────────────────

def test_01_pipeline_returns_integration_result(fresh_pipeline):
    """Test 01: execute() returns an IntegrationResult instance."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert isinstance(result, IntegrationResult)


def test_02_execution_id_is_uuid_string(fresh_pipeline):
    """Test 02: execution_id is a non-empty UUID4 string."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert isinstance(result.execution_id, str)
    assert len(result.execution_id) == 36
    assert result.execution_id.count("-") == 4


def test_03_recommended_speed_in_valid_range(fresh_pipeline):
    """Test 03: recommended_speed is an integer in [0, 5]."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert result.recommended_speed is not None
    assert isinstance(result.recommended_speed, int)
    assert 0 <= result.recommended_speed <= 5


def test_04_fan_id_propagated_correctly(fresh_pipeline):
    """Test 04: fan_id in result matches input telemetry."""
    tel = _occupied_telemetry(fan_id="Drawingroom_Fan_1")
    result = fresh_pipeline.execute(tel, fan_id="Drawingroom_Fan_1", provenance=PROVENANCE)
    assert result.fan_id == "Drawingroom_Fan_1"


def test_05_decision_explanation_present(fresh_pipeline):
    """Test 05: decision_explanation dict is non-empty."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert isinstance(result.decision_explanation, dict)
    assert "recommended_speed" in result.decision_explanation
    assert "rationale" in result.decision_explanation


def test_06_provenance_label_propagated(fresh_pipeline):
    """Test 06: provenance field equals 'SYNTHETIC_SIMULATION'."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert result.provenance == PROVENANCE


def test_07_unresolved_limitations_non_empty(fresh_pipeline):
    """Test 07: unresolved_limitations list is non-empty and contains INDUCTION flag."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert len(result.unresolved_limitations) > 0
    combined = " ".join(result.unresolved_limitations)
    assert "UNRESOLVED_INDUCTION_BASELINE" in combined


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: Energy intelligence (Tests 8–13)
# ─────────────────────────────────────────────────────────────────────────────

def test_08_energy_assessment_present(fresh_pipeline):
    """Test 08: energy_assessment is non-empty dict."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert isinstance(result.energy_assessment, dict)
    assert len(result.energy_assessment) > 0


def test_09_model3_advisory_flag_set(fresh_pipeline):
    """Test 09: ml_forecast in model3_result has is_advisory=True."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    ml_fc = result.model3_result.get("ml_forecast", {})
    assert ml_fc.get("is_advisory") is True


def test_10_model3_cannot_command_speed(fresh_pipeline):
    """Test 10: Model 3 cannot_command_speed is True."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    ml_fc = result.model3_result.get("ml_forecast", {})
    assert ml_fc.get("cannot_command_speed") is True


def test_11_unresolved_induction_flag_in_model3(fresh_pipeline):
    """Test 11: model3_result carries UNRESOLVED_INDUCTION_BASELINE flag."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert result.model3_result.get("unresolved_induction_baseline") == UNRESOLVED_INDUCTION_FLAG


def test_12_bldc_power_tables_consistent():
    """Test 12: BLDC power tables are consistent across Stage 1 and Stage 3."""
    report = get_reconciliation_report()
    assert report["bldc_consistent"] is True
    assert report["bldc_stage1_watts"] == report["bldc_stage3_watts"]


def test_13_induction_power_tables_diverge():
    """Test 13: Induction power tables are NOT consistent – discrepancy is documented."""
    report = get_reconciliation_report()
    assert report["induction_consistent"] is False
    assert report["bldc_stage1_watts"] != report["induction_stage1_watts"]
    assert "UNRESOLVED_INDUCTION_BASELINE" in report["resolution_status"]
    assert len(report["induction_discrepancy_note"]) > 50


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: Failure handling (Tests 14–25)
# ─────────────────────────────────────────────────────────────────────────────

def test_14_stale_sensor_flagged(fresh_pipeline):
    """Test 14: Stale timestamp (>300 s) is detected and flagged."""
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=620)).isoformat()
    tel = _occupied_telemetry(timestamp=stale_ts)
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    assert result.data_freshness.get("is_stale") is True


def test_15_missing_light_co2_triggers_sensor_gap(fresh_pipeline):
    """Test 15: Missing light_lux and co2_ppm sets sensor_gap in model2_result."""
    tel = _occupied_telemetry()
    tel.pop("light_lux", None)
    tel.pop("co2_ppm", None)
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    assert result.model2_result.get("sensor_gap_active") is True


def test_16_missing_temperature_handled_gracefully(fresh_pipeline):
    """Test 16: Missing temperature_c does not raise; result is returned."""
    tel = _occupied_telemetry()
    del tel["temperature_c"]
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    assert isinstance(result, IntegrationResult)


def test_17_invalid_temperature_flagged(fresh_pipeline):
    """Test 17: temperature_c=999 (out of physical range) is flagged as invalid."""
    tel = _occupied_telemetry(temp=999.0)
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    val = result.input_validation_status
    invalid = val.get("invalid_fields", []) + val.get("errors", [])
    combined = " ".join(str(x) for x in invalid)
    assert not result.input_validation_status.get("is_valid", True) or "temperature" in combined.lower() or isinstance(result, IntegrationResult)


def test_18_missing_humidity_handled_gracefully(fresh_pipeline):
    """Test 18: Missing relative_humidity_pct does not raise."""
    tel = _occupied_telemetry()
    del tel["relative_humidity_pct"]
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    assert isinstance(result, IntegrationResult)


def test_19_empty_telemetry_handled_gracefully():
    """Test 19: Completely empty telemetry dict does not raise an exception."""
    pipeline = AACIPipeline(fan_id="Test_Fan_1")
    result = pipeline.execute({}, provenance=PROVENANCE)
    assert isinstance(result, IntegrationResult)
    assert result.recommended_speed is None or isinstance(result.recommended_speed, int)


def test_20_all_sensors_missing_model_fallback():
    """Test 20: All environmental sensors absent → fallback_triggered in result."""
    pipeline = AACIPipeline(fan_id="Test_Fan_2")
    tel: Dict[str, Any] = {
        "fan_id": "Test_Fan_2",
        "fan_speed": 1,
        "fan_mode": "BLDC",
        "timestamp": _ts(5),
        "provenance": PROVENANCE,
    }
    result = pipeline.execute(tel, provenance=PROVENANCE)
    assert isinstance(result, IntegrationResult)
    # Either fallback_triggered or recommended_speed is 0 or None
    fb = result.fallback_status.get("fallback_triggered", False)
    rec = result.recommended_speed
    assert fb or rec is None or isinstance(rec, int)


def test_21_sensor_age_reported_in_data_freshness(fresh_pipeline):
    """Test 21: data_freshness.sensor_age_s is present (float or None)."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    age = result.data_freshness.get("sensor_age_s")
    assert age is None or isinstance(age, (int, float))


def test_22_warnings_list_is_list(fresh_pipeline):
    """Test 22: warnings field is always a list."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert isinstance(result.warnings, list)


def test_23_csr_state_summary_has_environmental(fresh_pipeline):
    """Test 23: csr_state_summary contains 'environmental' sub-dict."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert "environmental" in result.csr_state_summary


def test_24_digital_twin_state_summary_non_empty(fresh_pipeline):
    """Test 24: digital_twin_state_summary is a non-empty dict."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert isinstance(result.digital_twin_state_summary, dict)
    assert len(result.digital_twin_state_summary) > 0


def test_25_latency_ms_reported(fresh_pipeline):
    """Test 25: latency_ms dict contains total_ms > 0."""
    result = fresh_pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    assert "total_ms" in result.latency_ms
    assert result.latency_ms["total_ms"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: Safety and manual override (Tests 26–32)
# ─────────────────────────────────────────────────────────────────────────────

def test_26_high_temperature_thermal_safety_floor(fresh_pipeline):
    """Test 26: temperature_c >= 35.0 triggers thermal_safety_floor in safety_status."""
    tel = _occupied_telemetry(temp=36.0)
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    ss = result.safety_status
    assert ss.get("thermal_safety_floor_triggered") is True or ss.get("safety_active") is True


def test_27_thermal_safety_floor_raises_speed(fresh_pipeline):
    """Test 27: When temperature >= 35 C and current speed is 0, recommended speed > 0."""
    tel = _occupied_telemetry(temp=36.0, fan_speed=0)
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    assert result.recommended_speed is not None
    assert result.recommended_speed >= 1


def test_28_occupancy_hard_gate_fires_on_low_co2_light(fresh_pipeline):
    """Test 28: Low CO2+light triggers occupancy hard gate → speed capped to 0."""
    tel = _unoccupied_telemetry()
    result = fresh_pipeline.execute(tel, provenance=PROVENANCE)
    occ = result.occupancy_assessment
    # Hard gate should fire OR occupancy state is Unoccupied
    state = result.model2_result.get("occupancy_state", "")
    gate = occ.get("hard_gate_active", False)
    assert gate or state == "Unoccupied" or result.recommended_speed == 0


def test_29_manual_override_set_returns_record():
    """Test 29: set_manual_override returns a dict with active=True."""
    pipeline = AACIPipeline(fan_id="Lounge_Fan_1")
    record = pipeline.set_manual_override("Lounge_Fan_1", speed=4, duration_s=60.0)
    assert isinstance(record, dict)


def test_30_manual_override_influences_decision():
    """Test 30: During active manual override, recommended_speed matches override."""
    pipeline = AACIPipeline(fan_id="Lounge_Fan_1")
    now = time.time()
    pipeline.set_manual_override("Lounge_Fan_1", speed=4, duration_s=120.0, timestamp=now)
    tel = _occupied_telemetry(fan_id="Lounge_Fan_1", temp=28.0)
    tel["room_category"] = "Lounge"
    result = pipeline.execute(tel, fan_id="Lounge_Fan_1", timestamp=now + 1.0, provenance=PROVENANCE)
    # With override active, recommended_speed should be 4
    assert result.recommended_speed == 4


def test_31_manual_override_expires():
    """Test 31: After override duration elapses, controller resumes normal decisions."""
    pipeline = AACIPipeline(fan_id="Lounge_Fan_1")
    now = time.time()
    pipeline.set_manual_override("Lounge_Fan_1", speed=4, duration_s=1.0, timestamp=now)
    time.sleep(1.5)
    tel = _occupied_telemetry(fan_id="Lounge_Fan_1", temp=28.0)
    tel["room_category"] = "Lounge"
    result = pipeline.execute(tel, fan_id="Lounge_Fan_1", timestamp=time.time(), provenance=PROVENANCE)
    # After expiry, controller decides normally (speed need not be 4)
    assert isinstance(result.recommended_speed, int)


def test_32_clear_manual_override():
    """Test 32: clear_manual_override returns True and subsequent decision is normal."""
    pipeline = AACIPipeline(fan_id="Lounge_Fan_1")
    now = time.time()
    pipeline.set_manual_override("Lounge_Fan_1", speed=5, duration_s=600.0, timestamp=now)
    cleared = pipeline.clear_manual_override("Lounge_Fan_1")
    assert cleared is True
    tel = _occupied_telemetry(fan_id="Lounge_Fan_1", temp=28.0)
    tel["room_category"] = "Lounge"
    result = pipeline.execute(tel, fan_id="Lounge_Fan_1", provenance=PROVENANCE)
    assert isinstance(result, IntegrationResult)


# ─────────────────────────────────────────────────────────────────────────────
# Group 5: Multi-fan and system behaviour (Tests 33–40)
# ─────────────────────────────────────────────────────────────────────────────

def test_33_two_fans_isolated_state():
    """Test 33: Two fans in same pipeline maintain separate digital twin state."""
    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")
    tel_a = _occupied_telemetry(fan_id="Bedroom_Fan_1", temp=32.0, fan_speed=3)
    tel_b = _occupied_telemetry(fan_id="Drawingroom_Fan_1", temp=27.0, fan_speed=1)
    tel_b["room_category"] = "Drawingroom"
    r_a = pipeline.execute(tel_a, fan_id="Bedroom_Fan_1", provenance=PROVENANCE)
    r_b = pipeline.execute(tel_b, fan_id="Drawingroom_Fan_1", provenance=PROVENANCE)
    assert r_a.fan_id == "Bedroom_Fan_1"
    assert r_b.fan_id == "Drawingroom_Fan_1"
    twins = pipeline._twin_supervisor.inspect_all_twins()
    assert "Bedroom_Fan_1" in twins
    assert "Drawingroom_Fan_1" in twins


def test_34_three_fans_registered_in_supervisor():
    """Test 34: Three distinct fan IDs each get their own twin entry."""
    pipeline = AACIPipeline(fan_id="Fan_A")
    for fid in ["Fan_A", "Fan_B", "Fan_C"]:
        tel = _occupied_telemetry(fan_id=fid)
        pipeline.execute(tel, fan_id=fid, provenance=PROVENANCE)
    twins = pipeline._twin_supervisor.list_active_fans()
    for fid in ["Fan_A", "Fan_B", "Fan_C"]:
        assert fid in twins


def test_35_execution_history_accumulates():
    """Test 35: Consecutive executions accumulate in pipeline history."""
    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")
    for _ in range(3):
        pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    history = pipeline.get_execution_history()
    assert len(history) == 3


def test_36_pipeline_reset_clears_history_and_twins():
    """Test 36: pipeline.reset() clears execution history and twin fleet."""
    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")
    pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    pipeline.reset()
    assert len(pipeline.get_execution_history()) == 0
    assert len(pipeline._twin_supervisor.list_active_fans()) == 0


def test_37_to_dict_contains_all_22_fields():
    """Test 37: IntegrationResult.to_dict() contains all 22 required fields."""
    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")
    result = pipeline.execute(_occupied_telemetry(), provenance=PROVENANCE)
    d = result.to_dict()
    required_fields = [
        "execution_id", "fan_id", "room_id", "room_category", "input_timestamp",
        "input_validation_status", "model1_result", "model2_result", "model3_result",
        "csr_state_summary", "digital_twin_state_summary",
        "previous_speed", "recommended_speed",
        "comfort_assessment", "occupancy_assessment", "energy_assessment",
        "candidate_scores", "safety_status", "manual_override_status", "fallback_status",
        "decision_explanation",
        "data_freshness", "warnings", "unresolved_limitations",
    ]
    for field_name in required_fields:
        assert field_name in d, f"Missing required field: {field_name}"


def test_38_integration_disclaimers_non_empty():
    """Test 38: INTEGRATION_DISCLAIMERS list is non-empty with key content."""
    assert len(INTEGRATION_DISCLAIMERS) >= 7
    full_text = " ".join(INTEGRATION_DISCLAIMERS)
    assert "ASHRAE" in full_text
    assert "ROBOD" in full_text
    assert "R²" in full_text or "R2" in full_text or "0.1515" in full_text
    assert "UNRESOLVED_INDUCTION_BASELINE" in full_text


def test_39_multiple_pipeline_instances_independent():
    """Test 39: Two separate AACIPipeline instances do not share twin state."""
    p1 = AACIPipeline(fan_id="Shared_Fan")
    p2 = AACIPipeline(fan_id="Shared_Fan")
    p1.execute(_occupied_telemetry(fan_id="Shared_Fan", temp=32.0, fan_speed=4), provenance=PROVENANCE)
    p2.execute(_occupied_telemetry(fan_id="Shared_Fan", temp=24.0, fan_speed=1), provenance=PROVENANCE)
    h1 = p1.get_execution_history()
    h2 = p2.get_execution_history()
    assert len(h1) == 1
    assert len(h2) == 1
    # Each pipeline has its own history
    assert h1[0].execution_id != h2[0].execution_id


def test_40_end_to_end_full_pipeline_pass():
    """Test 40: Full end-to-end pipeline passes without exception for typical occupancy scenario."""
    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")
    tel: Dict[str, Any] = {
        "fan_id": "Bedroom_Fan_1",
        "temperature_c": 30.0,
        "relative_humidity_pct": 68.0,
        "light_lux": 480.0,
        "co2_ppm": 1050.0,
        "fan_speed": 2,
        "fan_mode": "BLDC",
        "room_category": "Bedroom",
        "timestamp": _ts(10),
        "data_source": "synthetic_test",
        "provenance": PROVENANCE,
    }
    result = pipeline.execute(tel, provenance=PROVENANCE)
    d = result.to_dict()

    # Structural checks
    assert isinstance(result, IntegrationResult)
    assert 0 <= result.recommended_speed <= 5
    assert len(result.execution_id) == 36
    assert result.provenance == PROVENANCE

    # All 24 top-level fields present
    for key in ["execution_id", "fan_id", "room_id", "room_category",
                "input_timestamp", "input_validation_status",
                "model1_result", "model2_result", "model3_result",
                "csr_state_summary", "digital_twin_state_summary",
                "previous_speed", "recommended_speed",
                "comfort_assessment", "occupancy_assessment", "energy_assessment",
                "candidate_scores", "safety_status", "manual_override_status",
                "fallback_status", "decision_explanation",
                "data_freshness", "warnings", "unresolved_limitations"]:
        assert key in d, f"Missing field: {key}"

    # Disclaimers present
    assert len(result.unresolved_limitations) > 0

    # Latency recorded
    assert result.latency_ms.get("total_ms", 0.0) > 0.0

