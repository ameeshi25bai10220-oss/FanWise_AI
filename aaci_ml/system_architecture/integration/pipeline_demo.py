"""
AACI Integration - 6 Pipeline Demonstration Scenarios
======================================================
Phase 5 - Stage 5: System Integration and Validation

Run standalone:
    python -m aaci_ml.system_architecture.integration.pipeline_demo

All synthetic inputs carry provenance="SYNTHETIC_SIMULATION".
No physical fan actuation occurs.  All speeds are software-scale 0-5.
No calibrated power savings claims are made.

Demos
-----
1. Occupied warm room   - CO2/Light confirms occupancy; temp rising.
2. Unoccupied room      - Low CO2/Light; occupancy hard gate fires.
3. Changing conditions  - Temperature / humidity varies across timesteps.
4. Manual override      - Set active then expired.
5. Sensor failure       - Stale data, missing sensors, staleness freeze.
6. Multi-fan isolation  - Bedroom_Fan_1 + Lounge_Fan_1 isolated state.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aaci_ml.system_architecture.integration.pipeline import AACIPipeline


PROVENANCE = "SYNTHETIC_SIMULATION"

DISCLAIMER = (
    "\n[NOTE] All inputs below are SYNTHETIC_SIMULATION data."
    "\n       All energy estimates are nominal research constants, uncalibrated."
    "\n       No physical fan actuation occurs."
    "\n       All speed values are software-scale 0-5 recommendations only.\n"
)


def _ts(offset_s: float = 0.0) -> str:
    """Returns ISO 8601 UTC timestamp offset by offset_s seconds from now."""
    dt = datetime.now(timezone.utc) - timedelta(seconds=offset_s)
    return dt.isoformat()


def _print_result(label: str, result: Any) -> None:
    """Prints a compact result summary."""
    print(f"\n  {'-' * 60}")
    print(f"  RESULT: {label}")
    print(f"  {'-' * 60}")
    d = result.to_dict()
    print(f"    execution_id      : {d['execution_id']}")
    print(f"    fan_id            : {d['fan_id']}")
    print(f"    room_category     : {d['room_category']}")
    print(f"    input_valid       : {d['input_validation_status'].get('is_valid')}")
    print(f"    is_stale          : {d['data_freshness'].get('is_stale')}")
    print(f"    previous_speed    : {d['previous_speed']}")
    print(f"    recommended_speed : {d['recommended_speed']}")
    print(f"    decision_mode     : {d['decision_explanation'].get('decision_mode')}")
    print(f"    rationale         : {d['decision_explanation'].get('rationale', '')[:100]}")
    occ = d["model2_result"]
    print(f"    occupancy_state   : {occ.get('occupancy_state')} (p={occ.get('occupancy_probability')})")
    print(f"    comfort_label     : {d['model1_result'].get('comfort_label')}")
    print(f"    fallback_triggered: {d['fallback_status'].get('fallback_triggered')}")
    print(f"    safety_active     : {d['safety_status'].get('safety_active')}")
    manual = d.get("manual_override_status")
    if manual:
        print(f"    manual_override   : {manual}")
    lat = d["latency_ms"]
    print(f"    latency_total_ms  : {lat.get('total_ms')} ms")
    if d["warnings"]:
        print(f"    warnings          : {d['warnings']}")
    print()


# -----------------------------------------------------------------------------
# Demo 1 - Occupied warm room
# -----------------------------------------------------------------------------

def demo_1_occupied_warm_room() -> None:
    print("\n" + "=" * 70)
    print("DEMO 1 - Occupied Warm Room")
    print("Scenario: Temperature rising, CO2 and light confirm occupancy.")
    print("=" * 70)
    print(DISCLAIMER)

    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")

    telemetry: Dict[str, Any] = {
        "fan_id": "Bedroom_Fan_1",
        "temperature_c": 31.0,
        "relative_humidity_pct": 65.0,
        "light_lux": 500.0,
        "co2_ppm": 1200.0,
        "fan_speed": 2,
        "fan_mode": "BLDC",
        "room_category": "Bedroom",
        "timestamp": _ts(10),
        "data_source": "synthetic_demo",
        "provenance": PROVENANCE,
    }

    result = pipeline.execute(telemetry, provenance=PROVENANCE)
    _print_result("Occupied Warm Room", result)


# -----------------------------------------------------------------------------
# Demo 2 - Unoccupied room
# -----------------------------------------------------------------------------

def demo_2_unoccupied_room() -> None:
    print("\n" + "=" * 70)
    print("DEMO 2 - Unoccupied Room")
    print("Scenario: Low CO2 and light -> occupancy hard gate fires -> speed=0.")
    print("=" * 70)
    print(DISCLAIMER)

    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")

    telemetry: Dict[str, Any] = {
        "fan_id": "Bedroom_Fan_1",
        "temperature_c": 28.0,
        "relative_humidity_pct": 55.0,
        "light_lux": 10.0,
        "co2_ppm": 420.0,
        "fan_speed": 3,
        "fan_mode": "BLDC",
        "room_category": "Bedroom",
        "timestamp": _ts(5),
        "data_source": "synthetic_demo",
        "provenance": PROVENANCE,
    }

    result = pipeline.execute(telemetry, provenance=PROVENANCE)
    _print_result("Unoccupied Room", result)


# -----------------------------------------------------------------------------
# Demo 3 - Changing conditions (3 timesteps)
# -----------------------------------------------------------------------------

def demo_3_changing_conditions() -> None:
    print("\n" + "=" * 70)
    print("DEMO 3 - Changing Conditions (3 timesteps)")
    print("Scenario: Temperature and humidity vary; speed recommendation adapts.")
    print("=" * 70)
    print(DISCLAIMER)

    pipeline = AACIPipeline(fan_id="Drawingroom_Fan_1")

    steps = [
        {"temperature_c": 26.0, "relative_humidity_pct": 50.0, "fan_speed": 1, "label": "Step 1 - Cool comfortable"},
        {"temperature_c": 30.0, "relative_humidity_pct": 70.0, "fan_speed": 2, "label": "Step 2 - Warm/humid"},
        {"temperature_c": 34.0, "relative_humidity_pct": 80.0, "fan_speed": 3, "label": "Step 3 - Hot/very humid"},
    ]

    for i, step in enumerate(steps):
        label = step.pop("label")
        print(f"\n  -- {label}")
        tel: Dict[str, Any] = {
            "fan_id": "Drawingroom_Fan_1",
            "light_lux": 400.0,
            "co2_ppm": 900.0,
            "fan_mode": "BLDC",
            "room_category": "Drawingroom",
            "timestamp": _ts(60 * (3 - i)),
            "data_source": "synthetic_demo",
            "provenance": PROVENANCE,
            **step,
        }
        result = pipeline.execute(tel, provenance=PROVENANCE)
        _print_result(label, result)
        time.sleep(0.05)


# -----------------------------------------------------------------------------
# Demo 4 - Manual override (set -> active -> expired)
# -----------------------------------------------------------------------------

def demo_4_manual_override() -> None:
    print("\n" + "=" * 70)
    print("DEMO 4 - Manual Override")
    print("Scenario: User sets speed=4 for 2 s; then override expires.")
    print("=" * 70)
    print(DISCLAIMER)

    pipeline = AACIPipeline(fan_id="Lounge_Fan_1")

    base_tel: Dict[str, Any] = {
        "fan_id": "Lounge_Fan_1",
        "temperature_c": 29.0,
        "relative_humidity_pct": 60.0,
        "light_lux": 350.0,
        "co2_ppm": 850.0,
        "fan_speed": 2,
        "fan_mode": "BLDC",
        "room_category": "Lounge",
        "timestamp": _ts(5),
        "data_source": "synthetic_demo",
        "provenance": PROVENANCE,
    }

    # Step 1 - baseline before override
    print("\n  Step 1: Baseline (no override)")
    r1 = pipeline.execute(base_tel, provenance=PROVENANCE)
    _print_result("Baseline", r1)

    # Step 2 - set manual override speed=4, expires in 2 s
    now = time.time()
    override_record = pipeline.set_manual_override(
        fan_id="Lounge_Fan_1", speed=4, duration_s=2.0, timestamp=now
    )
    print(f"\n  Override set: {override_record}")

    # Step 3 - execute while override is active
    print("\n  Step 3: Execute during override")
    base_tel["timestamp"] = _ts(0)
    r2 = pipeline.execute(base_tel, timestamp=now, provenance=PROVENANCE)
    _print_result("During Override", r2)

    # Step 4 - wait for override to expire, then execute
    time.sleep(2.5)
    print("\n  Step 4: Execute after override expired")
    base_tel["timestamp"] = _ts(0)
    r3 = pipeline.execute(base_tel, timestamp=time.time(), provenance=PROVENANCE)
    _print_result("After Override Expired", r3)


# -----------------------------------------------------------------------------
# Demo 5 - Sensor failure (stale and missing)
# -----------------------------------------------------------------------------

def demo_5_sensor_failure() -> None:
    print("\n" + "=" * 70)
    print("DEMO 5 - Sensor Failure")
    print("Scenario: Stale timestamp (>300 s old), missing CO2/light.")
    print("=" * 70)
    print(DISCLAIMER)

    pipeline = AACIPipeline(fan_id="Bedroom_Fan_2")

    # Packet timestamped 10 minutes ago -> stale
    stale_ts = _ts(offset_s=620)

    telemetry: Dict[str, Any] = {
        "fan_id": "Bedroom_Fan_2",
        "temperature_c": 29.0,
        "relative_humidity_pct": 62.0,
        # light_lux and co2_ppm intentionally absent
        "fan_speed": 2,
        "fan_mode": "BLDC",
        "room_category": "Bedroom",
        "timestamp": stale_ts,
        "data_source": "synthetic_demo",
        "provenance": PROVENANCE,
    }

    result = pipeline.execute(telemetry, provenance=PROVENANCE)
    _print_result("Stale + Missing Sensors", result)
    print(f"  is_stale          : {result.data_freshness.get('is_stale')}")
    print(f"  sensor_age_s      : {result.data_freshness.get('sensor_age_s')}")
    print(f"  stale_fields      : {result.data_freshness.get('stale_fields')}")


# -----------------------------------------------------------------------------
# Demo 6 - Multi-fan isolation
# -----------------------------------------------------------------------------

def demo_6_multi_fan() -> None:
    print("\n" + "=" * 70)
    print("DEMO 6 - Multi-Fan Isolation")
    print("Scenario: Bedroom_Fan_1 and Lounge_Fan_1 maintain isolated state.")
    print("=" * 70)
    print(DISCLAIMER)

    # Single shared pipeline instance manages both fans
    pipeline = AACIPipeline(fan_id="Bedroom_Fan_1")

    tel_bedroom: Dict[str, Any] = {
        "fan_id": "Bedroom_Fan_1",
        "temperature_c": 30.5,
        "relative_humidity_pct": 68.0,
        "light_lux": 480.0,
        "co2_ppm": 1100.0,
        "fan_speed": 3,
        "fan_mode": "BLDC",
        "room_category": "Bedroom",
        "timestamp": _ts(5),
        "data_source": "synthetic_demo",
        "provenance": PROVENANCE,
    }

    tel_lounge: Dict[str, Any] = {
        "fan_id": "Lounge_Fan_1",
        "temperature_c": 27.0,
        "relative_humidity_pct": 55.0,
        "light_lux": 200.0,
        "co2_ppm": 650.0,
        "fan_speed": 1,
        "fan_mode": "BLDC",
        "room_category": "Lounge",
        "timestamp": _ts(8),
        "data_source": "synthetic_demo",
        "provenance": PROVENANCE,
    }

    r_bed = pipeline.execute(tel_bedroom, fan_id="Bedroom_Fan_1", provenance=PROVENANCE)
    r_lou = pipeline.execute(tel_lounge, fan_id="Lounge_Fan_1", provenance=PROVENANCE)

    _print_result("Bedroom_Fan_1", r_bed)
    _print_result("Lounge_Fan_1", r_lou)

    # Verify isolation - twin states should differ
    twins = pipeline._twin_supervisor.inspect_all_twins()
    print(f"\n  Active twins      : {list(twins.keys())}")
    for fid, state in twins.items():
        spd = state.get("fan", {}).get("current_speed") or state.get("current_speed")
        print(f"    {fid}: current_speed={spd}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def run_all_demos() -> None:
    print("\n" + "=" * 70)
    print("  AACI Phase 5 - Stage 5: Pipeline Demonstrations")
    print("  All inputs are SYNTHETIC_SIMULATION. No hardware actuation.")
    print("=" * 70)

    demo_1_occupied_warm_room()
    demo_2_unoccupied_room()
    demo_3_changing_conditions()
    demo_4_manual_override()
    demo_5_sensor_failure()
    demo_6_multi_fan()

    print("\n" + "=" * 70)
    print("  All 6 demos completed.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_demos()

