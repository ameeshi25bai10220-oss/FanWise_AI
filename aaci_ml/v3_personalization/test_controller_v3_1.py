"""
AACI V3.1 – Controller Reliability Test Suite
=============================================
Atomberg Adaptive Comfort Intelligence (AACI)

Validates Priority 1 (Probability Validation) and Priority 2 (Motor Feedback Validation):
Part 1: Probability Distribution Validation
  P1. Valid distribution: Cold=0.10, Neutral=0.20, Hot=0.70 -> INCREASE
  P2. Missing class: Only Cold and Neutral provided -> HOLD_INVALID_PROBABILITY
  P3. Extra / Unknown class: "Warm" included -> HOLD_INVALID_PROBABILITY
  P4. Negative probability: Cold = -0.20 -> HOLD_INVALID_PROBABILITY
  P5. Probability > 1.0: Hot = 1.25 -> HOLD_INVALID_PROBABILITY
  P6. NaN probability: Cold = NaN -> HOLD_INVALID_PROBABILITY
  P7. Infinite probability: Hot = +inf -> HOLD_INVALID_PROBABILITY
  P8. Sum out of tolerance (Sum = 1.40 > 1.01) -> HOLD_INVALID_PROBABILITY
  P9. Sum out of tolerance (Sum = 0.60 < 0.99) -> HOLD_INVALID_PROBABILITY
  P10. Malformed non-dict type (list) -> HOLD_INVALID_PROBABILITY
  P11. Null / None probability input -> HOLD_INVALID_PROBABILITY
  P12. Boolean value inside dictionary -> HOLD_INVALID_PROBABILITY

Part 2: Motor Feedback & Tachometer Validation
  M1. Successful transition: commanded 3, measured 3 -> IDLE / Normal
  M2. Transition in progress: commanded 3, measured 2, t=5s (<15s) -> HOLD_MOTOR_TRANSITIONING
  M3. Persistent tachometer mismatch: commanded 3, measured 2, t=20s (>15s) -> HOLD_ACTUATOR_MISMATCH
  M4. Motor stall: commanded 3, measured 0, t=20s (>15s) -> HOLD_MOTOR_STALL
  M5. Missing tachometer reading (None) -> HOLD_TACHOMETER_FAULT
  M6. Missing tachometer reading (NaN) -> HOLD_TACHOMETER_FAULT
  M7. Invalid measured speed (99) -> HOLD_TELEMETRY_FAULT
  M8. Negative measured speed (-1) -> HOLD_TELEMETRY_FAULT
  M9. Manual override priority with valid feedback -> HOLD_MANUAL
"""

import os
import sys
import math
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V3_DIR = os.path.dirname(os.path.abspath(__file__))
if V3_DIR not in sys.path:
    sys.path.insert(0, V3_DIR)

from fan_controller_v3 import AACIFanControllerV3

def run_tests():
    print("=" * 85)
    print("AACI V3.1: CONTROLLER RELIABILITY TEST SUITE (PRIORITIES 1 & 2)")
    print("=" * 85)

    tests = []
    t_base = 1700000000.0

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: PROBABILITY DISTRIBUTION VALIDATION (PRIORITY 1)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Part 1: Probability Distribution Validation ---")

    # P1: Valid distribution
    ctrl = AACIFanControllerV3()
    ctrl.last_adjustment_time = t_base - 600
    res = ctrl.evaluate_decision("Hot", {"Cold": 0.10, "Neutral": 0.20, "Hot": 0.70}, air_temp_c=28.5, measured_fan_speed=2, current_timestamp=t_base)
    pass_p1 = (res["action"] == "INCREASE" and res["target_speed"] == 3)
    tests.append({"id": "P1", "name": "Valid Distribution (Cold=0.1, Neu=0.2, Hot=0.7)", "expected": "INCREASE", "actual": res["action"], "passed": pass_p1})

    # P2: Missing class
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.50, "Neutral": 0.50}, air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p2 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P2", "name": "Missing Class (Missing 'Hot')", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p2})

    # P3: Extra unexpected class
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Hot", {"Cold": 0.20, "Neutral": 0.40, "Hot": 0.40, "Warm": 0.0}, air_temp_c=28.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p3 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P3", "name": "Extra Unexpected Class ('Warm')", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p3})

    # P4: Negative probability
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Hot", {"Cold": -0.20, "Neutral": 0.40, "Hot": 0.80}, air_temp_c=28.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p4 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P4", "name": "Negative Probability (Cold = -0.20)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p4})

    # P5: Probability > 1.0
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Hot", {"Cold": 0.0, "Neutral": 0.0, "Hot": 1.25}, air_temp_c=28.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p5 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P5", "name": "Probability > 1.0 (Hot = 1.25)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p5})

    # P6: NaN probability
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Cold", {"Cold": float("nan"), "Neutral": 0.50, "Hot": 0.50}, air_temp_c=22.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p6 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P6", "name": "NaN Probability (Cold = NaN)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p6})

    # P7: Infinite probability
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Hot", {"Cold": 0.0, "Neutral": 0.0, "Hot": float("inf")}, air_temp_c=28.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p7 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P7", "name": "Infinite Probability (Hot = +inf)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p7})

    # P8: Sum exceeds 1.0 + tolerance
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.50, "Neutral": 0.50, "Hot": 0.40}, air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p8 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P8", "name": "Sum Exceeds 1.0 (Sum = 1.40)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p8})

    # P9: Sum below 1.0 - tolerance
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.20, "Neutral": 0.20, "Hot": 0.20}, air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p9 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P9", "name": "Sum Below 1.0 (Sum = 0.60)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p9})

    # P10: Malformed type (list instead of dict)
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", [0.2, 0.6, 0.2], air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p10 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P10", "name": "Malformed Type (List input)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p10})

    # P11: Null / None input
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", None, air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p11 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P11", "name": "Null Probability Input (None)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p11})

    # P12: Boolean value in dict
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": False, "Neutral": 0.50, "Hot": 0.50}, air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_p12 = (res["action"] == "HOLD_INVALID_PROBABILITY" and res["fault_detected"] is True)
    tests.append({"id": "P12", "name": "Boolean Value in Dict (Cold=False)", "expected": "HOLD_INVALID_PROBABILITY", "actual": res["action"], "passed": pass_p12})

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: MOTOR FEEDBACK & TACHOMETER VALIDATION (PRIORITY 2)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Part 2: Motor Feedback & Tachometer Validation ---")

    # M1: Successful speed transition (commanded = measured = 3)
    ctrl = AACIFanControllerV3()
    ctrl.commanded_speed = 3
    ctrl.commanded_speed_time = t_base - 30
    ctrl.last_adjustment_time = t_base - 600
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=3, current_timestamp=t_base)
    pass_m1 = (res["action"] == "HOLD_COMFORT" and ctrl.motor_state == "IDLE")
    tests.append({"id": "M1", "name": "Successful Transition (Cmd=3, Meas=3)", "expected": "HOLD_COMFORT", "actual": res["action"], "passed": pass_m1})

    # M2: Transition in progress (commanded = 3, measured = 2, elapsed = 5s < 15s)
    ctrl = AACIFanControllerV3()
    ctrl.commanded_speed = 3
    ctrl.commanded_speed_time = t_base - 5.0  # 5s elapsed
    ctrl.last_adjustment_time = t_base - 5.0
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_m2 = (res["action"] == "HOLD_MOTOR_TRANSITIONING" and ctrl.motor_state == "TRANSITIONING")
    tests.append({"id": "M2", "name": "Transition in Progress (Elapsed=5s < 15s)", "expected": "HOLD_MOTOR_TRANSITIONING", "actual": res["action"], "passed": pass_m2})

    # M3: Persistent tachometer mismatch (commanded = 3, measured = 2, elapsed = 20s > 15s)
    ctrl = AACIFanControllerV3()
    ctrl.commanded_speed = 3
    ctrl.commanded_speed_time = t_base - 20.0  # 20s elapsed
    ctrl.last_adjustment_time = t_base - 20.0
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=2, current_timestamp=t_base)
    pass_m3 = (res["action"] == "HOLD_ACTUATOR_MISMATCH" and ctrl.motor_state == "MISMATCH")
    tests.append({"id": "M3", "name": "Persistent Tachometer Mismatch (Elapsed=20s)", "expected": "HOLD_ACTUATOR_MISMATCH", "actual": res["action"], "passed": pass_m3})

    # M4: Motor stall (commanded = 3, measured = 0, elapsed = 20s > 15s)
    ctrl = AACIFanControllerV3()
    ctrl.commanded_speed = 3
    ctrl.commanded_speed_time = t_base - 20.0
    ctrl.last_adjustment_time = t_base - 20.0
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=0, current_timestamp=t_base)
    pass_m4 = (res["action"] == "HOLD_MOTOR_STALL" and ctrl.motor_state == "STALL" and res["fault_detected"] is True)
    tests.append({"id": "M4", "name": "Motor Stall Alert (Cmd=3, Meas=0, t=20s)", "expected": "HOLD_MOTOR_STALL", "actual": res["action"], "passed": pass_m4})

    # M5: Missing tachometer reading (None)
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=None, current_timestamp=t_base)
    pass_m5 = (res["action"] == "HOLD_TACHOMETER_FAULT" and res["fault_detected"] is True)
    tests.append({"id": "M5", "name": "Missing Tachometer Reading (None)", "expected": "HOLD_TACHOMETER_FAULT", "actual": res["action"], "passed": pass_m5})

    # M6: Missing tachometer reading (NaN)
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=float("nan"), current_timestamp=t_base)
    pass_m6 = (res["action"] == "HOLD_TACHOMETER_FAULT" and res["fault_detected"] is True)
    tests.append({"id": "M6", "name": "Missing Tachometer Reading (NaN)", "expected": "HOLD_TACHOMETER_FAULT", "actual": res["action"], "passed": pass_m6})

    # M7: Invalid measured speed (99)
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=99, current_timestamp=t_base)
    pass_m7 = (res["action"] == "HOLD_TELEMETRY_FAULT" and res["fault_detected"] is True)
    tests.append({"id": "M7", "name": "Invalid Measured Speed (Speed 99)", "expected": "HOLD_TELEMETRY_FAULT", "actual": res["action"], "passed": pass_m7})

    # M8: Negative measured speed (-1)
    ctrl = AACIFanControllerV3()
    res = ctrl.evaluate_decision("Neutral", {"Cold": 0.10, "Neutral": 0.80, "Hot": 0.10}, air_temp_c=25.0, measured_fan_speed=-1, current_timestamp=t_base)
    pass_m8 = (res["action"] == "HOLD_TELEMETRY_FAULT" and res["fault_detected"] is True)
    tests.append({"id": "M8", "name": "Negative Measured Speed (Speed -1)", "expected": "HOLD_TELEMETRY_FAULT", "actual": res["action"], "passed": pass_m8})

    # M9: Manual override priority
    ctrl = AACIFanControllerV3()
    ctrl.set_manual_override(speed=4, duration_mins=60, reason="WHITE_NOISE", current_timestamp=t_base)
    res = ctrl.evaluate_decision("Cold", {"Cold": 0.80, "Neutral": 0.10, "Hot": 0.10}, air_temp_c=22.0, measured_fan_speed=4, current_timestamp=t_base + 300)
    pass_m9 = (res["action"] == "HOLD_MANUAL" and res["commanded_speed"] == 4)
    tests.append({"id": "M9", "name": "Manual Override Priority (Hold Manual)", "expected": "HOLD_MANUAL", "actual": res["action"], "passed": pass_m9})

    # ─────────────────────────────────────────────────────────────────────────
    # PRINT RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'ID':<4} | {'Test Scenario':<44} | {'Expected':<26} | {'Actual':<26} | {'Status'}")
    print("-" * 112)
    all_passed = True
    for t in tests:
        status_str = "PASSED" if t["passed"] else "FAILED"
        if not t["passed"]:
            all_passed = False
        print(f"{t['id']:<4} | {t['name']:<44} | {t['expected']:<26} | {t['actual']:<26} | {status_str}")
    print("=" * 112)

    passed_count = sum(1 for t in tests if t["passed"])
    print(f"\nSummary: {passed_count}/{len(tests)} Tests Passed ({passed_count/len(tests)*100:.1f}%)")
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

