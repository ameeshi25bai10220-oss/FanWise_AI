"""
AACI V3 – Automated Fan Controller Scenario Test Suite
======================================================
Atomberg Adaptive Comfort Intelligence (AACI)

Rigorously verifies AACIFanControllerV3 across 16 critical production scenarios:
 1. Telemetry Fault: Speed 99 (must trigger HOLD_TELEMETRY_FAULT, not clamp to 5)
 2. Telemetry Fault: Speed -1 or negative (must trigger HOLD_TELEMETRY_FAULT)
 3. Sensor Fault: Missing temperature (None/NaN -> HOLD_SENSOR_FAULT)
 4. Sensor Fault: Impossible high temperature (62C -> HOLD_SENSOR_FAULT)
 5. Sensor Fault: Impossible low temperature (-15C -> HOLD_SENSOR_FAULT)
 6. Stale Telemetry: Sensor timestamp > 180s old -> HOLD_STALE_TELEMETRY
 7. High Confidence Hot: P(Hot)=0.70 >= tau_Hot(0.52) -> INCREASE
 8. High Confidence Cold: P(Cold)=0.65 >= tau_Cold(0.45) -> DECREASE
 9. High Confidence Neutral: P(Neutral)=0.60 >= tau_Neu(0.48) -> HOLD_COMFORT
10. Low Confidence / Ambiguous: P(Hot)=0.40 < tau_Hot(0.52) -> HOLD_UNCERTAIN
11. Narrow Probability Margin: P(Hot)=0.49, P(Neu)=0.47 (margin=0.02 < 0.12) -> HOLD_UNCERTAIN
12. Hysteresis Anti-Hunting: Hot prediction within 300s cooldown -> HOLD_COOLDOWN
13. Manual Override Active: User set Speed 4 with reason 'WHITE_NOISE' -> HOLD_MANUAL
14. Manual Override Expiration: After 120 mins, manual hold expires -> AUTO_RESUMED / actions unblocked
15. Cold Lockout Safety: Ta = 18.5C (<20C), Hot predicted -> HOLD_SAFETY / DECREASE_SAFETY
16. Extreme Heat Floor Safety: Ta = 33.0C (>=31C), Speed = 0 -> INCREASE_SAFETY to speed 2

Outputs:
  - controller_v3_test_report.csv
  - controller_v3_test_report.json
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fan_controller_v3 import AACIFanControllerV3

V3_DIR = os.path.dirname(os.path.abspath(__file__))

def run_scenario_tests():
    print("=" * 80)
    print("AACI V3: EXHAUSTIVE FAN CONTROLLER SCENARIO TEST SUITE")
    print("=" * 80)
    
    t_base = 1700000000.0  # Reference epoch timestamp
    
    scenarios = [
        # Scenario 1: Speed 99 telemetry fault
        {
            "id": 1,
            "name": "Telemetry Fault: Speed 99",
            "setup": lambda ctrl: setattr(ctrl, "current_speed", 2),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.1, "Neutral": 0.2, "Hot": 0.7},
                "air_temp_c": 28.0,
                "current_fan_speed": 99,
                "current_timestamp": t_base + 600,
            },
            "expected_action": "HOLD_TELEMETRY_FAULT",
            "verify": lambda res, ctrl: res["fault_detected"] is True and ctrl.mode == "FAULT",
        },
        # Scenario 2: Speed -1 telemetry fault
        {
            "id": 2,
            "name": "Telemetry Fault: Negative Speed (-1)",
            "setup": lambda ctrl: setattr(ctrl, "current_speed", 2),
            "inputs": {
                "predicted_class": "Cold",
                "probabilities": {"Cold": 0.7, "Neutral": 0.2, "Hot": 0.1},
                "air_temp_c": 23.0,
                "current_fan_speed": -1,
                "current_timestamp": t_base + 600,
            },
            "expected_action": "HOLD_TELEMETRY_FAULT",
            "verify": lambda res, ctrl: res["fault_detected"] is True and res["target_speed"] == 2,
        },
        # Scenario 3: Missing room temperature (None)
        {
            "id": 3,
            "name": "Sensor Fault: Missing Temperature (None)",
            "setup": lambda ctrl: setattr(ctrl, "current_speed", 2),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.1, "Neutral": 0.2, "Hot": 0.7},
                "air_temp_c": None,
                "current_fan_speed": 2,
                "current_timestamp": t_base + 600,
            },
            "expected_action": "HOLD_SENSOR_FAULT",
            "verify": lambda res, ctrl: res["fault_detected"] is True and "Missing room temperature" in res["reason"],
        },
        # Scenario 4: Impossible high temperature (62C)
        {
            "id": 4,
            "name": "Sensor Fault: Impossible High Temperature (62.0C)",
            "setup": lambda ctrl: setattr(ctrl, "current_speed", 2),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.1, "Neutral": 0.2, "Hot": 0.7},
                "air_temp_c": 62.0,
                "current_fan_speed": 2,
                "current_timestamp": t_base + 600,
            },
            "expected_action": "HOLD_SENSOR_FAULT",
            "verify": lambda res, ctrl: res["fault_detected"] is True and "62.0C" in res["reason"],
        },
        # Scenario 5: Impossible low temperature (-15C)
        {
            "id": 5,
            "name": "Sensor Fault: Impossible Low Temperature (-15.0C)",
            "setup": lambda ctrl: setattr(ctrl, "current_speed", 2),
            "inputs": {
                "predicted_class": "Cold",
                "probabilities": {"Cold": 0.8, "Neutral": 0.1, "Hot": 0.1},
                "air_temp_c": -15.0,
                "current_fan_speed": 2,
                "current_timestamp": t_base + 600,
            },
            "expected_action": "HOLD_SENSOR_FAULT",
            "verify": lambda res, ctrl: res["fault_detected"] is True and "-15.0C" in res["reason"],
        },
        # Scenario 6: Stale sensor telemetry (>180s)
        {
            "id": 6,
            "name": "Telemetry Stale: 250s Since Last Update",
            "setup": lambda ctrl: setattr(ctrl, "current_speed", 2),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.1, "Neutral": 0.2, "Hot": 0.7},
                "air_temp_c": 28.0,
                "current_fan_speed": 2,
                "sensor_timestamp": t_base,
                "current_timestamp": t_base + 250.0,
            },
            "expected_action": "HOLD_STALE_TELEMETRY",
            "verify": lambda res, ctrl: res["confidence_status"] == "STALE" and "250s" in res["reason"],
        },
        # Scenario 7: High confidence Hot -> INCREASE
        {
            "id": 7,
            "name": "High Confidence Hot (P=0.70 >= 0.52)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 2), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.10, "Neutral": 0.20, "Hot": 0.70},
                "air_temp_c": 28.5,
                "current_fan_speed": 2,
                "current_timestamp": t_base,
            },
            "expected_action": "INCREASE",
            "verify": lambda res, ctrl: res["target_speed"] == 3 and res["speed_delta"] == 1,
        },
        # Scenario 8: High confidence Cold -> DECREASE
        {
            "id": 8,
            "name": "High Confidence Cold (P=0.65 >= 0.45)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 3), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Cold",
                "probabilities": {"Cold": 0.65, "Neutral": 0.25, "Hot": 0.10},
                "air_temp_c": 23.0,
                "current_fan_speed": 3,
                "current_timestamp": t_base,
            },
            "expected_action": "DECREASE",
            "verify": lambda res, ctrl: res["target_speed"] == 2 and res["speed_delta"] == -1,
        },
        # Scenario 9: High confidence Neutral -> HOLD_COMFORT
        {
            "id": 9,
            "name": "High Confidence Neutral (P=0.60 >= 0.48)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 2), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Neutral",
                "probabilities": {"Cold": 0.20, "Neutral": 0.60, "Hot": 0.20},
                "air_temp_c": 25.0,
                "current_fan_speed": 2,
                "current_timestamp": t_base,
            },
            "expected_action": "HOLD_COMFORT",
            "verify": lambda res, ctrl: res["target_speed"] == 2 and res["speed_delta"] == 0,
        },
        # Scenario 10: Low confidence Hot (P=0.40 < tau_Hot=0.52)
        {
            "id": 10,
            "name": "Low Confidence Hot (P=0.40 < 0.52)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 2), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.35, "Neutral": 0.25, "Hot": 0.40},
                "air_temp_c": 27.0,
                "current_fan_speed": 2,
                "current_timestamp": t_base,
            },
            "expected_action": "HOLD_UNCERTAIN",
            "verify": lambda res, ctrl: res["target_speed"] == 2 and "below threshold" in res["reason"],
        },
        # Scenario 11: Narrow probability margin (margin=0.02 < 0.12)
        {
            "id": 11,
            "name": "Narrow Probability Margin (P_hot=0.49, P_neu=0.47)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 2), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.04, "Neutral": 0.47, "Hot": 0.49},
                "air_temp_c": 27.5,
                "current_fan_speed": 2,
                "current_timestamp": t_base,
            },
            "expected_action": "HOLD_UNCERTAIN",
            "verify": lambda res, ctrl: res["target_speed"] == 2 and "margin" in res["reason"],
        },
        # Scenario 12: Cooldown anti-hunting (within 300s)
        {
            "id": 12,
            "name": "Hysteresis Cooldown (100s since last change < 300s)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 3), setattr(ctrl, "last_adjustment_time", t_base - 100)),
            "inputs": {
                "predicted_class": "Hot",
                "probabilities": {"Cold": 0.10, "Neutral": 0.20, "Hot": 0.70},
                "air_temp_c": 29.0,
                "current_fan_speed": 3,
                "current_timestamp": t_base,
            },
            "expected_action": "HOLD_COOLDOWN",
            "verify": lambda res, ctrl: res["target_speed"] == 3 and "cooldown remaining" in res["reason"],
        },
        # Scenario 13: Manual override priority
        {
            "id": 13,
            "name": "Manual Override Priority (Speed 4 for White Noise)",
            "setup": lambda ctrl: ctrl.set_manual_override(speed=4, duration_mins=60, reason="White noise for sleeping", current_timestamp=t_base),
            "inputs": {
                "predicted_class": "Cold",
                "probabilities": {"Cold": 0.75, "Neutral": 0.15, "Hot": 0.10},
                "air_temp_c": 22.0,
                "current_fan_speed": 4,
                "current_timestamp": t_base + 300, # 5 mins into 60 min override
            },
            "expected_action": "HOLD_MANUAL",
            "verify": lambda res, ctrl: res["target_speed"] == 4 and "White noise" in res["reason"],
        },
        # Scenario 14: Manual override expiration
        {
            "id": 14,
            "name": "Manual Override Expiration (Resumes Auto After 60m)",
            "setup": lambda ctrl: (ctrl.set_manual_override(speed=4, duration_mins=60, reason="Temporary hold", current_timestamp=t_base), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Cold",
                "probabilities": {"Cold": 0.70, "Neutral": 0.20, "Hot": 0.10},
                "air_temp_c": 22.5,
                "current_fan_speed": 4,
                "current_timestamp": t_base + 3700, # 61 mins later -> expired!
            },
            "expected_action": "DECREASE",
            "verify": lambda res, ctrl: res["target_speed"] == 3 and res["action"] == "DECREASE",
        },
        # Scenario 15: Safety Guard 1 - Cold Lockout (<20C)
        {
            "id": 15,
            "name": "Safety Guard: Cold Lockout (Ta=18.5C < 20C)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 2), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Hot", # Erroneous hot vote in very cold room
                "probabilities": {"Cold": 0.10, "Neutral": 0.20, "Hot": 0.70},
                "air_temp_c": 18.5,
                "current_fan_speed": 2,
                "current_timestamp": t_base,
            },
            "expected_action": "DECREASE_SAFETY",
            "verify": lambda res, ctrl: res["target_speed"] == 1 and res["safety_guard_active"] is True and "Safety Guard Active" in res["reason"],
        },
        # Scenario 16: Safety Guard 2 - Extreme Heat Floor (Ta=33C, Speed 0)
        {
            "id": 16,
            "name": "Safety Guard: Extreme Heat Floor (Ta=33.0C, Speed=0)",
            "setup": lambda ctrl: (setattr(ctrl, "current_speed", 0), setattr(ctrl, "last_adjustment_time", t_base - 600)),
            "inputs": {
                "predicted_class": "Cold", # Erroneous cold vote in sweltering room
                "probabilities": {"Cold": 0.70, "Neutral": 0.20, "Hot": 0.10},
                "air_temp_c": 33.0,
                "current_fan_speed": 0,
                "current_timestamp": t_base,
            },
            "expected_action": "INCREASE_SAFETY",
            "verify": lambda res, ctrl: res["target_speed"] == 2 and res["safety_guard_active"] is True and "Extreme room heat" in res["reason"],
        },
    ]

    report = []
    all_passed = True

    print(f"\nRunning {len(scenarios)} validation scenarios...\n")
    print(f"{'ID':<3} | {'Scenario Name':<45} | {'Expected Action':<20} | {'Actual Action':<20} | {'Status'}")
    print("-" * 105)

    for sc in scenarios:
        ctrl = AACIFanControllerV3()
        if sc.get("setup"):
            sc["setup"](ctrl)
            
        res = ctrl.evaluate_decision(**sc["inputs"])
        actual_action = res.get("action")
        expected_action = sc["expected_action"]
        
        matches_action = (actual_action == expected_action)
        matches_verify = sc["verify"](res, ctrl) if "verify" in sc else True
        passed = matches_action and matches_verify
        
        if not passed:
            all_passed = False
            
        status = "PASSED" if passed else "FAILED"
        print(f"{sc['id']:<3} | {sc['name']:<45} | {expected_action:<20} | {actual_action:<20} | {status}")
        
        report.append({
            "scenario_id": sc["id"],
            "scenario_name": sc["name"],
            "expected_action": expected_action,
            "actual_action": actual_action,
            "status": status,
            "target_speed": res.get("target_speed"),
            "current_speed": res.get("current_speed"),
            "speed_delta": res.get("speed_delta"),
            "fault_detected": res.get("fault_detected"),
            "safety_guard_active": res.get("safety_guard_active"),
            "confidence_status": res.get("confidence_status"),
            "reason": res.get("reason"),
        })

    print("-" * 105)
    total_passed = sum(1 for r in report if r["status"] == "PASSED")
    print(f"Results: {total_passed}/{len(scenarios)} Scenarios Passed ({total_passed/len(scenarios)*100:.1f}%)")
    print("=" * 80)

    # Save CSV and JSON
    df_rep = pd.DataFrame(report)
    csv_path = os.path.join(V3_DIR, "controller_v3_test_report.csv")
    json_path = os.path.join(V3_DIR, "controller_v3_test_report.json")
    
    df_rep.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Saved test reports to:\n  - {csv_path}\n  - {json_path}")
    return all_passed

if __name__ == "__main__":
    success = run_scenario_tests()
    sys.exit(0 if success else 1)

