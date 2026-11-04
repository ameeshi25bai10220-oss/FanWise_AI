"""
FanWise AI - BLDC Motor Digital Twin Simulation Engine
======================================================
Simulates the physical electromechanical dynamics of an Atomberg 1200mm BLDC ceiling fan:
  - Tripartite speed architecture: Target Speed, Commanded Speed, Measured Tachometer Speed
  - Physical motor transition delay (15-second ramp-up / ramp-down dynamics)
  - Tachometer RPM response with realistic aerodynamic jitter (+-2 RPM)
  - Instantaneous electrical power consumption (documented Atomberg Renesa spec)
  - Anomaly & Fault Injection: Motor Stall, Speed Mismatch, Sensor Fault, Stale Telemetry

DISCLAIMER: This is a software digital twin simulating motor electromechanics,
not an actual physical hardware motor integration.
"""

from typing import Any, Dict, Optional
import time
import math
import random


# Documented Atomberg Renesa 1200mm BLDC Motor Physical Constants
BLDC_SPEED_TO_RPM = {
    0: 0,
    1: 130,
    2: 180,
    3: 240,
    4: 300,
    5: 360,
}

BLDC_SPEED_TO_WATTS = {
    0: 0.0,
    1: 5.0,
    2: 10.0,
    3: 16.0,
    4: 22.0,
    5: 28.0,
}

# Physical motor acceleration & deceleration characteristics
MOTOR_TRANSITION_TIMEOUT_S = 15.0  # Seconds required for complete multi-step transition


class FanDigitalTwin:
    """
    Virtual digital twin representing an Atomberg Renesa BLDC Ceiling Fan.
    Maintains internal physical state, motor temperature, transition timing,
    and simulated sensor/actuator health telemetry.
    """

    def __init__(self, fan_id: str = "Bedroom_Fan_1", model_name: str = "Atomberg Renesa 1200mm"):
        self.fan_id = fan_id
        self.model_name = model_name
        self.target_speed = 2
        self.commanded_speed = 2
        self.measured_speed = 2
        self.command_time = time.time()
        self.active_fault: str = "NONE"
        self.transition_start_time: Optional[float] = None
        self.simulated_stale_age_s: float = 0.0

    def step(
        self,
        new_commanded_speed: int,
        injected_fault: str = "NONE",
        elapsed_transition_s: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Executes a simulation step, evaluating motor ramp, tachometer feedback,
        power draw, and fault conditions.
        """
        self.active_fault = injected_fault.upper()
        self.target_speed = max(0, min(5, int(new_commanded_speed)))
        
        # 1. Handle Fault Injections
        if self.active_fault == "SENSOR_FAULT":
            return {
                "fan_id": self.fan_id,
                "model_name": self.model_name,
                "motor_state": "SENSOR_FAULT_LOCKOUT",
                "health_status": "FAULT_TRIGGERED",
                "fault_description": "Ambient sensor disconnected or returning impossible NaN values. Fail-safe minimum speed enforced.",
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "rpm": BLDC_SPEED_TO_RPM.get(self.measured_speed, 180),
                "power_w": BLDC_SPEED_TO_WATTS.get(self.measured_speed, 10.0),
                "transition_progress_pct": 100.0,
                "is_transitioning": False,
                "anomaly_alert": "ALARM: Ambient sensor telemetry missing. P1 Fail-safe engaged.",
                "is_digital_twin": True,
            }

        if self.active_fault == "STALE_TELEMETRY":
            self.simulated_stale_age_s = 345.0
            return {
                "fan_id": self.fan_id,
                "model_name": self.model_name,
                "motor_state": "STALENESS_FREEZE",
                "health_status": "COMMUNICATION_TIMEOUT",
                "fault_description": f"Telemetry age ({self.simulated_stale_age_s:.0f}s > 300s limit). Freezing current speed {self.commanded_speed} to avoid hunting.",
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "rpm": BLDC_SPEED_TO_RPM.get(self.measured_speed, 180),
                "power_w": BLDC_SPEED_TO_WATTS.get(self.measured_speed, 10.0),
                "transition_progress_pct": 100.0,
                "is_transitioning": False,
                "anomaly_alert": "WARNING: Telemetry stream stale. Controller frozen at last confirmed speed.",
                "is_digital_twin": True,
            }

        # 2. Motor Transition Physics (Ramp-up / Ramp-down)
        if self.commanded_speed != self.target_speed:
            self.commanded_speed = self.target_speed
            self.command_time = time.time()
            self.transition_start_time = time.time()

        is_transitioning = False
        progress_pct = 100.0

        if elapsed_transition_s < MOTOR_TRANSITION_TIMEOUT_S:
            is_transitioning = True
            progress_pct = round((elapsed_transition_s / MOTOR_TRANSITION_TIMEOUT_S) * 100.0, 1)
            # Tachometer is part-way between previous speed and new speed
            if self.commanded_speed > self.measured_speed:
                motor_state = "RAMPING_UP"
            else:
                motor_state = "RAMPING_DOWN"
        else:
            motor_state = "STEADY"
            self.measured_speed = self.commanded_speed

        # 3. Handle Mechanical / Tachometer Faults
        rpm_jitter = random.choice([-2, -1, 0, 1, 2])
        if self.active_fault == "MOTOR_STALL":
            # Commanded speed > 0, but rotor is locked / stalled
            measured_speed = 0
            rpm = 0
            power_w = 4.2  # Stall current draw
            motor_state = "STALL_DETECTED"
            anomaly = "CRITICAL: Motor rotor locked! Commanded speed > 0, but measured tachometer RPM = 0."
        elif self.active_fault == "SPEED_MISMATCH":
            # Commanded speed is high (e.g. 4), but tachometer indicates low speed (1)
            measured_speed = 1
            rpm = BLDC_SPEED_TO_RPM[1] + rpm_jitter
            power_w = BLDC_SPEED_TO_WATTS[1]
            motor_state = "TACHOMETER_MISMATCH"
            anomaly = f"WARNING: Speed mismatch! Controller commanded Speed {self.commanded_speed}, but tachometer reports Speed {measured_speed}."
        elif self.active_fault == "MANUAL_OVERRIDE":
            motor_state = "MANUAL_OVERRIDE_ACTIVE"
            measured_speed = self.commanded_speed
            rpm = BLDC_SPEED_TO_RPM[measured_speed] + rpm_jitter
            power_w = BLDC_SPEED_TO_WATTS[measured_speed]
            anomaly = f"INFO: User manual remote override active at Speed {measured_speed} (600s lockout)."
        else:
            # Normal steady or transitioning state
            if is_transitioning:
                measured_speed = self.measured_speed
                # Intermediate RPM approximation
                prev_rpm = BLDC_SPEED_TO_RPM.get(self.measured_speed, 180)
                target_rpm = BLDC_SPEED_TO_RPM.get(self.commanded_speed, 180)
                rpm = int(prev_rpm + (target_rpm - prev_rpm) * (progress_pct / 100.0)) + rpm_jitter
                power_w = round(BLDC_SPEED_TO_WATTS.get(self.commanded_speed, 10.0), 1)
                anomaly = f"Transitioning: BLDC motor ramping to Speed {self.commanded_speed} ({progress_pct:.0f}% complete, ~{rpm} RPM)."
            else:
                measured_speed = self.commanded_speed
                rpm = BLDC_SPEED_TO_RPM[measured_speed] + (rpm_jitter if measured_speed > 0 else 0)
                power_w = BLDC_SPEED_TO_WATTS[measured_speed]
                anomaly = "Normal operation. All motor telemetry parameters nominal."

        return {
            "fan_id": self.fan_id,
            "model_name": self.model_name,
            "motor_state": motor_state,
            "health_status": "FAULT" if "STALL" in motor_state or "MISMATCH" in motor_state else "HEALTHY",
            "commanded_speed": self.commanded_speed,
            "measured_speed": measured_speed,
            "target_speed": self.target_speed,
            "rpm": max(0, rpm),
            "power_w": power_w,
            "transition_progress_pct": progress_pct,
            "is_transitioning": is_transitioning,
            "anomaly_alert": anomaly,
            "active_fault": self.active_fault,
            "power_curve": BLDC_SPEED_TO_WATTS,
            "is_digital_twin": True,
            "disclaimer": "Software Digital Twin simulating electromechanical BLDC response; not physical motor hardware.",
        }
