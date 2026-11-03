"""
AACI V3.1 – Robust Uncertainty-Aware Adaptive Fan Controller
============================================================
Atomberg Adaptive Comfort Intelligence (AACI)

Advanced decision layer featuring:
  - Priority 1: Strict probability distribution validation (finite, [0,1], sum=1.0 +- 0.01)
  - Priority 2: Tripartite speed distinction (Target, Commanded, Measured) with transition
    timeout, motor stall, and tachometer mismatch detection
  - Sensor fault detection (missing/impossible/NaN temperatures, stale readings)
  - Class-specific dynamic confidence thresholds (tau_Hot=0.52, tau_Cold=0.45, tau_Neutral=0.48)
  - Probability margin gating (top vs 2nd >= 0.12)
  - Manual override priority with duration tracking and expiration
  - Configurable comfort policies (Cold lockout <20C, Heat floor >=31C)
  - Anti-hunting hysteresis (300-second cooldown) and single-step rate limiting
"""

import time
import math
import numpy as np
from typing import Dict, Any, Optional

_UNSET = object()

class AACIFanControllerV3:
    """
    Production-grade decision engine for Atomberg BLDC Ceiling Fans.
    Translates personalized comfort predictions and sensor telemetry into safe motor actions.
    """
    
    # Valid physical fan speeds
    VALID_SPEEDS = {0, 1, 2, 3, 4, 5}
    MIN_SPEED = 0  # OFF
    MAX_SPEED = 5  # Max RPM (~360 RPM)
    
    # Valid environmental sensor ranges
    MIN_VALID_TEMP_C = -10.0
    MAX_VALID_TEMP_C = 55.0
    MIN_VALID_RH     = 0.0
    MAX_VALID_RH     = 100.0
    MAX_VALID_VELOCITY = 5.0
    STALE_DATA_TIMEOUT_SECS = 180.0  # 3 minutes without telemetry = stale
    
    # Motor actuator specifications
    MOTOR_TRANSITION_TIMEOUT_SECS = 15.0  # 15 seconds for BLDC motor ramp-up/down
    
    # Probability validation tolerance
    PROBABILITY_SUM_TOLERANCE = 0.01  # |sum(p) - 1.0| <= 0.01
    REQUIRED_CLASSES = {"Cold", "Neutral", "Hot"}
    
    # Physical safety thresholds (Configurable comfort policies)
    COLD_LOCK_TEMP_C    = 20.0  # Cold lockout: prevent fan acceleration below 20C
    EXTREME_HEAT_TEMP_C = 31.0  # Extreme heat: ensure minimum speed 2 above 31C
    
    # Class-specific confidence thresholds
    THRESHOLDS = {
        "Cold":    0.45,
        "Neutral": 0.48,
        "Hot":     0.52,
    }
    MIN_PROB_MARGIN = 0.12  # Top probability minus second probability
    
    # Hysteresis & rate limiting
    COOLDOWN_SECONDS     = 300   # 5-minute cooldown between automated speed shifts
    MANUAL_OVERRIDE_SECS = 3600  # Default 1 hour manual hold
    
    def __init__(self, fan_model: str = "Atomberg Renesa"):
        self.fan_model = fan_model
        
        # Priority 2: Tripartite speed tracking
        self.target_speed = 2      # Desired speed computed by algorithm
        self.commanded_speed = 2   # Last speed actively sent to motor driver
        self.measured_speed = 2    # Speed reported by motor tachometer/sensors
        self.commanded_speed_time = 0.0  # Timestamp when commanded_speed was updated
        
        # Operational mode and state
        self.mode = "AUTO"  # "AUTO", "MANUAL", "FAULT", "TRANSITIONING"
        self.motor_state = "IDLE"  # "IDLE", "TRANSITIONING", "MISMATCH", "STALL", "FAULT"
        
        # Timing trackers
        self.last_adjustment_time = 0.0
        self.manual_override_until = 0.0
        self.manual_override_reason: Optional[str] = None
        self.total_adjustments = 0
        
    @property
    def current_speed(self) -> int:
        """Backwards compatibility alias for operational fan speed."""
        return self.commanded_speed

    @current_speed.setter
    def current_speed(self, val: int):
        self.commanded_speed = val
        self.measured_speed = val
        self.target_speed = val

    def set_manual_override(
        self,
        speed: int,
        duration_mins: int = 60,
        reason: Optional[str] = None,
        current_timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Sets an explicit manual user preference, pausing automated adjustments."""
        now = current_timestamp if current_timestamp is not None else time.time()
        if speed not in self.VALID_SPEEDS:
            return {
                "status": "REJECTED_INVALID_SPEED",
                "error": f"Speed {speed} is invalid. Must be one of {sorted(list(self.VALID_SPEEDS))}.",
                "current_speed": self.commanded_speed,
            }
            
        self.commanded_speed = speed
        self.target_speed = speed
        self.commanded_speed_time = now
        self.manual_override_until = now + (duration_mins * 60.0)
        self.manual_override_reason = reason
        self.mode = "MANUAL"
        
        return {
            "status": "MANUAL_ACTIVE",
            "current_speed": self.commanded_speed,
            "duration_mins": duration_mins,
            "expires_at": self.manual_override_until,
            "reason": reason or "User preference",
        }
        
    def clear_manual_override(self) -> Dict[str, Any]:
        """Clears manual override and reactivates automated comfort control."""
        self.manual_override_until = 0.0
        self.manual_override_reason = None
        self.mode = "AUTO"
        return {"status": "AUTO_RESUMED", "current_speed": self.commanded_speed}

    def _validate_probabilities(self, probabilities: Any) -> Optional[str]:
        """
        Priority 1: Rigorous validation of the input probability distribution.
        Returns None if valid, or an error message string if invalid.
        """
        if probabilities is None or not isinstance(probabilities, dict):
            return "Probabilities must be a non-null dictionary."
            
        # Check required classes exactly
        keys_set = set(probabilities.keys())
        if keys_set != self.REQUIRED_CLASSES:
            return f"Invalid probability classes. Required: {sorted(list(self.REQUIRED_CLASSES))}, Received: {sorted(list(keys_set))}."
            
        # Check each probability value
        sum_p = 0.0
        for cls_name, p in probabilities.items():
            if p is None or isinstance(p, bool) or not isinstance(p, (int, float)):
                return f"Probability for '{cls_name}' is not a numeric float/int: {p}."
            if not math.isfinite(p):
                return f"Probability for '{cls_name}' is non-finite (NaN or Inf): {p}."
            if p < 0.0 or p > 1.0:
                return f"Probability for '{cls_name}' is out of bounds [0.0, 1.0]: {p}."
            sum_p += float(p)
            
        # Check sum approximately 1.0
        if abs(sum_p - 1.0) > self.PROBABILITY_SUM_TOLERANCE:
            return f"Probabilities do not sum to 1.0 (Sum = {sum_p:.4f}, tolerance = +-{self.PROBABILITY_SUM_TOLERANCE})."
            
        return None

    def evaluate_decision(
        self,
        predicted_class: str,
        probabilities: Optional[Dict[str, float]],
        air_temp_c: Optional[float],
        relative_humidity: Optional[float] = 50.0,
        current_fan_speed: Any = _UNSET,
        measured_fan_speed: Any = _UNSET,
        current_air_velocity_ms: float = 0.2,
        sensor_timestamp: Optional[float] = None,
        current_timestamp: Optional[float] = None,
        user_preference_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Rigorously evaluates thermal comfort state, telemetry validity, motor feedback,
        and safety rules to issue deterministic fan speed actions.
        """
        now = current_timestamp if current_timestamp is not None else time.time()
        time_since_last_adjust = now - self.last_adjustment_time
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. STRICT TELEMETRY & ENVIRONMENTAL SENSOR SANITIZATION
        # ─────────────────────────────────────────────────────────────────────
        # 1a. Validate Room Temperature
        if air_temp_c is None or (isinstance(air_temp_c, float) and (np.isnan(air_temp_c) or np.isinf(air_temp_c))):
            self.mode = "FAULT"
            return {
                "action": "HOLD_SENSOR_FAULT",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": "Sensor Fault: Missing room temperature reading. Holding current speed safely.",
                "fault_detected": True,
                "safety_guard_active": True,
                "confidence_status": "FAULT",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }
            
        if not (self.MIN_VALID_TEMP_C <= air_temp_c <= self.MAX_VALID_TEMP_C):
            self.mode = "FAULT"
            return {
                "action": "HOLD_SENSOR_FAULT",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": f"Sensor Fault: Impossible temperature {air_temp_c:.1f}C (valid: {self.MIN_VALID_TEMP_C} to {self.MAX_VALID_TEMP_C}C). Holding speed.",
                "fault_detected": True,
                "safety_guard_active": True,
                "confidence_status": "FAULT",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }
            
        # 1b. Validate Stale Sensor Telemetry
        if sensor_timestamp is not None and (now - sensor_timestamp > self.STALE_DATA_TIMEOUT_SECS):
            stale_secs = int(now - sensor_timestamp)
            return {
                "action": "HOLD_STALE_TELEMETRY",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": f"Telemetry Stale: No update for {stale_secs}s (exceeds {int(self.STALE_DATA_TIMEOUT_SECS)}s timeout). Auto-adjustments suspended.",
                "fault_detected": True,
                "safety_guard_active": True,
                "confidence_status": "STALE",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }

        # ─────────────────────────────────────────────────────────────────────
        # 2. MOTOR FEEDBACK & TACHOMETER VALIDATION (PRIORITY 2)
        # ─────────────────────────────────────────────────────────────────────
        # Determine reported measured speed:
        # If neither argument provided, defaults to current internal commanded speed
        if measured_fan_speed is _UNSET and current_fan_speed is _UNSET:
            raw_measured = self.commanded_speed
        elif measured_fan_speed is not _UNSET:
            raw_measured = measured_fan_speed
        else:
            raw_measured = current_fan_speed
        
        # Missing tachometer reading
        if raw_measured is None or (isinstance(raw_measured, float) and (np.isnan(raw_measured) or np.isinf(raw_measured))):
            self.mode = "FAULT"
            self.motor_state = "FAULT"
            return {
                "action": "HOLD_TACHOMETER_FAULT",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": None,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": "Tachometer Fault: Missing motor speed telemetry. Auto-control held for safety.",
                "fault_detected": True,
                "safety_guard_active": True,
                "confidence_status": "FAULT",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }
            
        # Out-of-bounds speed reading (e.g. 99, -1)
        if raw_measured not in self.VALID_SPEEDS:
            self.mode = "FAULT"
            self.motor_state = "FAULT"
            return {
                "action": "HOLD_TELEMETRY_FAULT",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": raw_measured,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": f"Telemetry Fault: Measured fan speed {raw_measured} is invalid (valid: 0-5). Holding state to protect motor.",
                "fault_detected": True,
                "safety_guard_active": True,
                "confidence_status": "FAULT",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }
            
        self.measured_speed = int(raw_measured)
        
        # Check transition timeout and actuator mismatch
        if self.measured_speed != self.commanded_speed:
            time_in_transition = now - self.commanded_speed_time
            
            # Within transition window: allow motor time to ramp RPM
            if time_in_transition < self.MOTOR_TRANSITION_TIMEOUT_SECS:
                self.motor_state = "TRANSITIONING"
                rem_time = int(self.MOTOR_TRANSITION_TIMEOUT_SECS - time_in_transition)
                return {
                    "action": "HOLD_MOTOR_TRANSITIONING",
                    "target_speed": self.commanded_speed,
                    "commanded_speed": self.commanded_speed,
                    "measured_speed": self.measured_speed,
                    "current_speed": self.commanded_speed,
                    "speed_delta": 0,
                    "reason": f"Motor Transitioning: Commanded Speed {self.commanded_speed}, measured Speed {self.measured_speed} ({rem_time}s transition window remaining).",
                    "fault_detected": False,
                    "safety_guard_active": False,
                    "confidence_status": "TRANSITIONING",
                    "estimated_cooling_c": self._estimate_cooling(self.measured_speed),
                }
            else:
                # Persistent discrepancy after timeout: Motor Stall vs Actuator Mismatch
                if self.measured_speed == 0 and self.commanded_speed > 0:
                    self.motor_state = "STALL"
                    self.mode = "FAULT"
                    return {
                        "action": "HOLD_MOTOR_STALL",
                        "target_speed": self.commanded_speed,
                        "commanded_speed": self.commanded_speed,
                        "measured_speed": 0,
                        "current_speed": self.commanded_speed,
                        "speed_delta": 0,
                        "reason": f"Motor Stall Alert: Commanded Speed {self.commanded_speed}, but tachometer reports 0 RPM after {int(time_in_transition)}s. Disengaging auto commands.",
                        "fault_detected": True,
                        "safety_guard_active": True,
                        "confidence_status": "FAULT",
                        "estimated_cooling_c": 0.0,
                    }
                else:
                    self.motor_state = "MISMATCH"
                    return {
                        "action": "HOLD_ACTUATOR_MISMATCH",
                        "target_speed": self.commanded_speed,
                        "commanded_speed": self.commanded_speed,
                        "measured_speed": self.measured_speed,
                        "current_speed": self.commanded_speed,
                        "speed_delta": 0,
                        "reason": f"Actuator Mismatch Warning: Commanded Speed {self.commanded_speed}, but measured Speed is {self.measured_speed} after {int(time_in_transition)}s.",
                        "fault_detected": True,
                        "safety_guard_active": True,
                        "confidence_status": "MISMATCH",
                        "estimated_cooling_c": self._estimate_cooling(self.measured_speed),
                    }
        else:
            self.motor_state = "IDLE"
            if self.mode == "TRANSITIONING":
                self.mode = "AUTO"

        # ─────────────────────────────────────────────────────────────────────
        # 3. PROBABILITY VALIDATION (PRIORITY 1)
        # ─────────────────────────────────────────────────────────────────────
        prob_error = self._validate_probabilities(probabilities)
        if prob_error is not None:
            return {
                "action": "HOLD_INVALID_PROBABILITY",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": f"Probability Fault: {prob_error} Automated adjustments blocked.",
                "fault_detected": True,
                "safety_guard_active": True,
                "confidence_status": "INVALID_PROBABILITY",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }

        rh_eff = 50.0 if (relative_humidity is None or np.isnan(relative_humidity)) else float(np.clip(relative_humidity, 0.0, 100.0))

        # ─────────────────────────────────────────────────────────────────────
        # 4. MANUAL OVERRIDE CHECK
        # ─────────────────────────────────────────────────────────────────────
        if now < self.manual_override_until:
            remaining_mins = max(0, int((self.manual_override_until - now) / 60))
            return {
                "action": "HOLD_MANUAL",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": f"Manual override active ({remaining_mins} mins remaining). Reason: '{self.manual_override_reason or 'User preference'}'. Auto-control paused.",
                "fault_detected": False,
                "safety_guard_active": False,
                "confidence_status": "MANUAL",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }

        # ─────────────────────────────────────────────────────────────────────
        # 5. CLASS-SPECIFIC CONFIDENCE & MARGIN GATES
        # ─────────────────────────────────────────────────────────────────────
        confidence = float(probabilities.get(predicted_class, 0.0))
        sorted_probs = sorted(probabilities.values(), reverse=True)
        prob_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        
        required_threshold = self.THRESHOLDS.get(predicted_class, 0.50)
        is_uncertain = (confidence < required_threshold) or (prob_margin < self.MIN_PROB_MARGIN)
        confidence_status = "LOW_UNCERTAIN" if is_uncertain else ("HIGH" if confidence >= 0.65 else "MODERATE")

        # ─────────────────────────────────────────────────────────────────────
        # 6. CONFIGURABLE THERMAL POLICY GUARDS
        # ─────────────────────────────────────────────────────────────────────
        # Policy Guard 1: Cold Lockout (<20C)
        if air_temp_c < self.COLD_LOCK_TEMP_C:
            if self.commanded_speed > 1:
                new_speed = max(0, self.commanded_speed - 1)
                self.commanded_speed = new_speed
                self.target_speed = new_speed
                self.commanded_speed_time = now
                self.last_adjustment_time = now
                self.total_adjustments += 1
                return {
                    "action": "DECREASE_SAFETY",
                    "target_speed": new_speed,
                    "commanded_speed": new_speed,
                    "measured_speed": self.measured_speed,
                    "current_speed": new_speed,
                    "speed_delta": -1,
                    "reason": f"Safety Guard Active: Room temperature ({air_temp_c:.1f}C) below {self.COLD_LOCK_TEMP_C}C cold lockout. Throttling fan.",
                    "fault_detected": False,
                    "safety_guard_active": True,
                    "confidence_status": confidence_status,
                    "estimated_cooling_c": self._estimate_cooling(new_speed),
                }
            elif predicted_class == "Hot":
                return {
                    "action": "HOLD_SAFETY",
                    "target_speed": self.commanded_speed,
                    "commanded_speed": self.commanded_speed,
                    "measured_speed": self.measured_speed,
                    "current_speed": self.commanded_speed,
                    "speed_delta": 0,
                    "reason": f"Safety Guard Active: Room is cold ({air_temp_c:.1f}C). Suppressing conflicting Hot prediction.",
                    "fault_detected": False,
                    "safety_guard_active": True,
                    "confidence_status": confidence_status,
                    "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
                }

        # Policy Guard 2: Extreme Heat Minimum Floor (>=31C)
        if air_temp_c >= self.EXTREME_HEAT_TEMP_C and self.commanded_speed < 2:
            new_speed = 2
            self.commanded_speed = new_speed
            self.target_speed = new_speed
            self.commanded_speed_time = now
            self.last_adjustment_time = now
            self.total_adjustments += 1
            return {
                "action": "INCREASE_SAFETY",
                "target_speed": new_speed,
                "commanded_speed": new_speed,
                "measured_speed": self.measured_speed,
                "current_speed": new_speed,
                "speed_delta": +1,
                "reason": f"Safety Guard Active: Extreme room heat ({air_temp_c:.1f}C). Enforcing minimum speed 2 for convective relief.",
                "fault_detected": False,
                "safety_guard_active": True,
                "confidence_status": confidence_status,
                "estimated_cooling_c": self._estimate_cooling(new_speed),
            }

        # ─────────────────────────────────────────────────────────────────────
        # 7. HYSTERESIS COOLDOWN & UNCERTAINTY DAMPING
        # ─────────────────────────────────────────────────────────────────────
        if time_since_last_adjust < self.COOLDOWN_SECONDS:
            remaining_cool = int(self.COOLDOWN_SECONDS - time_since_last_adjust)
            return {
                "action": "HOLD_COOLDOWN",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": f"Hysteresis active: Fan speed adjusted recently ({remaining_cool}s cooldown remaining).",
                "fault_detected": False,
                "safety_guard_active": False,
                "confidence_status": confidence_status,
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }

        if is_uncertain:
            return {
                "action": "HOLD_UNCERTAIN",
                "target_speed": self.commanded_speed,
                "commanded_speed": self.commanded_speed,
                "measured_speed": self.measured_speed,
                "current_speed": self.commanded_speed,
                "speed_delta": 0,
                "reason": f"Prediction confidence ({confidence:.1%}) below threshold ({required_threshold:.1%}) or margin ({prob_margin:.2f}) narrow. Holding speed.",
                "fault_detected": False,
                "safety_guard_active": False,
                "confidence_status": "LOW_UNCERTAIN",
                "estimated_cooling_c": self._estimate_cooling(self.commanded_speed),
            }

        # ─────────────────────────────────────────────────────────────────────
        # 8. COMFORT-DRIVEN DECISION LOGIC
        # ─────────────────────────────────────────────────────────────────────
        speed_delta = 0
        action = "HOLD"
        reason = "Occupant is thermally comfortable. Current airflow maintains neutral state."

        if predicted_class == "Hot":
            if self.commanded_speed < self.MAX_SPEED:
                speed_delta = 1
                action = "INCREASE"
                reason = f"Hot discomfort predicted with {confidence:.1%} confidence (tau={required_threshold:.2f}). Increasing speed by 1 to enhance cooling."
            else:
                reason = "Hot discomfort predicted, but fan is already at maximum speed."
                
        elif predicted_class == "Cold":
            if self.commanded_speed > self.MIN_SPEED:
                speed_delta = -1
                action = "DECREASE"
                reason = f"Cold discomfort predicted with {confidence:.1%} confidence (tau={required_threshold:.2f}). Reducing speed by 1 to decrease draft."
            else:
                reason = "Cold discomfort predicted, but fan is already off."
                
        elif predicted_class == "Neutral":
            # Eco-mode optimization: If room is mild and fan is high, gently decrement
            if air_temp_c <= 24.0 and rh_eff <= 60.0 and self.commanded_speed >= 3:
                speed_delta = -1
                action = "DECREASE_ECO"
                reason = "Neutral comfort in mild room. Reducing fan speed by 1 to conserve energy."
            else:
                action = "HOLD_COMFORT"
                reason = f"Neutral comfort predicted ({confidence:.1%} confidence). Maintaining speed {self.commanded_speed}."

        new_target = max(self.MIN_SPEED, min(self.MAX_SPEED, self.commanded_speed + speed_delta))
        self.target_speed = new_target
        
        if speed_delta != 0:
            self.commanded_speed = new_target
            self.commanded_speed_time = now
            self.last_adjustment_time = now
            self.total_adjustments += 1

        return {
            "action": action,
            "target_speed": new_target,
            "commanded_speed": self.commanded_speed,
            "measured_speed": self.measured_speed,
            "current_speed": self.commanded_speed,
            "speed_delta": speed_delta,
            "reason": reason,
            "fault_detected": False,
            "safety_guard_active": False,
            "confidence_status": confidence_status,
            "estimated_cooling_c": self._estimate_cooling(new_target),
        }

    def _estimate_cooling(self, speed: int) -> float:
        velocity_map = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.50, 4: 0.75, 5: 1.10}
        v = velocity_map.get(speed, 0.2)
        if v <= 0.15:
            return 0.0
        return round(float(1.2 * np.sqrt(v - 0.15)), 2)
