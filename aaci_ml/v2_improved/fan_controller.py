"""
AACI – Adaptive Ceiling Fan Control Decision Layer
===================================================
Atomberg Adaptive Comfort Intelligence (AACI)

This module implements the intelligent control strategy that converts
thermal comfort ML predictions, confidence levels, and physical safety guards
into smooth, stable, energy-efficient fan speed commands.

Key Features:
  - Asymmetric safety guards against Cold-as-Hot classification errors
  - Confidence thresholding & uncertainty damping (abstention)
  - Speed change rate limiting (max +/- 1 step per adjustment)
  - Time-based hysteresis & anti-hunting cooldown (default 300s)
  - Thermal comfort convective cooling calculation
  - Manual override handling with auto-expiration
"""

import time
from typing import Dict, Any, Optional

class AACIFanController:
    """
    Intelligent decision layer for Atomberg BLDC Ceiling Fans.
    Translates ML thermal comfort probabilities into physical speed control.
    """
    
    # Speed bounds
    MIN_SPEED = 0  # Fan OFF
    MAX_SPEED = 5  # Highest speed (e.g. Atomberg Renesa ~360 RPM)
    
    # Safety thresholds (empirically derived from ASHRAE 55 and audit findings)
    COLD_LOCK_TEMP_C   = 20.0  # Prevent fan speed increases if Ta < 20C
    EXTREME_HEAT_TEMP_C = 31.0  # Guarantee minimum airflow if Ta > 31C
    HIGH_HUMIDITY_RH   = 75.0  # High humidity requiring convective assistance
    
    # Confidence & hysteresis constants
    CONFIDENCE_THRESHOLD = 0.50  # Below this, treat prediction as uncertain
    COOLDOWN_SECONDS     = 300   # 5-minute cooldown between automatic speed shifts
    MANUAL_OVERRIDE_SECS = 3600  # 1 hour manual hold before auto-resuming
    
    def __init__(self, fan_model: str = "Atomberg Renesa"):
        self.fan_model = fan_model
        self.current_speed = 2  # Default starting speed (moderate)
        self.last_adjustment_time = 0.0
        self.manual_override_until = 0.0
        self.mode = "AUTO"  # "AUTO", "MANUAL", "ECO", "SLEEP"
        self.history = []
        
    def set_manual_speed(self, speed: int, override_duration_mins: int = 60):
        """User manual override via remote or app."""
        speed = max(self.MIN_SPEED, min(self.MAX_SPEED, int(speed)))
        self.current_speed = speed
        self.manual_override_until = time.time() + (override_duration_mins * 60)
        self.mode = "MANUAL"
        return {
            "status": "MANUAL_SET",
            "current_speed": self.current_speed,
            "override_expires_in_mins": override_duration_mins
        }
        
    def clear_manual_override(self):
        """Restores automatic control mode."""
        self.manual_override_until = 0.0
        self.mode = "AUTO"
        
    def evaluate_decision(
        self,
        predicted_class: str,
        probabilities: Dict[str, float],
        air_temp_c: Optional[float],
        relative_humidity: Optional[float] = 50.0,
        current_air_velocity_ms: float = 0.2,
        current_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the optimal fan speed action based on comfort ML outputs
        and physical safety rules.
        """
        import numpy as np
        now = current_timestamp if current_timestamp is not None else time.time()
        time_since_last_adjust = now - self.last_adjustment_time
        
        # 0. Sanitize current speed state
        try:
            self.current_speed = max(self.MIN_SPEED, min(self.MAX_SPEED, int(self.current_speed)))
        except (ValueError, TypeError):
            self.current_speed = 2  # Safe default moderate speed
            
        # 0b. Validate sensor inputs (Handle missing / NaN readings)
        if air_temp_c is None or (isinstance(air_temp_c, float) and np.isnan(air_temp_c)):
            return {
                "action": "HOLD_SENSOR_FAULT",
                "target_speed": self.current_speed,
                "current_speed": self.current_speed,
                "speed_delta": 0,
                "reason": "Sensor fault: Missing or invalid room temperature reading. Holding current speed safely.",
                "safety_guard_active": True,
                "confidence_status": "FAULT",
                "estimated_cooling_c": self._estimate_cooling(self.current_speed),
            }
            
        rh_eff = 50.0 if (relative_humidity is None or np.isnan(relative_humidity)) else float(relative_humidity)
        
        # 1. Check Manual Override State
        if now < self.manual_override_until:
            remaining_mins = max(0, int((self.manual_override_until - now) / 60))
            return {
                "action": "HOLD_MANUAL",
                "target_speed": self.current_speed,
                "current_speed": self.current_speed,
                "speed_delta": 0,
                "reason": f"Manual override active ({remaining_mins} mins remaining). Auto-control paused.",
                "safety_guard_active": False,
                "confidence_status": "MANUAL",
                "estimated_cooling_c": self._estimate_cooling(self.current_speed),
            }
            
        # 2. Confidence and Uncertainty Assessment
        if not probabilities or not isinstance(probabilities, dict):
            probabilities = {cls: 0.3333 for cls in ["Cold", "Neutral", "Hot"]}
            
        confidence = float(probabilities.get(predicted_class, 0.0))
        sorted_probs = sorted(probabilities.values(), reverse=True)
        prob_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        
        is_uncertain = (confidence < self.CONFIDENCE_THRESHOLD) or (prob_margin < 0.12)
        confidence_status = "LOW_UNCERTAIN" if is_uncertain else ("HIGH" if confidence >= 0.65 else "MODERATE")
        
        # 3. Safety Guard 1: Cold Room Lockout (Protects against 21.27% Cold-as-Hot errors)
        if air_temp_c < self.COLD_LOCK_TEMP_C:
            if self.current_speed > 1:
                # Force decrement to low speed or off in cold rooms
                new_speed = max(0, self.current_speed - 1)
                self.current_speed = new_speed
                self.last_adjustment_time = now
                return {
                    "action": "DECREASE_SAFETY",
                    "target_speed": new_speed,
                    "current_speed": new_speed,
                    "speed_delta": -1,
                    "reason": f"Safety Guard Active: Room temperature ({air_temp_c:.1f}C) is below {self.COLD_LOCK_TEMP_C}C. Locking fan speed to prevent thermal distress.",
                    "safety_guard_active": True,
                    "confidence_status": confidence_status,
                    "estimated_cooling_c": self._estimate_cooling(new_speed),
                }
            elif predicted_class == "Hot":
                # Suppress false Hot prediction
                return {
                    "action": "HOLD_SAFETY",
                    "target_speed": self.current_speed,
                    "current_speed": self.current_speed,
                    "speed_delta": 0,
                    "reason": f"Safety Guard Active: Room is cold ({air_temp_c:.1f}C). Suppressing conflicting Hot prediction.",
                    "safety_guard_active": True,
                    "confidence_status": confidence_status,
                    "estimated_cooling_c": self._estimate_cooling(self.current_speed),
                }

        # 4. Safety Guard 2: High Temperature Safety Floor
        if air_temp_c >= self.EXTREME_HEAT_TEMP_C and self.current_speed < 2:
            new_speed = 2
            self.current_speed = new_speed
            self.last_adjustment_time = now
            return {
                "action": "INCREASE_SAFETY",
                "target_speed": new_speed,
                "current_speed": new_speed,
                "speed_delta": +1,
                "reason": f"Safety Guard Active: Extreme room heat ({air_temp_c:.1f}C). Enforcing minimum speed 2 for convective relief.",
                "safety_guard_active": True,
                "confidence_status": confidence_status,
                "estimated_cooling_c": self._estimate_cooling(new_speed),
            }

        # 5. Cooldown / Anti-Hunting Hysteresis
        if time_since_last_adjust < self.COOLDOWN_SECONDS:
            remaining_cool = int(self.COOLDOWN_SECONDS - time_since_last_adjust)
            return {
                "action": "HOLD_COOLDOWN",
                "target_speed": self.current_speed,
                "current_speed": self.current_speed,
                "speed_delta": 0,
                "reason": f"Hysteresis stabilization: Fan speed adjusted recently ({remaining_cool}s cooldown remaining).",
                "safety_guard_active": False,
                "confidence_status": confidence_status,
                "estimated_cooling_c": self._estimate_cooling(self.current_speed),
            }

        # 6. Low Confidence / Uncertainty Handling
        if is_uncertain:
            return {
                "action": "HOLD_UNCERTAIN",
                "target_speed": self.current_speed,
                "current_speed": self.current_speed,
                "speed_delta": 0,
                "reason": f"Prediction confidence ({confidence:.1%}) is low or margin is narrow. Maintaining speed to avoid erratic changes.",
                "safety_guard_active": False,
                "confidence_status": "LOW_UNCERTAIN",
                "estimated_cooling_c": self._estimate_cooling(self.current_speed),
            }

        # 7. Comfort-Driven Control Rules
        speed_delta = 0
        action = "HOLD"
        reason = "Occupant is thermally comfortable. Current airflow maintains neutral state."

        if predicted_class == "Hot":
            if self.current_speed < self.MAX_SPEED:
                speed_delta = 1
                action = "INCREASE"
                reason = f"Hot discomfort predicted with {confidence:.1%} confidence. Increasing speed by 1 to enhance convective cooling."
            else:
                reason = "Hot discomfort predicted, but fan is already at maximum speed."
                
        elif predicted_class == "Cold":
            if self.current_speed > self.MIN_SPEED:
                speed_delta = -1
                action = "DECREASE"
                reason = f"Cold discomfort predicted with {confidence:.1%} confidence. Reducing speed by 1 to decrease draft sensation."
            else:
                reason = "Cold discomfort predicted, but fan is already off."
                
        elif predicted_class == "Neutral":
            # Eco-mode optimization: If room is mild and fan is high, gently decrement to save energy
            if air_temp_c <= 24.0 and rh_eff <= 60.0 and self.current_speed >= 3:
                speed_delta = -1
                action = "DECREASE_ECO"
                reason = "Neutral comfort in mild room. Reducing fan speed by 1 to conserve energy without compromising comfort."
            else:
                action = "HOLD_COMFORT"
                reason = f"Neutral comfort predicted ({confidence:.1%} confidence). Maintaining speed {self.current_speed}."

        new_speed = max(self.MIN_SPEED, min(self.MAX_SPEED, self.current_speed + speed_delta))
        if speed_delta != 0:
            self.current_speed = new_speed
            self.last_adjustment_time = now

        return {
            "action": action,
            "target_speed": new_speed,
            "current_speed": new_speed,
            "speed_delta": speed_delta,
            "reason": reason,
            "safety_guard_active": False,
            "confidence_status": confidence_status,
            "estimated_cooling_c": self._estimate_cooling(new_speed),
        }

    def _estimate_cooling(self, speed: int) -> float:
        """
        Estimates the skin-surface convective cooling effect (Delta T in Celsius)
        corresponding to Atomberg BLDC fan speed levels.
        """
        # Speed 0: 0.05 m/s, Speed 1: 0.15 m/s, Speed 2: 0.30 m/s,
        # Speed 3: 0.50 m/s, Speed 4: 0.75 m/s, Speed 5: 1.10 m/s
        velocity_map = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.50, 4: 0.75, 5: 1.10}
        v = velocity_map.get(speed, 0.2)
        if v <= 0.15:
            return 0.0
        import numpy as np
        return round(float(1.2 * np.sqrt(v - 0.15)), 2)

