"""
AACI Adaptive Controller – Safety Guard
=======================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Enforces all priority safety constraints, staleness gates, and fallback logic:
- Priority P1: Manual override handling with thermal floor preservation.
- Priority P2: Sensor staleness freeze (age > 300 s) and sentinel detection.
- Priority P3: Thermal safety floor (T >= 35°C forces s >= 1).
- Priority P4: Occupancy hard gate (P(occupied) < 0.35 forces s = 0).
- Priority P5: Rate limiting (|delta_s| <= 2).
- Emergency safe default when all models and sensors fail.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.controller.constants import (
    CANDIDATE_SPEEDS,
    DECISION_MODE_EMERGENCY_DEFAULT,
    DECISION_MODE_MANUAL_OVERRIDE,
    DECISION_MODE_OCCUPANCY_GATE,
    DECISION_MODE_STALENESS_FREEZE,
    DECISION_MODE_THERMAL_SAFETY,
    RATE_LIMIT_MAX_DELTA_SPEED,
    SENSOR_STALENESS_THRESHOLD_SECONDS,
    THERMAL_SAFETY_MIN_SPEED,
    THERMAL_SAFETY_TEMP_THRESHOLD_C,
    WARM_CONDITION_THRESHOLD_C,
)


class SafetyGuard:
    """Enforces safety rules and fallback protections across controller decisions."""

    def __init__(
        self,
        staleness_threshold_s: float = SENSOR_STALENESS_THRESHOLD_SECONDS,
        thermal_floor_temp_c: float = THERMAL_SAFETY_TEMP_THRESHOLD_C,
        max_rate_delta: int = RATE_LIMIT_MAX_DELTA_SPEED,
    ):
        self.staleness_threshold_s = staleness_threshold_s
        self.thermal_floor_temp_c = thermal_floor_temp_c
        self.max_rate_delta = max_rate_delta

    def check_staleness(self, sensor_age_s: Optional[float], is_stale_flag: bool = False) -> bool:
        """Determines if sensor telemetry has exceeded the staleness threshold (>300s)."""
        if is_stale_flag:
            return True
        if sensor_age_s is not None and sensor_age_s > self.staleness_threshold_s:
            return True
        return False

    def evaluate_pre_decision_safety(
        self,
        csr: Union[CommonState, Dict[str, Any]],
        current_speed: int,
        manual_override: Optional[Dict[str, Any]] = None,
        all_models_unavailable: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes pre-decision safety checks (Priorities P1, P2, and Emergency Default):
        Returns a short-circuit decision dict if a safety trigger is active, else None.
        """
        csr_dict = csr.to_dict() if hasattr(csr, "to_dict") else dict(csr)
        env = csr_dict.get("environmental", {})
        temp = env.get("temperature_c") or csr_dict.get("temperature")
        sensor_age = env.get("sensor_age_s") or csr_dict.get("metadata", {}).get("sensor_age_s")
        is_stale = env.get("is_stale", False) or self.check_staleness(sensor_age)

        # 1. Priority P1: Manual Override
        if manual_override is not None and manual_override.get("active", False):
            ov_speed = int(manual_override["speed"])
            # Thermal safety floor still applies even during manual override (safety_and_fallback_design.md §8.4)
            if temp is not None and temp >= self.thermal_floor_temp_c and ov_speed == 0:
                ov_speed = THERMAL_SAFETY_MIN_SPEED
                notes = "Manual override to 0 overridden by thermal safety floor (T >= 35°C)."
            else:
                notes = f"Manual override active to speed {ov_speed}."

            return {
                "short_circuit": True,
                "recommended_speed": ov_speed,
                "decision_mode": DECISION_MODE_MANUAL_OVERRIDE,
                "fallback_triggered": False,
                "fallback_reason": None,
                "safety_notes": notes,
                "manual_override_status": manual_override,
            }

        # 2. Priority P2 / Emergency: All models unavailable AND sensor stale
        if all_models_unavailable and is_stale:
            # Emergency Safe Default (safety_and_fallback_design.md §7)
            if current_speed > 0:
                safe_speed = current_speed
            else:
                safe_speed = 1 if (temp is not None and temp >= WARM_CONDITION_THRESHOLD_C) else 1

            return {
                "short_circuit": True,
                "recommended_speed": safe_speed,
                "decision_mode": DECISION_MODE_EMERGENCY_DEFAULT,
                "fallback_triggered": True,
                "fallback_reason": "All models unavailable and sensor data stale (>300s). Emergency safe default activated.",
                "safety_notes": "Emergency safe default applied: maintained safe non-zero operating speed.",
                "manual_override_status": None,
            }

        # 3. Priority P2: Sensor Staleness Freeze
        if is_stale:
            return {
                "short_circuit": True,
                "recommended_speed": current_speed,
                "decision_mode": DECISION_MODE_STALENESS_FREEZE,
                "fallback_triggered": True,
                "fallback_reason": f"Sensor telemetry stale (age {sensor_age}s > {self.staleness_threshold_s}s). Decisions frozen.",
                "safety_notes": "Maintained last known confirmed speed to avoid acting on stale telemetry.",
                "manual_override_status": None,
            }

        return None

    def filter_candidate_speeds(
        self,
        current_speed: int,
        temperature_c: Optional[float],
        occupancy_hard_gate_active: bool,
        apply_rate_limiting: bool = True,
    ) -> Tuple[List[int], Optional[str]]:
        """
        Filters candidate speeds through Priorities P3, P4, and P5:
        - Priority P4: Occupancy Hard Gate (forces candidate {0})
        - Priority P3: Thermal Safety Floor (if T >= 35°C and candidate is {0}, forces {1})
        - Priority P5: Rate Limiting (|delta_s| <= 2)
        Returns: (filtered_candidate_speeds, constraint_mode_note)
        """
        # Step 1: Base candidates or Occupancy Hard Gate
        if occupancy_hard_gate_active:
            candidates = [0]
            mode_note = "Occupancy hard gate active (unoccupied)."
        else:
            candidates = list(CANDIDATE_SPEEDS)
            mode_note = None

        # Step 2: Priority P3: Thermal Safety Floor
        if temperature_c is not None and temperature_c >= self.thermal_floor_temp_c:
            if candidates == [0]:
                candidates = [THERMAL_SAFETY_MIN_SPEED]
                mode_note = "Thermal safety floor active (T >= 35°C): minimum speed 1 enforced over occupancy gate."
            else:
                # Disallow 0 when T >= 35°C
                candidates = [s for s in candidates if s >= THERMAL_SAFETY_MIN_SPEED]
                mode_note = "Thermal safety floor active (T >= 35°C): speed 0 excluded."

        # Step 3: Priority P5: Rate Limiting (|s - current_speed| <= 2)
        if apply_rate_limiting:
            rate_limited = [
                s for s in candidates if abs(s - current_speed) <= self.max_rate_delta
            ]
            if rate_limited:
                candidates = rate_limited
            else:
                # If rate limit filtered all (e.g. forced jump), pick closest
                closest = min(candidates, key=lambda s: abs(s - current_speed))
                candidates = [closest]

        return sorted(list(set(candidates))), mode_note

