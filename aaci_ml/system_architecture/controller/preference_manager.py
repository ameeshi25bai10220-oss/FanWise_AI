"""
AACI Adaptive Controller – Preference & Manual Override Manager
==============================================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Manages configurable controller weights and user manual overrides:
- Maintains per-fan manual override state with a default 600-second lockout timer.
- Handles override expiration and state transition back to adaptive control.
- Enforces strict isolation across fans in multi-fan installations.
- NEVER issues physical fan actuation commands.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional, Tuple

from aaci_ml.system_architecture.controller.constants import (
    DEFAULT_ALPHA_COMFORT_WEIGHT,
    DEFAULT_BETA_ENERGY_WEIGHT,
    DEFAULT_DELTA_OCCUPANCY_WEIGHT,
    DEFAULT_GAMMA_STABILITY_WEIGHT,
    MANUAL_OVERRIDE_LOCKOUT_SECONDS,
)


class PreferenceManager:
    """Manages controller weights and per-fan manual overrides in software state."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA_COMFORT_WEIGHT,
        beta: float = DEFAULT_BETA_ENERGY_WEIGHT,
        gamma: float = DEFAULT_GAMMA_STABILITY_WEIGHT,
        delta: float = DEFAULT_DELTA_OCCUPANCY_WEIGHT,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        # Per-fan override records: fan_id -> dict
        self._overrides: Dict[str, Dict[str, Any]] = {}

    def get_weights(self) -> Tuple[float, float, float, float]:
        """Returns current controller weights (alpha, beta, gamma, delta)."""
        return self.alpha, self.beta, self.gamma, self.delta

    def set_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> None:
        """Updates controller optimization weights."""
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)
        if gamma is not None:
            self.gamma = float(gamma)
        if delta is not None:
            self.delta = float(delta)

    def set_manual_override(
        self,
        fan_id: str,
        speed: int,
        duration_s: float = MANUAL_OVERRIDE_LOCKOUT_SECONDS,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Registers a user manual override in virtual software state.
        DOES NOT interact with or command physical hardware.
        """
        if not (0 <= speed <= 5):
            raise ValueError(f"Manual override speed must be in [0, 5], got {speed}")

        curr_ts = timestamp if timestamp is not None else time.time()
        expiry_ts = curr_ts + duration_s
        iso_start = datetime.fromtimestamp(curr_ts, timezone.utc).isoformat()
        iso_expiry = datetime.fromtimestamp(expiry_ts, timezone.utc).isoformat()

        record = {
            "fan_id": fan_id,
            "speed": speed,
            "start_time_s": curr_ts,
            "expiry_time_s": expiry_ts,
            "start_timestamp_iso": iso_start,
            "expiry_timestamp_iso": iso_expiry,
            "duration_s": duration_s,
            "active": True,
        }
        self._overrides[fan_id] = record
        return record

    def clear_manual_override(self, fan_id: str) -> bool:
        """Clears any active manual override for the specified fan."""
        if fan_id in self._overrides:
            self._overrides[fan_id]["active"] = False
            del self._overrides[fan_id]
            return True
        return False

    def get_manual_override(
        self,
        fan_id: str,
        current_time: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Checks if a manual override is currently active and unexpired.
        Returns the override record if active, else clears and returns None.
        """
        if fan_id not in self._overrides:
            return None

        record = self._overrides[fan_id]
        if not record.get("active", False):
            return None

        curr_ts = current_time if current_time is not None else time.time()
        if curr_ts >= record.get("expiry_time_s", 0.0):
            # Expired
            self.clear_manual_override(fan_id)
            return None

        return dict(record)

    def list_active_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Returns all currently active unexpired overrides across all fans."""
        now = time.time()
        active = {}
        for fan_id in list(self._overrides.keys()):
            ov = self.get_manual_override(fan_id, current_time=now)
            if ov is not None:
                active[fan_id] = ov
        return active

