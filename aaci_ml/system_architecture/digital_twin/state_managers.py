"""
AACI Fan Digital Twin – State Managers
======================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Implements state managers for:
1. FanStateManager: Fan operating speed, mode, pending transitions, manual override.
2. RoomStateManager: Room environmental condition, occupancy, sensor freshness, provenance.
3. EnergyStateManager: Strict 4-way segregated energy accounting:
   - Measured energy telemetry
   - Deterministic research-physics estimates
   - ML-predicted next-packet advisory forecast (LightGBM, R^2 approx 0.1515)
   - Future hardware-calibrated energy (reserved placeholder)

All calculations strictly maintain value provenance and disclaimers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple

from aaci_ml.system_architecture.digital_twin.constants import (
    DEFAULT_MANUAL_OVERRIDE_TIMEOUT_SECONDS,
    RESIDENTIAL_FAN_ROOM_CATEGORIES,
    SPEED_CHANGE_PENDING_TIMEOUT_SECONDS,
)

# Nominal power tables for research-physics estimates (W)
# Sourced from dataset documentation / nominal specifications (PROVISIONAL/UNCALIBRATED)
BLDC_POWER_TABLE_WATTS: Tuple[float, ...] = (0.0, 10.0, 15.0, 20.0, 27.0, 48.0)
INDUCTION_POWER_TABLE_WATTS: Tuple[float, ...] = (0.0, 16.0, 24.0, 36.0, 52.0, 75.0)

# Academic ROBOD room descriptions
ROBOD_ROOM_LABELS: Dict[int, str] = {
    1: "Lecture Room 1",
    2: "Lecture Room 2",
    3: "Seminar Room 1",
    4: "Seminar Room 2",
    5: "Server / Lab Room",
}


@dataclass
class FanStateManager:
    """Manages virtual operational state of a single fan."""

    fan_id: str = "Bedroom_Fan_1"
    room_category: str = "Bedroom"
    fan_mode: str = "BLDC"  # "BLDC" or "Induction"
    current_speed: int = 0
    proposed_speed: int = 0
    speed_change_pending: bool = False
    speed_change_pending_timestamp: Optional[float] = None
    manual_override_active: bool = False
    manual_override_speed: Optional[int] = None
    manual_override_expiry_timestamp: Optional[float] = None
    session_start_timestamp: Optional[str] = None
    last_state_update: Optional[str] = None
    state_validity: bool = True

    def __post_init__(self) -> None:
        if self.room_category == "Bedroom" and self.fan_id in RESIDENTIAL_FAN_ROOM_CATEGORIES:
            self.room_category = RESIDENTIAL_FAN_ROOM_CATEGORIES[self.fan_id]
        if self.session_start_timestamp is None:
            self.session_start_timestamp = datetime.now(timezone.utc).isoformat()
        if self.last_state_update is None:
            self.last_state_update = self.session_start_timestamp

    @property
    def is_on(self) -> bool:
        """True if current speed is greater than 0."""
        return self.current_speed > 0

    def set_speed(self, speed: int, timestamp: Optional[float] = None) -> None:
        """Sets the confirmed current fan speed directly (e.g. from telemetry)."""
        if not (0 <= speed <= 5):
            raise ValueError(f"Speed must be in [0, 5], got {speed}")
        self.current_speed = speed
        self.proposed_speed = speed
        self.speed_change_pending = False
        self.speed_change_pending_timestamp = None
        self.last_state_update = (
            datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )

    def propose_speed(self, speed: int, timestamp: Optional[float] = None) -> None:
        """Records a proposed speed awaiting physical/telemetry acknowledgement."""
        if not (0 <= speed <= 5):
            raise ValueError(f"Speed must be in [0, 5], got {speed}")
        self.proposed_speed = speed
        curr_ts = timestamp if timestamp is not None else time.time()
        if self.proposed_speed != self.current_speed:
            self.speed_change_pending = True
            self.speed_change_pending_timestamp = curr_ts
        else:
            self.speed_change_pending = False
            self.speed_change_pending_timestamp = None
        self.last_state_update = (
            datetime.fromtimestamp(curr_ts, timezone.utc).isoformat()
        )

    def confirm_speed(self, speed: int, timestamp: Optional[float] = None) -> None:
        """Confirms telemetry feedback matching or updating speed."""
        self.set_speed(speed, timestamp=timestamp)

    def set_manual_override(
        self,
        speed: int,
        duration_s: float = DEFAULT_MANUAL_OVERRIDE_TIMEOUT_SECONDS,
        timestamp: Optional[float] = None,
    ) -> None:
        """Sets user manual override speed flag (does NOT send hardware commands)."""
        if not (0 <= speed <= 5):
            raise ValueError(f"Override speed must be in [0, 5], got {speed}")
        curr_ts = timestamp if timestamp is not None else time.time()
        self.manual_override_active = True
        self.manual_override_speed = speed
        self.manual_override_expiry_timestamp = curr_ts + duration_s
        self.current_speed = speed
        self.proposed_speed = speed
        self.speed_change_pending = False
        self.last_state_update = (
            datetime.fromtimestamp(curr_ts, timezone.utc).isoformat()
        )

    def clear_manual_override(self, timestamp: Optional[float] = None) -> None:
        """Clears manual override."""
        self.manual_override_active = False
        self.manual_override_speed = None
        self.manual_override_expiry_timestamp = None
        curr_ts = timestamp if timestamp is not None else time.time()
        self.last_state_update = (
            datetime.fromtimestamp(curr_ts, timezone.utc).isoformat()
        )

    def is_override_active(self, current_time: Optional[float] = None) -> bool:
        """Checks whether manual override is currently active and unexpired."""
        if not self.manual_override_active:
            return False
        curr_ts = current_time if current_time is not None else time.time()
        if (
            self.manual_override_expiry_timestamp is not None
            and curr_ts >= self.manual_override_expiry_timestamp
        ):
            self.clear_manual_override(timestamp=curr_ts)
            return False
        return True

    def check_pending_timeout(self, current_time: Optional[float] = None) -> bool:
        """Checks if speed change acknowledgement timed out, resetting proposed speed."""
        if not self.speed_change_pending:
            return False
        curr_ts = current_time if current_time is not None else time.time()
        if (
            self.speed_change_pending_timestamp is not None
            and (curr_ts - self.speed_change_pending_timestamp) >= SPEED_CHANGE_PENDING_TIMEOUT_SECONDS
        ):
            self.proposed_speed = self.current_speed
            self.speed_change_pending = False
            self.speed_change_pending_timestamp = None
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes fan state to dictionary."""
        return {
            "fan_id": self.fan_id,
            "room_category": self.room_category,
            "fan_mode": self.fan_mode,
            "current_speed": self.current_speed,
            "proposed_speed": self.proposed_speed,
            "is_on": self.is_on,
            "speed_change_pending": self.speed_change_pending,
            "manual_override_active": self.manual_override_active,
            "manual_override_speed": self.manual_override_speed,
            "manual_override_expiry_timestamp": self.manual_override_expiry_timestamp,
            "session_start_timestamp": self.session_start_timestamp,
            "last_state_update": self.last_state_update,
            "state_validity": self.state_validity,
        }


@dataclass
class RoomStateManager:
    """Manages virtual environmental and occupancy state of the room."""

    temperature: Optional[float] = None
    relative_humidity: Optional[float] = None
    air_velocity: Optional[float] = None
    outdoor_monthly_temp_c: Optional[float] = None
    occupancy_classification: str = "Unknown"
    occupancy_probability: Optional[float] = None
    estimated_headcount: Optional[float] = None
    occupancy_feature_set: Optional[str] = None
    headcount_uncertainty_range: Optional[Tuple[float, float]] = None
    robod_room_id: Optional[int] = None
    robod_training_context: Optional[str] = None
    sensor_freshness_seconds: Optional[float] = None
    is_sensor_stale: bool = False
    missing_sensors: List[str] = field(default_factory=list)
    data_provenance: Dict[str, str] = field(default_factory=dict)
    comfort_assessment: Optional[str] = None
    comfort_probabilities_uncalibrated: bool = True
    airflow_mapping_provisional: bool = True

    def update_from_csr(self, csr_dict: Dict[str, Any]) -> None:
        """Updates room state from a validated Common State Representation dictionary or instance."""
        # Handle dict or dataclass
        state_dict = (
            csr_dict
            if isinstance(csr_dict, dict)
            else (csr_dict.to_dict() if hasattr(csr_dict, "to_dict") else vars(csr_dict))
        )

        env = state_dict.get("environmental") or {}
        occ = state_dict.get("occupancy") or state_dict.get("occupancy_signals") or {}
        prov = state_dict.get("provenance") or state_dict.get("metadata", {}).get("data_source") or {}
        health = state_dict.get("system_health") or state_dict.get("health") or {}
        meta = state_dict.get("metadata") or {}

        if "temperature_c" in env and env["temperature_c"] is not None:
            self.temperature = float(env["temperature_c"])
        elif "air_temp_c" in env and env["air_temp_c"] is not None:
            self.temperature = float(env["air_temp_c"])
        elif "temperature" in state_dict and state_dict["temperature"] is not None:
            self.temperature = float(state_dict["temperature"])

        if "relative_humidity_pct" in env and env["relative_humidity_pct"] is not None:
            self.relative_humidity = float(env["relative_humidity_pct"])
        elif "relative_humidity" in state_dict and state_dict["relative_humidity"] is not None:
            self.relative_humidity = float(state_dict["relative_humidity"])

        if "air_velocity_ms" in env:
            self.air_velocity = (
                float(env["air_velocity_ms"]) if env["air_velocity_ms"] is not None else None
            )
        elif "air_velocity" in state_dict:
            self.air_velocity = (
                float(state_dict["air_velocity"]) if state_dict["air_velocity"] is not None else None
            )

        if "outdoor_monthly_temp_c" in env:
            self.outdoor_monthly_temp_c = (
                float(env["outdoor_monthly_temp_c"])
                if env["outdoor_monthly_temp_c"] is not None
                else None
            )

        sensor_age = env.get("sensor_age_s") or meta.get("sensor_age_s") or meta.get("sensor_age_seconds")
        if sensor_age is not None:
            self.sensor_freshness_seconds = float(sensor_age)

        if "is_stale" in env:
            self.is_sensor_stale = bool(env["is_stale"])
        elif "sensor_stale" in state_dict:
            self.is_sensor_stale = bool(state_dict["sensor_stale"])

        missing = list(health.get("missing_sensors") or env.get("missing_sensors") or [])
        if not occ.get("light_available", True) and "light" not in missing:
            missing.append("light")
        if not occ.get("co2_available", True) and "co2" not in missing:
            missing.append("co2")
        self.missing_sensors = missing

        occ_class = occ.get("occupancy_classification") or occ.get("classification")
        if occ_class is not None:
            self.occupancy_classification = str(occ_class)

        occ_prob = occ.get("occupancy_probability") or occ.get("probability")
        if occ_prob is not None:
            self.occupancy_probability = float(occ_prob)

        if "estimated_headcount" in occ and occ["estimated_headcount"] is not None:
            self.estimated_headcount = float(occ["estimated_headcount"])
        if "feature_set" in occ and occ["feature_set"] is not None:
            self.occupancy_feature_set = str(occ["feature_set"])
        elif "feature_set_used" in occ and occ["feature_set_used"] is not None:
            self.occupancy_feature_set = str(occ["feature_set_used"])

        if "robod_room_id" in occ and occ["robod_room_id"] is not None:
            self.robod_room_id = int(occ["robod_room_id"])
            self.robod_training_context = (
                f"ROBOD academic training room {self.robod_room_id}: "
                f"{ROBOD_ROOM_LABELS.get(self.robod_room_id, 'Unknown')}"
            )

        # Merge data provenance
        if isinstance(prov, dict):
            for k, v in prov.items():
                self.data_provenance[k] = str(v)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes room state to dictionary."""
        return {
            "temperature_c": self.temperature,
            "relative_humidity_pct": self.relative_humidity,
            "air_velocity_ms": self.air_velocity,
            "outdoor_monthly_temp_c": self.outdoor_monthly_temp_c,
            "occupancy_classification": self.occupancy_classification,
            "occupancy_probability": self.occupancy_probability,
            "estimated_headcount": self.estimated_headcount,
            "occupancy_feature_set": self.occupancy_feature_set,
            "headcount_uncertainty_range": self.headcount_uncertainty_range,
            "robod_room_id": self.robod_room_id,
            "robod_training_context": self.robod_training_context,
            "sensor_freshness_seconds": self.sensor_freshness_seconds,
            "is_sensor_stale": self.is_sensor_stale,
            "missing_sensors": self.missing_sensors,
            "data_provenance": self.data_provenance,
            "comfort_assessment": self.comfort_assessment,
            "comfort_probabilities_uncalibrated": self.comfort_probabilities_uncalibrated,
            "airflow_mapping_provisional": self.airflow_mapping_provisional,
        }


@dataclass
class EnergyStateManager:
    """
    Manages energy state with strict 4-way segregation:
    1. Measured telemetry (from smart meter / hardware packet)
    2. Physics-derived research estimates (P(s) * dt / 3600)
    3. ML advisory forecasts (LightGBM, R^2 approx 0.1515, advisory only)
    4. Hardware calibrated energy (reserved placeholder)
    """

    # 1. Measured telemetry
    measured_espent_wh: Optional[float] = None
    measured_power_w: Optional[float] = None
    measured_history: List[Dict[str, Any]] = field(default_factory=list)

    # 2. Deterministic research-physics estimates
    estimated_power_w: float = 0.0
    cumulative_energy_session_wh: float = 0.0
    cumulative_energy_lifetime_wh: float = 0.0
    last_physics_update_timestamp: Optional[float] = None
    physics_estimate_source: str = "nominal_dataset_constants_uncalibrated"

    # 3. ML advisory forecast
    predicted_next_energy_wh: Optional[float] = None
    predicted_next_energy_source: str = "LightGBM_ML_ADVISORY"
    predicted_next_energy_r2_benchmark: float = 0.1515
    is_advisory: bool = True
    cannot_command_speed: bool = True
    ml_cold_start: bool = True
    forecast_history: List[Dict[str, Any]] = field(default_factory=list)

    # 4. Hardware calibrated energy (reserved placeholder)
    calibrated_measurement: Optional[float] = None
    is_hardware_validated: bool = False
    calibration_notes: str = "Reserved for future physical Atomberg test bench calibration"

    def update_physics_energy(
        self, current_speed: int, fan_mode: str, elapsed_seconds: float
    ) -> float:
        """
        Updates deterministic energy accumulator using nominal research physics constants:
        delta_E = P(s) * elapsed_seconds / 3600.0 (Wh).
        Returns delta_E.
        """
        table = (
            BLDC_POWER_TABLE_WATTS
            if fan_mode.upper() == "BLDC"
            else INDUCTION_POWER_TABLE_WATTS
        )
        speed_idx = max(0, min(current_speed, 5))
        power_w = table[speed_idx]
        self.estimated_power_w = power_w

        if elapsed_seconds > 0.0:
            delta_wh = (power_w * elapsed_seconds) / 3600.0
            self.cumulative_energy_session_wh += delta_wh
            self.cumulative_energy_lifetime_wh += delta_wh
            return delta_wh
        return 0.0

    def record_measured_telemetry(
        self,
        espent_wh: Optional[float],
        power_w: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Records ground-truth telemetry measurement without mixing into physics/ML."""
        if espent_wh is not None:
            self.measured_espent_wh = float(espent_wh)
        if power_w is not None:
            self.measured_power_w = float(power_w)

        ts = (
            datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )
        self.measured_history.append(
            {"timestamp": ts, "espent_wh": espent_wh, "power_w": power_w}
        )
        # Keep last 50 telemetry measurements
        if len(self.measured_history) > 50:
            self.measured_history.pop(0)

    def record_ml_forecast(self, adapter_envelope: Dict[str, Any]) -> None:
        """Records an advisory energy forecast from the Model 3 adapter."""
        # Extract energy prediction fields
        pred_val = adapter_envelope.get("predicted_energy_wh")
        if pred_val is None:
            pred_val = adapter_envelope.get("predicted_next_packet_energy_wh")

        if pred_val is not None:
            self.predicted_next_energy_wh = float(pred_val)
            self.ml_cold_start = False
        else:
            self.predicted_next_energy_wh = None

        if "is_advisory" in adapter_envelope:
            self.is_advisory = bool(adapter_envelope["is_advisory"])
        if "cannot_command_speed" in adapter_envelope:
            self.cannot_command_speed = bool(adapter_envelope["cannot_command_speed"])

        ts = adapter_envelope.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        self.forecast_history.append(
            {
                "timestamp": ts,
                "predicted_next_energy_wh": self.predicted_next_energy_wh,
                "is_advisory": self.is_advisory,
                "cannot_command_speed": self.cannot_command_speed,
                "r2_benchmark": self.predicted_next_energy_r2_benchmark,
            }
        )
        if len(self.forecast_history) > 50:
            self.forecast_history.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes energy state with 4-way segregation preserved."""
        return {
            "category_1_measured_telemetry": {
                "measured_espent_wh": self.measured_espent_wh,
                "measured_power_w": self.measured_power_w,
                "history_count": len(self.measured_history),
            },
            "category_2_physics_estimates": {
                "estimated_power_w": self.estimated_power_w,
                "cumulative_energy_session_wh": self.cumulative_energy_session_wh,
                "cumulative_energy_lifetime_wh": self.cumulative_energy_lifetime_wh,
                "physics_estimate_source": self.physics_estimate_source,
                "disclaimer": "PROVISIONAL: Based on nominal research constants; not validated Atomberg measurements.",
            },
            "category_3_ml_advisory_forecast": {
                "predicted_next_energy_wh": self.predicted_next_energy_wh,
                "predicted_next_energy_source": self.predicted_next_energy_source,
                "predicted_next_energy_r2_benchmark": self.predicted_next_energy_r2_benchmark,
                "is_advisory": self.is_advisory,
                "cannot_command_speed": self.cannot_command_speed,
                "ml_cold_start": self.ml_cold_start,
                "disclaimer": "Advisory only (R^2 approx 0.1515). Variable-horizon next packet forecast. Cannot command speed.",
            },
            "category_4_hardware_calibrated": {
                "calibrated_measurement": self.calibrated_measurement,
                "is_hardware_validated": self.is_hardware_validated,
                "calibration_notes": self.calibration_notes,
            },
        }
