"""
AACI Common State Representation (CSR) – Unified System State Object
===================================================================
Phase 5 – Stage 2: Common State Representation (CSR) Implementation

Represents the complete, validated system state across environmental readings,
occupancy detection, fan operating states, energy accounting, and system health.
Enforces strict value provenance (measured, assumed, estimated, predicted),
distinguishes research physics constants from real hardware measurements,
and preserves ROBOD training context without conflation with residential categories.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
import datetime

from .constants import (
    DEFAULT_STALENESS_THRESHOLD_SECONDS,
    ROBOD_ROOM_TYPES,
    RESEARCH_BLDC_POWER_MAP,
    RESEARCH_INDUCTION_POWER_MAP,
)
from .sensor_validator import CentralizedSensorValidator, ValidationResult


@dataclass
class EnvironmentalState:
    """Room environmental state."""
    air_temp_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    air_velocity_ms: Optional[float] = None
    air_velocity_source: str = "none"  # "measured_direct", "provisional_dryad_estimate", "none"
    outdoor_monthly_temp_c: Optional[float] = None
    measurement_timestamp: Optional[str] = None
    is_valid: bool = False
    is_stale: bool = False
    sensor_age_s: Optional[float] = None
    provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class OccupancyState:
    """Room occupancy intelligence state."""
    occupancy_classification: str = "Unknown"  # "Occupied", "Unoccupied", "Unknown"
    occupancy_binary: Optional[int] = None      # 1, 0, or None
    occupancy_probability: Optional[float] = None  # Calibrated probability
    estimated_headcount: Optional[float] = None
    headcount_rounded: Optional[int] = None
    density_category: Optional[str] = None
    uncertainty_limitations: str = (
        "Module 2 headcount interval uses a fixed empirical margin of +/- 1.5 persons; "
        "it is NOT a statistically derived confidence interval."
    )
    source_model_id: Optional[str] = None
    feature_set_used: Optional[str] = None
    light_available: bool = False
    co2_available: bool = False
    sensor_gap_active: bool = True
    is_uncertain: bool = True
    prediction_timestamp: Optional[str] = None
    degraded_status: bool = True
    model_available: bool = False
    robod_room_id: Optional[int] = None
    robod_room_type: Optional[str] = None
    robod_training_context_note: str = (
        "ROBOD room types (Lecture Room 1-2, Office Space 3-4, Library Space 5) "
        "represent the academic/commercial training dataset and must NOT be equated "
        "with residential Bedroom, Drawingroom, or Lounge spaces."
    )
    provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class FanOperatingState:
    """Fan operational telemetry state."""
    fan_id: str = "Unknown_Fan"
    current_speed: Optional[int] = None
    mode: str = "BLDC"
    op_time_s: Optional[float] = None
    last_observed_timestamp: Optional[str] = None
    telemetry_validity: bool = False
    data_freshness: bool = False
    sensor_age_s: Optional[float] = None
    provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class EnergyState:
    """
    Fan energy state.
    Strictly differentiates between measured telemetry, nominal research calculations,
    advisory ML forecasts, and future hardware calibrations.
    """
    # 1. Measured telemetry energy from IoT packet
    measured_espent_wh: Optional[float] = None
    measured_espent_source: str = "firmware_telemetry_packet"

    # 2. Deterministic research-dataset physics estimates
    physics_candidate_table: List[Dict[str, Any]] = field(default_factory=list)
    physics_power_map_source: str = (
        "nominal_research_dataset_constants (NOT validated Atomberg hardware measurement)"
    )

    # 3. Machine learning predicted next-packet energy (advisory)
    ml_predicted_next_energy_wh: Optional[float] = None
    ml_model_used: Optional[str] = None
    ml_forecast_horizon: str = "next_observed_packet_variable"
    ml_forecast_is_advisory: bool = True
    ml_forecast_cannot_command_speed: bool = True
    ml_forecast_r2_benchmark: float = 0.1515

    # 4. Future hardware-calibrated energy values
    hardware_calibrated_energy_wh: Optional[float] = None
    hardware_calibration_source: Optional[str] = None

    provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemHealthState:
    """System diagnostic, validity, and error state."""
    overall_state_valid: bool = False
    thermal_adapter_available: bool = False
    occupancy_adapter_available: bool = False
    energy_adapter_available: bool = False
    missing_sensors: List[str] = field(default_factory=list)
    invalid_inputs: List[str] = field(default_factory=list)
    stale_sensors: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    is_degraded: bool = False
    degraded_reason: Optional[str] = None


@dataclass
class CommonState:
    """
    The unified Common State Representation (CSR) for AACI.
    Encapsulates all verified sensor, operational, intelligence, and health signals.
    """
    environmental: EnvironmentalState = field(default_factory=EnvironmentalState)
    occupancy: OccupancyState = field(default_factory=OccupancyState)
    fan_state: FanOperatingState = field(default_factory=FanOperatingState)
    energy: EnergyState = field(default_factory=EnergyState)
    health: SystemHealthState = field(default_factory=SystemHealthState)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts CommonState to a structured dictionary compatible with
        Stage 1 adapter consumption and JSON serialization.
        """
        # Build backward-compatible structure for Stage 1 adapters
        return {
            "timestamp": self.environmental.measurement_timestamp or self.fan_state.last_observed_timestamp,
            "environmental": {
                "temperature_c": self.environmental.air_temp_c,
                "relative_humidity_pct": self.environmental.relative_humidity_pct,
                "air_velocity_ms": self.environmental.air_velocity_ms,
                "outdoor_monthly_temp_c": self.environmental.outdoor_monthly_temp_c,
                "air_velocity_source": self.environmental.air_velocity_source,
                "is_valid": self.environmental.is_valid,
                "is_stale": self.environmental.is_stale,
            },
            "occupancy_signals": {
                "light_lux": self.metadata.get("light_lux"),
                "co2_ppm": self.metadata.get("co2_ppm"),
                "light_available": self.occupancy.light_available,
                "co2_available": self.occupancy.co2_available,
                "sensor_gap_active": self.occupancy.sensor_gap_active,
                "occupancy_classification": self.occupancy.occupancy_classification,
                "occupancy_probability": self.occupancy.occupancy_probability,
            },
            "fan_state": {
                "fan_id": self.fan_state.fan_id,
                "current_speed": self.fan_state.current_speed,
                "mode": self.fan_state.mode,
                "op_time_s": self.fan_state.op_time_s,
                "espent_wh": self.energy.measured_espent_wh,
                "telemetry_validity": self.fan_state.telemetry_validity,
                "data_freshness": self.fan_state.data_freshness,
            },
            "energy": {
                "measured_espent_wh": self.energy.measured_espent_wh,
                "physics_candidate_table": self.energy.physics_candidate_table,
                "ml_predicted_next_energy_wh": self.energy.ml_predicted_next_energy_wh,
                "ml_forecast_is_advisory": self.energy.ml_forecast_is_advisory,
                "hardware_calibrated_energy_wh": self.energy.hardware_calibrated_energy_wh,
            },
            "metadata": {
                "fan_id": self.fan_state.fan_id,
                "room_id": self.fan_state.fan_id,
                "room_category": self.metadata.get("room_category", "Unknown"),
                "sensor_age_seconds": self.environmental.sensor_age_s or self.fan_state.sensor_age_s,
                "sensor_age_s": self.environmental.sensor_age_s or self.fan_state.sensor_age_s,
                "data_source": self.metadata.get("data_source", "aaci_common_state"),
            },
            "system_health": asdict(self.health),
        }

    def __getitem__(self, key: str) -> Any:
        """Enables dict-like subscript access for Stage 1 adapter interoperability."""
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(f"Key '{key}' not found in CommonState representation.")

    def __contains__(self, key: str) -> bool:
        """Enables 'in' operator checks for Stage 1 adapter interoperability."""
        return key in self.to_dict()

    def keys(self):
        """Enables dict-like .keys() iteration."""
        return self.to_dict().keys()

    def items(self):
        """Enables dict-like .items() iteration."""
        return self.to_dict().items()

    def get(self, key: str, default: Any = None) -> Any:
        """Enables dict-like .get() access for Stage 1 adapter interoperability."""
        d = self.to_dict()
        return d.get(key, default)

    @staticmethod
    def _extract_first_not_none(*sources: Any) -> Any:
        """Extracts first non-None value from sources, preserving zero and false."""
        for s in sources:
            if s is not None:
                return s
        return None

    @classmethod
    def from_telemetry(
        cls,
        raw_telemetry: Dict[str, Any],
        validator: Optional[CentralizedSensorValidator] = None,
    ) -> "CommonState":
        """
        Primary factory constructor: ingests raw telemetry, applies centralized validation,
        and constructs a validated, provenance-tracked CommonState object.
        """
        val_engine = validator or CentralizedSensorValidator()
        v_res = val_engine.validate_raw_telemetry(raw_telemetry)

        # Unpack telemetry
        env_raw = raw_telemetry.get("environmental", {}) if isinstance(raw_telemetry.get("environmental"), dict) else raw_telemetry
        fan_raw = raw_telemetry.get("fan_state", {}) if isinstance(raw_telemetry.get("fan_state"), dict) else raw_telemetry
        meta_raw = raw_telemetry.get("metadata", {}) if isinstance(raw_telemetry.get("metadata"), dict) else raw_telemetry
        occ_raw = raw_telemetry.get("occupancy_signals", {}) if isinstance(raw_telemetry.get("occupancy_signals"), dict) else raw_telemetry

        fan_id_val = cls._extract_first_not_none(
            meta_raw.get("fan_id"), meta_raw.get("room_id"),
            raw_telemetry.get("fan_id"), raw_telemetry.get("room_id"), "Unknown_Fan"
        )
        fan_id = str(fan_id_val).strip()

        # Parse timestamp
        parsed_dt, iso_ts = val_engine.parse_timestamp(
            cls._extract_first_not_none(
                meta_raw.get("timestamp"), raw_telemetry.get("timestamp"),
                raw_telemetry.get("datetime"), raw_telemetry.get("Date")
            )
        )

        # 1. Environmental State
        raw_temp = cls._extract_first_not_none(
            env_raw.get("temperature_c"), env_raw.get("air_temp_c"),
            raw_telemetry.get("temperature"), raw_telemetry.get("temperature_c"), raw_telemetry.get("air_temp_c")
        )
        raw_rh = cls._extract_first_not_none(
            env_raw.get("relative_humidity_pct"), env_raw.get("relative_humidity"),
            raw_telemetry.get("humidity"), raw_telemetry.get("relative_humidity_pct"), raw_telemetry.get("relative_humidity")
        )
        raw_vel = cls._extract_first_not_none(
            env_raw.get("air_velocity_ms"), raw_telemetry.get("air_velocity_ms")
        )
        raw_outdoor = cls._extract_first_not_none(
            env_raw.get("outdoor_monthly_temp_c"), raw_telemetry.get("outdoor_monthly_temp_c")
        )

        env_prov = {}
        temp_val = None
        if "temperature" not in v_res.invalid_fields and "temperature" not in v_res.missing_fields and raw_temp is not None:
            try:
                temp_val = float(raw_temp)
                env_prov["air_temp_c"] = "measured"
            except (ValueError, TypeError):
                temp_val = None

        rh_val = None
        if "relative_humidity" not in v_res.invalid_fields and "relative_humidity" not in v_res.missing_fields and raw_rh is not None:
            try:
                rh_val = float(raw_rh)
                env_prov["relative_humidity_pct"] = "measured"
            except (ValueError, TypeError):
                rh_val = None

        vel_val = None
        vel_source = "none"
        if "air_velocity_ms" not in v_res.invalid_fields and raw_vel is not None:
            try:
                vel_val = float(raw_vel)
                vel_source = "measured_direct"
                env_prov["air_velocity_ms"] = "measured"
            except (ValueError, TypeError):
                vel_val = None

        outdoor_val = None
        if raw_outdoor is not None:
            try:
                outdoor_val = float(raw_outdoor)
                env_prov["outdoor_monthly_temp_c"] = "measured"
            except (ValueError, TypeError):
                outdoor_val = None

        env_state = EnvironmentalState(
            air_temp_c=temp_val,
            relative_humidity_pct=rh_val,
            air_velocity_ms=vel_val,
            air_velocity_source=vel_source,
            outdoor_monthly_temp_c=outdoor_val,
            measurement_timestamp=iso_ts,
            is_valid=("temperature" not in v_res.invalid_fields and "relative_humidity" not in v_res.invalid_fields and "environmental" not in v_res.invalid_fields),
            is_stale=v_res.is_stale,
            sensor_age_s=v_res.sensor_age_s,
            provenance=env_prov,
        )

        # 2. Fan Operating State
        raw_speed = cls._extract_first_not_none(
            fan_raw.get("current_speed"), raw_telemetry.get("current_speed"),
            raw_telemetry.get("fan_speed"), raw_telemetry.get("speed")
        )
        raw_mode = cls._extract_first_not_none(
            fan_raw.get("mode"), raw_telemetry.get("fan_mode"),
            raw_telemetry.get("mode"), "BLDC"
        )
        raw_op_time = cls._extract_first_not_none(
            fan_raw.get("op_time_s"), raw_telemetry.get("op_time_s"),
            raw_telemetry.get("opTime")
        )

        fan_prov = {"fan_id": "measured"}
        speed_val = None
        if "current_speed" not in v_res.invalid_fields and raw_speed is not None:
            try:
                speed_val = int(raw_speed)
                fan_prov["current_speed"] = "measured"
            except (ValueError, TypeError):
                speed_val = None

        mode_val = "Induction" if str(raw_mode).strip().lower() == "induction" else "BLDC"
        fan_prov["mode"] = "measured" if ("mode" in raw_telemetry or "mode" in fan_raw) else "assumed"

        op_time_val = None
        if "op_time_s" not in v_res.invalid_fields and raw_op_time is not None:
            try:
                op_time_val = float(raw_op_time)
                fan_prov["op_time_s"] = "measured"
            except (ValueError, TypeError):
                op_time_val = None

        fan_state = FanOperatingState(
            fan_id=fan_id,
            current_speed=speed_val,
            mode=mode_val,
            op_time_s=op_time_val,
            last_observed_timestamp=iso_ts,
            telemetry_validity=v_res.is_valid,
            data_freshness=(not v_res.is_stale),
            sensor_age_s=v_res.sensor_age_s,
            provenance=fan_prov,
        )

        # 3. Energy State
        raw_espent = cls._extract_first_not_none(
            fan_raw.get("espent_wh"), raw_telemetry.get("espent_wh"),
            raw_telemetry.get("eSpent")
        )
        energy_prov = {}
        espent_val = None
        if "espent_wh" not in v_res.invalid_fields and raw_espent is not None:
            try:
                espent_val = float(raw_espent)
                energy_prov["measured_espent_wh"] = "measured"
            except (ValueError, TypeError):
                espent_val = None

        # Build nominal research physics candidate table if op_time is valid
        candidate_table = []
        power_map = RESEARCH_INDUCTION_POWER_MAP if mode_val == "Induction" else RESEARCH_BLDC_POWER_MAP
        dt_duration = op_time_val if (op_time_val is not None and op_time_val >= 0) else 14.4

        for s in range(6):
            p_w = power_map[s]
            e_wh = round(p_w * dt_duration / 3600.0, 4)
            candidate_table.append({
                "speed": s,
                "power_w": p_w,
                "energy_wh_physics": e_wh,
                "source": "nominal_research_dataset_constants",
            })
        energy_prov["physics_candidate_table"] = "estimated"

        energy_state = EnergyState(
            measured_espent_wh=espent_val,
            measured_espent_source="firmware_telemetry_packet" if espent_val is not None else "none",
            physics_candidate_table=candidate_table,
            physics_power_map_source="nominal_research_dataset_constants (NOT validated Atomberg hardware measurement)",
            ml_predicted_next_energy_wh=None,  # Populated via energy adapter envelope
            ml_model_used=None,
            ml_forecast_horizon="next_observed_packet_variable",
            ml_forecast_is_advisory=True,
            ml_forecast_cannot_command_speed=True,
            ml_forecast_r2_benchmark=0.1515,
            hardware_calibrated_energy_wh=None,
            hardware_calibration_source=None,
            provenance=energy_prov,
        )

        # 4. Occupancy State (Initial raw state prior to adapter enrichment)
        raw_light = occ_raw.get("light_lux", raw_telemetry.get("light_lux", raw_telemetry.get("Light")))
        raw_co2 = occ_raw.get("co2_ppm", raw_telemetry.get("co2_ppm", raw_telemetry.get("CO2")))

        light_avail = ("light_lux" not in v_res.invalid_fields and raw_light is not None)
        co2_avail = ("co2_ppm" not in v_res.invalid_fields and raw_co2 is not None)
        sensor_gap = not (light_avail and co2_avail)

        occ_state = OccupancyState(
            occupancy_classification="Unknown",
            occupancy_binary=None,
            occupancy_probability=None,
            estimated_headcount=None,
            headcount_rounded=None,
            density_category=None,
            source_model_id=None,
            feature_set_used="set_b" if sensor_gap else "set_a",
            light_available=light_avail,
            co2_available=co2_avail,
            sensor_gap_active=sensor_gap,
            is_uncertain=True,
            prediction_timestamp=None,
            degraded_status=sensor_gap,
            model_available=False,
            robod_room_id=None,
            robod_room_type=None,
            provenance={"light_available": "measured", "co2_available": "measured"},
        )

        # 5. System Health
        health_state = SystemHealthState(
            overall_state_valid=v_res.is_valid,
            thermal_adapter_available=False,
            occupancy_adapter_available=False,
            energy_adapter_available=False,
            missing_sensors=v_res.missing_fields,
            invalid_inputs=v_res.invalid_fields,
            stale_sensors=v_res.stale_fields,
            validation_errors=v_res.errors,
            validation_warnings=v_res.warnings,
            is_degraded=(sensor_gap or v_res.is_stale or not v_res.is_valid),
            degraded_reason="; ".join(v_res.errors + v_res.warnings) if (v_res.errors or v_res.warnings) else None,
        )

        # Metadata
        meta_dict = {
            "room_category": meta_raw.get("room_category", raw_telemetry.get("room_category", "Unknown")),
            "data_source": meta_raw.get("data_source", raw_telemetry.get("data_source", "mendeley_telemetry")),
            "light_lux": raw_light if light_avail else None,
            "co2_ppm": raw_co2 if co2_avail else None,
            "validation_timestamp": datetime.datetime.now().isoformat(),
        }

        return cls(
            environmental=env_state,
            occupancy=occ_state,
            fan_state=fan_state,
            energy=energy_state,
            health=health_state,
            metadata=meta_dict,
        )

    def update_from_adapter_envelope(self, envelope: Any) -> None:
        """
        Ingests an AdapterOutputEnvelope from any Stage 1 adapter, updates corresponding
        state components, attaches verified provenance, and updates system health diagnostics.
        """
        # Read either dataclass fields or dict keys
        adapter_name = getattr(envelope, "adapter_name", None) or (envelope.get("adapter_name") if isinstance(envelope, dict) else None)
        norm_payload = getattr(envelope, "normalized_payload", None) or (envelope.get("normalized_payload", {}) if isinstance(envelope, dict) else {})
        model_avail = getattr(envelope, "model_availability", None)
        is_model_avail = getattr(model_avail, "is_available", True) if model_avail else True

        if adapter_name == "ThermalComfortAdapter":
            self.health.thermal_adapter_available = is_model_avail
            # Update provisional airflow velocity if estimated by comfort adapter
            vel = norm_payload.get("air_velocity_used_ms")
            vel_mapping = norm_payload.get("air_velocity_mapping", "none")
            if vel is not None and self.environmental.air_velocity_ms is None:
                self.environmental.air_velocity_ms = vel
                self.environmental.air_velocity_source = vel_mapping
                self.environmental.provenance["air_velocity_ms"] = "estimated"
            self.metadata["comfort_label"] = norm_payload.get("comfort_label")
            self.metadata["comfort_probabilities"] = norm_payload.get("comfort_probabilities")
            self.metadata["comfort_is_uncertain"] = norm_payload.get("is_uncertain")

        elif adapter_name == "OccupancyAdapter":
            self.health.occupancy_adapter_available = is_model_avail
            self.occupancy.occupancy_classification = norm_payload.get("occupancy_state", "Unknown")
            self.occupancy.occupancy_binary = norm_payload.get("occupancy_binary")
            self.occupancy.occupancy_probability = norm_payload.get("occupancy_probability")
            self.occupancy.estimated_headcount = norm_payload.get("headcount_estimate")
            self.occupancy.headcount_rounded = norm_payload.get("headcount_rounded")
            self.occupancy.density_category = norm_payload.get("density_category")
            self.occupancy.source_model_id = getattr(envelope, "model_id", "AACI_Model_2_Occupancy")
            self.occupancy.feature_set_used = norm_payload.get("feature_set_used")
            self.occupancy.sensor_gap_active = norm_payload.get("sensor_gap_active", True)
            self.occupancy.is_uncertain = norm_payload.get("is_uncertain", True)
            self.occupancy.prediction_timestamp = getattr(envelope, "timestamp", None)
            self.occupancy.model_available = is_model_avail
            self.occupancy.degraded_status = not is_model_avail or self.occupancy.sensor_gap_active

            # Resolve ROBOD room context
            raw_mod = getattr(envelope, "raw_model_output", {}) or {}
            mod2_raw = raw_mod.get("module2", {}) if isinstance(raw_mod, dict) else {}
            room_ctx = mod2_raw.get("room_context", {}) if isinstance(mod2_raw, dict) else {}
            if room_ctx and "room_id" in room_ctx:
                rid = int(room_ctx["room_id"])
                self.occupancy.robod_room_id = rid
                self.occupancy.robod_room_type = ROBOD_ROOM_TYPES.get(rid, "Generic Room")

            self.occupancy.provenance["occupancy_binary"] = "predicted"
            self.occupancy.provenance["occupancy_probability"] = "predicted"
            if self.occupancy.estimated_headcount is not None:
                self.occupancy.provenance["estimated_headcount"] = "predicted"

        elif adapter_name == "EnergyIntelligenceAdapter":
            self.health.energy_adapter_available = is_model_avail
            cand_table = norm_payload.get("candidate_evaluation_table", [])
            if cand_table:
                self.energy.physics_candidate_table = cand_table
                self.energy.provenance["physics_candidate_table"] = "estimated"

            ml_fc = norm_payload.get("ml_forecast", {})
            self.energy.ml_predicted_next_energy_wh = ml_fc.get("predicted_next_energy_wh")
            self.energy.ml_model_used = ml_fc.get("model_used")
            self.energy.ml_forecast_is_advisory = ml_fc.get("is_advisory", True)
            self.energy.ml_forecast_cannot_command_speed = ml_fc.get("cannot_command_speed", True)
            self.energy.provenance["ml_predicted_next_energy_wh"] = "predicted"
