"""
AACI Integration – AACIPipeline End-to-End Orchestrator
=======================================================
Phase 5 – Stage 5: System Integration and Validation

AACIPipeline is the single end-to-end coordinator that:
  1. Validates raw telemetry via CentralizedSensorValidator.
  2. Constructs CommonState (CSR) from validated telemetry.
  3. Updates the Fan Digital Twin for the target fan.
  4. Invokes the AdaptiveController to generate a DecisionExplanation.
  5. Assembles a fully populated IntegrationResult with all 22 required fields.
  6. Maintains per-fan twin isolation via a shared TwinSupervisor.

FROZEN COMPONENT BOUNDARY
--------------------------
This module imports from, but does NOT modify:
    aaci_ml.system_architecture.adapters.*
    aaci_ml.system_architecture.common_state.*
    aaci_ml.system_architecture.digital_twin.*
    aaci_ml.system_architecture.controller.*

Incompatibility notes
---------------------
- CommonState.from_telemetry() does not accept fan_id/data_provenance kwargs;
  these must be embedded in the raw_telemetry dict before calling.
- AdaptiveController.decide() accepts CommonState or Dict[str,Any] as `csr`.
- TwinSupervisor.route_csr() returns List[StateTransitionRecord]; we do not need
  the records here (twin state is updated as a side-effect).
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aaci_ml.system_architecture.adapters.thermal_comfort_adapter import ThermalComfortAdapter
from aaci_ml.system_architecture.adapters.occupancy_adapter import OccupancyAdapter
from aaci_ml.system_architecture.adapters.energy_adapter import EnergyIntelligenceAdapter
from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.common_state.sensor_validator import CentralizedSensorValidator
from aaci_ml.system_architecture.controller.adaptive_controller import AdaptiveController
from aaci_ml.system_architecture.digital_twin.twin_supervisor import TwinSupervisor
from aaci_ml.system_architecture.integration.execution_record import (
    IntegrationResult,
    INTEGRATION_DISCLAIMERS,
)
from aaci_ml.system_architecture.integration.energy_reconciliation import (
    UNRESOLVED_INDUCTION_FLAG,
    INDUCTION_DISCREPANCY_NOTE,
)


class AACIPipeline:
    """
    End-to-end AACI software pipeline orchestrator.

    One instance per deployment context.  For multi-fan isolation, all fans
    managed by this pipeline share one TwinSupervisor instance (and therefore
    one isolated FanDigitalTwin per fan_id).

    Parameters
    ----------
    fan_id : str, optional
        Default fan ID applied when raw_telemetry does not carry one.
    alpha, beta, gamma, delta : float
        Controller scoring weights passed to AdaptiveController.
    history_capacity : int
        Maximum historical snapshots per twin (passed at twin creation).
    """

    def __init__(
        self,
        fan_id: str = "Default_Fan_1",
        alpha: float = 1.5,
        beta: float = 0.8,
        gamma: float = 0.5,
        delta: float = 1.0,
        history_capacity: int = 50,
    ):
        self.default_fan_id = fan_id
        self.history_capacity = history_capacity

        # Shared infrastructure
        self._twin_supervisor = TwinSupervisor()
        self._sensor_validator = CentralizedSensorValidator()

        # Shared adapters (re-used across calls; stateless except model weights)
        self._comfort_adapter = ThermalComfortAdapter()
        self._occupancy_adapter = OccupancyAdapter()
        self._energy_adapter = EnergyIntelligenceAdapter()

        # AdaptiveController receives shared supervisor and adapters
        self._controller = AdaptiveController(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            twin_supervisor=self._twin_supervisor,
            comfort_adapter=self._comfort_adapter,
            occupancy_adapter=self._occupancy_adapter,
            energy_adapter=self._energy_adapter,
        )

        # Execution history
        self._history: List[IntegrationResult] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def execute(
        self,
        raw_telemetry: Dict[str, Any],
        fan_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        provenance: str = "TELEMETRY",
    ) -> IntegrationResult:
        """
        Run one complete AACI pipeline cycle.

        Parameters
        ----------
        raw_telemetry : dict
            Raw sensor telemetry.  At minimum should contain temperature_c and
            relative_humidity_pct.  May include light_lux, co2_ppm, fan_speed,
            fan_mode, timestamp (ISO 8601), room_category, etc.
        fan_id : str, optional
            Overrides raw_telemetry['fan_id'] or the pipeline default.
        timestamp : float, optional
            Unix epoch seconds.  If None, current wall clock is used.
        provenance : str
            'SYNTHETIC_SIMULATION' for synthetic/demo inputs; 'TELEMETRY' otherwise.

        Returns
        -------
        IntegrationResult
            Fully populated 22-field result.
        """
        t_start = time.perf_counter()
        exec_id = str(uuid.uuid4())
        warnings: List[str] = []

        # ── 0. Resolve fan_id ────────────────────────────────────────────────
        resolved_fan_id = (
            fan_id
            or raw_telemetry.get("fan_id")
            or self.default_fan_id
        )

        # Embed fan_id/data_provenance into telemetry dict for CommonState
        telemetry = dict(raw_telemetry)
        telemetry["fan_id"] = resolved_fan_id
        telemetry["data_provenance"] = provenance
        if "room_category" not in telemetry:
            telemetry["room_category"] = "Unknown"

        # ── 1. Input validation ──────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            v_res = self._sensor_validator.validate_raw_telemetry(telemetry)
        except Exception as exc:
            warnings.append(f"Sensor validator raised exception: {exc}")
            v_res = None

        input_validation_status: Dict[str, Any] = {}
        if v_res is not None:
            input_validation_status = {
                "is_valid": v_res.is_valid,
                "errors": v_res.errors,
                "warnings": v_res.warnings,
                "missing_fields": v_res.missing_fields,
                "invalid_fields": v_res.invalid_fields,
                "stale_fields": v_res.stale_fields,
                "is_stale": v_res.is_stale,
                "sensor_age_s": v_res.sensor_age_s,
            }
        else:
            input_validation_status = {
                "is_valid": False,
                "errors": ["Validator exception; proceeding in degraded mode."],
                "warnings": warnings[:],
                "missing_fields": [],
                "invalid_fields": [],
                "stale_fields": [],
                "is_stale": False,
                "sensor_age_s": None,
            }
        t_validation_ms = (time.perf_counter() - t0) * 1000.0

        # ── 2. Build CSR ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            csr = CommonState.from_telemetry(raw_telemetry=telemetry)
        except Exception as exc:
            warnings.append(f"CommonState.from_telemetry raised exception: {exc}")
            csr = CommonState()  # empty fallback

        t_csr_ms = (time.perf_counter() - t0) * 1000.0

        # Snapshot CSR fields for the result
        csr_summary = self._summarise_csr(csr)

        # ── 3. Update Digital Twin ───────────────────────────────────────────
        t0 = time.perf_counter()
        twin = self._twin_supervisor.get_or_create_twin(resolved_fan_id)
        try:
            self._twin_supervisor.route_csr(
                csr=csr,
                fan_id=resolved_fan_id,
                data_provenance=provenance,
            )
        except Exception as exc:
            warnings.append(f"TwinSupervisor.route_csr raised exception: {exc}")

        twin_snapshot = {}
        try:
            twin_snapshot = twin.inspect_state()
        except Exception as exc:
            warnings.append(f"twin.inspect_state() raised exception: {exc}")

        t_twin_ms = (time.perf_counter() - t0) * 1000.0

        # Previous speed (read from twin state before decision)
        previous_speed = None
        try:
            previous_speed = twin_snapshot.get("fan", {}).get("current_speed")
            if previous_speed is None:
                previous_speed = twin_snapshot.get("current_speed")
        except Exception:
            pass

        # ── 4. AdaptiveController decision ───────────────────────────────────
        t0 = time.perf_counter()
        decision = None
        try:
            decision = self._controller.decide(
                csr=csr,
                fan_id=resolved_fan_id,
                timestamp=timestamp,
                apply_rate_limiting=True,
            )
        except Exception as exc:
            warnings.append(f"AdaptiveController.decide raised exception: {exc}")

        t_controller_ms = (time.perf_counter() - t0) * 1000.0

        # ── 5. Direct adapter invocations for model result fields ─────────────
        t0 = time.perf_counter()
        model1_result = self._invoke_comfort_adapter(telemetry, warnings)
        model2_result = self._invoke_occupancy_adapter(telemetry, warnings)
        model3_result = self._invoke_energy_adapter(telemetry, warnings)
        t_adapter_ms = (time.perf_counter() - t0) * 1000.0

        # ── 6. Assemble IntegrationResult ─────────────────────────────────────
        total_ms = (time.perf_counter() - t_start) * 1000.0

        result = IntegrationResult(
            execution_id=exec_id,
            fan_id=resolved_fan_id,
            room_id=csr.occupancy.robod_room_id if hasattr(csr, "occupancy") else None,
            room_category=(
                csr.metadata.get("room_category", "Unknown")
                if hasattr(csr, "metadata") else "Unknown"
            ),
            input_timestamp=(
                telemetry.get("timestamp")
                or datetime.now(timezone.utc).isoformat()
            ),
            input_validation_status=input_validation_status,
            model1_result=model1_result,
            model2_result=model2_result,
            model3_result=model3_result,
            csr_state_summary=csr_summary,
            digital_twin_state_summary=twin_snapshot,
            previous_speed=previous_speed,
            recommended_speed=decision.recommended_speed if decision else None,
            comfort_assessment=decision.comfort_assessment if decision else {},
            occupancy_assessment=decision.occupancy_assessment if decision else {},
            energy_assessment=decision.energy_assessment if decision else {},
            candidate_scores=decision.candidate_evaluations if decision else [],
            safety_status=decision.safety_status if decision else {},
            manual_override_status=decision.manual_override_status if decision else None,
            fallback_status=decision.fallback_status if decision else {"fallback_triggered": True, "fallback_reason": "Controller unavailable"},
            decision_explanation=decision.to_dict() if decision else {},
            data_freshness={
                "sensor_age_s": input_validation_status.get("sensor_age_s"),
                "is_stale": input_validation_status.get("is_stale", False),
                "stale_fields": input_validation_status.get("stale_fields", []),
                "staleness_threshold_s": 300.0,
            },
            warnings=warnings,
            unresolved_limitations=list(INTEGRATION_DISCLAIMERS) + [INDUCTION_DISCREPANCY_NOTE],
            latency_ms={
                "total_ms": round(total_ms, 2),
                "validation_ms": round(t_validation_ms, 2),
                "csr_ms": round(t_csr_ms, 2),
                "twin_ms": round(t_twin_ms, 2),
                "controller_ms": round(t_controller_ms, 2),
                "adapter_ms": round(t_adapter_ms, 2),
            },
            provenance=provenance,
        )

        self._history.append(result)
        return result

    def set_manual_override(
        self,
        fan_id: str,
        speed: int,
        duration_s: float = 600.0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Delegate manual override to AdaptiveController."""
        return self._controller.set_manual_override(
            fan_id=fan_id, speed=speed, duration_s=duration_s, timestamp=timestamp
        )

    def clear_manual_override(self, fan_id: str) -> bool:
        """Delegate manual override clear to AdaptiveController."""
        return self._controller.clear_manual_override(fan_id)

    def get_execution_history(self) -> List[IntegrationResult]:
        """Returns list of all IntegrationResult objects from this pipeline instance."""
        return list(self._history)

    def reset(self) -> None:
        """Resets digital twin fleet and execution history. Does NOT reset adapters."""
        self._twin_supervisor.clear_all()
        self._history.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _summarise_csr(self, csr: CommonState) -> Dict[str, Any]:
        """Extracts a compact summary of the CommonState for IntegrationResult."""
        try:
            env = csr.environmental
            occ = csr.occupancy
            fan = csr.fan_state
            energy = csr.energy
            health = csr.health
            meta = csr.metadata
        except AttributeError:
            return {"error": "CommonState fields missing; fallback CSR used."}

        return {
            "environmental": {
                "air_temp_c": getattr(env, "air_temp_c", None),
                "relative_humidity_pct": getattr(env, "relative_humidity_pct", None),
                "air_velocity_ms": getattr(env, "air_velocity_ms", None),
                "air_velocity_source": getattr(env, "air_velocity_source", "none"),
                "outdoor_monthly_temp_c": getattr(env, "outdoor_monthly_temp_c", None),
                "is_valid": getattr(env, "is_valid", False),
                "is_stale": getattr(env, "is_stale", False),
                "sensor_age_s": getattr(env, "sensor_age_s", None),
            },
            "occupancy": {
                "occupancy_classification": getattr(occ, "occupancy_classification", "Unknown"),
                "occupancy_probability": getattr(occ, "occupancy_probability", None),
                "occupancy_binary": getattr(occ, "occupancy_binary", None),
                "estimated_headcount": getattr(occ, "estimated_headcount", None),
                "robod_room_id": getattr(occ, "robod_room_id", None),
                "robod_room_type": getattr(occ, "robod_room_type", None),
                "model_available": getattr(occ, "model_available", False),
                "degraded_status": getattr(occ, "degraded_status", True),
            },
            "fan_state": {
                "fan_id": getattr(fan, "fan_id", "Unknown"),
                "current_speed": getattr(fan, "current_speed", None),
                "mode": getattr(fan, "mode", "BLDC"),
            },
            "energy": {
                "estimated_power_w": getattr(energy, "estimated_power_w", None),
                "bldc_nominal_power_w": getattr(energy, "bldc_nominal_power_w", None),
                "induction_nominal_power_w": getattr(energy, "induction_nominal_power_w", None),
                "hardware_calibrated_power_w": getattr(energy, "hardware_calibrated_power_w", None),
            },
            "health": {
                "overall_state_valid": getattr(health, "overall_state_valid", False),
                "thermal_adapter_available": getattr(health, "thermal_adapter_available", False),
                "occupancy_adapter_available": getattr(health, "occupancy_adapter_available", False),
                "energy_adapter_available": getattr(health, "energy_adapter_available", False),
                "is_degraded": getattr(health, "is_degraded", True),
            },
            "metadata": dict(meta) if isinstance(meta, dict) else {},
        }

    def _invoke_comfort_adapter(
        self, telemetry: Dict[str, Any], warnings: List[str]
    ) -> Dict[str, Any]:
        """Invokes ThermalComfortAdapter and returns a summary dict."""
        try:
            env_input = {
                "temperature_c": telemetry.get("temperature_c"),
                "relative_humidity_pct": telemetry.get("relative_humidity_pct"),
                "air_velocity_ms": telemetry.get("air_velocity_ms"),
                "outdoor_monthly_temp_c": telemetry.get("outdoor_monthly_temp_c"),
                "fan_speed": telemetry.get("fan_speed"),
                "clo": telemetry.get("clo"),
                "met": telemetry.get("met"),
                "building_type": telemetry.get("building_type"),
                "cooling_strategy": telemetry.get("cooling_strategy"),
            }
            envelope = self._comfort_adapter.predict(env_input)
            payload = getattr(envelope, "normalized_payload", {}) or {}
            avail = getattr(envelope, "model_availability", None)
            return {
                "comfort_label": payload.get("comfort_label"),
                "comfort_probabilities": payload.get("comfort_probabilities"),
                "is_uncertain": payload.get("is_uncertain"),
                "air_velocity_used_ms": payload.get("air_velocity_used_ms"),
                "provisional_mapping_flag": payload.get("provisional_dryad_mapping", True),
                "model_available": getattr(avail, "is_available", True) if avail else True,
                "disclaimers": payload.get("disclaimers", []),
            }
        except Exception as exc:
            warnings.append(f"ThermalComfortAdapter.predict failed: {exc}")
            return {"model_available": False, "error": str(exc)}

    def _invoke_occupancy_adapter(
        self, telemetry: Dict[str, Any], warnings: List[str]
    ) -> Dict[str, Any]:
        """Invokes OccupancyAdapter and returns a summary dict."""
        try:
            occ_input = {
                "temperature_c": telemetry.get("temperature_c"),
                "relative_humidity_pct": telemetry.get("relative_humidity_pct"),
                "light_lux": telemetry.get("light_lux"),
                "co2_ppm": telemetry.get("co2_ppm"),
                "fan_id": telemetry.get("fan_id", self.default_fan_id),
            }
            envelope = self._occupancy_adapter.predict(occ_input)
            payload = getattr(envelope, "normalized_payload", {}) or {}
            avail = getattr(envelope, "model_availability", None)
            return {
                "occupancy_state": payload.get("occupancy_state"),
                "occupancy_binary": payload.get("occupancy_binary"),
                "occupancy_probability": payload.get("occupancy_probability"),
                "headcount_estimate": payload.get("headcount_estimate"),
                "headcount_rounded": payload.get("headcount_rounded"),
                "density_category": payload.get("density_category"),
                "sensor_gap_active": payload.get("sensor_gap_active", True),
                "is_uncertain": payload.get("is_uncertain", True),
                "feature_set_used": payload.get("feature_set_used"),
                "model_available": getattr(avail, "is_available", True) if avail else True,
            }
        except Exception as exc:
            warnings.append(f"OccupancyAdapter.predict failed: {exc}")
            return {"model_available": False, "error": str(exc)}

    def _invoke_energy_adapter(
        self, telemetry: Dict[str, Any], warnings: List[str]
    ) -> Dict[str, Any]:
        """Invokes EnergyIntelligenceAdapter and returns a summary dict."""
        try:
            energy_input = {
                "temperature_c": telemetry.get("temperature_c"),
                "relative_humidity_pct": telemetry.get("relative_humidity_pct"),
                "fan_speed": telemetry.get("fan_speed"),
                "fan_mode": telemetry.get("fan_mode", "BLDC"),
                "light_lux": telemetry.get("light_lux"),
                "co2_ppm": telemetry.get("co2_ppm"),
                "timestamp": telemetry.get("timestamp"),
            }
            envelope = self._energy_adapter.predict(energy_input)
            payload = getattr(envelope, "normalized_payload", {}) or {}
            avail = getattr(envelope, "model_availability", None)
            ml_fc = payload.get("ml_forecast", {})
            return {
                "candidate_evaluation_table": payload.get("candidate_evaluation_table", []),
                "ml_forecast": {
                    "predicted_next_energy_wh": ml_fc.get("predicted_next_energy_wh"),
                    "model_used": ml_fc.get("model_used"),
                    "is_advisory": ml_fc.get("is_advisory", True),
                    "cannot_command_speed": ml_fc.get("cannot_command_speed", True),
                },
                "model_available": getattr(avail, "is_available", True) if avail else True,
                "unresolved_induction_baseline": UNRESOLVED_INDUCTION_FLAG,
            }
        except Exception as exc:
            warnings.append(f"EnergyIntelligenceAdapter.predict failed: {exc}")
            return {
                "model_available": False,
                "error": str(exc),
                "unresolved_induction_baseline": UNRESOLVED_INDUCTION_FLAG,
            }

