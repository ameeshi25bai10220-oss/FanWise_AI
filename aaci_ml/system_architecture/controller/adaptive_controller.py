"""
AACI Adaptive Controller – Main Orchestration Runtime
=====================================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Coordinates multi-objective fan speed selection:
- Integrates Model 1 (Comfort), Model 2 (Occupancy), and Model 3 (Energy).
- Enforces strict safety priorities (P1-P5) and fallback handling.
- Manages user manual overrides without physical fan actuation.
- Updates and reads from Fan Digital Twin virtual runtime.
- Produces fully explainable DecisionExplanation outputs.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from aaci_ml.system_architecture.adapters.energy_adapter import EnergyIntelligenceAdapter
from aaci_ml.system_architecture.adapters.occupancy_adapter import OccupancyAdapter
from aaci_ml.system_architecture.adapters.thermal_comfort_adapter import ThermalComfortAdapter
from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.controller.comfort_scorer import ComfortScorer
from aaci_ml.system_architecture.controller.constants import (
    CANDIDATE_SPEEDS,
    DECISION_MODE_ADAPTIVE,
    DECISION_MODE_OCCUPANCY_GATE,
    DECISION_MODE_THERMAL_SAFETY,
    DEFAULT_ALPHA_COMFORT_WEIGHT,
    DEFAULT_BETA_ENERGY_WEIGHT,
    DEFAULT_DELTA_OCCUPANCY_WEIGHT,
    DEFAULT_GAMMA_STABILITY_WEIGHT,
    MANDATORY_DISCLAIMERS,
)
from aaci_ml.system_architecture.controller.decision_engine import DecisionEngine
from aaci_ml.system_architecture.controller.decision_explanation import DecisionExplanation
from aaci_ml.system_architecture.controller.energy_scorer import EnergyScorer
from aaci_ml.system_architecture.controller.occupancy_scorer import OccupancyScorer
from aaci_ml.system_architecture.controller.preference_manager import PreferenceManager
from aaci_ml.system_architecture.controller.safety_guard import SafetyGuard
from aaci_ml.system_architecture.digital_twin.fan_digital_twin import FanDigitalTwin
from aaci_ml.system_architecture.digital_twin.twin_supervisor import TwinSupervisor


class AdaptiveController:
    """Main orchestration controller for adaptive ceiling fan recommendations."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA_COMFORT_WEIGHT,
        beta: float = DEFAULT_BETA_ENERGY_WEIGHT,
        gamma: float = DEFAULT_GAMMA_STABILITY_WEIGHT,
        delta: float = DEFAULT_DELTA_OCCUPANCY_WEIGHT,
        twin_supervisor: Optional[TwinSupervisor] = None,
        comfort_adapter: Optional[ThermalComfortAdapter] = None,
        occupancy_adapter: Optional[OccupancyAdapter] = None,
        energy_adapter: Optional[EnergyIntelligenceAdapter] = None,
    ):
        # Component subsystems
        self.preference_manager = PreferenceManager(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        self.comfort_scorer = ComfortScorer(adapter=comfort_adapter)
        self.occupancy_scorer = OccupancyScorer(adapter=occupancy_adapter)
        self.energy_scorer = EnergyScorer(adapter=energy_adapter)
        self.safety_guard = SafetyGuard()
        self.decision_engine = DecisionEngine(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        self.twin_supervisor = twin_supervisor or TwinSupervisor()

    def get_or_create_twin(self, fan_id: str) -> FanDigitalTwin:
        """Retrieves or creates a digital twin for the specified fan."""
        return self.twin_supervisor.get_or_create_twin(fan_id)

    def set_manual_override(
        self,
        fan_id: str,
        speed: int,
        duration_s: float = 600.0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Sets manual override flag for a specific fan in software state."""
        record = self.preference_manager.set_manual_override(
            fan_id=fan_id, speed=speed, duration_s=duration_s, timestamp=timestamp
        )
        twin = self.get_or_create_twin(fan_id)
        twin.set_manual_override(speed=speed, duration_s=duration_s, timestamp=timestamp)
        return record

    def clear_manual_override(self, fan_id: str) -> bool:
        """Clears manual override flag for a specific fan."""
        cleared = self.preference_manager.clear_manual_override(fan_id)
        twin = self.twin_supervisor.get_twin(fan_id)
        if twin is not None:
            twin.clear_manual_override()
        return cleared

    def decide(
        self,
        csr: Union[CommonState, Dict[str, Any]],
        fan_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        apply_rate_limiting: bool = True,
    ) -> DecisionExplanation:
        """
        Executes a complete decision cycle for the target fan and room:
        1. Identifies fan and retrieves/updates digital twin.
        2. Evaluates Priority P1 (Manual Override) and P2 (Sensor Staleness/Sentinels).
        3. Evaluates energy and occupancy intelligence.
        4. Filters candidate speeds through Priorities P3, P4, and P5.
        5. Evaluates thermal comfort across filtered candidates.
        6. Solves multi-objective optimization problem.
        7. Records decision in Digital Twin and returns structured DecisionExplanation.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        iso_timestamp = (
            datetime.fromtimestamp(curr_time, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )

        # 1. Unpack CSR and identify target fan
        csr_dict = csr.to_dict() if hasattr(csr, "to_dict") else dict(csr)
        target_fan_id = (
            fan_id
            or csr_dict.get("metadata", {}).get("fan_id")
            or csr_dict.get("fan_state", {}).get("fan_id")
            or csr_dict.get("fan_id")
            or "Bedroom_Fan_1"
        )
        twin = self.get_or_create_twin(target_fan_id)
        current_speed = twin.fan_state.current_speed
        fan_mode = twin.fan_mode

        # Check manual override in preference manager
        active_override = self.preference_manager.get_manual_override(
            target_fan_id, current_time=curr_time
        )

        # Extract environmental properties
        env = csr_dict.get("environmental", {})
        temp = env.get("temperature_c") if env.get("temperature_c") is not None else csr_dict.get("temperature")
        rh = env.get("relative_humidity_pct") if env.get("relative_humidity_pct") is not None else csr_dict.get("relative_humidity")
        sensor_age_val = env.get("sensor_age_s")
        if sensor_age_val is None:
            sensor_age_val = csr_dict.get("metadata", {}).get("sensor_age_s")
        if sensor_age_val is None:
            sensor_age_val = 0.0
        sensor_age = float(sensor_age_val)
        is_stale = env.get("is_stale", False) or (sensor_age > 300.0)

        # 2. Pre-Decision Safety Checks (P1 Override, P2 Staleness / Emergency Default)
        pre_safety = self.safety_guard.evaluate_pre_decision_safety(
            csr=csr,
            current_speed=current_speed,
            manual_override=active_override,
            all_models_unavailable=False,
        )

        if pre_safety is not None and pre_safety.get("short_circuit"):
            rec_speed = pre_safety["recommended_speed"]
            mode = pre_safety["decision_mode"]
            notes = pre_safety.get("safety_notes", "")

            # Update Digital Twin
            twin.record_decision(
                proposed_speed=rec_speed,
                rationale=notes,
                decision_mode=mode,
                fallback_triggered=pre_safety.get("fallback_triggered", False),
                timestamp=curr_time,
            )

            return DecisionExplanation(
                recommended_speed=rec_speed,
                previous_speed=current_speed,
                speed_change=rec_speed - current_speed,
                decision_timestamp=iso_timestamp,
                decision_mode=mode,
                rationale=notes,
                comfort_assessment={"status": "BYPASSED_DUE_TO_SAFETY_OR_OVERRIDE"},
                occupancy_assessment={"status": "BYPASSED_DUE_TO_SAFETY_OR_OVERRIDE"},
                energy_assessment={"status": "BYPASSED_DUE_TO_SAFETY_OR_OVERRIDE"},
                safety_status={
                    "safety_active": True,
                    "staleness_detected": is_stale,
                    "sensor_age_s": sensor_age,
                },
                manual_override_status=active_override,
                fallback_status={
                    "fallback_triggered": pre_safety.get("fallback_triggered", False),
                    "fallback_reason": pre_safety.get("fallback_reason"),
                },
                candidate_evaluations=[],
                provenance_info={"source": "SAFETY_GUARD_SHORT_CIRCUIT"},
            )

        # 3. Energy Scoring (All candidates 0-5)
        energy_eval = self.energy_scorer.score_candidates(
            candidate_speeds=list(CANDIDATE_SPEEDS),
            fan_mode=fan_mode,
            csr=csr,
        )

        # 4. Occupancy Scoring
        occupancy_eval = self.occupancy_scorer.score_candidates(
            candidate_speeds=list(CANDIDATE_SPEEDS),
            energy_scores=energy_eval["energy_scores"],
            csr=csr,
        )
        occupancy_hard_gate = occupancy_eval.get("hard_gate_active", False)

        # 5. Filter Candidate Speeds (P3 Thermal Safety, P4 Occupancy Gate, P5 Rate Limit)
        candidates, constraint_note = self.safety_guard.filter_candidate_speeds(
            current_speed=current_speed,
            temperature_c=temp,
            occupancy_hard_gate_active=occupancy_hard_gate,
            apply_rate_limiting=apply_rate_limiting,
        )

        # 6. Thermal Comfort Scoring across filtered candidate speeds
        comfort_eval = self.comfort_scorer.score_candidates(
            candidate_speeds=candidates,
            current_speed=current_speed,
            csr=csr,
        )

        # 7. Multi-Objective Cost Optimization
        alpha, beta, gamma, delta = self.preference_manager.get_weights()
        self.decision_engine.alpha = alpha
        self.decision_engine.beta = beta
        self.decision_engine.gamma = gamma
        self.decision_engine.delta = delta

        winning_speed, candidate_records, rationale = self.decision_engine.evaluate_candidates(
            candidate_speeds=candidates,
            current_speed=current_speed,
            comfort_eval=comfort_eval,
            occupancy_eval=occupancy_eval,
            energy_eval=energy_eval,
        )

        # Determine decision mode
        if constraint_note and "Thermal safety floor" in constraint_note:
            mode = DECISION_MODE_THERMAL_SAFETY
        elif constraint_note and "Occupancy hard gate" in constraint_note:
            mode = DECISION_MODE_OCCUPANCY_GATE
        else:
            mode = DECISION_MODE_ADAPTIVE

        if constraint_note:
            rationale = f"[{constraint_note}] {rationale}"

        # 8. Record in Digital Twin
        twin.record_decision(
            proposed_speed=winning_speed,
            rationale=rationale,
            decision_mode=mode,
            cost_score=candidate_records[0]["total_cost"] if candidate_records else None,
            comfort_signal=comfort_eval.get("candidate_scores", {}).get(winning_speed),
            occupancy_signal=occupancy_eval,
            energy_signal=energy_eval.get("ml_advisory"),
            fallback_triggered=(not comfort_eval["model_available"] or not occupancy_eval["model_available"]),
            timestamp=curr_time,
        )

        # 9. Build and return structured DecisionExplanation
        return DecisionExplanation(
            recommended_speed=winning_speed,
            previous_speed=current_speed,
            speed_change=winning_speed - current_speed,
            decision_timestamp=iso_timestamp,
            decision_mode=mode,
            rationale=rationale,
            comfort_assessment=comfort_eval,
            occupancy_assessment=occupancy_eval,
            energy_assessment=energy_eval,
            safety_status={
                "safety_active": bool(constraint_note),
                "constraint_applied": constraint_note,
                "staleness_detected": False,
                "thermal_safety_floor_triggered": (temp is not None and temp >= 35.0),
            },
            manual_override_status=None,
            fallback_status={
                "fallback_triggered": (not comfort_eval["model_available"] or not occupancy_eval["model_available"]),
                "fallback_reason": (
                    "Model fallback active: "
                    + (
                        "Comfort unavailable; " if not comfort_eval["model_available"] else ""
                    )
                    + (
                        "Occupancy unavailable; " if not occupancy_eval["model_available"] else ""
                    )
                ) if (not comfort_eval["model_available"] or not occupancy_eval["model_available"]) else None,
            },
            candidate_evaluations=candidate_records,
            provenance_info={
                "csr_provenance": csr_dict.get("metadata", {}).get("data_source", "common_state"),
                "fan_id": target_fan_id,
                "room_category": twin.room_category,
            },
        )
