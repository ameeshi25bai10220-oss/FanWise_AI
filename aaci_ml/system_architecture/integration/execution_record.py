"""
AACI Integration – IntegrationResult Output Contract
=====================================================
Phase 5 – Stage 5: System Integration and Validation

Defines the structured 22-field output contract for a single end-to-end
pipeline execution.  Every field is documented with its source, type, and
provenance semantics.

This module must NOT import from, or modify, any frozen component.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory integration-level disclaimers
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_DISCLAIMERS: List[str] = [
    "Model 1 (Thermal Comfort) is trained on ASHRAE DB-II commercial/office data; "
    "predictions for residential Indian ceiling-fan deployment are unvalidated.",
    "Model 1 softmax probabilities are uncalibrated and are NOT calibrated confidence scores.",
    "Air velocity estimates use provisional Dryad open-access mapping; "
    "no validated Atomberg hardware calibration exists.",
    "Model 2 (Occupancy) is trained on ROBOD academic dataset (rooms 1-5: "
    "Lecture Rooms, Office Space, Library Space); ROBOD room IDs must NOT be "
    "equated with residential Bedroom / Drawingroom / Lounge categories.",
    "Model 3 (Energy) R²≈0.1515, MAE≈1.19 Wh; it is advisory only and "
    "CANNOT command fan speed.",
    "All power estimates are nominal research-dataset constants; no "
    "hardware-calibrated Atomberg energy data is available.",
    "UNRESOLVED_INDUCTION_BASELINE: Two diverging induction power tables exist "
    "across Stage 1 and Stage 3 of this codebase; neither is validated.",
    "No physical fan actuation occurs; all speed values are software-scale 0-5 "
    "recommendations only.",
]


@dataclass
class IntegrationResult:
    """
    22-field structured output contract for one complete AACI pipeline execution.

    Field groups
    ─────────────
    Identity:        execution_id, fan_id, room_id, room_category, input_timestamp
    Validation:      input_validation_status
    Model outputs:   model1_result, model2_result, model3_result
    System state:    csr_state_summary, digital_twin_state_summary
    Decision:        previous_speed, recommended_speed
    Assessments:     comfort_assessment, occupancy_assessment, energy_assessment
    Scoring:         candidate_scores
    Status flags:    safety_status, manual_override_status, fallback_status
    Explanation:     decision_explanation
    Metadata:        data_freshness, warnings, unresolved_limitations
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    execution_id: str = ""
    """Unique execution identifier (UUID4 string)."""

    fan_id: str = "Unknown_Fan"
    """Fan device identifier passed to the pipeline."""

    room_id: Optional[int] = None
    """ROBOD room ID (1-5), if resolved from occupancy adapter. Research context only."""

    room_category: str = "Unknown"
    """Fan room category: Bedroom, Drawingroom, Lounge, or Unknown."""

    input_timestamp: Optional[str] = None
    """ISO 8601 timestamp of the input telemetry packet."""

    # ── Validation ───────────────────────────────────────────────────────────
    input_validation_status: Dict[str, Any] = field(default_factory=dict)
    """
    Sensor validation result. Keys: is_valid, errors, warnings,
    missing_fields, invalid_fields, stale_fields, is_stale.
    """

    # ── Model outputs (raw adapter envelope summaries) ───────────────────────
    model1_result: Dict[str, Any] = field(default_factory=dict)
    """
    ThermalComfortAdapter normalized_payload summary including:
    comfort_label, comfort_probabilities, is_uncertain, air_velocity_used_ms,
    provisional_mapping_flag, model_available.
    """

    model2_result: Dict[str, Any] = field(default_factory=dict)
    """
    OccupancyAdapter normalized_payload summary including:
    occupancy_state, occupancy_binary, occupancy_probability,
    headcount_estimate, sensor_gap_active, model_available.
    """

    model3_result: Dict[str, Any] = field(default_factory=dict)
    """
    EnergyIntelligenceAdapter normalized_payload summary including:
    candidate_evaluation_table, ml_forecast, model_available.
    Advisory only; R²≈0.1515.
    """

    # ── System state ─────────────────────────────────────────────────────────
    csr_state_summary: Dict[str, Any] = field(default_factory=dict)
    """
    CommonState summary:
    environmental (temp, humidity, air_vel), occupancy, fan_state,
    energy, health, metadata.
    """

    digital_twin_state_summary: Dict[str, Any] = field(default_factory=dict)
    """
    FanDigitalTwin inspect_state() snapshot at the time of the decision.
    """

    # ── Decision ─────────────────────────────────────────────────────────────
    previous_speed: Optional[int] = None
    """Fan speed at start of this decision cycle (software scale 0-5)."""

    recommended_speed: Optional[int] = None
    """Recommended fan speed output (software scale 0-5). NOT a hardware command."""

    # ── Assessments ──────────────────────────────────────────────────────────
    comfort_assessment: Dict[str, Any] = field(default_factory=dict)
    """ComfortScorer output: candidate_scores, model_available, disclaimers."""

    occupancy_assessment: Dict[str, Any] = field(default_factory=dict)
    """OccupancyScorer output: hard_gate_active, scores, model_available."""

    energy_assessment: Dict[str, Any] = field(default_factory=dict)
    """EnergyScorer output: energy_scores, ml_advisory, model_available."""

    # ── Scoring ──────────────────────────────────────────────────────────────
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)
    """Per-candidate speed cost breakdown from DecisionEngine."""

    # ── Status flags ─────────────────────────────────────────────────────────
    safety_status: Dict[str, Any] = field(default_factory=dict)
    """SafetyGuard outcome: safety_active, constraint_applied, thermal_safety_floor."""

    manual_override_status: Optional[Dict[str, Any]] = None
    """
    PreferenceManager override record if active, else None.
    Keys: active, speed, expires_at, remaining_s.
    """

    fallback_status: Dict[str, Any] = field(default_factory=dict)
    """
    Fallback status: fallback_triggered, fallback_reason.
    Triggered when Model 1 or Model 2 is unavailable.
    """

    # ── Explanation ──────────────────────────────────────────────────────────
    decision_explanation: Dict[str, Any] = field(default_factory=dict)
    """
    Full DecisionExplanation.to_dict() output including rationale,
    decision_mode, disclaimers, and provenance_info.
    """

    # ── Metadata ─────────────────────────────────────────────────────────────
    data_freshness: Dict[str, Any] = field(default_factory=dict)
    """
    Sensor data freshness: sensor_age_s, is_stale, stale_fields,
    staleness_threshold_s.
    """

    warnings: List[str] = field(default_factory=list)
    """Non-fatal warnings accumulated during pipeline execution."""

    unresolved_limitations: List[str] = field(
        default_factory=lambda: list(INTEGRATION_DISCLAIMERS)
    )
    """
    Mandatory unresolved-limitation notices that accompany every result.
    Includes UNRESOLVED_INDUCTION_BASELINE and all model disclaimer strings.
    """

    # ── Latency (bonus) ──────────────────────────────────────────────────────
    latency_ms: Dict[str, float] = field(default_factory=dict)
    """
    Per-step wall-clock latency in milliseconds.
    Keys: total_ms, csr_ms, twin_ms, controller_ms, adapter_ms.
    """

    provenance: str = "UNKNOWN"
    """
    Top-level data provenance label.
    'SYNTHETIC_SIMULATION' for synthetic/demo inputs; 'TELEMETRY' for live sensor data.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Serializes IntegrationResult to a JSON-compatible dictionary."""
        return {
            "execution_id": self.execution_id,
            "fan_id": self.fan_id,
            "room_id": self.room_id,
            "room_category": self.room_category,
            "input_timestamp": self.input_timestamp,
            "input_validation_status": self.input_validation_status,
            "model1_result": self.model1_result,
            "model2_result": self.model2_result,
            "model3_result": self.model3_result,
            "csr_state_summary": self.csr_state_summary,
            "digital_twin_state_summary": self.digital_twin_state_summary,
            "previous_speed": self.previous_speed,
            "recommended_speed": self.recommended_speed,
            "comfort_assessment": self.comfort_assessment,
            "occupancy_assessment": self.occupancy_assessment,
            "energy_assessment": self.energy_assessment,
            "candidate_scores": self.candidate_scores,
            "safety_status": self.safety_status,
            "manual_override_status": self.manual_override_status,
            "fallback_status": self.fallback_status,
            "decision_explanation": self.decision_explanation,
            "data_freshness": self.data_freshness,
            "warnings": self.warnings,
            "unresolved_limitations": self.unresolved_limitations,
            "latency_ms": self.latency_ms,
            "provenance": self.provenance,
        }

