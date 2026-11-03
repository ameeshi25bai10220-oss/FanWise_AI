"""
AACI Adaptive Controller Package
================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Exports the complete Adaptive Controller suite:
- AdaptiveController: Main orchestration controller.
- ComfortScorer: Thermal comfort evaluator and piecewise DiscomfortPenalty calculator.
- OccupancyScorer: Occupancy relevance and OccupancyBonus evaluator.
- EnergyScorer: Relative energy scorer and ML advisory forecast inspector.
- DecisionEngine: Multi-objective candidate cost evaluator and tie-breaker.
- SafetyGuard: Priority safety rules (P1-P5) and fallback enforcer.
- PreferenceManager: User weights and per-fan manual override manager.
- DecisionExplanation: Structured decision output contract.
- Constants and operational limits.
"""

from aaci_ml.system_architecture.controller.adaptive_controller import AdaptiveController
from aaci_ml.system_architecture.controller.comfort_scorer import ComfortScorer
from aaci_ml.system_architecture.controller.constants import (
    CANDIDATE_SPEEDS,
    COMFORT_UNCERTAINTY_HIGH_THRESHOLD,
    COMFORT_UNCERTAINTY_SEVERE_THRESHOLD,
    COMFORT_UNIFORM_PENALTY_FALLBACK,
    DECISION_MODE_ADAPTIVE,
    DECISION_MODE_EMERGENCY_DEFAULT,
    DECISION_MODE_MANUAL_OVERRIDE,
    DECISION_MODE_OCCUPANCY_GATE,
    DECISION_MODE_STALENESS_FREEZE,
    DECISION_MODE_THERMAL_SAFETY,
    DEFAULT_ALPHA_COMFORT_WEIGHT,
    DEFAULT_BETA_ENERGY_WEIGHT,
    DEFAULT_DELTA_OCCUPANCY_WEIGHT,
    DEFAULT_GAMMA_STABILITY_WEIGHT,
    MANDATORY_DISCLAIMERS,
    MANUAL_OVERRIDE_LOCKOUT_SECONDS,
    OCCUPANCY_HARD_GATE_PROB_THRESHOLD,
    OCCUPANCY_UNCERTAINTY_HIGH,
    OCCUPANCY_UNCERTAINTY_LOW,
    RATE_LIMIT_MAX_DELTA_SPEED,
    SENSOR_STALENESS_THRESHOLD_SECONDS,
    THERMAL_SAFETY_MIN_SPEED,
    THERMAL_SAFETY_TEMP_THRESHOLD_C,
    TIE_BREAK_COST_THRESHOLD,
    WARM_CONDITION_THRESHOLD_C,
)
from aaci_ml.system_architecture.controller.decision_engine import DecisionEngine
from aaci_ml.system_architecture.controller.decision_explanation import DecisionExplanation
from aaci_ml.system_architecture.controller.energy_scorer import EnergyScorer
from aaci_ml.system_architecture.controller.occupancy_scorer import OccupancyScorer
from aaci_ml.system_architecture.controller.preference_manager import PreferenceManager
from aaci_ml.system_architecture.controller.safety_guard import SafetyGuard

__all__ = [
    "AdaptiveController",
    "ComfortScorer",
    "OccupancyScorer",
    "EnergyScorer",
    "DecisionEngine",
    "SafetyGuard",
    "PreferenceManager",
    "DecisionExplanation",
    "DEFAULT_ALPHA_COMFORT_WEIGHT",
    "DEFAULT_BETA_ENERGY_WEIGHT",
    "DEFAULT_GAMMA_STABILITY_WEIGHT",
    "DEFAULT_DELTA_OCCUPANCY_WEIGHT",
    "CANDIDATE_SPEEDS",
    "RATE_LIMIT_MAX_DELTA_SPEED",
    "OCCUPANCY_HARD_GATE_PROB_THRESHOLD",
    "OCCUPANCY_UNCERTAINTY_LOW",
    "OCCUPANCY_UNCERTAINTY_HIGH",
    "THERMAL_SAFETY_TEMP_THRESHOLD_C",
    "THERMAL_SAFETY_MIN_SPEED",
    "WARM_CONDITION_THRESHOLD_C",
    "SENSOR_STALENESS_THRESHOLD_SECONDS",
    "MANUAL_OVERRIDE_LOCKOUT_SECONDS",
    "TIE_BREAK_COST_THRESHOLD",
    "COMFORT_UNCERTAINTY_HIGH_THRESHOLD",
    "COMFORT_UNCERTAINTY_SEVERE_THRESHOLD",
    "COMFORT_UNIFORM_PENALTY_FALLBACK",
    "DECISION_MODE_ADAPTIVE",
    "DECISION_MODE_MANUAL_OVERRIDE",
    "DECISION_MODE_THERMAL_SAFETY",
    "DECISION_MODE_OCCUPANCY_GATE",
    "DECISION_MODE_STALENESS_FREEZE",
    "DECISION_MODE_EMERGENCY_DEFAULT",
    "MANDATORY_DISCLAIMERS",
]

