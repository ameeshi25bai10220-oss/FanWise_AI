"""
AACI Adaptive Controller – Constants & Operational Limits
=========================================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Defines hyperparameter weights, constraint limits, priority checks,
and mandatory disclaimers for multi-objective fan speed selection.
"""

from typing import Tuple

# Multi-objective optimization default weights
# Sourced from adaptive_controller_design.md §3
DEFAULT_ALPHA_COMFORT_WEIGHT: float = 1.5      # Thermal comfort primary objective
DEFAULT_BETA_ENERGY_WEIGHT: float = 0.8        # Energy efficiency secondary
DEFAULT_GAMMA_STABILITY_WEIGHT: float = 0.5    # Speed transition stability penalty
DEFAULT_DELTA_OCCUPANCY_WEIGHT: float = 1.0    # Occupancy bonus weight

# Physical candidate speeds
CANDIDATE_SPEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# Rate limiting: maximum speed delta per normal cycle
RATE_LIMIT_MAX_DELTA_SPEED: int = 2

# Occupancy threshold limits
OCCUPANCY_HARD_GATE_PROB_THRESHOLD: float = 0.35
OCCUPANCY_UNCERTAINTY_LOW: float = 0.35
OCCUPANCY_UNCERTAINTY_HIGH: float = 0.65

# Thermal safety floor
THERMAL_SAFETY_TEMP_THRESHOLD_C: float = 35.0
THERMAL_SAFETY_MIN_SPEED: int = 1

# Warm conditions floor for emergency safe default
WARM_CONDITION_THRESHOLD_C: float = 28.0

# Sensor staleness threshold
SENSOR_STALENESS_THRESHOLD_SECONDS: float = 300.0

# Manual override lockout duration
MANUAL_OVERRIDE_LOCKOUT_SECONDS: float = 600.0

# Tie-break threshold for stability
TIE_BREAK_COST_THRESHOLD: float = 0.05

# Comfort uncertainty thresholds (ASHRAE XGBoost uncalibrated softmax)
COMFORT_UNCERTAINTY_HIGH_THRESHOLD: float = 0.60
COMFORT_UNCERTAINTY_SEVERE_THRESHOLD: float = 0.40
COMFORT_UNIFORM_PENALTY_FALLBACK: float = 0.33

# Decision modes
DECISION_MODE_ADAPTIVE: str = "adaptive"
DECISION_MODE_MANUAL_OVERRIDE: str = "manual_override"
DECISION_MODE_THERMAL_SAFETY: str = "thermal_safety_override"
DECISION_MODE_OCCUPANCY_GATE: str = "occupancy_hard_gate"
DECISION_MODE_STALENESS_FREEZE: str = "staleness_freeze"
DECISION_MODE_EMERGENCY_DEFAULT: str = "emergency_safe_default"
DECISION_MODE_UNAVAILABLE: str = "unavailable"

# Mandatory regulatory & architecture disclaimers
MANDATORY_DISCLAIMERS: Tuple[str, ...] = (
    "SOFTWARE RECOMMENDATION ONLY: Controller outputs a software recommendation scale (0-5); no physical fan actuation is performed.",
    "UNCALIBRATED COMFORT PROBABILITIES: Comfort probabilities are uncalibrated XGBoost softmax outputs from ASHRAE Database II; not calibrated Bayesian posteriors.",
    "ADVISORY ENERGY FORECAST: Model 3 LightGBM forecast (R^2 approx 0.1515) is advisory only; cannot command speed or guarantee energy savings.",
    "PROVISIONAL AIRFLOW & NOMINAL POWER: Airflow velocity is mapped from provisional Dryad dataset; power values are nominal research constants, not validated Atomberg bench measurements.",
    "ACADEMIC ROBOD CONTEXT: Headcount and spatial occupancy predictions originate from academic ROBOD environments and must not be conflated with residential room ground truth.",
)

