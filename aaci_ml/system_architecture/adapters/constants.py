"""
AACI System Architecture – Adapter Constants
============================================
Phase 5 – Stage 1: Adapter Layer Implementation

Defines physical, mathematical, and provisional constants used by AACI model adapters.
All provisional values are explicitly documented and flagged.
"""

from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Thermal Comfort Intelligence Constants
# ─────────────────────────────────────────────────────────────────────────────

# PROVISIONAL speed-to-air-velocity mapping
# Values sourced from Dryad open-access dataset (measured on non-Atomberg fan at 1.1m height).
# NOT a validated Atomberg calibration. Must always be flagged as PROVISIONAL.
PROVISIONAL_SPEED_VELOCITY_MAP: Dict[int, float] = {
    0: 0.050,  # ASHRAE still-air baseline (m/s) — not fan-generated
    1: 0.112,  # Speed 1 provisional air velocity (m/s)
    2: 0.239,  # Speed 2 provisional air velocity (m/s)
    3: 0.293,  # Speed 3 provisional air velocity (m/s)
    4: 0.410,  # Speed 4 provisional air velocity (m/s)
    5: 0.527,  # Speed 5 provisional air velocity (m/s)
}

# Model 1 confidence uncertainty threshold
# Softmax winning probability below 0.60 is treated as uncertain comfort
COMFORT_UNCERTAINTY_THRESHOLD: float = 0.60

# Model 1 training domain assumptions (logged as acknowledged mismatches)
MODEL1_DEFAULT_ASSUMPTIONS = {
    "building_type": "Office",
    "cooling_strategy": "Air Conditioned",
    "clo": 0.6,
    "met": 1.2,
    "domain_mismatch_note": (
        "Model 1 trained on ASHRAE DB II commercial/office data. "
        "building_type='Office' and cooling_strategy='Air Conditioned' are "
        "acknowledged approximations for residential Indian ceiling fan deployment."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Occupancy Intelligence Constants
# ─────────────────────────────────────────────────────────────────────────────

# Occupancy probability uncertainty band [0.35, 0.65]
OCCUPANCY_UNCERTAINTY_BAND_LOW: float = 0.35
OCCUPANCY_UNCERTAINTY_BAND_HIGH: float = 0.65

# Hard occupancy gate threshold
OCCUPANCY_GATE_THRESHOLD: float = 0.35

# Default fan ID to integer room ID mapping for Module 2 Spatial Predictor
DEFAULT_FAN_TO_ROOM_MAPPING: Dict[str, int] = {
    "Bedroom_Fan_1": 1,
    "Bedroom_Fan_2": 2,
    "Bedroom_Fan_3": 1,
    "Drawingroom_Fan_1": 3,
    "Drawingroom_Fan_2": 4,
    "Drawingroom_Fan_3": 3,
    "Lounge_Fan_1": 5,
    "Lounge_Fan_2": 5,
    "Lounge_Fan_3": 5,
}

# ─────────────────────────────────────────────────────────────────────────────
# Model 3: Energy Intelligence Constants
# ─────────────────────────────────────────────────────────────────────────────

# Deterministic fan power ratings by speed (Watts)
BLDC_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 10.0,
    2: 15.0,
    3: 20.0,
    4: 27.0,
    5: 48.0,
}

INDUCTION_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 24.0,
    2: 34.0,
    3: 55.0,
    4: 64.0,
    5: 96.0,
}

# Model 3 ML forecaster benchmarks and metadata
MODEL3_ML_BENCHMARK_R2: float = 0.1515
MODEL3_ML_BENCHMARK_MAE_WH: float = 1.1873
MODEL3_FORECAST_HORIZON: str = "next_observed_packet_variable"

# ─────────────────────────────────────────────────────────────────────────────
# Common Sensor Validity & Staleness Limits
# ─────────────────────────────────────────────────────────────────────────────

MAX_SENSOR_STALENESS_SECONDS: float = 300.0

VALID_TEMP_RANGE_C = (-10.0, 60.0)
VALID_HUMIDITY_RANGE_PCT = (0.0, 100.0)
VALID_AIR_VELOCITY_RANGE_MS = (0.0, 5.0)
VALID_LIGHT_RANGE_LUX = (0.0, 100000.0)
VALID_CO2_RANGE_PPM = (200.0, 10000.0)
VALID_FAN_SPEEDS = (0, 1, 2, 3, 4, 5)

# Sentinel error values indicating hardware sensor fault
SENSOR_SENTINEL_VALUES = [255, 255.0]

