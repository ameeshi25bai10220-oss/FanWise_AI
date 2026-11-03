"""
AACI Common State Representation – Constants & Validation Thresholds
====================================================================
Phase 5 – Stage 2: Common State Representation (CSR) Implementation

Centralizes sensor physical validity boundaries, sentinel definitions,
freshness thresholds, and ROBOD room type metadata.
"""

from typing import Dict, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# Physical & Sensor Range Boundaries
# ─────────────────────────────────────────────────────────────────────────────

# Plausible human indoor environmental ranges
PHYSICAL_TEMP_RANGE_C: Tuple[float, float] = (-10.0, 60.0)
PHYSICAL_HUMIDITY_RANGE_PCT: Tuple[float, float] = (0.0, 100.0)
PHYSICAL_AIR_VELOCITY_RANGE_MS: Tuple[float, float] = (0.0, 5.0)

# Optional sensor ranges
PHYSICAL_LIGHT_RANGE_LUX: Tuple[float, float] = (0.0, 100000.0)
PHYSICAL_CO2_RANGE_PPM: Tuple[float, float] = (200.0, 10000.0)

# Hardware sentinel values indicating sensor disconnection / fault
SENSOR_SENTINEL_VALUES: List[float] = [255.0, 255]

# Valid discrete fan speeds
VALID_FAN_SPEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# Valid fan operating modes
VALID_FAN_MODES: Tuple[str, ...] = ("BLDC", "Induction")

# ─────────────────────────────────────────────────────────────────────────────
# Time & Staleness Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Design default staleness threshold in seconds.
# Documented as a design choice for ceiling fan comfort telemetry (sampling interval ~30-150s),
# NOT a universal physical law. Configurable per deployment.
DEFAULT_STALENESS_THRESHOLD_SECONDS: float = 300.0

# ─────────────────────────────────────────────────────────────────────────────
# ROBOD Room Training Metadata
# ─────────────────────────────────────────────────────────────────────────────

# Original training domain room types from the ROBOD dataset.
# Sourced from commercial/academic buildings (NOT residential Bedroom/Drawingroom/Lounge).
ROBOD_ROOM_TYPES: Dict[int, str] = {
    1: "Lecture Room 1",
    2: "Lecture Room 2",
    3: "Office Space 3",
    4: "Office Space 4",
    5: "Library Space 5",
}

# ─────────────────────────────────────────────────────────────────────────────
# Nominal Research Dataset Power Constants
# ─────────────────────────────────────────────────────────────────────────────

# Nominal power constants from research datasets (Watts).
# Strictly separated from actual measured power and future hardware-calibrated profiles.
RESEARCH_BLDC_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 10.0,
    2: 15.0,
    3: 20.0,
    4: 27.0,
    5: 48.0,
}

RESEARCH_INDUCTION_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 24.0,
    2: 34.0,
    3: 55.0,
    4: 64.0,
    5: 96.0,
}

