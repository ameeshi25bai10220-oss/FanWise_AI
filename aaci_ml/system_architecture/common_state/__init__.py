"""
AACI System Architecture – Common State Representation Package
=============================================================
Phase 5 – Stage 2: Common State Representation (CSR) Implementation

Exports the validated CommonState representation, centralized sensor validator,
and related telemetry diagnostic structures.
"""

from .constants import (
    PHYSICAL_TEMP_RANGE_C,
    PHYSICAL_HUMIDITY_RANGE_PCT,
    PHYSICAL_AIR_VELOCITY_RANGE_MS,
    PHYSICAL_LIGHT_RANGE_LUX,
    PHYSICAL_CO2_RANGE_PPM,
    SENSOR_SENTINEL_VALUES,
    VALID_FAN_SPEEDS,
    VALID_FAN_MODES,
    DEFAULT_STALENESS_THRESHOLD_SECONDS,
    ROBOD_ROOM_TYPES,
    RESEARCH_BLDC_POWER_MAP,
    RESEARCH_INDUCTION_POWER_MAP,
)
from .sensor_validator import (
    CentralizedSensorValidator,
    ValidationResult,
    SensorValidationError,
)
from .common_state import (
    CommonState,
    EnvironmentalState,
    OccupancyState,
    FanOperatingState,
    EnergyState,
    SystemHealthState,
)

__all__ = [
    "CommonState",
    "EnvironmentalState",
    "OccupancyState",
    "FanOperatingState",
    "EnergyState",
    "SystemHealthState",
    "CentralizedSensorValidator",
    "ValidationResult",
    "SensorValidationError",
    "PHYSICAL_TEMP_RANGE_C",
    "PHYSICAL_HUMIDITY_RANGE_PCT",
    "PHYSICAL_AIR_VELOCITY_RANGE_MS",
    "PHYSICAL_LIGHT_RANGE_LUX",
    "PHYSICAL_CO2_RANGE_PPM",
    "SENSOR_SENTINEL_VALUES",
    "VALID_FAN_SPEEDS",
    "VALID_FAN_MODES",
    "DEFAULT_STALENESS_THRESHOLD_SECONDS",
    "ROBOD_ROOM_TYPES",
    "RESEARCH_BLDC_POWER_MAP",
    "RESEARCH_INDUCTION_POWER_MAP",
]

