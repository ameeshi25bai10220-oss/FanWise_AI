"""
AACI System Architecture – Model Adapters Package
=================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Exports production adapters for the three frozen AACI intelligence models:
  - ThermalComfortAdapter (Model 1)
  - OccupancyAdapter (Model 2)
  - EnergyIntelligenceAdapter (Model 3)
  - AdapterOutputEnvelope and contracts
"""

from .contract import (
    AdapterOutputEnvelope,
    InputValidityStatus,
    InputFreshnessInfo,
    ModelAvailabilityStatus,
    ValueProvenance,
    DegradedState,
)
from .constants import (
    PROVISIONAL_SPEED_VELOCITY_MAP,
    COMFORT_UNCERTAINTY_THRESHOLD,
    OCCUPANCY_UNCERTAINTY_BAND_LOW,
    OCCUPANCY_UNCERTAINTY_BAND_HIGH,
    BLDC_POWER_MAP,
    INDUCTION_POWER_MAP,
    MODEL3_ML_BENCHMARK_R2,
    MODEL3_ML_BENCHMARK_MAE_WH,
)
from .thermal_comfort_adapter import ThermalComfortAdapter
from .occupancy_adapter import OccupancyAdapter
from .energy_adapter import EnergyIntelligenceAdapter

__all__ = [
    "ThermalComfortAdapter",
    "OccupancyAdapter",
    "EnergyIntelligenceAdapter",
    "AdapterOutputEnvelope",
    "InputValidityStatus",
    "InputFreshnessInfo",
    "ModelAvailabilityStatus",
    "ValueProvenance",
    "DegradedState",
    "PROVISIONAL_SPEED_VELOCITY_MAP",
    "COMFORT_UNCERTAINTY_THRESHOLD",
    "OCCUPANCY_UNCERTAINTY_BAND_LOW",
    "OCCUPANCY_UNCERTAINTY_BAND_HIGH",
    "BLDC_POWER_MAP",
    "INDUCTION_POWER_MAP",
    "MODEL3_ML_BENCHMARK_R2",
    "MODEL3_ML_BENCHMARK_MAE_WH",
]

