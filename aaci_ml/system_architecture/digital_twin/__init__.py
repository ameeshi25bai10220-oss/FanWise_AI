"""
AACI Fan Digital Twin Runtime
=============================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Exports the complete Digital Twin runtime package:
- FanDigitalTwin: Core virtual twin representation of a single fan and room.
- TwinSupervisor: Fleet manager for multi-fan installations.
- DigitalTwinSimulator: Replay and synthetic scenario simulator.
- State Managers: FanStateManager, RoomStateManager, EnergyStateManager.
- Transition Engine: StateTransitionEngine, StateTransitionRecord.
- History Manager: HistoricalStateManager.
- Constants and configuration.
"""

from aaci_ml.system_architecture.digital_twin.constants import (
    DEFAULT_MANUAL_OVERRIDE_TIMEOUT_SECONDS,
    DEFAULT_SNAPSHOT_HISTORY_CAPACITY,
    DEFAULT_TRANSITION_HISTORY_CAPACITY,
    RESIDENTIAL_FAN_ROOM_CATEGORIES,
    SPEED_CHANGE_PENDING_TIMEOUT_SECONDS,
    SYNTHETIC_SIMULATION_PROVENANCE_TAG,
    TRANSITION_TRIGGERS,
)
from aaci_ml.system_architecture.digital_twin.fan_digital_twin import FanDigitalTwin
from aaci_ml.system_architecture.digital_twin.history_manager import HistoricalStateManager
from aaci_ml.system_architecture.digital_twin.simulator import DigitalTwinSimulator
from aaci_ml.system_architecture.digital_twin.state_managers import (
    BLDC_POWER_TABLE_WATTS,
    INDUCTION_POWER_TABLE_WATTS,
    EnergyStateManager,
    FanStateManager,
    RoomStateManager,
)
from aaci_ml.system_architecture.digital_twin.transition_engine import (
    StateTransitionEngine,
    StateTransitionRecord,
)
from aaci_ml.system_architecture.digital_twin.twin_supervisor import TwinSupervisor

__all__ = [
    "FanDigitalTwin",
    "TwinSupervisor",
    "DigitalTwinSimulator",
    "FanStateManager",
    "RoomStateManager",
    "EnergyStateManager",
    "StateTransitionEngine",
    "StateTransitionRecord",
    "HistoricalStateManager",
    "BLDC_POWER_TABLE_WATTS",
    "INDUCTION_POWER_TABLE_WATTS",
    "TRANSITION_TRIGGERS",
    "SYNTHETIC_SIMULATION_PROVENANCE_TAG",
    "DEFAULT_TRANSITION_HISTORY_CAPACITY",
    "DEFAULT_SNAPSHOT_HISTORY_CAPACITY",
    "SPEED_CHANGE_PENDING_TIMEOUT_SECONDS",
    "DEFAULT_MANUAL_OVERRIDE_TIMEOUT_SECONDS",
    "RESIDENTIAL_FAN_ROOM_CATEGORIES",
]

