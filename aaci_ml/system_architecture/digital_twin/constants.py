"""
AACI Fan Digital Twin – Constants & Configuration
=================================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Defines operational limits, history buffer capacities, event triggers,
and provenance markers for virtual twin simulations.
"""

from typing import Dict, List, Tuple

# Transition history capacity (circular buffer depth)
DEFAULT_TRANSITION_HISTORY_CAPACITY: int = 20

# Snapshot history capacity (circular buffer depth)
DEFAULT_SNAPSHOT_HISTORY_CAPACITY: int = 50

# Speed change pending acknowledgement timeout (seconds)
SPEED_CHANGE_PENDING_TIMEOUT_SECONDS: float = 120.0

# Manual override lockout duration default (seconds)
DEFAULT_MANUAL_OVERRIDE_TIMEOUT_SECONDS: float = 600.0

# Supported state transition event trigger types
TRANSITION_TRIGGERS: Tuple[str, ...] = (
    "INITIALIZATION",
    "FAN_SWITCHED_ON",
    "FAN_SWITCHED_OFF",
    "SPEED_CHANGED",
    "OCCUPANCY_STATE_CHANGED",
    "ENVIRONMENTAL_CHANGED",
    "ENERGY_TELEMETRY_RECEIVED",
    "SENSOR_STALE",
    "SENSOR_RECOVERED",
    "ADAPTER_UNAVAILABLE",
    "MANUAL_OVERRIDE_SET",
    "MANUAL_OVERRIDE_EXPIRED",
    "SAFETY_FLOOR_TRIGGERED",
)

# Explicit provenance tag for synthetic simulation signals
SYNTHETIC_SIMULATION_PROVENANCE_TAG: str = "SYNTHETIC_SIMULATION"

# Default fan ID to room category mapping (residential context)
RESIDENTIAL_FAN_ROOM_CATEGORIES: Dict[str, str] = {
    "Bedroom_Fan_1": "Bedroom",
    "Bedroom_Fan_2": "Bedroom",
    "Bedroom_Fan_3": "Bedroom",
    "Drawingroom_Fan_1": "Drawingroom",
    "Drawingroom_Fan_2": "Drawingroom",
    "Drawingroom_Fan_3": "Drawingroom",
    "Lounge_Fan_1": "Lounge",
    "Lounge_Fan_2": "Lounge",
    "Lounge_Fan_3": "Lounge",
}

