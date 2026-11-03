"""
AACI Fan Digital Twin – State Transition Engine
===============================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Detects, verifies, and records operational transitions for virtual twin states:
- Speed change events (ON, OFF, SPEED_CHANGED)
- Occupancy condition transitions (Occupied, Unoccupied, Unknown)
- Sensor health transitions (STALE, RECOVERED, ADAPTER_UNAVAILABLE)
- Manual override life-cycle transitions (SET, EXPIRED)
- Value provenance preservation across all transition records.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aaci_ml.system_architecture.digital_twin.constants import (
    SYNTHETIC_SIMULATION_PROVENANCE_TAG,
    TRANSITION_TRIGGERS,
)


@dataclass
class StateTransitionRecord:
    """Immutable record of an operational state transition."""

    timestamp: str
    fan_id: str
    triggering_event: str
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    data_provenance: str
    validity_status: str = "VALID"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes record to JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp,
            "fan_id": self.fan_id,
            "triggering_event": self.triggering_event,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "data_provenance": self.data_provenance,
            "validity_status": self.validity_status,
            "notes": self.notes,
        }


class StateTransitionEngine:
    """Analyzes twin state differentials to detect and log operational transitions."""

    def __init__(self, fan_id: str):
        self.fan_id = fan_id

    def create_record(
        self,
        triggering_event: str,
        previous_state: Dict[str, Any],
        new_state: Dict[str, Any],
        data_provenance: str = "TELEMETRY",
        validity_status: str = "VALID",
        notes: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> StateTransitionRecord:
        """Constructs a validated StateTransitionRecord."""
        if triggering_event not in TRANSITION_TRIGGERS and not triggering_event.startswith("CUSTOM_"):
            triggering_event = "CUSTOM_" + triggering_event

        ts = timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat()

        # Check if provenance is synthetic simulation
        if data_provenance == SYNTHETIC_SIMULATION_PROVENANCE_TAG:
            validity_status = "SIMULATION"

        return StateTransitionRecord(
            timestamp=ts,
            fan_id=self.fan_id,
            triggering_event=triggering_event,
            previous_state=previous_state,
            new_state=new_state,
            data_provenance=data_provenance,
            validity_status=validity_status,
            notes=notes,
        )

    def detect_transitions(
        self,
        previous_state: Optional[Dict[str, Any]],
        current_state: Dict[str, Any],
        trigger_hint: Optional[str] = None,
        data_provenance: str = "TELEMETRY",
        timestamp: Optional[str] = None,
    ) -> List[StateTransitionRecord]:
        """
        Compares previous and current states to detect triggered events.
        Returns a list of StateTransitionRecords.
        """
        records: List[StateTransitionRecord] = []
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat()

        # Handle initialization
        if previous_state is None:
            init_record = self.create_record(
                triggering_event=trigger_hint or "INITIALIZATION",
                previous_state={},
                new_state=current_state,
                data_provenance=data_provenance,
                validity_status="VALID",
                notes="Initial twin state establishment",
                timestamp=ts,
            )
            records.append(init_record)
            return records

        # 1. Manual override events
        old_override = previous_state.get(
            "manual_override_active",
            previous_state.get("fan", {}).get("manual_override_active", False),
        )
        new_override = current_state.get(
            "manual_override_active",
            current_state.get("fan", {}).get("manual_override_active", False),
        )
        if not old_override and new_override:
            records.append(
                self.create_record(
                    triggering_event="MANUAL_OVERRIDE_SET",
                    previous_state={"manual_override_active": False},
                    new_state={
                        "manual_override_active": True,
                        "speed": current_state.get("current_speed", current_state.get("fan", {}).get("current_speed")),
                    },
                    data_provenance="USER_INTERACTION",
                    notes="User manual override activated",
                    timestamp=ts,
                )
            )
        elif old_override and not new_override:
            records.append(
                self.create_record(
                    triggering_event="MANUAL_OVERRIDE_EXPIRED",
                    previous_state={"manual_override_active": True},
                    new_state={"manual_override_active": False},
                    data_provenance="SYSTEM_TIMER",
                    notes="Manual override expired",
                    timestamp=ts,
                )
            )

        # 2. Fan speed changes
        old_speed = previous_state.get(
            "current_speed",
            previous_state.get("fan", {}).get("current_speed", 0),
        )
        new_speed = current_state.get(
            "current_speed",
            current_state.get("fan", {}).get("current_speed", 0),
        )

        if old_speed != new_speed:
            if old_speed == 0 and new_speed > 0:
                event = "FAN_SWITCHED_ON"
            elif old_speed > 0 and new_speed == 0:
                event = "FAN_SWITCHED_OFF"
            else:
                event = "SPEED_CHANGED"

            records.append(
                self.create_record(
                    triggering_event=event,
                    previous_state={"current_speed": old_speed},
                    new_state={"current_speed": new_speed},
                    data_provenance=data_provenance,
                    notes=f"Speed transition: {old_speed} -> {new_speed}",
                    timestamp=ts,
                )
            )

        # 3. Occupancy transitions
        old_occ = previous_state.get(
            "room_state", {}
        ).get(
            "occupancy_classification",
            previous_state.get("occupancy_classification", "Unknown"),
        )
        new_occ = current_state.get(
            "room_state", {}
        ).get(
            "occupancy_classification",
            current_state.get("occupancy_classification", "Unknown"),
        )
        if old_occ != new_occ and new_occ != "Unknown":
            records.append(
                self.create_record(
                    triggering_event="OCCUPANCY_STATE_CHANGED",
                    previous_state={"occupancy_classification": old_occ},
                    new_state={"occupancy_classification": new_occ},
                    data_provenance=data_provenance,
                    notes=f"Occupancy transition: {old_occ} -> {new_occ}",
                    timestamp=ts,
                )
            )

        # 4. Sensor staleness / recovery
        old_stale = previous_state.get(
            "data_quality", {}
        ).get(
            "sensor_stale",
            previous_state.get("room_state", {}).get("is_sensor_stale", False),
        )
        new_stale = current_state.get(
            "data_quality", {}
        ).get(
            "sensor_stale",
            current_state.get("room_state", {}).get("is_sensor_stale", False),
        )
        if not old_stale and new_stale:
            records.append(
                self.create_record(
                    triggering_event="SENSOR_STALE",
                    previous_state={"is_sensor_stale": False},
                    new_state={"is_sensor_stale": True},
                    data_provenance=data_provenance,
                    validity_status="STALE",
                    notes="Sensors marked stale due to age timeout",
                    timestamp=ts,
                )
            )
        elif old_stale and not new_stale:
            records.append(
                self.create_record(
                    triggering_event="SENSOR_RECOVERED",
                    previous_state={"is_sensor_stale": True},
                    new_state={"is_sensor_stale": False},
                    data_provenance=data_provenance,
                    validity_status="VALID",
                    notes="Fresh telemetry received; sensor staleness cleared",
                    timestamp=ts,
                )
            )

        # 5. Environmental changes (temperature or humidity update)
        old_temp = previous_state.get("room_state", {}).get("temperature_c")
        new_temp = current_state.get("room_state", {}).get("temperature_c")
        if old_temp != new_temp and new_temp is not None:
            records.append(
                self.create_record(
                    triggering_event="ENVIRONMENTAL_CHANGED",
                    previous_state={"temperature_c": old_temp},
                    new_state={"temperature_c": new_temp},
                    data_provenance=data_provenance,
                    notes=f"Temperature reading updated: {old_temp} -> {new_temp}",
                    timestamp=ts,
                )
            )

        # 5. Explicit trigger hint if provided and not already logged
        if trigger_hint and not any(r.triggering_event == trigger_hint for r in records):
            records.append(
                self.create_record(
                    triggering_event=trigger_hint,
                    previous_state=previous_state,
                    new_state=current_state,
                    data_provenance=data_provenance,
                    timestamp=ts,
                )
            )

        return records
