"""
AACI Fan Digital Twin – Fleet Supervisor
========================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Manages a multi-fan fleet of independent FanDigitalTwin instances:
- One instance per fan device, keyed strictly by fan_id.
- Isolated state lifecycle across fans.
- Dynamic instantiation upon observing new fan IDs.
- Multi-fan fleet inspection and routing.
"""

from typing import Any, Dict, List, Optional, Union

from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.digital_twin.constants import RESIDENTIAL_FAN_ROOM_CATEGORIES
from aaci_ml.system_architecture.digital_twin.fan_digital_twin import FanDigitalTwin
from aaci_ml.system_architecture.digital_twin.transition_engine import StateTransitionRecord


class TwinSupervisor:
    """Manages active fleet of FanDigitalTwin instances."""

    def __init__(self):
        self._twins: Dict[str, FanDigitalTwin] = {}

    def get_or_create_twin(
        self,
        fan_id: str,
        room_category: Optional[str] = None,
        fan_mode: str = "BLDC",
    ) -> FanDigitalTwin:
        """Retrieves existing twin for fan_id or instantiates a new one."""
        if fan_id not in self._twins:
            if room_category is None:
                room_category = RESIDENTIAL_FAN_ROOM_CATEGORIES.get(fan_id, "Bedroom")
            self._twins[fan_id] = FanDigitalTwin(
                fan_id=fan_id,
                room_category=room_category,
                fan_mode=fan_mode,
            )
        return self._twins[fan_id]

    def get_twin(self, fan_id: str) -> Optional[FanDigitalTwin]:
        """Returns twin instance for fan_id if present, else None."""
        return self._twins.get(fan_id)

    def list_active_fans(self) -> List[str]:
        """Returns sorted list of registered fan IDs."""
        return sorted(list(self._twins.keys()))

    def route_csr(
        self,
        csr: Union[CommonState, Dict[str, Any]],
        fan_id: Optional[str] = None,
        data_provenance: str = "CSR_ROUTED",
    ) -> List[StateTransitionRecord]:
        """Routes a Common State Representation to the target fan twin."""
        target_id = fan_id
        if target_id is None:
            if isinstance(csr, CommonState):
                target_id = csr.fan_id
            elif isinstance(csr, dict):
                target_id = csr.get("fan_id") or csr.get("fan", {}).get("fan_id")

        if not target_id:
            target_id = "Default_Fan_1"

        twin = self.get_or_create_twin(target_id)
        return twin.update_from_csr(csr, data_provenance=data_provenance)

    def route_telemetry(
        self,
        raw_telemetry: Dict[str, Any],
        fan_id: Optional[str] = None,
        data_provenance: str = "TELEMETRY",
    ) -> List[StateTransitionRecord]:
        """Routes raw telemetry packet to target fan twin."""
        target_id = fan_id or raw_telemetry.get("fan_id") or "Default_Fan_1"
        twin = self.get_or_create_twin(target_id)
        return twin.update_from_telemetry(
            raw_telemetry=raw_telemetry, data_provenance=data_provenance
        )

    def inspect_all_twins(self) -> Dict[str, Dict[str, Any]]:
        """Returns current virtual state for all managed fan twins."""
        return {fan_id: twin.inspect_state() for fan_id, twin in self._twins.items()}

    def clear_all(self) -> None:
        """Clears all twin instances."""
        self._twins.clear()

