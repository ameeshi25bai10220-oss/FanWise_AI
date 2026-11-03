"""
AACI Fan Digital Twin – Historical State Manager
================================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Maintains bounded circular buffers for:
- State transitions (default N=20)
- Periodic full-state snapshots (default N=50)

Enforces chronological ordering, per-fan isolation, duplicate event rejection,
and structured state recovery.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from aaci_ml.system_architecture.digital_twin.constants import (
    DEFAULT_SNAPSHOT_HISTORY_CAPACITY,
    DEFAULT_TRANSITION_HISTORY_CAPACITY,
)
from aaci_ml.system_architecture.digital_twin.transition_engine import StateTransitionRecord


class HistoricalStateManager:
    """Manages history buffers and temporal ordering for a single fan twin."""

    def __init__(
        self,
        fan_id: str,
        transition_capacity: int = DEFAULT_TRANSITION_HISTORY_CAPACITY,
        snapshot_capacity: int = DEFAULT_SNAPSHOT_HISTORY_CAPACITY,
    ):
        self.fan_id = fan_id
        self.transition_capacity = transition_capacity
        self.snapshot_capacity = snapshot_capacity
        self._transitions: Deque[StateTransitionRecord] = deque(maxlen=transition_capacity)
        self._snapshots: Deque[Dict[str, Any]] = deque(maxlen=snapshot_capacity)
        # Seen transition keys for duplicate rejection: (timestamp, triggering_event)
        self._seen_transition_keys: Deque[str] = deque(maxlen=100)

    def _make_key(self, record: StateTransitionRecord) -> str:
        return f"{record.timestamp}::{record.triggering_event}"

    def append_transition(self, record: StateTransitionRecord) -> bool:
        """
        Appends a state transition record.
        Rejects duplicate transitions with identical timestamp and trigger event.
        Maintains chronological order.
        Returns True if appended, False if rejected as duplicate.
        """
        if record.fan_id != self.fan_id:
            raise ValueError(f"Fan ID mismatch: expected {self.fan_id}, got {record.fan_id}")

        key = self._make_key(record)
        if key in self._seen_transition_keys:
            return False

        self._seen_transition_keys.append(key)

        # Check for out-of-order insertion
        if self._transitions and record.timestamp < self._transitions[-1].timestamp:
            # Insert maintaining chronological sort
            items = list(self._transitions)
            items.append(record)
            items.sort(key=lambda r: r.timestamp)
            # Rebuild deque trimmed to maxlen
            self._transitions.clear()
            for r in items[-self.transition_capacity:]:
                self._transitions.append(r)
        else:
            self._transitions.append(record)

        return True

    def record_snapshot(self, snapshot: Dict[str, Any], timestamp: Optional[str] = None) -> None:
        """Records a periodic full-state snapshot."""
        ts = (
            timestamp
            if timestamp is not None
            else snapshot.get("last_updated", datetime.now(timezone.utc).isoformat())
        )
        snap = dict(snapshot)
        snap["snapshot_timestamp"] = ts
        snap["fan_id"] = self.fan_id

        # Check for out-of-order snapshot
        if self._snapshots and ts < self._snapshots[-1].get("snapshot_timestamp", ""):
            items = list(self._snapshots)
            items.append(snap)
            items.sort(key=lambda s: s.get("snapshot_timestamp", ""))
            self._snapshots.clear()
            for s in items[-self.snapshot_capacity:]:
                self._snapshots.append(s)
        else:
            self._snapshots.append(snap)

    def get_transitions(self) -> List[StateTransitionRecord]:
        """Returns all transitions in chronological order."""
        return list(self._transitions)

    def get_recent_transitions(self, n: int = 5) -> List[StateTransitionRecord]:
        """Returns the most recent n transitions."""
        items = list(self._transitions)
        return items[-n:] if n <= len(items) else items

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Returns all snapshots in chronological order."""
        return list(self._snapshots)

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent snapshot if available."""
        return self._snapshots[-1] if self._snapshots else None

    def clear(self) -> None:
        """Clears all history buffers."""
        self._transitions.clear()
        self._snapshots.clear()
        self._seen_transition_keys.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes history manager state to dictionary."""
        return {
            "fan_id": self.fan_id,
            "transition_count": len(self._transitions),
            "transition_capacity": self.transition_capacity,
            "snapshot_count": len(self._snapshots),
            "snapshot_capacity": self.snapshot_capacity,
            "transitions": [t.to_dict() for t in self._transitions],
            "snapshots": list(self._snapshots),
        }

