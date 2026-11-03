"""
AACI Fan Digital Twin – Core Runtime
====================================
Phase 5 – Stage 3: Fan Digital Twin Runtime Implementation

Virtual digital twin runtime for a ceiling fan:
- Maintains virtual fan state, room state, and energy accounting.
- Consumes Common State Representation (CSR) and adapter envelopes.
- Enforces strict 4-way energy segregation.
- Logs transition records and circular snapshot history.
- Operates strictly as a software state object (no hardware actuation).
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Union

from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.digital_twin.constants import (
    DEFAULT_MANUAL_OVERRIDE_TIMEOUT_SECONDS,
    RESIDENTIAL_FAN_ROOM_CATEGORIES,
    SYNTHETIC_SIMULATION_PROVENANCE_TAG,
)
from aaci_ml.system_architecture.digital_twin.history_manager import HistoricalStateManager
from aaci_ml.system_architecture.digital_twin.state_managers import (
    EnergyStateManager,
    FanStateManager,
    RoomStateManager,
)
from aaci_ml.system_architecture.digital_twin.transition_engine import (
    StateTransitionEngine,
    StateTransitionRecord,
)


class FanDigitalTwin:
    """
    Virtual in-memory digital twin of a single ceiling fan and room.
    Matches the schema defined in fan_digital_twin_design.md §3.1.
    """

    def __init__(
        self,
        fan_id: str = "Bedroom_Fan_1",
        room_category: Optional[str] = None,
        fan_mode: str = "BLDC",
        initial_speed: int = 0,
        transition_capacity: int = 20,
        snapshot_capacity: int = 50,
    ):
        self.fan_id = fan_id
        if room_category is None:
            room_category = RESIDENTIAL_FAN_ROOM_CATEGORIES.get(fan_id, "Bedroom")
        self.room_category = room_category
        self.fan_mode = fan_mode

        # Component managers
        self.fan_state = FanStateManager(
            fan_id=fan_id,
            room_category=room_category,
            fan_mode=fan_mode,
            current_speed=initial_speed,
            proposed_speed=initial_speed,
        )
        self.room_state = RoomStateManager()
        self.energy_state = EnergyStateManager()
        self.transition_engine = StateTransitionEngine(fan_id=fan_id)
        self.history_manager = HistoricalStateManager(
            fan_id=fan_id,
            transition_capacity=transition_capacity,
            snapshot_capacity=snapshot_capacity,
        )

        # Operational tracking
        self.last_decision: Optional[Dict[str, Any]] = None
        self.last_update_timestamp: float = time.time()
        self.last_updated_iso: str = datetime.now(timezone.utc).isoformat()

        # Data quality flags
        self.data_quality: Dict[str, Any] = {
            "last_sensor_age_s": 0.0,
            "sensor_stale": False,
            "sentinel_255_detected": False,
            "light_available": False,
            "co2_available": False,
            "sensor_gap_active": True,
            "ml_cold_start": True,
        }

        # Provisional & disclaimer flags
        self.provisional_flags: Dict[str, Any] = {
            "airflow_mapping_provisional": True,
            "airflow_source": "Dryad_open_access_dataset_1.1m_height",
            "comfort_probabilities_uncalibrated": True,
            "energy_forecast_r2": 0.1515,
            "hardware_validation_performed": False,
        }

        # Initialize energy physics baseline
        self.energy_state.update_physics_energy(initial_speed, fan_mode, 0.0)

        # Log initial state
        init_snapshot = self.inspect_state()
        transitions = self.transition_engine.detect_transitions(
            previous_state=None,
            current_state=init_snapshot,
            trigger_hint="INITIALIZATION",
            data_provenance="INITIALIZATION",
        )
        for t in transitions:
            self.history_manager.append_transition(t)
        self.history_manager.record_snapshot(init_snapshot)

    def update_from_csr(
        self,
        csr: Union[CommonState, Dict[str, Any]],
        timestamp: Optional[float] = None,
        data_provenance: str = "CSR_TELEMETRY",
    ) -> List[StateTransitionRecord]:
        """
        Updates virtual twin state from a Common State Representation.
        Updates room state, fan speed (if confirmed in telemetry),
        accumulates physics energy, detects transitions, and appends to history.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        elapsed_s = max(0.0, curr_time - self.last_update_timestamp)
        self.last_update_timestamp = curr_time
        self.last_updated_iso = (
            datetime.fromtimestamp(curr_time, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )

        old_snapshot = self.inspect_state()

        # 1. Update Room State
        if isinstance(csr, CommonState):
            csr_dict = csr.to_dict()
        else:
            csr_dict = csr
        self.room_state.update_from_csr(csr_dict)

        # 2. Update Fan Speed if present in telemetry / CSR
        raw_fan_speed = csr_dict.get("fan_state", {}).get("current_speed")
        if raw_fan_speed is None:
            raw_fan_speed = csr_dict.get("fan", {}).get("current_speed")
        if raw_fan_speed is None:
            raw_fan_speed = csr_dict.get("current_speed")
        if raw_fan_speed is None:
            raw_fan_speed = csr_dict.get("fan_speed")
        if raw_fan_speed is not None and not self.fan_state.is_override_active(curr_time):
            self.fan_state.confirm_speed(int(raw_fan_speed), timestamp=curr_time)

        # Check pending timeout
        self.fan_state.check_pending_timeout(curr_time)

        # 3. Update Physics Energy Accumulator
        self.energy_state.update_physics_energy(
            self.fan_state.current_speed, self.fan_mode, elapsed_s
        )

        # 4. Check raw telemetry energy
        telemetry_wh = csr_dict.get("energy", {}).get("measured_espent_wh")
        if telemetry_wh is None:
            telemetry_wh = csr_dict.get("espent_wh")
        if telemetry_wh is not None:
            self.energy_state.record_measured_telemetry(
                telemetry_wh, timestamp=curr_time
            )

        # 5. Update Data Quality
        env_state = csr_dict.get("environmental", {})
        meta_state = csr_dict.get("metadata", {})
        occ_state = csr_dict.get("occupancy_signals", csr_dict.get("occupancy", {}))
        health_state = csr_dict.get("system_health", csr_dict.get("health", {}))

        sensor_age = env_state.get("sensor_age_s") or meta_state.get("sensor_age_s") or meta_state.get("sensor_age_seconds", 0.0)
        is_stale = env_state.get("is_stale") if "is_stale" in env_state else (sensor_age > 300.0)
        missing_sensors = health_state.get("missing_sensors", env_state.get("missing_sensors", []))

        self.data_quality["last_sensor_age_s"] = float(sensor_age or 0.0)
        self.data_quality["sensor_stale"] = bool(is_stale)

        if "light_available" in occ_state:
            self.data_quality["light_available"] = bool(occ_state["light_available"])
        else:
            self.data_quality["light_available"] = "light" not in missing_sensors

        if "co2_available" in occ_state:
            self.data_quality["co2_available"] = bool(occ_state["co2_available"])
        else:
            self.data_quality["co2_available"] = "co2" not in missing_sensors

        if "sensor_gap_active" in occ_state:
            self.data_quality["sensor_gap_active"] = bool(occ_state["sensor_gap_active"])
        else:
            self.data_quality["sensor_gap_active"] = (
                not self.data_quality["light_available"]
                or not self.data_quality["co2_available"]
            )
        self.data_quality["ml_cold_start"] = self.energy_state.ml_cold_start

        # 6. Detect & Record Transitions
        new_snapshot = self.inspect_state()
        transitions = self.transition_engine.detect_transitions(
            previous_state=old_snapshot,
            current_state=new_snapshot,
            data_provenance=data_provenance,
            timestamp=self.last_updated_iso,
        )
        for t in transitions:
            self.history_manager.append_transition(t)

        self.history_manager.record_snapshot(new_snapshot, timestamp=self.last_updated_iso)
        return transitions

    def update_from_telemetry(
        self,
        raw_telemetry: Dict[str, Any],
        timestamp: Optional[float] = None,
        data_provenance: str = "TELEMETRY",
    ) -> List[StateTransitionRecord]:
        """Convenience method: constructs CSR from telemetry dictionary, then updates twin."""
        telemetry = dict(raw_telemetry)
        if "fan_id" not in telemetry:
            telemetry["fan_id"] = self.fan_id
        if "provenance" not in telemetry:
            telemetry["provenance"] = data_provenance

        if timestamp is None:
            if "timestamp_s" in raw_telemetry:
                timestamp = float(raw_telemetry["timestamp_s"])
            elif "timestamp" in raw_telemetry and isinstance(raw_telemetry["timestamp"], (int, float)):
                timestamp = float(raw_telemetry["timestamp"])

        if timestamp is not None and "timestamp" not in telemetry:
            telemetry["timestamp"] = (
                datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            )
        csr = CommonState.from_telemetry(raw_telemetry=telemetry)
        return self.update_from_csr(
            csr=csr, timestamp=timestamp, data_provenance=data_provenance
        )

    def update_from_adapter_envelope(
        self,
        adapter_type: str,
        envelope: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> List[StateTransitionRecord]:
        """
        Enriches the Digital Twin with prediction envelopes from frozen adapters:
        - 'comfort': Model 1 thermal comfort assessment & class probabilities
        - 'occupancy': Model 2 occupancy classification, headcount & degraded status
        - 'energy': Model 3 advisory energy forecast (is_advisory=True, R^2 approx 0.1515)
        """
        curr_time = timestamp if timestamp is not None else time.time()
        self.last_updated_iso = (
            datetime.fromtimestamp(curr_time, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )
        old_snapshot = self.inspect_state()
        atype = adapter_type.lower()

        if "comfort" in atype:
            if "predicted_comfort_label" in envelope:
                self.room_state.comfort_assessment = envelope["predicted_comfort_label"]
            elif "comfort_class" in envelope:
                self.room_state.comfort_assessment = envelope["comfort_class"]

        elif "occupancy" in atype:
            if "occupancy_state" in envelope:
                self.room_state.occupancy_classification = envelope["occupancy_state"]
            elif "classification" in envelope:
                self.room_state.occupancy_classification = envelope["classification"]
            if "estimated_headcount" in envelope:
                self.room_state.estimated_headcount = envelope["estimated_headcount"]
            if "feature_set_used" in envelope:
                self.room_state.occupancy_feature_set = envelope["feature_set_used"]

        elif "energy" in atype:
            self.energy_state.record_ml_forecast(envelope)
            self.data_quality["ml_cold_start"] = self.energy_state.ml_cold_start

        new_snapshot = self.inspect_state()
        transitions = self.transition_engine.detect_transitions(
            previous_state=old_snapshot,
            current_state=new_snapshot,
            trigger_hint=f"ADAPTER_UPDATE_{atype.upper()}",
            data_provenance=envelope.get("data_provenance", "ADAPTER_ENVELOPE"),
            timestamp=self.last_updated_iso,
        )
        for t in transitions:
            self.history_manager.append_transition(t)

        self.history_manager.record_snapshot(new_snapshot, timestamp=self.last_updated_iso)
        return transitions

    def set_manual_override(
        self,
        speed: int,
        duration_s: float = DEFAULT_MANUAL_OVERRIDE_TIMEOUT_SECONDS,
        timestamp: Optional[float] = None,
    ) -> StateTransitionRecord:
        """
        Sets user manual override flag in virtual state.
        DOES NOT send physical actuation commands to hardware.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        ts_iso = (
            datetime.fromtimestamp(curr_time, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )
        old_snapshot = self.inspect_state()
        self.fan_state.set_manual_override(speed, duration_s=duration_s, timestamp=curr_time)
        new_snapshot = self.inspect_state()

        record = self.transition_engine.create_record(
            triggering_event="MANUAL_OVERRIDE_SET",
            previous_state={"current_speed": old_snapshot["current_speed"]},
            new_state={"current_speed": speed, "manual_override_active": True},
            data_provenance="USER_INTERACTION",
            notes=f"Manual override set to speed {speed} for {duration_s}s",
            timestamp=ts_iso,
        )
        self.history_manager.append_transition(record)
        self.history_manager.record_snapshot(new_snapshot, timestamp=ts_iso)
        return record

    def clear_manual_override(self, timestamp: Optional[float] = None) -> Optional[StateTransitionRecord]:
        """Clears manual override state flag."""
        if not self.fan_state.manual_override_active:
            return None
        curr_time = timestamp if timestamp is not None else time.time()
        ts_iso = (
            datetime.fromtimestamp(curr_time, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )
        old_snapshot = self.inspect_state()
        self.fan_state.clear_manual_override(timestamp=curr_time)
        new_snapshot = self.inspect_state()

        record = self.transition_engine.create_record(
            triggering_event="MANUAL_OVERRIDE_EXPIRED",
            previous_state={"manual_override_active": True},
            new_state={"manual_override_active": False},
            data_provenance="USER_INTERACTION",
            notes="Manual override cleared",
            timestamp=ts_iso,
        )
        self.history_manager.append_transition(record)
        self.history_manager.record_snapshot(new_snapshot, timestamp=ts_iso)
        return record

    def record_decision(
        self,
        proposed_speed: int,
        rationale: str = "",
        decision_mode: str = "adaptive",
        cost_score: Optional[float] = None,
        comfort_signal: Optional[Dict[str, Any]] = None,
        occupancy_signal: Optional[Dict[str, Any]] = None,
        energy_signal: Optional[Dict[str, Any]] = None,
        fallback_triggered: bool = False,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Records a decision outcome from the controller.
        Sets proposed_speed and speed_change_pending flag if proposed_speed != current_speed.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        ts_iso = (
            datetime.fromtimestamp(curr_time, timezone.utc).isoformat()
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )
        self.fan_state.propose_speed(proposed_speed, timestamp=curr_time)
        self.last_decision = {
            "timestamp": ts_iso,
            "proposed_speed": proposed_speed,
            "decision_mode": decision_mode,
            "rationale": rationale,
            "cost_score": cost_score,
            "comfort_signal": comfort_signal or {},
            "occupancy_signal": occupancy_signal or {},
            "energy_signal": energy_signal or {},
            "fallback_triggered": fallback_triggered,
        }

    def confirm_speed(self, speed: int, timestamp: Optional[float] = None) -> List[StateTransitionRecord]:
        """Confirms telemetry feedback matching proposed or updated speed."""
        curr_time = timestamp if timestamp is not None else time.time()
        old_snapshot = self.inspect_state()
        self.fan_state.confirm_speed(speed, timestamp=curr_time)
        new_snapshot = self.inspect_state()

        transitions = self.transition_engine.detect_transitions(
            previous_state=old_snapshot,
            current_state=new_snapshot,
            data_provenance="TELEMETRY_CONFIRMATION",
            timestamp=datetime.fromtimestamp(curr_time, timezone.utc).isoformat(),
        )
        for t in transitions:
            self.history_manager.append_transition(t)
        self.history_manager.record_snapshot(new_snapshot)
        return transitions

    def inspect_state(self) -> Dict[str, Any]:
        """
        Serializes full virtual state strictly compliant with:
        fan_digital_twin_design.md §3.1.
        """
        transitions = self.history_manager.get_transitions()
        formatted_history = [
            {
                "timestamp": t.timestamp,
                "trigger": t.triggering_event,
                "provenance": t.data_provenance,
                "validity_status": t.validity_status,
                "notes": t.notes,
                "previous_state": t.previous_state,
                "new_state": t.new_state,
            }
            for t in transitions
        ]

        return {
            "fan_id": self.fan_id,
            "room_category": self.room_category,
            "fan_mode": self.fan_mode,
            "current_speed": self.fan_state.current_speed,
            "proposed_speed": self.fan_state.proposed_speed,
            "is_on": self.fan_state.is_on,
            "speed_change_pending": self.fan_state.speed_change_pending,
            "estimated_power_w": self.energy_state.estimated_power_w,
            "predicted_next_energy_wh": self.energy_state.predicted_next_energy_wh,
            "predicted_next_energy_source": self.energy_state.predicted_next_energy_source,
            "predicted_next_energy_r2_benchmark": self.energy_state.predicted_next_energy_r2_benchmark,
            "cumulative_energy_session_wh": round(
                self.energy_state.cumulative_energy_session_wh, 4
            ),
            "cumulative_energy_lifetime_wh": round(
                self.energy_state.cumulative_energy_lifetime_wh, 4
            ),
            "session_start_timestamp": self.fan_state.session_start_timestamp,
            "state_transition_history": formatted_history,
            "last_decision": self.last_decision,
            "last_updated": self.last_updated_iso,
            "data_quality": dict(self.data_quality),
            "provisional_flags": dict(self.provisional_flags),
            "room_state": self.room_state.to_dict(),
            "energy_accounting_segregated": self.energy_state.to_dict(),
        }
