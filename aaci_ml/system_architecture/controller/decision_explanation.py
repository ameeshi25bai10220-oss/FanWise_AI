"""
AACI Adaptive Controller – Decision Explanation & Output Contract
=================================================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Defines the structured explanation and serialization contract for controller decisions.
Ensures every recommendation includes:
- Recommendation on a software scale (0-5)
- Previous speed & transition delta
- Timestamp and decision mode
- Modular comfort, occupancy, energy, safety, and override assessments
- Candidate scoring breakdown and tie-break explanations
- Value provenance and mandatory disclaimers
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aaci_ml.system_architecture.controller.constants import MANDATORY_DISCLAIMERS


@dataclass
class DecisionExplanation:
    """Structured decision output contract matching Phase 5 Stage 4 requirements."""

    recommended_speed: int
    previous_speed: Optional[int]
    speed_change: Optional[int]
    decision_timestamp: str
    decision_mode: str
    rationale: str
    comfort_assessment: Dict[str, Any]
    occupancy_assessment: Dict[str, Any]
    energy_assessment: Dict[str, Any]
    safety_status: Dict[str, Any]
    manual_override_status: Optional[Dict[str, Any]] = None
    fallback_status: Dict[str, Any] = field(default_factory=dict)
    candidate_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    provenance_info: Dict[str, Any] = field(default_factory=dict)
    disclaimers: List[str] = field(default_factory=lambda: list(MANDATORY_DISCLAIMERS))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes decision explanation to standard JSON-compatible dictionary."""
        return {
            "recommended_speed": self.recommended_speed,
            "previous_speed": self.previous_speed,
            "speed_change": self.speed_change,
            "decision_timestamp": self.decision_timestamp,
            "decision_mode": self.decision_mode,
            "rationale": self.rationale,
            "comfort_assessment": self.comfort_assessment,
            "occupancy_assessment": self.occupancy_assessment,
            "energy_assessment": self.energy_assessment,
            "safety_status": self.safety_status,
            "manual_override_status": self.manual_override_status,
            "fallback_status": self.fallback_status,
            "candidate_evaluations": self.candidate_evaluations,
            "provenance_info": self.provenance_info,
            "disclaimers": self.disclaimers,
        }

