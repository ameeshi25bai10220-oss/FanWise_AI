"""
AACI System Architecture – Common Adapter Output Contract
=========================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Defines the common output envelope for all AACI intelligence model adapters.
Enforces strict provenance distinctions (measured vs. assumed vs. estimated vs. predicted)
and ensures no calibrated uncertainty is fabricated when absent.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
import datetime


@dataclass
class InputValidityStatus:
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


@dataclass
class InputFreshnessInfo:
    sensor_age_seconds: Optional[float] = None
    is_stale: bool = False
    max_staleness_threshold_s: float = 300.0


@dataclass
class ModelAvailabilityStatus:
    is_available: bool = True
    reason_if_unavailable: Optional[str] = None


@dataclass
class ValueProvenance:
    measured: Dict[str, Any] = field(default_factory=dict)
    assumed: Dict[str, Any] = field(default_factory=dict)
    estimated: Dict[str, Any] = field(default_factory=dict)
    predicted: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradedState:
    is_degraded: bool = False
    degradation_reason: Optional[str] = None
    fallback_triggered: bool = False
    fallback_strategy: Optional[str] = None


@dataclass
class AdapterOutputEnvelope:
    """
    Unified output contract returned by all AACI model adapters.
    Strictly differentiates between measured, assumed, estimated, and predicted values.
    """
    model_id: str
    adapter_name: str
    prediction: Any
    prediction_unit: Optional[str]
    timestamp: Optional[str] = None
    input_validity: InputValidityStatus = field(default_factory=InputValidityStatus)
    input_freshness: InputFreshnessInfo = field(default_factory=InputFreshnessInfo)
    model_availability: ModelAvailabilityStatus = field(default_factory=ModelAvailabilityStatus)
    missing_inputs: List[str] = field(default_factory=list)
    value_provenance: ValueProvenance = field(default_factory=ValueProvenance)
    limitations: List[str] = field(default_factory=list)
    degraded_state: DegradedState = field(default_factory=DegradedState)
    raw_model_output: Optional[Dict[str, Any]] = None
    normalized_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire envelope hierarchy into a serializable JSON-compatible dict."""
        return asdict(self)

