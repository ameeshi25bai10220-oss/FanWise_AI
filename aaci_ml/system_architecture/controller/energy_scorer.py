"""
AACI Adaptive Controller – Energy Scorer
========================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Evaluates candidate fan speeds for energy efficiency:
- Computes relative EnergyScore(s) normalized to [0, 1] using deterministic research physics tables.
- Evaluates Model 3 LightGBM next-packet advisory energy forecast.
- Enforces strict advisory constraints: cannot command speed, cannot guarantee savings (R^2 approx 0.1515).
- Falls back gracefully to deterministic physics-only scoring during cold-start or adapter unavailability.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from aaci_ml.system_architecture.adapters.constants import BLDC_POWER_MAP, INDUCTION_POWER_MAP
from aaci_ml.system_architecture.adapters.energy_adapter import EnergyIntelligenceAdapter
from aaci_ml.system_architecture.common_state.common_state import CommonState


class EnergyScorer:
    """Evaluates multi-speed physics energy scores and ML advisory forecasts."""

    def __init__(self, adapter: Optional[EnergyIntelligenceAdapter] = None):
        self.adapter = adapter or EnergyIntelligenceAdapter()

    def score_candidates(
        self,
        candidate_speeds: List[int],
        fan_mode: str = "BLDC",
        csr: Optional[Union[CommonState, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates EnergyScore(s) for each candidate speed:
        EnergyScore(s) = (P(s) - P_min) / (P_max - P_min) for s in [1..5], and 0.0 for s=0.
        Also retrieves the Model 3 advisory forecast if available.
        """
        power_map = BLDC_POWER_MAP if str(fan_mode).upper() == "BLDC" else INDUCTION_POWER_MAP
        non_zero_powers = [power_map[s] for s in range(1, 6)]
        p_min = min(non_zero_powers)
        p_max = max(non_zero_powers)
        p_range = max(1e-5, p_max - p_min)

        energy_scores: Dict[int, float] = {}
        candidate_power_w: Dict[int, float] = {}

        for s in candidate_speeds:
            p_w = power_map.get(s, 0.0)
            candidate_power_w[s] = p_w
            if s == 0:
                energy_scores[0] = 0.0
            else:
                score = (p_w - p_min) / p_range
                energy_scores[s] = round(max(0.0, min(1.0, score)), 4)

        # Retrieve Model 3 advisory forecast if CSR provided
        ml_forecast_wh: Optional[float] = None
        ml_available = False
        ml_cold_start = True

        if csr is not None:
            csr_dict = csr.to_dict() if hasattr(csr, "to_dict") else dict(csr)
            try:
                pred_envelope = self.adapter.predict(csr_dict)
                if hasattr(pred_envelope, "normalized_payload"):
                    pred = pred_envelope.normalized_payload
                    ml_available = pred_envelope.model_availability.is_available
                elif isinstance(pred_envelope, dict):
                    pred = pred_envelope.get("normalized_payload", pred_envelope)
                    ml_available = pred_envelope.get("model_available", True)
                else:
                    pred = {}
                    ml_available = False

                if ml_available:
                    ml_forecast_wh = pred.get("predicted_next_energy_wh") or pred.get("predicted_energy_wh")
                    ml_cold_start = bool(pred.get("cold_start", False))
            except Exception:
                ml_available = False
                ml_cold_start = True

        return {
            "energy_scores": energy_scores,
            "candidate_power_w": candidate_power_w,
            "fan_mode": fan_mode,
            "power_map_source": "nominal_research_dataset_constants_uncalibrated",
            "ml_advisory": {
                "predicted_next_energy_wh": ml_forecast_wh,
                "is_advisory": True,
                "cannot_command_speed": True,
                "r2_benchmark": 0.1515,
                "ml_available": ml_available,
                "ml_cold_start": ml_cold_start,
                "disclaimer": "Advisory only (R^2 approx 0.1515). Variable-horizon next packet forecast. Cannot command speed.",
            },
            "summary": (
                f"Relative energy scored via {fan_mode} physics map (P_min={p_min}W, P_max={p_max}W). "
                f"Advisory ML next packet forecast: {ml_forecast_wh if ml_forecast_wh is not None else 'N/A'} Wh."
            ),
        }
