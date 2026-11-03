"""
AACI Adaptive Controller – Thermal Comfort Scorer
=================================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Evaluates candidate fan speeds against thermal comfort predictions from Model 1:
- Computes piecewise DiscomfortPenalty(s) relative to current speed.
- Evaluates per candidate speed using provisional airflow mapping.
- Treats XGBoost softmax outputs strictly as uncalibrated probabilities.
- Implements uncertainty downweighting (alpha_eff) when max probability < 0.60 or < 0.40.
- Falls back to uniform DiscomfortPenalty = 0.33 when Model 1 is unavailable.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from aaci_ml.system_architecture.adapters.constants import PROVISIONAL_SPEED_VELOCITY_MAP
from aaci_ml.system_architecture.adapters.thermal_comfort_adapter import ThermalComfortAdapter
from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.controller.constants import (
    COMFORT_UNCERTAINTY_HIGH_THRESHOLD,
    COMFORT_UNCERTAINTY_SEVERE_THRESHOLD,
    COMFORT_UNIFORM_PENALTY_FALLBACK,
)


class ComfortScorer:
    """Evaluates multi-speed candidate discomfort penalties."""

    def __init__(self, adapter: Optional[ThermalComfortAdapter] = None):
        self.adapter = adapter or ThermalComfortAdapter()

    def score_candidates(
        self,
        candidate_speeds: List[int],
        current_speed: int,
        csr: Union[CommonState, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates DiscomfortPenalty(s) for each candidate speed:
        - s > current_speed: penalty = P(Cold)
        - s < current_speed: penalty = P(Hot)
        - s == current_speed: penalty = 1.0 - P(Neutral)
        """
        results: Dict[int, Dict[str, Any]] = {}
        is_model_available = True
        overall_uncertainty = False
        min_alpha_mult = 1.0
        discomfort_penalties: Dict[int, float] = {}
        explanations: List[str] = []

        # Convert CSR to dict if necessary
        csr_dict = csr.to_dict() if hasattr(csr, "to_dict") else dict(csr)
        env = csr_dict.get("environmental", {})
        temp = env.get("temperature_c") or csr_dict.get("temperature")
        rh = env.get("relative_humidity_pct") or csr_dict.get("relative_humidity")

        # If temperature or humidity missing / invalid, comfort model cannot be invoked
        if temp is None or rh is None or temp == 255 or rh == 255 or temp <= 0 or rh <= 0:
            is_model_available = False
            for s in candidate_speeds:
                discomfort_penalties[s] = COMFORT_UNIFORM_PENALTY_FALLBACK
                results[s] = {
                    "speed": s,
                    "discomfort_penalty": COMFORT_UNIFORM_PENALTY_FALLBACK,
                    "model_available": False,
                    "predicted_label": "Unknown",
                    "probabilities_uncalibrated": None,
                    "max_probability": 0.0,
                    "is_uncertain": True,
                    "alpha_effective_mult": 0.33,
                    "rationale": "Thermal comfort model unavailable due to missing/invalid environmental sensors. Uniform 0.33 penalty applied.",
                }
            return {
                "candidate_scores": results,
                "discomfort_penalties": discomfort_penalties,
                "model_available": False,
                "overall_uncertain": True,
                "effective_alpha_mult": 0.33,
                "comfort_summary": "Thermal comfort model unavailable. Comfort-neutral scoring applied.",
            }

        # Model 1 available: evaluate candidate speeds
        for s in candidate_speeds:
            prov_v = PROVISIONAL_SPEED_VELOCITY_MAP.get(s, 0.20)
            candidate_inputs = dict(csr_dict)
            if "environmental" in candidate_inputs and isinstance(candidate_inputs["environmental"], dict):
                candidate_inputs["environmental"] = dict(candidate_inputs["environmental"])
                candidate_inputs["environmental"]["air_velocity_ms"] = prov_v
            else:
                candidate_inputs["environmental"] = {"temperature_c": temp, "relative_humidity_pct": rh, "air_velocity_ms": prov_v}

            if "fan_state" in candidate_inputs and isinstance(candidate_inputs["fan_state"], dict):
                candidate_inputs["fan_state"] = dict(candidate_inputs["fan_state"])
                candidate_inputs["fan_state"]["current_speed"] = s
            else:
                candidate_inputs["fan_state"] = {"current_speed": s}

            candidate_inputs["air_velocity_ms"] = prov_v
            candidate_inputs["fan_speed"] = s

            try:
                prediction = self.adapter.predict(candidate_inputs)
                if hasattr(prediction, "normalized_payload"):
                    payload = prediction.normalized_payload
                    model_avail = prediction.model_availability.is_available
                elif isinstance(prediction, dict):
                    payload = prediction.get("normalized_payload", prediction)
                    model_avail = prediction.get("model_available", True)
                else:
                    payload = {}
                    model_avail = False

                if not model_avail or payload.get("fallback_triggered") and payload.get("comfort_label") == "Unknown":
                    raise RuntimeError("Model 1 inference returned unavailable status")

                probs = payload.get("comfort_probabilities") or payload.get("class_probabilities_uncalibrated") or {}
                p_cold = float(probs.get("Cold", 0.0))
                p_neutral = float(probs.get("Neutral", 0.0))
                p_hot = float(probs.get("Hot", 0.0))
                max_p = float(payload.get("max_probability") or max([p_cold, p_neutral, p_hot] or [0.0]))
                label = payload.get("comfort_label") or payload.get("predicted_comfort_label", "Neutral")

                # Determine effective alpha multiplier
                if max_p < COMFORT_UNCERTAINTY_SEVERE_THRESHOLD:
                    alpha_mult = 0.25
                    is_unc = True
                elif max_p < COMFORT_UNCERTAINTY_HIGH_THRESHOLD:
                    alpha_mult = 0.50
                    is_unc = True
                else:
                    alpha_mult = 1.0
                    is_unc = False

                if is_unc:
                    overall_uncertainty = True
                if alpha_mult < min_alpha_mult:
                    min_alpha_mult = alpha_mult

                # Piecewise DiscomfortPenalty
                if s > current_speed:
                    penalty = p_cold
                    branch = "s > s_current (Cold penalty)"
                elif s < current_speed:
                    penalty = p_hot
                    branch = "s < s_current (Hot penalty)"
                else:
                    penalty = max(0.0, 1.0 - p_neutral)
                    branch = "s == s_current (1 - P(Neutral))"

                discomfort_penalties[s] = round(penalty, 4)
                results[s] = {
                    "speed": s,
                    "discomfort_penalty": round(penalty, 4),
                    "branch_evaluated": branch,
                    "model_available": True,
                    "predicted_label": label,
                    "probabilities_uncalibrated": probs,
                    "max_probability": round(max_p, 4),
                    "is_uncertain": is_unc,
                    "alpha_effective_mult": alpha_mult,
                    "provisional_air_velocity_ms": prov_v,
                }
            except Exception as e:
                # Fallback for individual candidate failure
                is_model_available = False
                overall_uncertainty = True
                min_alpha_mult = 0.33
                discomfort_penalties[s] = COMFORT_UNIFORM_PENALTY_FALLBACK
                results[s] = {
                    "speed": s,
                    "discomfort_penalty": COMFORT_UNIFORM_PENALTY_FALLBACK,
                    "model_available": False,
                    "error": str(e),
                    "is_uncertain": True,
                    "alpha_effective_mult": 0.33,
                    "rationale": "Exception during Model 1 evaluation; uniform 0.33 penalty applied.",
                }

        summary = (
            "Thermal comfort evaluated per candidate speed using provisional airflow mapping."
            if is_model_available
            else "Thermal comfort model unavailable. Comfort-neutral scoring applied."
        )

        return {
            "candidate_scores": results,
            "discomfort_penalties": discomfort_penalties,
            "model_available": is_model_available,
            "overall_uncertain": overall_uncertainty,
            "effective_alpha_mult": min_alpha_mult,
            "comfort_summary": summary,
        }
