"""
AACI Adaptive Controller – Occupancy Scorer
===========================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Evaluates candidate fan speeds against occupancy intelligence from Model 2:
- Computes OccupancyBonus(s) rewarding energy-efficient choices based on occupancy certainty.
- Determines whether the Occupancy Hard Gate applies (P(occupied) < 0.35).
- Handles degraded set_b operation during residential sensor gaps (missing light/CO2).
- Preserves ROBOD academic room ID training context without conflating with residential rooms.
- Falls back safely when Model 2 is unavailable (gate not applied, bonus = 0.0).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from aaci_ml.system_architecture.adapters.occupancy_adapter import OccupancyAdapter
from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.controller.constants import (
    OCCUPANCY_HARD_GATE_PROB_THRESHOLD,
    OCCUPANCY_UNCERTAINTY_HIGH,
    OCCUPANCY_UNCERTAINTY_LOW,
)


class OccupancyScorer:
    """Evaluates occupancy relevance, hard gate status, and multi-speed occupancy bonuses."""

    def __init__(self, adapter: Optional[OccupancyAdapter] = None):
        self.adapter = adapter or OccupancyAdapter()

    def score_candidates(
        self,
        candidate_speeds: List[int],
        energy_scores: Dict[int, float],
        csr: Union[CommonState, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculates OccupancyBonus(s) for each candidate speed:
        - If occupied (P >= 0.65): bonus = P(occupied) * (1 - EnergyScore(s))
        - If unoccupied hard gate (P < 0.35): bonus = 0.0
        - If uncertain (0.35 <= P < 0.65): bonus = 0.5 * (1 - EnergyScore(s))
        - If Model 2 unavailable: bonus = 0.0, hard gate disabled
        """
        bonuses: Dict[int, float] = {}
        csr_dict = csr.to_dict() if hasattr(csr, "to_dict") else dict(csr)

        # 1. Attempt prediction with OccupancyAdapter or inspect provided occupancy signals
        occ_signals = csr_dict.get("occupancy_signals") or csr_dict.get("occupancy") or {}
        if "occupancy_probability" in occ_signals and occ_signals["occupancy_probability"] is not None:
            pred = occ_signals
            model_avail = True
        else:
            try:
                pred_envelope = self.adapter.predict(csr_dict)
                if hasattr(pred_envelope, "normalized_payload"):
                    pred = pred_envelope.normalized_payload
                    model_avail = pred_envelope.model_availability.is_available
                elif isinstance(pred_envelope, dict):
                    pred = pred_envelope.get("normalized_payload", pred_envelope)
                    model_avail = pred_envelope.get("model_available", True)
                else:
                    pred = {}
                    model_avail = False
            except Exception:
                pred = {}
                model_avail = False

        if not model_avail:
            for s in candidate_speeds:
                bonuses[s] = 0.0
            return {
                "occupancy_bonuses": bonuses,
                "hard_gate_active": False,
                "model_available": False,
                "occupancy_classification": "Unknown",
                "occupancy_probability": None,
                "is_uncertain": True,
                "effective_delta_mult": 0.0,
                "headcount_estimate": None,
                "feature_set_used": None,
                "robod_context": "None",
                "summary": "Occupancy model unavailable. Occupancy gate not applied. Conservative energy-efficient recommendation used.",
            }

        # Model 2 available
        occ_state = pred.get("occupancy_state", "Unknown")
        occ_prob = float(pred.get("occupancy_probability", 0.50))
        occ_bin = pred.get("occupancy_binary", 1 if occ_prob >= 0.50 else 0)
        is_uncertain = bool(pred.get("is_uncertain", False)) or (
            OCCUPANCY_UNCERTAINTY_LOW <= occ_prob <= OCCUPANCY_UNCERTAINTY_HIGH
        )
        headcount = pred.get("estimated_headcount")
        feat_set = pred.get("feature_set_used", "set_b")
        robod_room = pred.get("robod_room_id")

        # 2. Check Occupancy Hard Gate: unoccupied AND P < 0.35
        hard_gate_active = (occ_bin == 0 and occ_prob < OCCUPANCY_HARD_GATE_PROB_THRESHOLD)

        # 3. Calculate OccupancyBonus(s)
        delta_mult = 0.50 if is_uncertain else 1.0

        for s in candidate_speeds:
            e_score = energy_scores.get(s, 0.0)
            if hard_gate_active:
                bonuses[s] = 0.0
            elif occ_prob >= OCCUPANCY_UNCERTAINTY_HIGH:
                # Confidently occupied
                bonuses[s] = round(occ_prob * (1.0 - e_score), 4)
            elif is_uncertain:
                # Uncertain band [0.35, 0.65]
                bonuses[s] = round(0.50 * (1.0 - e_score), 4)
            else:
                bonuses[s] = 0.0

        robod_desc = (
            f"Academic ROBOD Room {robod_room} (Module 2 Spatial Context)"
            if robod_room is not None
            else "Standard telemetry context (Residential)"
        )

        summary = (
            f"Occupancy gate active (Unoccupied, P={occ_prob:.2f}). Speed forced to 0."
            if hard_gate_active
            else f"Occupancy state: {occ_state} (P={occ_prob:.2f}, feature set: {feat_set})."
        )

        return {
            "occupancy_bonuses": bonuses,
            "hard_gate_active": hard_gate_active,
            "model_available": True,
            "occupancy_classification": occ_state,
            "occupancy_probability": round(occ_prob, 4),
            "is_uncertain": is_uncertain,
            "effective_delta_mult": delta_mult,
            "headcount_estimate": headcount,
            "feature_set_used": feat_set,
            "robod_context": robod_desc,
            "summary": summary,
        }
