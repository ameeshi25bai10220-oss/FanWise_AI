"""
AACI Adaptive Controller – Decision Engine
==========================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Solves the multi-objective optimization problem to select optimal fan speed:
C(s) = alpha_eff * DiscomfortPenalty(s)
     + beta      * EnergyScore(s)
     + gamma     * Delta_s_penalty(s)
     - delta_eff * OccupancyBonus(s)

Features:
- Normalized speed change penalty Delta_s_penalty(s) = |s - s_current| / 5.
- Stability tie-breaking: retains current speed if top-two candidate scores differ by < 0.05.
- Produces full candidate evaluation tables and human-readable rationale.
"""

from typing import Any, Dict, List, Optional, Tuple

from aaci_ml.system_architecture.controller.constants import (
    DEFAULT_ALPHA_COMFORT_WEIGHT,
    DEFAULT_BETA_ENERGY_WEIGHT,
    DEFAULT_DELTA_OCCUPANCY_WEIGHT,
    DEFAULT_GAMMA_STABILITY_WEIGHT,
    TIE_BREAK_COST_THRESHOLD,
)


class DecisionEngine:
    """Executes multi-objective cost evaluation and candidate speed selection."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA_COMFORT_WEIGHT,
        beta: float = DEFAULT_BETA_ENERGY_WEIGHT,
        gamma: float = DEFAULT_GAMMA_STABILITY_WEIGHT,
        delta: float = DEFAULT_DELTA_OCCUPANCY_WEIGHT,
        tie_break_threshold: float = TIE_BREAK_COST_THRESHOLD,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.tie_break_threshold = float(tie_break_threshold)

    def evaluate_candidates(
        self,
        candidate_speeds: List[int],
        current_speed: int,
        comfort_eval: Dict[str, Any],
        occupancy_eval: Dict[str, Any],
        energy_eval: Dict[str, Any],
        alpha_override: Optional[float] = None,
        beta_override: Optional[float] = None,
        gamma_override: Optional[float] = None,
        delta_override: Optional[float] = None,
    ) -> Tuple[int, List[Dict[str, Any]], str]:
        """
        Calculates total cost C(s) for each candidate speed, applies tie-breaking,
        and constructs the decision rationale.
        Returns: (winning_speed, candidate_evaluations, rationale)
        """
        # Apply base or overridden weights
        alpha = self.alpha if alpha_override is None else float(alpha_override)
        beta = self.beta if beta_override is None else float(beta_override)
        gamma = self.gamma if gamma_override is None else float(gamma_override)
        delta = self.delta if delta_override is None else float(delta_override)

        # Apply effective uncertainty adjustments
        alpha_mult = comfort_eval.get("effective_alpha_mult", 1.0)
        delta_mult = occupancy_eval.get("effective_delta_mult", 1.0)
        alpha_eff = round(alpha * alpha_mult, 4)
        delta_eff = round(delta * delta_mult, 4)

        discomfort_penalties = comfort_eval.get("discomfort_penalties", {})
        energy_scores = energy_eval.get("energy_scores", {})
        occupancy_bonuses = occupancy_eval.get("occupancy_bonuses", {})
        candidate_power_w = energy_eval.get("candidate_power_w", {})

        candidate_records: List[Dict[str, Any]] = []

        for s in candidate_speeds:
            # 1. Terms
            d_pen = discomfort_penalties.get(s, 0.33)
            e_score = energy_scores.get(s, 0.0)
            delta_s_pen = round(abs(s - current_speed) / 5.0, 4)
            occ_bonus = occupancy_bonuses.get(s, 0.0)

            # 2. Total cost C(s)
            cost = (
                (alpha_eff * d_pen)
                + (beta * e_score)
                + (gamma * delta_s_pen)
                - (delta_eff * occ_bonus)
            )
            cost = round(cost, 4)

            p_w = candidate_power_w.get(s, 0.0)
            candidate_records.append(
                {
                    "speed": s,
                    "power_w": p_w,
                    "discomfort_penalty": d_pen,
                    "energy_score": e_score,
                    "delta_s_penalty": delta_s_pen,
                    "occupancy_bonus": occ_bonus,
                    "total_cost": cost,
                    "formula": (
                        f"{alpha_eff}*{d_pen} + {beta}*{e_score} + {gamma}*{delta_s_pen} - {delta_eff}*{occ_bonus}"
                    ),
                }
            )

        # Sort candidates by cost ascending
        candidate_records.sort(key=lambda r: r["total_cost"])

        # Best candidate
        best_candidate = candidate_records[0]
        winning_speed = best_candidate["speed"]
        rationale_notes: List[str] = []

        # Check for tie-break stability rule:
        # If top-2 difference < tie_break_threshold, prefer current_speed
        if len(candidate_records) > 1:
            second_candidate = candidate_records[1]
            diff = round(abs(best_candidate["total_cost"] - second_candidate["total_cost"]), 4)
            if diff < self.tie_break_threshold:
                # Check if second candidate is current_speed and best is not
                if second_candidate["speed"] == current_speed and best_candidate["speed"] != current_speed:
                    winning_speed = current_speed
                    rationale_notes.append(
                        f"Stability tie-break applied: top score diff ({diff:.4f} < {self.tie_break_threshold}) "
                        f"favors retaining current speed {current_speed} over speed {best_candidate['speed']}."
                    )
                elif best_candidate["speed"] == current_speed:
                    rationale_notes.append(
                        f"Current speed {current_speed} selected with close margin (diff {diff:.4f})."
                    )

        # Construct explanation rationale
        comfort_label = comfort_eval.get("candidate_scores", {}).get(winning_speed, {}).get("predicted_label", "Neutral")
        occ_state = occupancy_eval.get("occupancy_classification", "Unknown")
        occ_prob = occupancy_eval.get("occupancy_probability")

        base_rationale = (
            f"Selected Speed {winning_speed} (Objective Cost: {best_candidate['total_cost']:.4f}). "
            f"Comfort: {comfort_label} (penalty {best_candidate['discomfort_penalty']:.2f}, alpha_eff={alpha_eff}). "
            f"Energy: Score {best_candidate['energy_score']:.2f} ({best_candidate['power_w']} W). "
            f"Occupancy: {occ_state} (P={occ_prob if occ_prob is not None else 'N/A'}, bonus {best_candidate['occupancy_bonus']:.2f})."
        )
        if rationale_notes:
            base_rationale += " " + " ".join(rationale_notes)

        return winning_speed, candidate_records, base_rationale

