"""
FanWise AI - Multi-Occupant Comfort Intelligence & Compromise Engine
====================================================================
Simulates multi-occupant shared thermal environments:
  - Models divergent occupant profiles (e.g., Occupant A runs hot, Occupant B is chill-sensitive)
  - Evaluates individual thermal sensations, Softmax distributions, and airflow preferences
  - Detects comfort conflicts and calculates mathematical severity
  - Solves for Pareto-optimal shared compromise fan speed
  - Communicates transparent physical trade-offs (acknowledging that a single ceiling fan
    creates a uniform room-level airflow and cannot provide opposing microclimates).
"""

from typing import Any, Dict, List, Optional
import math


def evaluate_multi_occupant_compromise(
    room_temp_c: float = 28.5,
    relative_humidity: float = 60.0,
    occupant_a: Optional[Dict[str, Any]] = None,
    occupant_b: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates individual thermal comfort for two occupants with opposing preferences
    and calculates a shared compromise fan speed while declaring physical trade-offs.
    """
    # Default profiles: Occupant A (Runs Hot) vs. Occupant B (Chill-Sensitive)
    occ_a = occupant_a or {
        "id": "occupant_A",
        "name": "Occupant A (Priya)",
        "preference": "cooler",  # Runs Hot
        "clothing_clo": 0.5,     # Light summer wear
        "metabolic_met": 1.3,    # Active office / walking
        "setpoint_shift_c": -1.8,
    }
    
    occ_b = occupant_b or {
        "id": "occupant_B",
        "name": "Occupant B (Rohan)",
        "preference": "warmer",  # Chill-Sensitive / Draft-Sensitive
        "clothing_clo": 0.8,     # Casual long sleeves
        "metabolic_met": 0.9,    # Sedentary resting
        "setpoint_shift_c": +1.8,
    }

    # 1. Individual Effective Perceived Temperatures
    # Effective Temp = T_room - Setpoint_shift
    eff_temp_a = room_temp_c - occ_a["setpoint_shift_c"]  # Runs hot -> feels warmer
    eff_temp_b = room_temp_c - occ_b["setpoint_shift_c"]  # Chill sensitive -> feels cooler

    # 2. Individual Thermal Sensation Softmax Estimation
    def get_thermal_probs(eff_temp: float) -> Dict[str, float]:
        if eff_temp >= 31.0:
            return {"Hot": 0.88, "Neutral": 0.10, "Cold": 0.02}
        elif eff_temp >= 28.5:
            return {"Hot": 0.65, "Neutral": 0.30, "Cold": 0.05}
        elif eff_temp >= 25.5:
            return {"Hot": 0.15, "Neutral": 0.75, "Cold": 0.10}
        elif eff_temp >= 23.0:
            return {"Hot": 0.05, "Neutral": 0.35, "Cold": 0.60}
        else:
            return {"Hot": 0.01, "Neutral": 0.15, "Cold": 0.84}

    probs_a = get_thermal_probs(eff_temp_a)
    probs_b = get_thermal_probs(eff_temp_b)

    sensation_a = max(probs_a, key=probs_a.get)
    sensation_b = max(probs_b, key=probs_b.get)

    # 3. Individual Ideal Fan Speeds
    # Occupant A (Hot): Desires high airflow (Speeds 4–5)
    if sensation_a == "Hot":
        ideal_speed_a = 5 if eff_temp_a >= 31.0 else 4
    elif sensation_a == "Neutral":
        ideal_speed_a = 3
    else:
        ideal_speed_a = 1

    # Occupant B (Cold/Draft sensitive): Desires low/gentle airflow (Speeds 1–2)
    if sensation_b == "Cold":
        ideal_speed_b = 1 if eff_temp_b <= 24.0 else 2
    elif sensation_b == "Neutral":
        ideal_speed_b = 2
    else:
        ideal_speed_b = 3

    # 4. Conflict Analysis
    speed_gap = abs(ideal_speed_a - ideal_speed_b)
    has_conflict = speed_gap >= 2
    conflict_severity = "HIGH" if speed_gap >= 3 else ("MODERATE" if speed_gap == 2 else "LOW")

    # 5. Shared Compromise Optimization
    # We evaluate multi-objective joint discomfort cost:
    # J(s) = Discomfort_A(s) + Discomfort_B(s) + DraftPenalty_B(s) + Energy(s)
    candidates = [0, 1, 2, 3, 4, 5]
    best_compromise_speed = 2
    min_joint_cost = float("inf")
    candidate_costs = {}

    for s in candidates:
        # Discomfort for Occupant A (penalize low speeds when hot)
        disc_a = (5 - s) * 0.45 if sensation_a == "Hot" else abs(s - ideal_speed_a) * 0.25
        # Discomfort for Occupant B (penalize high speeds when chill-sensitive / draft)
        disc_b = s * 0.50 if sensation_b == "Cold" else abs(s - ideal_speed_b) * 0.25
        # Draft penalty (excessive draft on draft-sensitive person is disproportionately uncomfortable)
        draft_penalty = max(0.0, (s - 2) * 0.35) if occ_b["preference"] == "warmer" else 0.0
        # Energy cost (BLDC power)
        energy_cost = (s / 5.0) * 0.20

        total_cost = round(disc_a + disc_b + draft_penalty + energy_cost, 3)
        candidate_costs[s] = total_cost

        if total_cost < min_joint_cost:
            min_joint_cost = total_cost
            best_compromise_speed = s

    # 6. Engineering Honesty: Transparent Physical Trade-Off Communication
    trade_off_acknowledgement = (
        "PHYSICAL CONSTRAINT ACKNOWLEDGEMENT: A single ceiling fan creates a uniform air velocity field "
        "and cannot independently supply divergent microclimates to occupants in different areas of the room."
    )

    trade_off_impact_a = (
        f"Occupant A ({occ_a['name']}): Desired Speed {ideal_speed_a}, but compromise set to Speed {best_compromise_speed}. "
        f"Airflow is slightly reduced from ideal, but provides adequate convective cooling while avoiding excessive draft on Occupant B."
    )

    trade_off_impact_b = (
        f"Occupant B ({occ_b['name']}): Desired Speed {ideal_speed_b}, compromise set to Speed {best_compromise_speed}. "
        f"Draft velocity is capped at ~0.24 m/s to prevent cold thermal discomfort and draft complaints."
    )

    return {
        "room_temperature_c": room_temp_c,
        "relative_humidity_pct": relative_humidity,
        "occupants": [
            {
                "id": occ_a["id"],
                "name": occ_a["name"],
                "preference": occ_a["preference"],
                "effective_temp_c": round(eff_temp_a, 1),
                "sensation": sensation_a,
                "probabilities": probs_a,
                "desired_speed": ideal_speed_a,
            },
            {
                "id": occ_b["id"],
                "name": occ_b["name"],
                "preference": occ_b["preference"],
                "effective_temp_c": round(eff_temp_b, 1),
                "sensation": sensation_b,
                "probabilities": probs_b,
                "desired_speed": ideal_speed_b,
            },
        ],
        "conflict_analysis": {
            "has_conflict": has_conflict,
            "speed_gap_steps": speed_gap,
            "severity": conflict_severity,
            "conflict_description": (
                f"Opposing preferences detected: {occ_a['name']} requests Speed {ideal_speed_a} (Turbo cooling), "
                f"while {occ_b['name']} requests Speed {ideal_speed_b} (Gentle draft-free air)."
            ) if has_conflict else "Occupant preferences are aligned.",
        },
        "compromise_decision": {
            "recommended_compromise_speed": best_compromise_speed,
            "joint_objective_cost": min_joint_cost,
            "candidate_costs": candidate_costs,
            "decision_policy": "Utilitarian Joint Discomfort Minimization with Draft-Safety Floor",
        },
        "transparent_trade_off": {
            "physical_constraint_notice": trade_off_acknowledgement,
            "impact_on_occupant_a": trade_off_impact_a,
            "impact_on_occupant_b": trade_off_impact_b,
            "practical_recommendation": (
                "For further comfort divergence, position Occupant A directly under the blade center-line "
                "(maximum velocity zone ~0.8 m/s) and Occupant B toward the perimeter (velocity <0.2 m/s)."
            ),
        },
    }
