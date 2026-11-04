"""
FanWise AI - Enhanced Lightweight Room Simulator Pipeline
==========================================================
Interactive Room Simulator Engine for Judges & Hackathon Demonstrations:
  - Model 1: Thermal Comfort Intelligence (900 XGBoost trees Softmax)
  - Model 2: Occupancy Intelligence (100 XGBoost trees + Ridge Headcount)
  - Model 3: Energy Intelligence (Physics Engine + LightGBM Advisory)
  - Personalization Layer (Bayesian-inspired setpoint & logit shift)
  - Fan Digital Twin Virtual Runtime & Physics State
  - Multi-Objective Adaptive Controller & Safety Guard (P1-P5)
  - Full Decision Lifecycle Stepper for "Glass Box" Explainability
"""

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Reference Tables
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATE_SPEEDS = (0, 1, 2, 3, 4, 5)

PROVISIONAL_SPEED_VELOCITY_MAP: Dict[int, float] = {
    0: 0.050,
    1: 0.112,
    2: 0.239,
    3: 0.293,
    4: 0.410,
    5: 0.527,
}

BLDC_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 10.0,
    2: 15.0,
    3: 20.0,
    4: 27.0,
    5: 48.0,
}

INDUCTION_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 24.0,
    2: 34.0,
    3: 55.0,
    4: 64.0,
    5: 96.0,
}

TARIFF_INR_PER_KWH = 7.0  # Standard Indian residential tariff baseline

MANDATORY_DISCLAIMERS: List[str] = [
    "Model 1 (Thermal Comfort) trained on ASHRAE DB-II commercial/office dataset; unvalidated for residential Indian deployment.",
    "Model 1 softmax probabilities are uncalibrated and are NOT calibrated confidence scores.",
    "Air velocity estimates use provisional Dryad open-access mapping; no validated Atomberg hardware calibration exists.",
    "Model 2 (Occupancy) trained on ROBOD academic dataset; ROBOD academic spaces must not be equated with residential spaces.",
    "Model 3 (Energy) R2~0.1515, MAE~1.19 Wh; advisory only and CANNOT command fan speed.",
    "All power estimates are nominal research constants; no physical hardware-calibrated Atomberg energy data available.",
    "UNRESOLVED_INDUCTION_BASELINE: Two diverging induction power tables exist in codebase.",
    "No physical fan actuation occurs; all speed values are software-scale 0-5 recommendations only.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Fast Math & Tree Walker Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_xgb_node(node: Dict, x: np.ndarray) -> float:
    """Traverses a single XGBoost JSON tree node."""
    if "leaf" in node:
        return float(node["leaf"])
    f_idx = int(node["split"].replace("f", ""))
    val = x[f_idx]
    if math.isnan(val):
        target_id = node["missing"]
    elif val < node["split_condition"]:
        target_id = node["yes"]
    else:
        target_id = node["no"]
    for child in node.get("children", []):
        if child["nodeid"] == target_id:
            return _eval_xgb_node(child, x)
    return 0.0


def _eval_lgbm_node(node: Dict, x: np.ndarray) -> float:
    """Traverses a single LightGBM JSON tree node."""
    if "leaf_value" in node:
        return float(node["leaf_value"])
    f_idx = int(node["split_feature"])
    val = x[f_idx] if f_idx < len(x) else 0.0
    thresh = float(node["threshold"])
    if math.isnan(val) or val <= thresh:
        return _eval_lgbm_node(node["left_child"], x)
    else:
        return _eval_lgbm_node(node["right_child"], x)


def calculate_humidity_ratio(temp_c: float, rh_pct: float, pressure_kpa: float = 101.325) -> float:
    """Computes psychrometric Humidity Ratio (kg water vapor per kg dry air)."""
    e_sat = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
    e_act = (rh_pct / 100.0) * e_sat
    p_hpa = pressure_kpa * 10.0
    w = 0.622 * e_act / max(1.0, (p_hpa - e_act))
    return float(w)


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Simulator Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class LightweightPipeline:
    """Production-grade, zero-heavy-dependency simulator engine for FanWise AI."""

    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = os.path.join(os.path.dirname(__file__), "models_bundle.json")
        
        with open(bundle_path, "r", encoding="utf-8") as f:
            self.bundle = json.load(f)

        self._twins: Dict[str, Dict[str, Any]] = {}
        self._manual_overrides: Dict[str, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []

    def set_manual_override(self, fan_id: str, speed: int, duration_s: float = 600.0) -> Dict[str, Any]:
        now = time.time()
        record = {
            "fan_id": fan_id,
            "speed": int(speed),
            "start_time_s": now,
            "expiry_time_s": now + duration_s,
            "duration_s": duration_s,
            "active": True,
        }
        self._manual_overrides[fan_id] = record
        return record

    def clear_manual_override(self, fan_id: str) -> bool:
        if fan_id in self._manual_overrides:
            del self._manual_overrides[fan_id]
            return True
        return False

    def execute(
        self,
        raw_telemetry: Dict[str, Any],
        fan_id: Optional[str] = None,
        provenance: str = "SIMULATION",
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        exec_id = str(uuid.uuid4())
        warnings: List[str] = []

        # ── 0. Parse Environmental & Occupant Parameters ───────────────────────
        resolved_fan_id = fan_id or raw_telemetry.get("fan_id", "Bedroom_Fan_1")
        temp = float(raw_telemetry.get("temperature_c", 28.0))
        rh = float(raw_telemetry.get("relative_humidity_pct", 60.0))
        outdoor_temp = float(raw_telemetry.get("outdoor_temp_c", 30.0))
        current_speed = int(raw_telemetry.get("fan_speed", 2))
        current_speed = max(0, min(5, current_speed))
        fan_mode = str(raw_telemetry.get("fan_mode", "BLDC")).upper()
        room_category = str(raw_telemetry.get("room_category", "Bedroom"))
        
        # Air Velocity: Custom or Auto from Fan Speed
        air_vel_input = raw_telemetry.get("air_velocity_ms")
        if air_vel_input is not None and float(air_vel_input) >= 0.0:
            vel_used = float(air_vel_input)
            vel_source = "User / Anemometer Direct"
        else:
            vel_used = PROVISIONAL_SPEED_VELOCITY_MAP.get(current_speed, 0.239)
            vel_source = f"Provisional Dryad (Speed {current_speed})"

        # Occupants & Sensor Inference
        occupants_count = raw_telemetry.get("occupants_count")
        if occupants_count is not None:
            occupants_count = int(occupants_count)
            # Infer realistic Light & CO2 levels from occupant count if not manually passed
            if occupants_count == 0:
                light = float(raw_telemetry.get("light_lux", 12.0))
                co2 = float(raw_telemetry.get("co2_ppm", 415.0))
            else:
                light = float(raw_telemetry.get("light_lux", 250.0 + occupants_count * 80.0))
                co2 = float(raw_telemetry.get("co2_ppm", 500.0 + occupants_count * 250.0))
        else:
            light = float(raw_telemetry["light_lux"]) if "light_lux" in raw_telemetry and raw_telemetry["light_lux"] is not None else 450.0
            co2 = float(raw_telemetry["co2_ppm"]) if "co2_ppm" in raw_telemetry and raw_telemetry["co2_ppm"] is not None else 850.0
            occupants_count = 1 if co2 >= 550.0 or light >= 100.0 else 0

        # Personalization & Behavioral Parameters
        setpoint_shift_c = float(raw_telemetry.get("setpoint_shift_c", 0.0))
        comfort_pref = str(raw_telemetry.get("comfort_preference", "neutral")).lower()
        if comfort_pref == "cooler":
            setpoint_shift_c = -1.8
        elif comfort_pref == "warmer":
            setpoint_shift_c = 1.8
            
        clo = float(raw_telemetry.get("clothing_clo", 0.6))
        met = float(raw_telemetry.get("metabolic_met", 1.2))
        operating_mode = str(raw_telemetry.get("operating_mode", "auto")).lower()
        manual_target_speed = raw_telemetry.get("manual_speed")
        if manual_target_speed is not None:
            manual_target_speed = int(manual_target_speed)

        # Staleness check
        is_stale = False
        sensor_age_s = 0.0
        ts_str = raw_telemetry.get("timestamp")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                sensor_age_s = (datetime.now(timezone.utc) - dt).total_seconds()
                if sensor_age_s > 300.0:
                    is_stale = True
            except Exception:
                pass

        # ── 1. Model 1: Thermal Comfort Intelligence ───────────────────────────
        # Apply setpoint shift to perceived temperature
        effective_temp = temp - setpoint_shift_c
        m1_base = self._predict_model1(effective_temp, rh, vel_used, outdoor_temp, clo, met)
        
        # Bayesian-inspired logit shift for individual personalization
        m1_personalized = self._apply_personalization(m1_base, setpoint_shift_c, comfort_pref)

        # ── 2. Model 2: Occupancy Intelligence ─────────────────────────────────
        m2_res = self._predict_model2(temp, rh, light, co2, occupants_count)
        occupancy_prob = m2_res["occupancy_probability"]
        occupancy_hard_gate = (occupants_count == 0) or (occupancy_prob < 0.35)

        # ── 3. Model 3: Energy Intelligence ───────────────────────────────────
        power_table = BLDC_POWER_MAP if fan_mode == "BLDC" else INDUCTION_POWER_MAP
        cand_table = [
            {
                "speed": s,
                "power_w": power_table[s],
                "energy_wh_physics": round(power_table[s] * 0.25, 2),
                "induction_equiv_w": INDUCTION_POWER_MAP[s],
                "bldc_savings_w": INDUCTION_POWER_MAP[s] - BLDC_POWER_MAP[s],
                "source": "physics_engine_deterministic",
            }
            for s in CANDIDATE_SPEEDS
        ]
        m3_res = self._predict_model3(temp, rh, current_speed, cand_table)

        # ── 4. Controller Logic & Safety Priorities (P1-P5) ────────────────────
        decision_mode = "adaptive"
        rationale = ""
        safety_active = False
        thermal_floor_active = False
        active_priority = "Normal"
        decision_steps = []

        # Decision Stepper: Step 1 (Sensors)
        decision_steps.append({
            "step": 1,
            "title": "Sensor Telemetry Ingestion",
            "status": "PASS",
            "detail": f"Room: {temp:.1f}°C, {rh:.0f}% RH | Outdoor: {outdoor_temp:.1f}°C | Airflow: {vel_used:.3f} m/s ({vel_source})",
        })

        # Decision Stepper: Step 2 (Occupancy Gate)
        if occupancy_hard_gate:
            decision_steps.append({
                "step": 2,
                "title": "Occupancy Intelligence & Gating",
                "status": "GATE_TRIGGERED",
                "detail": f"Occupants: {occupants_count} | Confidence: {occupancy_prob:.1%} < 35% threshold -> Hard Gate Activated",
            })
        else:
            decision_steps.append({
                "step": 2,
                "title": "Occupancy Intelligence & Gating",
                "status": "PASS",
                "detail": f"Occupants detected: {m2_res['headcount_rounded']} person(s) | Confidence: {occupancy_prob:.1%} (Gate Inactive)",
            })

        # Check Manual Operating Mode or Active Override (P2)
        if operating_mode == "manual" and manual_target_speed is not None:
            winning_speed = manual_target_speed
            decision_mode = "manual_override"
            active_priority = "P2 - Manual Direct Command"
            safety_active = True
            rationale = f"Manual Operating Mode active: Fan set directly to Speed {winning_speed}."
            decision_steps.append({
                "step": 3,
                "title": "Safety Guard & Control Mode",
                "status": "MANUAL_ENFORCED",
                "detail": f"Judge commanded Speed {winning_speed} in Manual Mode (overrides AI auto-pilot).",
            })
        else:
            override = self._manual_overrides.get(resolved_fan_id)
            if override and override.get("active"):
                if time.time() < override["expiry_time_s"]:
                    winning_speed = override["speed"]
                    decision_mode = "manual_override"
                    active_priority = "P2 - Manual Temporary Override"
                    safety_active = True
                    rationale = f"Temporary manual override active to Speed {winning_speed}."
                    decision_steps.append({
                        "step": 3,
                        "title": "Safety Guard & Control Mode",
                        "status": "OVERRIDE_ACTIVE",
                        "detail": f"Temporary 600s lockout active at Speed {winning_speed}.",
                    })
                else:
                    del self._manual_overrides[resolved_fan_id]

        if decision_mode not in ("manual_override",):
            # Priority P1: Staleness Freeze
            if is_stale:
                winning_speed = current_speed
                decision_mode = "staleness_freeze"
                active_priority = "P1 - Sensor Staleness Freeze"
                safety_active = True
                rationale = f"Telemetry age ({sensor_age_s:.0f}s > 300s limit). Frozen at confirmed Speed {current_speed}."
                decision_steps.append({
                    "step": 3,
                    "title": "Safety Guard & Priority Hierarchy",
                    "status": "STALENESS_FREEZE",
                    "detail": "Data is stale (>300s). Freezing current speed to prevent erratic changes.",
                })

            # Priority P4: Occupancy Hard Gate
            elif occupancy_hard_gate:
                winning_speed = 0
                decision_mode = "occupancy_hard_gate"
                active_priority = "P4 - Occupancy Hard Gate"
                safety_active = True
                rationale = "[Occupancy Hard Gate Active.] Space is unoccupied (0 occupants / P < 0.35). Fan shut off (Speed 0) for zero waste."
                decision_steps.append({
                    "step": 3,
                    "title": "Safety Guard & Priority Hierarchy",
                    "status": "GATE_ENFORCED",
                    "detail": "P4 Hard Gate enforced: Recommended Speed 0.",
                })

            # Priority P3: Thermal Safety Floor (T >= 35°C)
            elif temp >= 35.0:
                thermal_floor_active = True
                safety_active = True
                decision_mode = "thermal_safety"
                active_priority = "P3 - Thermal Safety Floor"
                candidates = [s for s in CANDIDATE_SPEEDS if s >= 1]
                winning_speed, cost_matrix = self._optimize_speed(candidates, current_speed, m1_personalized, power_table)
                rationale = f"[P3 Thermal Safety Floor Active (Room Temp {temp:.1f}°C >= 35°C).] Minimum fan speed floor >= 1 enforced for heat protection."
                decision_steps.append({
                    "step": 3,
                    "title": "Safety Guard & Priority Hierarchy",
                    "status": "THERMAL_FLOOR",
                    "detail": f"P3 Thermal Safety Floor Active: Room temp {temp:.1f}°C >= 35°C prohibits Speed 0.",
                })

            # Normal Multi-Objective Adaptive Optimization
            else:
                candidates = list(CANDIDATE_SPEEDS)
                winning_speed, cost_matrix = self._optimize_speed(candidates, current_speed, m1_personalized, power_table)
                active_priority = "P-Normal Adaptive"
                comfort_label = m1_personalized["comfort_label"]
                pref_note = f" (Adjusted for {comfort_pref} preference)" if comfort_pref != "neutral" else ""
                rationale = f"Selected Speed {winning_speed}. Thermal Sensation: {comfort_label}{pref_note}. Optimized global cost across comfort penalty, BLDC energy consumption, and stability."
                decision_steps.append({
                    "step": 3,
                    "title": "Safety Guard & Priority Hierarchy",
                    "status": "NORMAL_PASS",
                    "detail": "All safety priorities satisfied. Transitioning to multi-objective optimization.",
                })

        # Decision Stepper: Step 4 (Multi-Objective Optimization)
        if decision_mode in ("adaptive", "thermal_safety"):
            decision_steps.append({
                "step": 4,
                "title": "Multi-Objective Cost Minimization",
                "status": "OPTIMIZED",
                "detail": f"Selected Speed {winning_speed} as global minimum cost among candidates {candidates}.",
            })
        else:
            decision_steps.append({
                "step": 4,
                "title": "Multi-Objective Cost Minimization",
                "status": "BYPASS_BY_SAFETY",
                "detail": f"Safety constraint or manual command bypassed cost optimization to enforce Speed {winning_speed}.",
            })

        # ── 5. Energy Economics & ROI Calculations ────────────────────────────
        current_bldc_w = BLDC_POWER_MAP.get(winning_speed, 0.0)
        current_induction_w = INDUCTION_POWER_MAP.get(winning_speed, 0.0)
        watts_saved = max(0.0, current_induction_w - current_bldc_w)
        # Assuming 16 hours daily runtime
        daily_kwh_bldc = (current_bldc_w * 16.0) / 1000.0
        daily_kwh_induction = (current_induction_w * 16.0) / 1000.0
        daily_savings_inr = (daily_kwh_induction - daily_kwh_bldc) * TARIFF_INR_PER_KWH
        annual_savings_inr = daily_savings_inr * 365.0

        # ── 6. Digital Twin Virtual State Update ───────────────────────────────
        twin_state = {
            "fan_id": resolved_fan_id,
            "room_category": room_category,
            "current_speed": winning_speed,
            "previous_speed": current_speed,
            "fan_mode": fan_mode,
            "operating_mode": operating_mode,
            "power_w": current_bldc_w if fan_mode == "BLDC" else current_induction_w,
            "estimated_rpm": winning_speed * 70,  # ~350 RPM max
            "daily_kwh": round(daily_kwh_bldc, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._twins[resolved_fan_id] = twin_state

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # ── 7. Build Full 22-Field IntegrationResult ───────────────────────────
        result = {
            "execution_id": exec_id,
            "fan_id": resolved_fan_id,
            "room_id": 1,
            "room_category": room_category,
            "input_timestamp": ts_str or datetime.now(timezone.utc).isoformat(),
            "input_validation_status": {
                "is_valid": True,
                "is_stale": is_stale,
                "errors": [],
                "warnings": warnings,
            },
            "model1_result": m1_personalized,
            "model2_result": m2_res,
            "model3_result": m3_res,
            "csr_state_summary": {
                "environmental": {
                    "air_temp_c": temp,
                    "relative_humidity_pct": rh,
                    "outdoor_temp_c": outdoor_temp,
                    "air_velocity_ms": vel_used,
                    "air_velocity_source": vel_source,
                },
                "occupancy": m2_res,
                "fan_state": {
                    "current_speed": winning_speed,
                    "previous_speed": current_speed,
                    "mode": fan_mode,
                    "operating_mode": operating_mode,
                },
            },
            "digital_twin_state_summary": twin_state,
            "previous_speed": current_speed,
            "recommended_speed": winning_speed,
            "comfort_assessment": m1_personalized,
            "occupancy_assessment": {
                "hard_gate_active": occupancy_hard_gate,
                "occupancy_probability": occupancy_prob,
                "headcount": m2_res["headcount_rounded"],
            },
            "energy_assessment": {
                "bldc_power_w": current_bldc_w,
                "induction_equiv_w": current_induction_w,
                "watts_saved": watts_saved,
                "daily_savings_inr": round(daily_savings_inr, 2),
                "annual_savings_inr": round(annual_savings_inr, 0),
                "fan_mode": fan_mode,
            },
            "personalization_assessment": {
                "preference": comfort_pref,
                "setpoint_shift_c": setpoint_shift_c,
                "clothing_clo": clo,
                "metabolic_met": met,
                "effective_temp_c": round(effective_temp, 2),
            },
            "candidate_scores": cand_table,
            "safety_status": {
                "safety_active": safety_active,
                "active_priority": active_priority,
                "thermal_safety_floor_triggered": thermal_floor_active,
                "is_stale": is_stale,
            },
            "manual_override_status": self._manual_overrides.get(resolved_fan_id),
            "fallback_status": {"fallback_triggered": False},
            "decision_explanation": {
                "recommended_speed": winning_speed,
                "previous_speed": current_speed,
                "decision_mode": decision_mode,
                "active_priority": active_priority,
                "rationale": rationale,
                "decision_steps": decision_steps,
            },
            "data_freshness": {
                "sensor_age_s": sensor_age_s,
                "is_stale": is_stale,
            },
            "warnings": warnings,
            "unresolved_limitations": MANDATORY_DISCLAIMERS,
            "latency_ms": {"total_ms": round(total_ms, 2)},
            "provenance": provenance,
        }

        self._history.append(result)
        return result

    def _predict_model1(
        self,
        temp: float,
        rh: float,
        vel: float,
        outdoor_temp: float = 30.0,
        clo: float = 0.6,
        met: float = 1.2,
    ) -> Dict[str, Any]:
        m1 = self.bundle["model1"]
        trees = m1["trees"]
        mean = np.array(m1["num_mean"])
        scale = np.array(m1["num_scale"])
        
        raw_num = np.array([temp, rh, vel, outdoor_temp, clo, met], dtype=float)
        scaled_num = (raw_num - mean) / scale
        
        # Season: Summer; Building: Office; Cooling: Air Conditioned
        seasons = m1["categories"][0]
        buildings = m1["categories"][1]
        coolings = m1["categories"][2]
        
        ohe_season = [1.0 if s == "Summer" else 0.0 for s in seasons]
        ohe_bldg = [1.0 if b == "Office" else 0.0 for b in buildings]
        ohe_cool = [1.0 if c == "Air Conditioned" else 0.0 for c in coolings]
        
        x = np.concatenate([scaled_num, ohe_season, ohe_bldg, ohe_cool])
        
        raw_scores = [0.0, 0.0, 0.0]
        for idx, tree in enumerate(trees):
            cls_idx = idx % 3
            raw_scores[cls_idx] += _eval_xgb_node(tree, x)
            
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / np.sum(exp_scores)
        classes = m1["classes"]
        best_cls = classes[int(np.argmax(probs))]
        
        return {
            "comfort_label": best_cls,
            "comfort_probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(3)},
            "raw_logits": [round(float(s), 4) for s in raw_scores],
            "is_uncertain": float(np.max(probs)) < 0.60,
            "air_velocity_used_ms": vel,
            "provisional_mapping_flag": True,
            "model_available": True,
        }

    def _apply_personalization(
        self,
        m1_base: Dict[str, Any],
        setpoint_shift_c: float,
        comfort_pref: str,
    ) -> Dict[str, Any]:
        """Applies Bayesian-inspired logit adjustments based on occupant comfort preference."""
        if setpoint_shift_c == 0.0 and comfort_pref == "neutral":
            return m1_base

        classes = ["Cold", "Neutral", "Hot"]
        probs = np.array([m1_base["comfort_probabilities"][c] for c in classes], dtype=float)
        base_logits = np.log(probs + 1e-6)

        # Logit shift: negative shift_c means occupant runs warm -> wants more cooling
        # Increase 'Hot' logit and decrease 'Cold' logit
        logit_bias = np.array([
            0.5 * setpoint_shift_c,      # Cold logit increases if shift > 0 (chill-sensitive)
            -0.1 * abs(setpoint_shift_c),# Neutral slightly compresses
            -0.5 * setpoint_shift_c      # Hot logit increases if shift < 0 (runs hot)
        ])

        new_logits = base_logits + logit_bias
        exp_l = np.exp(new_logits - np.max(new_logits))
        new_probs = exp_l / np.sum(exp_l)
        best_idx = int(np.argmax(new_probs))

        res = dict(m1_base)
        res["comfort_label"] = classes[best_idx]
        res["comfort_probabilities"] = {classes[i]: round(float(new_probs[i]), 4) for i in range(3)}
        res["is_uncertain"] = float(np.max(new_probs)) < 0.60
        res["personalization_applied"] = {
            "preference": comfort_pref,
            "setpoint_shift_c": setpoint_shift_c,
            "shift_effect": "Shifted towards Cooler airflow" if setpoint_shift_c < 0 else "Shifted towards Warmer ambient",
        }
        return res

    def _predict_model2(
        self,
        temp: float,
        rh: float,
        light: float,
        co2: float,
        forced_occupants: Optional[int] = None,
    ) -> Dict[str, Any]:
        m2 = self.bundle["model2"]
        hum_ratio = calculate_humidity_ratio(temp, rh)
        
        cfg = m2["set_a"]
        raw_feats = np.array([temp, rh, light, co2, hum_ratio], dtype=float)
        feature_set = "set_a"
            
        mean = np.array(cfg["mean"])
        scale = np.array(cfg["scale"])
        x = (raw_feats - mean) / scale
        
        # Evaluate XGBoost binary trees
        raw_margin = 0.0
        for tree in cfg["trees"]:
            raw_margin += _eval_xgb_node(tree, x)
            
        prob = 1.0 / (1.0 + math.exp(-raw_margin))
        
        # Override probability if forced_occupants explicitly set to 0
        if forced_occupants == 0:
            prob = 0.04
            is_occupied = False
            headcount = 0.0
        else:
            is_occupied = prob >= 0.5
            # Module 2 Ridge Headcount
            if forced_occupants is not None and forced_occupants > 0:
                headcount = float(forced_occupants)
                prob = max(prob, 0.92)
                is_occupied = True
            elif is_occupied:
                ridge = m2["ridge"]
                coef = np.array(ridge["coef"])
                intercept = ridge["intercept"]
                hc_feats = np.array([temp, rh, light, co2, 45.0, 5.0], dtype=float)
                hc_scaled = (hc_feats - np.array(ridge["scaler_mean"])) / np.array(ridge["scaler_scale"])
                headcount = float(np.dot(hc_scaled, coef) + intercept)
                headcount = max(1.0, min(10.0, headcount))
            else:
                headcount = 0.0

        return {
            "occupancy_state": "Occupied" if is_occupied else "Unoccupied",
            "occupancy_binary": 1 if is_occupied else 0,
            "occupancy_probability": round(prob, 4),
            "headcount_estimate": round(headcount, 2) if is_occupied else 0.0,
            "headcount_rounded": int(round(headcount)) if is_occupied else 0,
            "density_category": "Empty" if headcount == 0 else ("Low (1-2)" if headcount <= 2 else "Medium (3-4)"),
            "sensor_gap_active": False,
            "is_uncertain": 0.35 <= prob <= 0.65,
            "feature_set_used": feature_set,
            "model_available": True,
        }

    def _predict_model3(self, temp: float, rh: float, speed: int, cand_table: List[Dict]) -> Dict[str, Any]:
        m3 = self.bundle["model3"]
        trees = m3["trees"]
        
        power = BLDC_POWER_MAP.get(speed, 20.0)
        x = np.array([temp, rh, speed, power, power, power], dtype=float)
        
        forecast_wh = 0.0
        for tree in trees:
            forecast_wh += _eval_lgbm_node(tree["tree_structure"], x)
            
        forecast_wh = max(1.0, min(15.0, round(forecast_wh, 2)))
        
        return {
            "candidate_evaluation_table": cand_table,
            "ml_forecast": {
                "predicted_next_energy_wh": forecast_wh,
                "model_used": "LightGBM",
                "is_advisory": True,
                "cannot_command_speed": True,
            },
            "model_available": True,
            "unresolved_induction_baseline": "UNRESOLVED_INDUCTION_BASELINE",
        }

    def _optimize_speed(
        self,
        candidates: List[int],
        current_speed: int,
        comfort_eval: Dict[str, Any],
        power_table: Dict[int, float],
    ) -> Tuple[int, Dict[int, float]]:
        """Finds candidate speed with minimum multi-objective cost and returns candidate cost breakdown."""
        best_speed = candidates[0]
        min_cost = float("inf")
        comfort_label = comfort_eval["comfort_label"]
        probs = comfort_eval["comfort_probabilities"]
        cost_breakdown = {}

        for s in candidates:
            # 1. Thermal discomfort penalty
            if comfort_label == "Hot":
                # Being too hot penalizes lower speeds heavily
                c_pen = max(0.0, (5 - s) * 0.35 * probs.get("Hot", 0.6))
            elif comfort_label == "Cold":
                # Being too cold penalizes higher speeds heavily
                c_pen = max(0.0, s * 0.40 * probs.get("Cold", 0.6))
            else:
                # Neutral prefers moderate speeds (1-3)
                c_pen = abs(s - 2) * 0.20

            # 2. Energy consumption cost normalized against max power
            e_cost = (power_table.get(s, 0.0) / 48.0) * 0.45

            # 3. Stability penalty (penalizes frequent rapid speed changes)
            s_pen = abs(s - current_speed) * 0.15

            total_cost = (1.5 * c_pen) + (0.8 * e_cost) + (0.5 * s_pen)
            cost_breakdown[s] = round(total_cost, 4)

            if total_cost < min_cost:
                min_cost = total_cost
                best_speed = s

        return best_speed, cost_breakdown
