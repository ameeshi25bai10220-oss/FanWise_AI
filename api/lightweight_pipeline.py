"""
FanWise AI - Lightweight Serverless Pipeline for Vercel
========================================================
Runs real inference for all 3 trained models using compact tree walkers and pure NumPy:
  - Model 1: Thermal Comfort (900 XGBoost trees, Softmax across Cold/Neutral/Hot)
  - Model 2: Occupancy Intelligence (100 XGBoost trees, psychrometric HumidityRatio, Ridge Headcount)
  - Model 3: Energy Intelligence (Physics Engine table + 100 LightGBM forecast trees)
  - Fan Digital Twin Virtual Runtime
  - Multi-Objective Adaptive Controller & Safety Guard (P1-P5)

Zero heavy C++ binaries (No xgboost.so, scipy.so, or lightgbm.so needed).
Bundle size is ~55 MB (well below Vercel's 500 MB limit).
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
# Constants
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

MANDATORY_DISCLAIMERS: List[str] = [
    "Model 1 (Thermal Comfort) is trained on ASHRAE DB-II commercial/office data; unvalidated for residential Indian deployment.",
    "Model 1 softmax probabilities are uncalibrated and are NOT calibrated confidence scores.",
    "Air velocity estimates use provisional Dryad open-access mapping; no validated Atomberg hardware calibration exists.",
    "Model 2 (Occupancy) is trained on ROBOD academic dataset (rooms 1-5); ROBOD rooms must not be equated with residential spaces.",
    "Model 3 (Energy) R2~0.1515, MAE~1.19 Wh; it is advisory only and CANNOT command fan speed.",
    "All power estimates are nominal research-dataset constants; no hardware-calibrated Atomberg energy data is available.",
    "UNRESOLVED_INDUCTION_BASELINE: Two diverging induction power tables exist in codebase.",
    "No physical fan actuation occurs; all speed values are software-scale 0-5 recommendations only.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Tree Walker Helpers
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
# Main Lightweight Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class LightweightPipeline:
    """High-performance, zero-heavy-dependency runner for the 3-model AACI pipeline."""

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
        provenance: str = "TELEMETRY",
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        exec_id = str(uuid.uuid4())
        warnings: List[str] = []

        # 0. Fan and Room resolution
        resolved_fan_id = fan_id or raw_telemetry.get("fan_id", "Bedroom_Fan_1")
        temp = float(raw_telemetry.get("temperature_c", 28.0))
        rh = float(raw_telemetry.get("relative_humidity_pct", 60.0))
        light = raw_telemetry.get("light_lux")
        co2 = raw_telemetry.get("co2_ppm")
        current_speed = int(raw_telemetry.get("fan_speed", 2))
        fan_mode = str(raw_telemetry.get("fan_mode", "BLDC")).upper()
        room_category = str(raw_telemetry.get("room_category", "Bedroom"))
        
        # Check staleness
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

        # 1. Model 1: Thermal Comfort
        vel_used = PROVISIONAL_SPEED_VELOCITY_MAP.get(current_speed, 0.239)
        m1_res = self._predict_model1(temp, rh, vel_used)

        # 2. Model 2: Occupancy Intelligence
        m2_res = self._predict_model2(temp, rh, light, co2)
        occupancy_prob = m2_res["occupancy_probability"]
        occupancy_hard_gate = occupancy_prob < 0.35

        # 3. Model 3: Energy Intelligence
        power_table = BLDC_POWER_MAP if fan_mode == "BLDC" else INDUCTION_POWER_MAP
        cand_table = [
            {
                "speed": s,
                "power_w": power_table[s],
                "energy_wh_physics": round(power_table[s] * 0.25, 2),
                "source": "physics_engine_deterministic",
            }
            for s in CANDIDATE_SPEEDS
        ]
        m3_res = self._predict_model3(temp, rh, current_speed, cand_table)

        # 4. Safety Guard & Priorities (P1-P5)
        decision_mode = "adaptive"
        rationale = ""
        safety_active = False
        thermal_floor_active = False

        # Check Active Manual Override (P2)
        override = self._manual_overrides.get(resolved_fan_id)
        if override and override.get("active"):
            if time.time() < override["expiry_time_s"]:
                winning_speed = override["speed"]
                decision_mode = "manual_override"
                rationale = f"Manual override active to speed {winning_speed}."
                safety_active = True
            else:
                del self._manual_overrides[resolved_fan_id]

        if decision_mode != "manual_override":
            # P1: Sensor Staleness Freeze
            if is_stale:
                winning_speed = current_speed
                decision_mode = "staleness_freeze"
                rationale = "Maintained last known confirmed speed to avoid acting on stale telemetry."
                safety_active = True
            
            # P4: Occupancy Hard Gate
            elif occupancy_hard_gate:
                winning_speed = 0
                decision_mode = "occupancy_hard_gate"
                rationale = "[Occupancy hard gate active (unoccupied).] Space is unoccupied (P < 0.35); speed reduced to 0."
                safety_active = True
                
            # P3: Thermal Safety Floor
            elif temp >= 35.0:
                thermal_floor_active = True
                safety_active = True
                decision_mode = "thermal_safety"
                # Filter candidates: minimum speed is 1 (cannot turn off)
                candidates = [s for s in CANDIDATE_SPEEDS if s >= 1]
                winning_speed = self._optimize_speed(candidates, current_speed, m1_res, power_table)
                rationale = f"[Thermal safety floor active (T >= 35°C).] Minimum fan speed 1 enforced for heat protection."
            
            # Normal Adaptive Optimization
            else:
                candidates = list(CANDIDATE_SPEEDS)
                winning_speed = self._optimize_speed(candidates, current_speed, m1_res, power_table)
                comfort_state = m1_res["comfort_label"]
                rationale = f"Selected Speed {winning_speed}. Comfort: {comfort_state}. Optimized for thermal comfort and energy efficiency."

        # 5. Digital Twin Update
        twin_state = {
            "fan_id": resolved_fan_id,
            "room_category": room_category,
            "current_speed": winning_speed,
            "previous_speed": current_speed,
            "power_w": power_table.get(winning_speed, 0.0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._twins[resolved_fan_id] = twin_state

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Build 22-field IntegrationResult
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
            "model1_result": m1_res,
            "model2_result": m2_res,
            "model3_result": m3_res,
            "csr_state_summary": {
                "environmental": {"air_temp_c": temp, "relative_humidity_pct": rh},
                "occupancy": m2_res,
                "fan_state": {"current_speed": winning_speed, "mode": fan_mode},
            },
            "digital_twin_state_summary": twin_state,
            "previous_speed": current_speed,
            "recommended_speed": winning_speed,
            "comfort_assessment": m1_res,
            "occupancy_assessment": {
                "hard_gate_active": occupancy_hard_gate,
                "occupancy_probability": occupancy_prob,
            },
            "energy_assessment": {
                "estimated_power_w": power_table.get(winning_speed, 0.0),
                "fan_mode": fan_mode,
            },
            "candidate_scores": cand_table,
            "safety_status": {
                "safety_active": safety_active,
                "thermal_safety_floor_triggered": thermal_floor_active,
                "is_stale": is_stale,
            },
            "manual_override_status": self._manual_overrides.get(resolved_fan_id),
            "fallback_status": {"fallback_triggered": False},
            "decision_explanation": {
                "recommended_speed": winning_speed,
                "previous_speed": current_speed,
                "decision_mode": decision_mode,
                "rationale": rationale,
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

    def _predict_model1(self, temp: float, rh: float, vel: float) -> Dict[str, Any]:
        m1 = self.bundle["model1"]
        trees = m1["trees"]
        mean = np.array(m1["num_mean"])
        scale = np.array(m1["num_scale"])
        
        raw_num = np.array([temp, rh, vel, 28.0, 0.6, 1.2], dtype=float)
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
            "is_uncertain": float(np.max(probs)) < 0.60,
            "air_velocity_used_ms": vel,
            "provisional_mapping_flag": True,
            "model_available": True,
            "disclaimers": [],
        }

    def _predict_model2(self, temp: float, rh: float, light: Optional[float], co2: Optional[float]) -> Dict[str, Any]:
        m2 = self.bundle["model2"]
        has_extended = (light is not None and co2 is not None)
        
        hum_ratio = calculate_humidity_ratio(temp, rh)
        
        if has_extended:
            cfg = m2["set_a"]
            raw_feats = np.array([temp, rh, float(light), float(co2), hum_ratio], dtype=float)
            feature_set = "set_a"
        else:
            cfg = m2["set_b"]
            raw_feats = np.array([temp, rh, hum_ratio, 14.0], dtype=float)
            feature_set = "set_b"
            
        mean = np.array(cfg["mean"])
        scale = np.array(cfg["scale"])
        x = (raw_feats - mean) / scale
        
        # Evaluate XGBoost binary trees
        raw_margin = 0.0
        for tree in cfg["trees"]:
            raw_margin += _eval_xgb_node(tree, x)
            
        prob = 1.0 / (1.0 + math.exp(-raw_margin))
        is_occupied = prob >= 0.5
        
        # Module 2 Ridge Headcount
        headcount = 0.0
        if is_occupied:
            ridge = m2["ridge"]
            coef = np.array(ridge["coef"])
            intercept = ridge["intercept"]
            # Scale environmental features for headcount
            hc_feats = np.array([temp, rh, float(light or 100.0), float(co2 or 400.0), 45.0, 5.0], dtype=float)
            hc_scaled = (hc_feats - np.array(ridge["scaler_mean"])) / np.array(ridge["scaler_scale"])
            headcount = float(np.dot(hc_scaled, coef) + intercept)
            headcount = max(1.0, min(10.0, headcount))

        return {
            "occupancy_state": "Occupied" if is_occupied else "Unoccupied",
            "occupancy_binary": 1 if is_occupied else 0,
            "occupancy_probability": round(prob, 4),
            "headcount_estimate": round(headcount, 2) if is_occupied else None,
            "headcount_rounded": int(round(headcount)) if is_occupied else None,
            "density_category": "low (1-3 persons)" if headcount <= 3 else "medium",
            "sensor_gap_active": not has_extended,
            "is_uncertain": 0.35 <= prob <= 0.65,
            "feature_set_used": feature_set,
            "model_available": True,
        }

    def _predict_model3(self, temp: float, rh: float, speed: int, cand_table: List[Dict]) -> Dict[str, Any]:
        m3 = self.bundle["model3"]
        trees = m3["trees"]
        
        # Features: [temp, rh, speed, power_w, lag1, lag2]
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
    ) -> int:
        """Finds candidate speed with minimum multi-objective cost."""
        best_speed = candidates[0]
        min_cost = float("inf")
        comfort_label = comfort_eval["comfort_label"]

        for s in candidates:
            # Comfort penalty
            if comfort_label == "Hot":
                c_pen = max(0.0, (4 - s) * 0.3)
            elif comfort_label == "Cold":
                c_pen = max(0.0, (s - 1) * 0.3)
            else:
                c_pen = abs(s - 2) * 0.2

            # Energy cost
            e_cost = (power_table[s] / 48.0) * 0.4

            # Stability penalty (favor staying at current speed)
            s_pen = abs(s - current_speed) * 0.15

            total_cost = (1.5 * c_pen) + (0.8 * e_cost) + (0.5 * s_pen)
            if total_cost < min_cost:
                min_cost = total_cost
                best_speed = s

        return best_speed
