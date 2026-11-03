"""
AACI V2 – Production Thermal Comfort Prediction & Fan Control Interface
========================================================================
Atomberg Adaptive Comfort Intelligence (AACI)

This module provides the inference interface for the improved V2 model.
It accepts the same environmental inputs as V1, automatically computes the
biometeorological engineered features, performs inference, and integrates
with the AACIFanController decision layer.

Usage:
    from predict_v2 import predict_comfort_v2

    result = predict_comfort_v2(
        air_temp_c=28.5,
        relative_humidity=65.0,
        air_velocity_ms=0.3,
        outdoor_monthly_temp_c=32.0,
        current_fan_speed=2,
    )
    print(result["predicted_class"])     # "Hot"
    print(result["decision"]["action"])  # "INCREASE"
    print(result["decision"]["target_speed"]) # 3
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any

from fan_controller import AACIFanController

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH    = os.path.join(_DIR, "aaci_v2_model.joblib")
_PREPROC_PATH  = os.path.join(_DIR, "aaci_v2_preprocessor.joblib")
_METADATA_PATH = os.path.join(_DIR, "aaci_v2_metadata.json")

# Singletons for fast repeated inference
_model        = None
_preprocessor = None
_metadata     = None
_controller   = None

def _load_artifacts():
    global _model, _preprocessor, _metadata, _controller
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"V2 Model not found at {_MODEL_PATH}. Run 04_comprehensive_experiments.py first.")
        _model        = joblib.load(_MODEL_PATH)
        _preprocessor = joblib.load(_PREPROC_PATH)
        with open(_METADATA_PATH, "r") as f:
            _metadata = json.load(f)
        _controller   = AACIFanController()

def compute_biomet_features_single(ta: Optional[float], rh: Optional[float], v: Optional[float], tout: Optional[float]):
    """Computes the 9 physics-based biometeorological features for a single sample."""
    ta_eff = 24.0 if (ta is None or (isinstance(ta, float) and np.isnan(ta))) else float(ta)
    rh_eff = 50.0 if (rh is None or (isinstance(rh, float) and np.isnan(rh))) else float(rh)
    v_eff = float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v)) and v > 0) else 0.1
    tout_eff = float(tout) if (tout is not None and not (isinstance(tout, float) and np.isnan(tout))) else ta_eff

    # 1. Vapor pressure (Magnus-Tetens)
    es = 6.112 * np.exp((17.67 * ta_eff) / (ta_eff + 243.5))
    e  = (rh_eff / 100.0) * es
    
    # 2. Dew point
    alpha = ((17.67 * ta_eff) / (ta_eff + 243.5)) + np.log(np.clip(rh_eff / 100.0, 1e-4, 1.0))
    tdew = (243.5 * alpha) / (17.67 - alpha)
    
    # 3. Steadman Apparent Temperature
    t_apparent = ta_eff + 0.33 * e - 0.70 * v_eff - 4.0
    
    # 4. Humidex
    humidex = ta_eff + (5.0 / 9.0) * (e - 10.0)
    
    # 5. Convective Cooling Effect & Effective Temperature
    cooling_effect = 1.2 * np.sqrt(np.maximum(v_eff - 0.15, 0.0))
    t_effective = ta_eff - cooling_effect
    
    # 6. Thermal gradient
    delta_tout = tout_eff - ta_eff
    
    # 7. Convective Heat Removal Potential
    conv_exchange = v_eff * (36.5 - ta_eff)
    
    # 8. Evaporative cooling potential
    evap_potential = np.maximum(es - e, 0.0) * np.sqrt(np.maximum(v_eff, 0.05))
    
    return {
        "vapor_pressure_hpa": round(float(e), 3),
        "dew_point_c": round(float(tdew), 2),
        "apparent_temp_c": round(float(t_apparent), 2),
        "humidex": round(float(humidex), 2),
        "cooling_effect_c": round(float(cooling_effect), 2),
        "effective_temp_c": round(float(t_effective), 2),
        "delta_tout_tin": round(float(delta_tout), 2),
        "convective_heat_exchange": round(float(conv_exchange), 3),
        "evaporative_cooling_potential": round(float(evap_potential), 3),
    }

def predict_comfort_v2(
    air_temp_c: float,
    relative_humidity: float,
    air_velocity_ms: float = 0.2,
    outdoor_monthly_temp_c: Optional[float] = None,
    clo: float = 0.6,
    met: float = 1.2,
    season: str = "Summer",
    building_type: str = "Office",
    cooling_strategy: str = "Air Conditioned",
    current_fan_speed: int = 2,
    controller_instance: Optional[AACIFanController] = None,
    current_timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Predicts thermal comfort class using the V2 model and evaluates
    the adaptive ceiling fan control decision.
    """
    _load_artifacts()
    ctrl = controller_instance if controller_instance is not None else _controller
    ctrl.current_speed = current_fan_speed

    CLASS_NAMES = _metadata["class_names"]

    # Compute engineered features
    eng_feats = compute_biomet_features_single(
        ta=air_temp_c,
        rh=relative_humidity,
        v=air_velocity_ms,
        tout=outdoor_monthly_temp_c
    )

    # Build input row matching preprocessor feature expectations
    input_row = {
        "Air temperature (C)":                air_temp_c,
        "Relative humidity (%)":              relative_humidity,
        "Air velocity (m/s)":                 air_velocity_ms,
        "Outdoor monthly air temperature (C)": outdoor_monthly_temp_c,
        "Clo":                                clo,
        "Met":                                met,
        "vapor_pressure_hpa":                 eng_feats["vapor_pressure_hpa"],
        "dew_point_c":                        eng_feats["dew_point_c"],
        "apparent_temp_c":                    eng_feats["apparent_temp_c"],
        "humidex":                            eng_feats["humidex"],
        "cooling_effect_c":                   eng_feats["cooling_effect_c"],
        "effective_temp_c":                   eng_feats["effective_temp_c"],
        "delta_tout_tin":                     eng_feats["delta_tout_tin"],
        "convective_heat_exchange":           eng_feats["convective_heat_exchange"],
        "evaporative_cooling_potential":      eng_feats["evaporative_cooling_potential"],
        "Season":                             season,
        "Building type":                      building_type,
        "Cooling startegy_building level":    cooling_strategy,
    }

    df_in = pd.DataFrame([input_row])
    X_proc = _preprocessor.transform(df_in)

    # Inference
    class_id = int(_model.predict(X_proc)[0])
    probs = _model.predict_proba(X_proc)[0]

    predicted_class = CLASS_NAMES[class_id]
    confidence = float(probs[class_id])
    probabilities = {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)}

    # Evaluate adaptive fan control decision layer
    decision = ctrl.evaluate_decision(
        predicted_class=predicted_class,
        probabilities=probabilities,
        air_temp_c=air_temp_c,
        relative_humidity=relative_humidity,
        current_air_velocity_ms=air_velocity_ms,
        current_timestamp=current_timestamp,
    )

    return {
        "predicted_class":   predicted_class,
        "class_id":          class_id,
        "probabilities":     probabilities,
        "confidence":        round(confidence, 4),
        "decision":          decision,
        "recommendation":    decision["reason"],
        "model_name":        _metadata["model_name"],
        "engineered_features": eng_feats,
        "inputs_used": {
            "air_temp_c":        air_temp_c,
            "relative_humidity": relative_humidity,
            "air_velocity_ms":   air_velocity_ms,
            "outdoor_temp_c":    outdoor_monthly_temp_c,
            "clo":               clo,
            "met":               met,
            "season":            season,
            "building_type":     building_type,
            "cooling_strategy":  cooling_strategy,
            "current_fan_speed": current_fan_speed,
        }
    }


if __name__ == "__main__":
    print("AACI V2 – End-to-End Prediction & Fan Control Test")
    print("=" * 60)
    
    test_scenarios = [
        ("Hot Summer Office", 31.5, 68.0, 0.2, 36.0, 2),
        ("Cold Winter Room (Safety Lock Test)", 18.5, 45.0, 0.4, 14.0, 3),
        ("Comfortable Neutral Room", 24.0, 50.0, 0.2, 28.0, 2),
        ("High Humidity Stuffy Room", 28.0, 80.0, 0.15, 30.0, 1),
    ]
    
    ctrl = AACIFanController()
    
    for label, ta, rh, v, tout, spd in test_scenarios:
        res = predict_comfort_v2(
            air_temp_c=ta,
            relative_humidity=rh,
            air_velocity_ms=v,
            outdoor_monthly_temp_c=tout,
            current_fan_speed=spd,
            controller_instance=ctrl
        )
        print(f"\nScenario: {label}")
        print(f"  Inputs:  Ta={ta}C, RH={rh}%, v={v}m/s, FanSpeed={spd}")
        print(f"  Comfort: {res['predicted_class']} (Confidence: {res['confidence']:.1%})")
        print(f"  Action:  {res['decision']['action']} -> Target Speed: {res['decision']['target_speed']}")
        print(f"  Reason:  {res['decision']['reason']}")
        print(f"  Cooling: Perceived Delta T = -{res['decision']['estimated_cooling_c']}C")

