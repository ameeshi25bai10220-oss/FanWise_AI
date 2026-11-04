"""
FanWise AI / AACI - Room Simulator API & Web Application
=========================================================
Interactive serverless application for Vercel deployment:
  - Model 1: Thermal Comfort Intelligence (900 XGBoost trees Softmax)
  - Model 2: Occupancy Intelligence (Modules 1 & 2 Spatial Headcount)
  - Model 3: Energy Intelligence (Physics Engine + LightGBM Advisory)
  - Personalization Layer (Bayesian setpoint and logit adaptation)
  - Digital Twin State & Multi-Objective Adaptive Controller

Endpoints:
  GET  /                     -> Interactive Room Simulator Dashboard
  GET  /api/health           -> System health, models status & platform diagnostics
  POST /api/recommend        -> Real-time Room Simulation & 22-field AACI execution
  POST /api/override         -> Set manual user override
  POST /api/override/clear   -> Clear manual user override
  GET  /api/twins            -> Inspect fleet of virtual fan digital twins
"""

import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flask import Flask, jsonify, render_template, request

# Import lightweight serverless pipeline
try:
    from api.lightweight_pipeline import LightweightPipeline
    pipeline_class = LightweightPipeline
    engine_type = "Lightweight Pure-NumPy Simulator Engine (~55MB)"
except Exception:
    from aaci_ml.system_architecture.integration.pipeline import AACIPipeline
    pipeline_class = AACIPipeline
    engine_type = "Full C++ AACIPipeline"

# Initialize Flask app
templates_dir = os.path.join(ROOT_DIR, "templates")
app = Flask(__name__, template_folder=templates_dir)

# Global pipeline instance
_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = pipeline_class()
    return _pipeline_instance


@app.after_request
def add_cors_headers(response):
    """Enable CORS for browser clients."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def home():
    """Serves the interactive Room Simulator dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Returns system status, model availability, and platform diagnostics."""
    pipe = get_pipeline()
    return jsonify({
        "status": "healthy",
        "service": "FanWise AI - AACI Room Simulator",
        "version": "Phase 5 - Stage 5.1 Interactive Simulator",
        "platform": "Vercel Serverless Python",
        "engine": engine_type,
        "models": {
            "model_1_comfort": "AVAILABLE (ASHRAE DB-II 900-Tree XGBoost)",
            "model_2_occupancy": "AVAILABLE (Module 1 XGBoost + Module 2 Spatial Headcount)",
            "model_3_energy": "AVAILABLE (Physics Engine + LightGBM Advisory)",
            "personalization": "ACTIVE (Bayesian Logit & Setpoint Adaptation)",
        },
        "digital_twin": "ACTIVE",
        "adaptive_controller": "ACTIVE",
    })


@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def recommend():
    """
    Simulates room conditions and executes the full AACI pipeline.
    Returns 22-field IntegrationResult with decision steps & energy ROI.
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    payload = request.get_json(silent=True) or {}
    pipe = get_pipeline()

    telemetry: Dict[str, Any] = {
        "fan_id": payload.get("fan_id", "Bedroom_Fan_1"),
        "temperature_c": float(payload.get("temperature_c", 28.0)),
        "relative_humidity_pct": float(payload.get("relative_humidity_pct", 60.0)),
        "outdoor_temp_c": float(payload.get("outdoor_temp_c", 30.0)),
        "air_velocity_ms": float(payload["air_velocity_ms"]) if "air_velocity_ms" in payload and payload["air_velocity_ms"] is not None else None,
        "occupants_count": int(payload["occupants_count"]) if "occupants_count" in payload and payload["occupants_count"] is not None else 1,
        "occupancy_timeout_s": float(payload.get("occupancy_timeout_s", 300.0)),
        "vacant_duration_s": float(payload.get("vacant_duration_s", 0.0)),
        "light_lux": float(payload["light_lux"]) if "light_lux" in payload and payload["light_lux"] is not None else None,
        "co2_ppm": float(payload["co2_ppm"]) if "co2_ppm" in payload and payload["co2_ppm"] is not None else None,
        "comfort_preference": str(payload.get("comfort_preference", "neutral")),
        "setpoint_shift_c": float(payload.get("setpoint_shift_c", 0.0)),
        "clothing_clo": float(payload.get("clothing_clo", 0.6)),
        "metabolic_met": float(payload.get("metabolic_met", 1.2)),
        "operating_mode": str(payload.get("operating_mode", "auto")),
        "manual_speed": int(payload["manual_speed"]) if "manual_speed" in payload and payload["manual_speed"] is not None else None,
        "fan_speed": int(payload.get("fan_speed", 2)),
        "fan_mode": str(payload.get("fan_mode", "BLDC")).upper(),
        "room_category": payload.get("room_category", "Bedroom"),
        "data_source": "room_simulator_interactive",
        "provenance": payload.get("provenance", "SIMULATION"),
    }

    try:
        result = pipe.execute(
            raw_telemetry=telemetry,
            fan_id=telemetry["fan_id"],
            provenance=telemetry["provenance"],
        )
        out = result if isinstance(result, dict) else result.to_dict()
        return jsonify(out), 200
    except Exception as exc:
        return jsonify({
            "error": "Simulator execution failed",
            "details": str(exc),
            "fallback_speed": 2,
        }), 500


@app.route("/api/override", methods=["POST"])
def set_override():
    """Sets a temporary manual user override."""
    payload = request.get_json(silent=True) or {}
    fan_id = payload.get("fan_id", "Bedroom_Fan_1")
    speed = int(payload.get("speed", 3))
    duration_s = float(payload.get("duration_s", 600.0))

    pipe = get_pipeline()
    record = pipe.set_manual_override(fan_id=fan_id, speed=speed, duration_s=duration_s)
    return jsonify({
        "status": "success",
        "message": f"Manual override set to speed {speed} for {duration_s}s",
        "override_record": record,
    }), 200


@app.route("/api/override/clear", methods=["POST"])
def clear_override():
    """Clears active manual user override."""
    payload = request.get_json(silent=True) or {}
    fan_id = payload.get("fan_id", "Bedroom_Fan_1")

    pipe = get_pipeline()
    cleared = pipe.clear_manual_override(fan_id=fan_id)
    return jsonify({
        "status": "success",
        "cleared": cleared,
        "fan_id": fan_id,
    }), 200


@app.route("/api/twins", methods=["GET"])
def get_twins():
    """Returns digital twin virtual state for all managed fans."""
    pipe = get_pipeline()
    twins_state = getattr(pipe, "_twins", {})
    return jsonify({
        "active_twins_count": len(twins_state),
        "twins": twins_state,
    }), 200


@app.route("/api/energy-comparison", methods=["GET", "POST", "OPTIONS"])
def energy_comparison():
    """
    Evaluates 24-hour energy consumption across three operating strategies:
      1. Conventional Fixed-Speed (e.g. Speed 4 continuous)
      2. User Manual Operation (empirical residential RDD behavior)
      3. FanWise Adaptive AI (Model 1 Comfort + Model 2 Occupancy Gate + Model 3 BLDC)
    
    Returns hourly timeline, cumulative Wh, percentage savings, and unoccupied savings.
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    from api.energy_comparison import simulate_energy_comparison

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.args

    scenario = str(payload.get("scenario", "workday")).lower()
    fixed_speed = int(payload.get("fixed_speed", 4))
    tariff = float(payload.get("tariff_inr_per_kwh", 7.50))

    try:
        report = simulate_energy_comparison(
            scenario=scenario,
            fixed_speed_setting=fixed_speed,
            tariff_inr_per_kwh=tariff,
        )
        return jsonify(report), 200
    except Exception as exc:
        return jsonify({
            "error": "Energy comparison simulation failed",
            "details": str(exc),
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
