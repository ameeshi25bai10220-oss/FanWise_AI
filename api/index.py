"""
FanWise AI / AACI - Vercel Serverless API & Web Application
============================================================
Lightweight serverless runtime for Vercel deployment:
  - Model 1: Thermal Comfort Intelligence (ASHRAE DB-II 900 XGBoost trees)
  - Model 2: Occupancy Intelligence (Modules 1 & 2 Spatial)
  - Model 3: Energy Intelligence (Physics Engine + LightGBM Advisory)
  - Fan Digital Twin Virtual Runtime & TwinSupervisor
  - Multi-Objective Adaptive Controller & Safety Guard (P1-P5)

Routes:
  GET  /                     -> Interactive Web Dashboard
  GET  /api/health           -> System health and model availability diagnostics
  POST /api/recommend        -> Complete 22-field AACI pipeline execution
  POST /api/override         -> Set manual user override (P2 priority)
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

# Try importing LightweightPipeline (fast serverless runner ~55MB bundle)
# Fallback to AACIPipeline if heavy ML dependencies are present
try:
    from api.lightweight_pipeline import LightweightPipeline
    pipeline_class = LightweightPipeline
    engine_type = "Lightweight Pure-NumPy Engine (~55MB)"
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
    """Enable CORS for external web requests or headless clients."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def home():
    """Serves the main interactive dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Returns system status, model availability, and platform diagnostics."""
    pipe = get_pipeline()
    return jsonify({
        "status": "healthy",
        "service": "FanWise AI - AACI Multi-Model Pipeline",
        "version": "Phase 5 - Stage 5",
        "platform": "Vercel Serverless Python",
        "engine": engine_type,
        "models": {
            "model_1_comfort": "AVAILABLE (ASHRAE DB-II XGBoost)",
            "model_2_occupancy": "AVAILABLE (Module 1 XGBoost + Module 2 Spatial)",
            "model_3_energy": "AVAILABLE (Physics Engine + LightGBM Advisory)",
        },
        "digital_twin": "ACTIVE",
        "adaptive_controller": "ACTIVE",
    })


@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def recommend():
    """
    Executes the full end-to-end AACI pipeline on incoming telemetry.
    Returns the complete 22-field IntegrationResult payload.
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    payload = request.get_json(silent=True) or {}
    pipe = get_pipeline()

    telemetry: Dict[str, Any] = {
        "fan_id": payload.get("fan_id", "Bedroom_Fan_1"),
        "temperature_c": float(payload.get("temperature_c", 30.5)),
        "relative_humidity_pct": float(payload.get("relative_humidity_pct", 65.0)),
        "light_lux": float(payload["light_lux"]) if "light_lux" in payload and payload["light_lux"] is not None else None,
        "co2_ppm": float(payload["co2_ppm"]) if "co2_ppm" in payload and payload["co2_ppm"] is not None else None,
        "fan_speed": int(payload.get("fan_speed", 2)),
        "fan_mode": payload.get("fan_mode", "BLDC"),
        "room_category": payload.get("room_category", "Bedroom"),
        "data_source": "vercel_api_request",
        "provenance": payload.get("provenance", "TELEMETRY"),
    }

    try:
        result = pipe.execute(
            raw_telemetry=telemetry,
            fan_id=telemetry["fan_id"],
            provenance=telemetry["provenance"],
        )
        # Handle dict or dataclass
        out = result if isinstance(result, dict) else result.to_dict()
        return jsonify(out), 200
    except Exception as exc:
        return jsonify({
            "error": "Pipeline execution failed",
            "details": str(exc),
            "fallback_speed": 2,
        }), 500


@app.route("/api/override", methods=["POST"])
def set_override():
    """Sets a manual user override for a specific fan."""
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
