"""
AACI Module 2: Unified Runtime Predictor API.
Provides an edge-ready inference engine combining Module 2A (Headcount) and Module 2B (Pattern Intelligence).

Outputs:
  - estimated_headcount (float >= 0.0)
  - headcount_rounded (int)
  - occupancy_state ('unoccupied' or 'occupied')
  - density_category ('empty', 'low', 'medium', 'high')
  - uncertainty_band (lower_bound, upper_bound)
  - room_utilization_context (typical baseline for room typology)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from robod_loader import FEATURE_GROUPS, ROOM_TYPES

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))


class Module2SpatialPredictor:
    """Production runtime inference engine for AACI Module 2 (Spatial Headcount)."""

    def __init__(
        self,
        candidate_model_name: str = "rf_multimodal",
        room_id: int = 1
    ):
        self.candidate_model_name = candidate_model_name
        self.room_id = room_id
        self.room_type = ROOM_TYPES.get(room_id, "Generic Room")

        # Load metadata
        meta_path = os.path.join(MODELS_DIR, "headcount_metadata.json")
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        model_info = self.metadata.get(candidate_model_name, {})
        self.algorithm = model_info.get("algorithm", "random_forest")
        self.feature_group = model_info.get("feature_group", "group4_multimodal")
        self.expected_features = FEATURE_GROUPS.get(self.feature_group, [])

        # Load model and scaler
        model_path = os.path.join(MODELS_DIR, f"headcount_{candidate_model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Candidate model artifact not found: {model_path}")
        self.model = joblib.load(model_path)

        scaler_path = os.path.join(MODELS_DIR, f"headcount_scaler_{candidate_model_name}.joblib")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # Empirical residual uncertainty per model
        self.uncertainty_margin = 1.50 # default +/- 1.5 persons for 90% band

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """Aligns input dictionary with expected feature columns, providing physical defaults."""
        row = {}
        for col in self.expected_features:
            if col in inputs:
                row[col] = float(inputs[col])
            else:
                # Provide reasonable default if sensor is not connected
                if "air_temperature" in col:
                    row[col] = float(inputs.get("air_temperature [Celsius]", inputs.get("Temperature", 26.0)))
                elif "indoor_relative_humidity" in col:
                    row[col] = float(inputs.get("indoor_relative_humidity [%]", inputs.get("Humidity", 72.0)))
                elif "indoor_co2" in col:
                    row[col] = float(inputs.get("indoor_co2 [ppm]", inputs.get("CO2", 450.0)))
                elif "sound_pressure" in col:
                    row[col] = float(inputs.get("sound_pressure_level [dba]", 50.0))
                elif "wifi" in col:
                    row[col] = float(inputs.get("wifi_connected_devices [number]", 0.0))
                elif "illuminance" in col:
                    row[col] = float(inputs.get("illuminance [lux]", 20.0))
                elif "voc" in col:
                    row[col] = float(inputs.get("voc [ppb]", 20.0))
                elif "pm2.5" in col:
                    row[col] = float(inputs.get("pm2.5 [mu_g/m3]", 5.0))
                elif "dry_bulb_temp" in col:
                    row[col] = float(inputs.get("dry_bulb_temp [Celsius]", 26.5))
                elif "outdoor_relative_humidity" in col:
                    row[col] = float(inputs.get("outdoor_relative_humidity [%]", 95.0))
                elif "outdoor_co2" in col:
                    row[col] = float(inputs.get("outdoor_co2 [ppm]", 460.0))
                elif "solar_radiation" in col:
                    row[col] = float(inputs.get("global_horizontal_solar_radiation [W/m2]", 0.0))
                elif "pressure" in col:
                    row[col] = float(inputs.get("baromatic_pressure [hPa]", 1005.0))
                elif "hour_sin" in col:
                    h = inputs.get("hour", 14)
                    row[col] = np.sin(2 * np.pi * h / 24.0)
                elif "hour_cos" in col:
                    h = inputs.get("hour", 14)
                    row[col] = np.cos(2 * np.pi * h / 24.0)
                elif "dow_sin" in col:
                    dow = inputs.get("day_of_week", 2)
                    row[col] = np.sin(2 * np.pi * dow / 7.0)
                elif "dow_cos" in col:
                    dow = inputs.get("day_of_week", 2)
                    row[col] = np.cos(2 * np.pi * dow / 7.0)
                elif "is_weekend" in col:
                    row[col] = float(inputs.get("is_weekend", 0.0))
                else:
                    row[col] = 0.0
        return pd.DataFrame([row])[self.expected_features]

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generates continuous headcount estimate and spatial categorical load."""
        X_df = self._prepare_inputs(inputs)

        if self.scaler is not None:
            X_proc = self.scaler.transform(X_df)
        else:
            X_proc = X_df.values

        raw_pred = float(self.model.predict(X_proc)[0])
        headcount_est = max(0.0, raw_pred)
        headcount_int = int(round(headcount_est))

        # Categorize density
        if headcount_int == 0:
            density = "empty"
            state = "unoccupied"
        elif headcount_int <= 3:
            density = "low (1-3 persons)"
            state = "occupied"
        elif headcount_int <= 10:
            density = "medium (4-10 persons)"
            state = "occupied"
        else:
            density = "high (>10 persons)"
            state = "occupied"

        lower_bound = max(0, int(round(headcount_est - self.uncertainty_margin)))
        upper_bound = int(round(headcount_est + self.uncertainty_margin))

        return {
            "estimated_headcount": round(headcount_est, 2),
            "headcount_rounded": headcount_int,
            "occupancy_state": state,
            "density_category": density,
            "uncertainty_90pct_interval": [lower_bound, upper_bound],
            "model_used": self.candidate_model_name,
            "room_context": {
                "room_id": self.room_id,
                "room_type": self.room_type
            }
        }


if __name__ == "__main__":
    predictor = Module2SpatialPredictor(candidate_model_name="rf_multimodal", room_id=1)
    sample = {
        "air_temperature [Celsius]": 24.5,
        "indoor_relative_humidity [%]": 62.0,
        "indoor_co2 [ppm]": 850.0,
        "sound_pressure_level [dba]": 62.0,
        "wifi_connected_devices [number]": 15.0,
        "illuminance [lux]": 350.0,
        "hour": 15
    }
    res = predictor.predict(sample)
    print("Module 2 Predictor Sample Output:")
    print(res)
