"""
AACI Module 1: OccupancyPredictor Inference API.
Provides an edge-ready, decoupled runtime inference interface for room occupancy detection.

Features:
  - Supports all 4 algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost)
    across all 3 feature sets (Set A, Set B, Set C).
  - Validates physical sensor ranges (Temperature, Humidity, Light, CO2).
  - Automatically calculates psychrometric HumidityRatio from Temp and RH if omitted.
  - Returns well-calibrated occupancy probability P(occupied) in [0.0, 1.0].
  - Applies calibrated decision threshold tau* with configurable override.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional

from data_loader import FEATURE_SETS

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def calculate_humidity_ratio(temperature_c: float, relative_humidity_pct: float, pressure_kpa: float = 101.325) -> float:
    """
    Computes psychrometric Humidity Ratio (kg water vapor per kg dry air)
    using standard saturation vapor pressure equations (Magnus-Tetens).
    """
    # Saturation vapor pressure in hPa
    e_sat = 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    # Actual vapor pressure in hPa
    e_act = (relative_humidity_pct / 100.0) * e_sat
    # Pressure in hPa
    p_hpa = pressure_kpa * 10.0
    # Humidity ratio W = 0.622 * e / (P - e)
    w = 0.622 * e_act / max(1.0, (p_hpa - e_act))
    return float(w)


class OccupancyPredictor:
    """Production runtime inference engine for AACI Occupancy Intelligence Module 1."""

    def __init__(
        self,
        algorithm: str = "random_forest",
        feature_set: str = "set_a",
        use_calibrated: bool = True,
        threshold_override: Optional[float] = None
    ):
        self.algorithm = algorithm.lower()
        self.feature_set = feature_set.lower()
        self.use_calibrated = use_calibrated
        
        # Load metadata to get optimal threshold
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                
        model_key = f"{self.algorithm}_{self.feature_set}"
        model_info = self.metadata.get("models", {}).get(model_key, {})
        default_tau = model_info.get("recommended_production_threshold", 0.50)
        
        self.threshold = threshold_override if threshold_override is not None else default_tau
        self.expected_features = FEATURE_SETS.get(self.feature_set, [])
        
        # Load artifacts
        suffix = "_calibrated.joblib" if use_calibrated else ".joblib"
        model_filename = f"module1_{self.algorithm}_{self.feature_set}{suffix}"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
            
        self.model = joblib.load(model_path)
        
        # Load scaler if Logistic Regression
        scaler_filename = f"preprocessor_{self.feature_set}.joblib"
        scaler_path = os.path.join(MODELS_DIR, scaler_filename)
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) and self.algorithm == "logistic_regression" else None

    def _validate_and_prepare_input(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """Validates physical ranges and formats input into an aligned single-row DataFrame."""
        temp = float(inputs.get("Temperature", inputs.get("temperature", 21.0)))
        rh = float(inputs.get("Humidity", inputs.get("humidity", 30.0)))
        
        # Validate physical limits
        if not (-10.0 <= temp <= 60.0):
            raise ValueError(f"Unphysical Temperature reading: {temp} deg C")
        if not (0.0 <= rh <= 100.0):
            raise ValueError(f"Unphysical Relative Humidity reading: {rh}%")
            
        row = {"Temperature": temp, "Humidity": rh}
        
        # Psychrometric Humidity Ratio
        if "HumidityRatio" in self.expected_features:
            if "HumidityRatio" in inputs or "humidity_ratio" in inputs:
                row["HumidityRatio"] = float(inputs.get("HumidityRatio", inputs.get("humidity_ratio")))
            else:
                row["HumidityRatio"] = calculate_humidity_ratio(temp, rh)
                
        # Light
        if "Light" in self.expected_features:
            light = float(inputs.get("Light", inputs.get("light", 0.0)))
            if light < 0.0:
                raise ValueError(f"Negative Light reading: {light} lux")
            row["Light"] = light
            
        # CO2
        if "CO2" in self.expected_features:
            co2 = float(inputs.get("CO2", inputs.get("co2", 400.0)))
            if co2 < 200.0 or co2 > 10000.0:
                raise ValueError(f"Unphysical CO2 reading: {co2} ppm")
            row["CO2"] = co2
            
        # Hour
        if "Hour" in self.expected_features:
            hour = inputs.get("Hour", inputs.get("hour", None))
            if hour is None and "date" in inputs:
                hour = pd.to_datetime(inputs["date"]).hour
            elif hour is None:
                hour = 12
            row["Hour"] = int(hour)
            
        df = pd.DataFrame([row])[self.expected_features]
        return df

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs occupancy inference on sensor inputs.
        Returns:
          {
            "occupied": int (0 or 1),
            "probability": float (0.0 to 1.0),
            "threshold": float,
            "confidence": float,
            "algorithm": str,
            "feature_set": str
          }
        """
        X_df = self._validate_and_prepare_input(inputs)
        
        if self.scaler is not None:
            X_proc = self.scaler.transform(X_df)
        else:
            X_proc = X_df.values
            
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_proc)[0]
            prob_occupied = float(probs[1])
        else:
            prob_occupied = float(self.model.predict(X_proc)[0])
            
        prob_occupied = min(max(prob_occupied, 0.0), 1.0)
        is_occupied = int(prob_occupied >= self.threshold)
        confidence = float(prob_occupied if is_occupied == 1 else (1.0 - prob_occupied))
        
        return {
            "occupied": is_occupied,
            "probability": round(prob_occupied, 4),
            "threshold": round(self.threshold, 4),
            "confidence": round(confidence, 4),
            "algorithm": self.algorithm,
            "feature_set": self.feature_set
        }


if __name__ == "__main__":
    predictor = OccupancyPredictor(algorithm="random_forest", feature_set="set_a")
    
    # Test case 1: Typical occupied daytime office meeting
    occupied_input = {
        "Temperature": 22.5,
        "Humidity": 28.0,
        "Light": 450.0,
        "CO2": 950.0
    }
    res1 = predictor.predict(occupied_input)
    print("Test Case 1 (Occupied Meeting):", res1)
    
    # Test case 2: Empty room at night (lights off, ambient CO2)
    empty_input = {
        "Temperature": 20.0,
        "Humidity": 25.0,
        "Light": 0.0,
        "CO2": 420.0
    }
    res2 = predictor.predict(empty_input)
    print("Test Case 2 (Empty Room Night):", res2)

