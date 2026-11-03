"""
AACI – Thermal Comfort Prediction Function
==========================================
Atomberg Adaptive Comfort Intelligence (AACI)

Usage:
    from predict import predict_comfort

    result = predict_comfort(
        air_temp_c=27.5,
        relative_humidity=60.0,
        air_velocity_ms=0.3,
        outdoor_monthly_temp_c=32.0,
        clo=0.5,
        met=1.2,
        season="Summer",
        building_type="Office",
        cooling_strategy="Air Conditioned",
    )
    print(result)
    # {
    #   "predicted_class": "Hot",
    #   "class_id": 2,
    #   "probabilities": {"Cold": 0.03, "Neutral": 0.18, "Hot": 0.79},
    #   "confidence": 0.79,
    #   "recommendation": "Consider increasing fan speed or reducing temperature."
    # }
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Paths to saved artifacts
# ─────────────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH     = os.path.join(_DIR, "aaci_comfort_model.joblib")
_PREPROC_PATH   = os.path.join(_DIR, "aaci_preprocessor.joblib")
_METADATA_PATH  = os.path.join(_DIR, "aaci_model_metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load model and preprocessor (loads once, reuses on subsequent calls)
# ─────────────────────────────────────────────────────────────────────────────
_model      = None
_preprocessor = None
_metadata   = None

def _load_artifacts():
    global _model, _preprocessor, _metadata
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {_MODEL_PATH}. "
                "Run 02_train_model.py first."
            )
        _model        = joblib.load(_MODEL_PATH)
        _preprocessor = joblib.load(_PREPROC_PATH)
        with open(_METADATA_PATH) as f:
            _metadata = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# AACI recommendation engine
# ─────────────────────────────────────────────────────────────────────────────
_RECOMMENDATIONS = {
    "Cold": (
        "Occupant is likely too cold. "
        "Reduce fan speed or increase room temperature to improve comfort."
    ),
    "Neutral": (
        "Occupant is thermally comfortable. "
        "Maintain current fan speed and temperature settings."
    ),
    "Hot": (
        "Occupant is likely too warm. "
        "Increase fan speed or reduce room temperature to improve comfort."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────────────────────────────────────
def predict_comfort(
    air_temp_c: float,
    relative_humidity: float,
    air_velocity_ms: float,
    outdoor_monthly_temp_c: Optional[float] = None,
    clo: float = 0.6,
    met: float = 1.2,
    season: str = "Summer",
    building_type: str = "Office",
    cooling_strategy: str = "Air Conditioned",
) -> dict:
    """
    Predict indoor thermal comfort class using the AACI model.

    Parameters
    ----------
    air_temp_c : float
        Indoor air temperature in Celsius (e.g., 27.5).
    relative_humidity : float
        Relative humidity percentage (0–100, e.g., 60.0).
    air_velocity_ms : float
        Air velocity at occupant level in m/s (e.g., 0.3 for fan on).
    outdoor_monthly_temp_c : float, optional
        Outdoor monthly average temperature in Celsius (defaults to None → imputed).
    clo : float
        Clothing insulation value (default 0.6 = typical indoor clothing).
    met : float
        Metabolic rate (default 1.2 = seated, light office work).
    season : str
        One of: "Summer", "Winter", "Spring", "Autumn".
    building_type : str
        One of: "Office", "Classroom", "Multifamily housing", "Others", "Senior center".
    cooling_strategy : str
        One of: "Air Conditioned", "Naturally Ventilated", "Mixed Mode", "NA".

    Returns
    -------
    dict with keys:
        predicted_class : str     ("Cold", "Neutral", or "Hot")
        class_id        : int     (0, 1, or 2)
        probabilities   : dict    {class_name: probability}
        confidence      : float   (probability of the predicted class)
        recommendation  : str     (AACI fan control suggestion)
        model_name      : str     (name of the underlying model)
        inputs_used     : dict    (echo of the input values)
    """
    _load_artifacts()

    CLASS_NAMES = _metadata["class_names"]  # ["Cold", "Neutral", "Hot"]
    NUM_FEATS   = _metadata["numeric_features"]
    CAT_FEATS   = _metadata["categorical_features"]

    # Build input DataFrame matching the training feature order
    input_data = {
        "Air temperature (C)":               air_temp_c,
        "Relative humidity (%)":             relative_humidity,
        "Air velocity (m/s)":                air_velocity_ms,
        "Outdoor monthly air temperature (C)": outdoor_monthly_temp_c,  # may be NaN
        "Clo":                               clo,
        "Met":                               met,
        "Season":                            season,
        "Building type":                     building_type,
        "Cooling startegy_building level":   cooling_strategy,
    }

    input_df = pd.DataFrame([input_data])

    # Apply trained preprocessor
    X_processed = _preprocessor.transform(input_df)

    # Predict class and probabilities
    class_id = int(_model.predict(X_processed)[0])
    probs     = _model.predict_proba(X_processed)[0]  # shape (3,)

    predicted_class = CLASS_NAMES[class_id]
    confidence      = float(probs[class_id])

    probabilities = {
        cls: round(float(p), 4)
        for cls, p in zip(CLASS_NAMES, probs)
    }

    return {
        "predicted_class": predicted_class,
        "class_id":        class_id,
        "probabilities":   probabilities,
        "confidence":      round(confidence, 4),
        "recommendation":  _RECOMMENDATIONS[predicted_class],
        "model_name":      _metadata["model_name"],
        "inputs_used": {
            "air_temp_c":           air_temp_c,
            "relative_humidity":    relative_humidity,
            "air_velocity_ms":      air_velocity_ms,
            "outdoor_temp_c":       outdoor_monthly_temp_c,
            "clo":                  clo,
            "met":                  met,
            "season":               season,
            "building_type":        building_type,
            "cooling_strategy":     cooling_strategy,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke test when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("AACI Prediction Function – Demo")
    print("=" * 50)

    test_cases = [
        {
            "label": "Hot summer day, slow fan",
            "air_temp_c": 32.0,
            "relative_humidity": 70.0,
            "air_velocity_ms": 0.1,
            "outdoor_monthly_temp_c": 38.0,
            "season": "Summer",
        },
        {
            "label": "Comfortable office, moderate fan",
            "air_temp_c": 24.5,
            "relative_humidity": 50.0,
            "air_velocity_ms": 0.2,
            "outdoor_monthly_temp_c": 28.0,
            "season": "Summer",
        },
        {
            "label": "Cool room, high fan speed",
            "air_temp_c": 20.0,
            "relative_humidity": 35.0,
            "air_velocity_ms": 0.8,
            "outdoor_monthly_temp_c": 18.0,
            "season": "Winter",
        },
    ]

    for case in test_cases:
        label = case.pop("label")
        result = predict_comfort(**case)
        print(f"\n[{label}]")
        print(f"  Input  : Ta={case.get('air_temp_c')}C, RH={case.get('relative_humidity')}%, "
              f"v={case.get('air_velocity_ms')}m/s")
        print(f"  Result : {result['predicted_class']}  (confidence={result['confidence']:.1%})")
        print(f"  Probs  : Cold={result['probabilities']['Cold']:.3f} | "
              f"Neutral={result['probabilities']['Neutral']:.3f} | "
              f"Hot={result['probabilities']['Hot']:.3f}")
        print(f"  Action : {result['recommendation']}")

