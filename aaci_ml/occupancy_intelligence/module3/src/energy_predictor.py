"""
AACI Module 3: Energy Intelligence Predictor API
Provides production-grade inference for smart fan energy consumption, operating duration,
and physical BLDC energy savings vs induction fan baselines.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rdd_loader import POWER_LOOKUP, SAVED_LOOKUP, INDUCTION_LOOKUP

MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), "models")
EVAL_DIR = os.path.join(os.path.dirname(SRC_DIR), "evaluation")


class EnergyIntelligencePredictor:
    """
    Independent runtime predictor for smart fan energy consumption, operating duration,
    and baseline induction motor energy savings.
    """
    
    def __init__(self, fan_id: str = "Bedroom_Fan_1", models_dir: str = MODELS_DIR):
        self.fan_id = fan_id
        self.models_dir = models_dir
        self.pipeline = None
        self._load_pipeline(fan_id)
        
    def _load_pipeline(self, fan_id: str):
        """Load trained model pipeline for specified fan."""
        model_path = os.path.join(self.models_dir, f"{fan_id}_rf_energy_pipeline.joblib")
        if os.path.exists(model_path):
            try:
                self.pipeline = joblib.load(model_path)
            except Exception as e:
                self.pipeline = None
        else:
            self.pipeline = None
            
    @staticmethod
    def get_power_ratings(speed: int) -> Dict[str, float]:
        """
        Return physical power consumption and induction baseline ratings in Watts.
        """
        spd = int(np.clip(speed, 0, 5))
        bldc_w = POWER_LOOKUP.get(spd, 0.0)
        ind_w = INDUCTION_LOOKUP.get(spd, 0.0)
        saved_w = SAVED_LOOKUP.get(spd, 0.0)
        
        return {
            "speed": spd,
            "bldc_power_W": bldc_w,
            "induction_baseline_power_W": ind_w,
            "power_saved_rate_W": saved_w,
            "saving_ratio_pct": round(saved_w / ind_w * 100.0, 1) if ind_w > 0 else 0.0
        }

    @staticmethod
    def calculate_exact_energy_and_savings(
        speed: int,
        duration_seconds: float
    ) -> Dict[str, float]:
        """
        Calculate deterministic firmware-level energy consumption and induction baseline savings.
        
        Parameters
        ----------
        speed : int
            Fan speed (0 to 5).
        duration_seconds : float
            Operating time in seconds.
            
        Returns
        -------
        Dict[str, float]
            Energy spent and saved in Joules and Watt-hours.
        """
        spd = int(np.clip(speed, 0, 5))
        dur = max(0.0, float(duration_seconds))
        
        power_w = POWER_LOOKUP.get(spd, 0.0)
        ind_w = INDUCTION_LOOKUP.get(spd, 0.0)
        saved_rate_w = SAVED_LOOKUP.get(spd, 0.0)
        
        e_spent_joules = power_w * dur
        e_saved_joules = saved_rate_w * dur
        e_induction_joules = ind_w * dur
        
        return {
            "speed": spd,
            "duration_seconds": round(dur, 1),
            "bldc_energy_spent_joules": round(e_spent_joules, 2),
            "bldc_energy_spent_Wh": round(e_spent_joules / 3600.0, 4),
            "induction_baseline_joules": round(e_induction_joules, 2),
            "induction_baseline_Wh": round(e_induction_joules / 3600.0, 4),
            "energy_saved_joules": round(e_saved_joules, 2),
            "energy_saved_Wh": round(e_saved_joules / 3600.0, 4),
            "energy_saved_pct": round(saved_rate_w / ind_w * 100.0, 1) if ind_w > 0 else 0.0
        }

    def predict_next_interval_energy(
        self,
        temperature: float,
        humidity: float,
        current_speed: int = 1,
        current_op_time: float = 150.0,
        timestamp: Optional[datetime] = None,
        mode: int = 0
    ) -> Dict[str, Any]:
        """
        Predict expected energy consumption (Wh) and operating duration for the next interval.
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        sin_day = np.sin(2 * np.pi * day_of_week / 7.0)
        cos_day = np.cos(2 * np.pi * day_of_week / 7.0)
        
        current_speed = int(np.clip(current_speed, 0, 5))
        lag_spent_wh = (POWER_LOOKUP.get(current_speed, 0.0) * current_op_time) / 3600.0
        
        # Prepare feature dict
        feats = {
            'temperature': float(temperature),
            'humidity': float(humidity),
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'sin_day': sin_day,
            'cos_day': cos_day,
            'is_weekend': is_weekend,
            'mode': int(mode),
            'lag_speed_1': float(current_speed),
            'lag_opTime_1': float(current_op_time),
            'lag_eSpent_Wh_1': float(lag_spent_wh),
            'lag_temp_1': float(temperature),
            'lag_hum_1': float(humidity),
            'roll_mean_speed_3': float(current_speed),
            'roll_mean_temp_3': float(temperature),
            'roll_mean_hum_3': float(humidity)
        }
        
        if self.pipeline is not None and "energy_model" in self.pipeline:
            model = self.pipeline["energy_model"]
            cols = self.pipeline["feature_cols"]
            X = np.array([[feats[c] for c in cols]])
            pred_energy_wh = float(np.clip(model.predict(X)[0], 0.0, None))
            
            dur_model = self.pipeline.get("duration_model")
            if dur_model is not None:
                pred_duration_sec = float(np.clip(dur_model.predict(X)[0], 0.0, 900.0))
            else:
                pred_duration_sec = float(current_op_time)
        else:
            # Deterministic empirical fallback
            pred_energy_wh = float(lag_spent_wh)
            pred_duration_sec = float(current_op_time)
            
        pred_energy_joules = pred_energy_wh * 3600.0
        
        # Modeled savings relative to induction baseline at current speed
        ind_w = INDUCTION_LOOKUP.get(current_speed, 0.0)
        ind_energy_wh = ind_w * (pred_duration_sec / 3600.0)
        saved_energy_wh = max(0.0, ind_energy_wh - pred_energy_wh)
        
        return {
            "fan_id": self.fan_id,
            "predicted_energy_Wh": round(pred_energy_wh, 4),
            "predicted_energy_Joules": round(pred_energy_joules, 2),
            "predicted_duration_seconds": round(pred_duration_sec, 1),
            "modeled_induction_baseline_Wh": round(ind_energy_wh, 4),
            "modeled_energy_saved_Wh": round(saved_energy_wh, 4),
            "bldc_rated_power_W": POWER_LOOKUP.get(current_speed, 0.0),
            "is_fan_running": bool(current_speed > 0)
        }

