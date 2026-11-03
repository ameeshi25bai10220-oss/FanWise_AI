"""
AACI System Architecture – Energy Intelligence Adapter (Model 3)
================================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Wraps both the deterministic physics engine and the frozen LightGBM ML forecaster from Model 3.
Maintains strict separation between:
  1. Deterministic physics engine: authoritative candidate speed power & energy calculations for known speed and duration.
  2. Machine learning forecaster: purely advisory next-observed-packet energy forecast (R2 ~ 0.1515).

Preserves exact 26-feature schema, per-fan causal rolling buffer, and zero future leakage.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union
import datetime
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Model 3 directory isolation & sys.path resolution for preprocessor unpickling
MODEL3_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "aaci_ml", "v3_energy_intelligence"))
if MODEL3_DIR not in sys.path:
    sys.path.insert(0, MODEL3_DIR)

from .contract import (
    AdapterOutputEnvelope,
    InputValidityStatus,
    InputFreshnessInfo,
    ModelAvailabilityStatus,
    ValueProvenance,
    DegradedState,
)
from .constants import (
    BLDC_POWER_MAP,
    INDUCTION_POWER_MAP,
    MODEL3_ML_BENCHMARK_R2,
    MODEL3_ML_BENCHMARK_MAE_WH,
    MODEL3_FORECAST_HORIZON,
    VALID_TEMP_RANGE_C,
    VALID_HUMIDITY_RANGE_PCT,
    VALID_FAN_SPEEDS,
    SENSOR_SENTINEL_VALUES,
    MAX_SENSOR_STALENESS_SECONDS,
)


class TelemetryObservation:
    """Historical single-packet telemetry record for causal feature buffer."""
    def __init__(
        self,
        dt: datetime.datetime,
        temperature: float,
        humidity: float,
        speed: int,
        espent_wh: float,
    ):
        self.dt = dt
        self.temperature = temperature
        self.humidity = humidity
        self.speed = speed
        self.espent_wh = espent_wh


class EnergyIntelligenceAdapter:
    """
    Production adapter wrapping frozen Model 3 Energy Intelligence.
    Provides candidate speed physics evaluations and advisory ML next-packet energy forecasts.
    """

    MODEL_ID = "AACI_Model_3_Energy_LightGBM_PhysicsEngine"
    ADAPTER_NAME = "EnergyIntelligenceAdapter"

    NUMERIC_FEATURES = [
        "temperature", "humidity", "sin_hour", "cos_hour", "sin_month", "cos_month",
        "day_of_week", "is_weekend", "is_night", "temp_lag_1", "temp_lag_2",
        "humidity_lag_1", "humidity_lag_2", "temp_diff_1_2", "humidity_diff_1_2",
        "temp_rolling_mean_3", "humidity_rolling_mean_3", "temp_rolling_mean_6",
        "humidity_rolling_mean_6", "speed_lag_1", "speed_lag_2", "speed_changed",
        "eSpent_wh_lag_1", "eSpent_wh_lag_2", "eSpent_wh_rolling_sum_3",
        "eSpent_wh_rolling_sum_12"
    ]

    def __init__(
        self,
        models_dir: Optional[str] = None,
        max_buffer_depth: int = 20,
    ):
        self.models_dir = models_dir or os.path.join(MODEL3_DIR, "models")
        self.max_buffer_depth = max_buffer_depth
        self.buffers: Dict[str, List[TelemetryObservation]] = {}
        self._ml_model = None
        self._preprocessor = None
        self._artifacts_loaded = False
        self._load_error: Optional[str] = None

    def _ensure_artifacts_loaded(self) -> bool:
        """Loads LightGBM model and preprocessor joblib artifacts lazily."""
        if self._artifacts_loaded:
            return True
        if self._load_error is not None:
            return False

        model_path = os.path.join(self.models_dir, "aaci_energy_lightgbm_model.joblib")
        preproc_path = os.path.join(self.models_dir, "aaci_energy_preprocessor.joblib")

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Energy ML model not found at {model_path}")
            if not os.path.exists(preproc_path):
                raise FileNotFoundError(f"Energy preprocessor not found at {preproc_path}")

            # Ensure MODEL3_DIR is on sys.path for unpickling src.preprocessing
            if MODEL3_DIR not in sys.path:
                sys.path.insert(0, MODEL3_DIR)

            self._ml_model = joblib.load(model_path)
            self._preprocessor = joblib.load(preproc_path)
            self._artifacts_loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def update_history(
        self,
        fan_id: str,
        dt: datetime.datetime,
        temperature: float,
        humidity: float,
        speed: int,
        espent_wh: float,
    ) -> None:
        """Appends a validated historical observation to the fan's causal buffer."""
        if fan_id not in self.buffers:
            self.buffers[fan_id] = []
        obs = TelemetryObservation(dt, temperature, humidity, speed, espent_wh)
        self.buffers[fan_id].append(obs)
        # Keep buffer bounded to max_buffer_depth
        if len(self.buffers[fan_id]) > self.max_buffer_depth:
            self.buffers[fan_id] = self.buffers[fan_id][-self.max_buffer_depth:]

    def clear_history(self, fan_id: Optional[str] = None) -> None:
        """Clears buffer for testing or device reset."""
        if fan_id:
            self.buffers.pop(fan_id, None)
        else:
            self.buffers.clear()

    @staticmethod
    def _parse_timestamp(raw_ts: Any) -> datetime.datetime:
        """Parses timestamp into datetime object."""
        if isinstance(raw_ts, datetime.datetime):
            return raw_ts
        if isinstance(raw_ts, datetime.date):
            return datetime.datetime(raw_ts.year, raw_ts.month, raw_ts.day, 12, 0)
        if isinstance(raw_ts, str):
            try:
                return datetime.datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except Exception:
                try:
                    return datetime.datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
        return datetime.datetime.now()

    @staticmethod
    def _resolve_room_category(fan_id: str, room_category_input: Optional[str]) -> str:
        """Determines room category with fallback to 'Bedroom'."""
        if room_category_input in ["Bedroom", "Drawingroom", "Lounge"]:
            return room_category_input
        lower_id = fan_id.lower()
        if "drawing" in lower_id:
            return "Drawingroom"
        elif "lounge" in lower_id:
            return "Lounge"
        return "Bedroom"

    def _build_feature_row(
        self,
        fan_id: str,
        current_dt: datetime.datetime,
        current_temp: float,
        current_rh: float,
        room_category: str,
    ) -> Optional[pd.DataFrame]:
        """
        Builds the exact 26-feature single-row DataFrame using strictly retrospective lagged values.
        Returns None if buffer has insufficient history (cold start).
        """
        history = self.buffers.get(fan_id, [])
        # We require at least 2 historical observations to compute lag_1 and lag_2
        if len(history) < 2:
            return None

        # Sort history by datetime just to be strictly chronological
        sorted_hist = sorted(history, key=lambda x: x.dt)

        lag_1 = sorted_hist[-1]
        lag_2 = sorted_hist[-2]

        hour_val = current_dt.hour + current_dt.minute / 60.0
        dow_val = current_dt.weekday()
        month_val = current_dt.month

        # Cyclic temporal
        sin_hour = np.sin(2.0 * np.pi * hour_val / 24.0)
        cos_hour = np.cos(2.0 * np.pi * hour_val / 24.0)
        sin_month = np.sin(2.0 * np.pi * (month_val - 1) / 12.0)
        cos_month = np.cos(2.0 * np.pi * (month_val - 1) / 12.0)
        day_of_week = dow_val
        is_weekend = int(dow_val >= 5)
        is_night = int((current_dt.hour >= 22) or (current_dt.hour < 6))

        # Lags
        temp_lag_1 = lag_1.temperature
        temp_lag_2 = lag_2.temperature
        humidity_lag_1 = lag_1.humidity
        humidity_lag_2 = lag_2.humidity
        temp_diff_1_2 = temp_lag_1 - temp_lag_2
        humidity_diff_1_2 = humidity_lag_1 - humidity_lag_2

        # Rolling means of retrospective temperature and humidity
        temps_hist = [h.temperature for h in sorted_hist]
        rh_hist = [h.humidity for h in sorted_hist]

        temp_rolling_mean_3 = float(np.mean(temps_hist[-3:]))
        humidity_rolling_mean_3 = float(np.mean(rh_hist[-3:]))
        temp_rolling_mean_6 = float(np.mean(temps_hist[-6:]))
        humidity_rolling_mean_6 = float(np.mean(rh_hist[-6:]))

        # Operational lags
        speed_lag_1 = float(lag_1.speed)
        speed_lag_2 = float(lag_2.speed)
        speed_changed = float(speed_lag_1 != speed_lag_2)

        espent_hist = [h.espent_wh for h in sorted_hist]
        espent_wh_lag_1 = float(lag_1.espent_wh)
        espent_wh_lag_2 = float(lag_2.espent_wh)
        espent_wh_rolling_sum_3 = float(np.sum(espent_hist[-3:]))
        espent_wh_rolling_sum_12 = float(np.sum(espent_hist[-12:]))

        row = {
            "temperature": current_temp,
            "humidity": current_rh,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "sin_month": sin_month,
            "cos_month": cos_month,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "temp_lag_1": temp_lag_1,
            "temp_lag_2": temp_lag_2,
            "humidity_lag_1": humidity_lag_1,
            "humidity_lag_2": humidity_lag_2,
            "temp_diff_1_2": temp_diff_1_2,
            "humidity_diff_1_2": humidity_diff_1_2,
            "temp_rolling_mean_3": temp_rolling_mean_3,
            "humidity_rolling_mean_3": humidity_rolling_mean_3,
            "temp_rolling_mean_6": temp_rolling_mean_6,
            "humidity_rolling_mean_6": humidity_rolling_mean_6,
            "speed_lag_1": speed_lag_1,
            "speed_lag_2": speed_lag_2,
            "speed_changed": speed_changed,
            "eSpent_wh_lag_1": espent_wh_lag_1,
            "eSpent_wh_lag_2": espent_wh_lag_2,
            "eSpent_wh_rolling_sum_3": espent_wh_rolling_sum_3,
            "eSpent_wh_rolling_sum_12": espent_wh_rolling_sum_12,
            "room_category": room_category,
        }

        return pd.DataFrame([row])

    def predict(self, inputs: Dict[str, Any]) -> AdapterOutputEnvelope:
        """
        Main energy inference entry point.

        Accepts either flat dict or hierarchical Common State Representation (CSR):
        Flat keys:
          - 'current_speed' or 'fan_speed' (int 0..5, optional, default 3)
          - 'fan_mode' or 'mode' (str, 'BLDC' or 'Induction', default 'BLDC')
          - 'op_time_s' (float, optional, default 14.4s for standard single packet)
          - 'espent_wh' or 'eSpent' (float, optional, past observed packet energy)
          - 'temperature' or 'temperature_c' (float, required for ML forecast)
          - 'humidity' or 'relative_humidity_pct' (float, required for ML forecast)
          - 'fan_id' or 'room_id' (str, fan identifier for lag buffer)
          - 'room_category' (str, optional, 'Bedroom' / 'Drawingroom' / 'Lounge')
          - 'timestamp' or 'datetime' (str or datetime, optional)
          - 'sensor_age_seconds' (float, optional)
        """
        validity = InputValidityStatus()
        freshness = InputFreshnessInfo(max_staleness_threshold_s=MAX_SENSOR_STALENESS_SECONDS)
        provenance = ValueProvenance()
        limitations = []
        missing_inputs = []

        # 1. Unpack inputs
        env = inputs.get("environmental", {}) if isinstance(inputs.get("environmental"), dict) else inputs
        fan_state = inputs.get("fan_state", {}) if isinstance(inputs.get("fan_state"), dict) else inputs
        meta = inputs.get("metadata", {}) if isinstance(inputs.get("metadata"), dict) else inputs

        raw_temp = env.get("temperature_c", inputs.get("temperature", inputs.get("temperature_c")))
        raw_rh = env.get("relative_humidity_pct", inputs.get("humidity", inputs.get("relative_humidity_pct")))
        raw_speed = fan_state.get("current_speed", inputs.get("current_speed", inputs.get("fan_speed", 3)))
        raw_mode = fan_state.get("mode", inputs.get("fan_mode", inputs.get("mode", "BLDC")))
        raw_op_time = fan_state.get("op_time_s", inputs.get("op_time_s", 14.4))
        raw_espent = fan_state.get("espent_wh", inputs.get("espent_wh", inputs.get("eSpent", 0.08)))
        raw_fan_id = meta.get("fan_id", meta.get("room_id", inputs.get("fan_id", inputs.get("room_id", "Bedroom_Fan_1"))))
        raw_room_cat = meta.get("room_category", inputs.get("room_category"))
        raw_ts = meta.get("timestamp", inputs.get("timestamp", inputs.get("datetime")))
        sensor_age = meta.get("sensor_age_seconds", inputs.get("sensor_age_seconds"))

        # Freshness
        if sensor_age is not None:
            try:
                age_val = float(sensor_age)
                freshness.sensor_age_seconds = age_val
                freshness.is_stale = age_val > MAX_SENSOR_STALENESS_SECONDS
                if freshness.is_stale:
                    validity.validation_warnings.append(
                        f"Sensor data is stale ({age_val:.1f}s > {MAX_SENSOR_STALENESS_SECONDS}s)"
                    )
            except (ValueError, TypeError):
                validity.validation_warnings.append(f"Invalid sensor_age_seconds value: {sensor_age}")

        fan_id = str(raw_fan_id)
        mode = "Induction" if str(raw_mode).strip().lower() == "induction" else "BLDC"
        power_map = INDUCTION_POWER_MAP if mode == "Induction" else BLDC_POWER_MAP

        provenance.assumed["fan_mode"] = mode
        if mode == "Induction":
            validity.validation_warnings.append("Fan mode set to Induction; BLDC is primary standard.")

        # Op time in seconds
        try:
            op_time_s = max(0.1, float(raw_op_time))
            provenance.measured["op_time_s"] = op_time_s
        except (ValueError, TypeError):
            op_time_s = 14.4
            provenance.assumed["op_time_s"] = 14.4
            validity.validation_warnings.append(f"Invalid op_time_s: {raw_op_time}; using 14.4s default")

        # Speed validation
        try:
            current_speed = int(raw_speed)
            if current_speed not in VALID_FAN_SPEEDS:
                validity.validation_warnings.append(f"Current speed {raw_speed} not in (0-5); clamping to nearest")
                current_speed = min(max(current_speed, 0), 5)
            provenance.measured["current_speed"] = current_speed
        except (ValueError, TypeError):
            current_speed = 3
            provenance.assumed["current_speed"] = 3
            validity.validation_warnings.append(f"Invalid speed {raw_speed}; default to 3")

        # 2. Physics Engine Candidate Evaluations (Authoritative, Deterministic)
        # Authoritative calculations for all speeds 0..5
        candidate_eval_table: List[Dict[str, Any]] = []
        for s in range(6):
            power_w = power_map[s]
            # Energy (Wh) = Power (W) * Time (s) / 3600 (s/h)
            energy_wh = round(power_w * op_time_s / 3600.0, 4)
            candidate_eval_table.append({
                "speed": s,
                "power_w": power_w,
                "energy_wh_physics": energy_wh,
                "source": "physics_engine_deterministic",
            })

        provenance.estimated["candidate_evaluation_table"] = candidate_eval_table
        physics_engine_available = True

        # 3. Machine Learning Forecaster (Advisory Next-Packet Energy)
        # Rules:
        # - The ML forecaster is strictly ADVISORY.
        # - It must never override deterministic energy calculations.
        # - It must never directly command fan speed.
        limitations.append(
            "Model 3 ML next-packet energy prediction is purely advisory (R2 ~ 0.1515). "
            "It must not be used as an authoritative energy calculation or as a direct fan-speed command."
        )

        dt = self._parse_timestamp(raw_ts)
        room_cat = self._resolve_room_category(fan_id, raw_room_cat)

        # Validate environmental inputs for ML
        temp_val: Optional[float] = None
        if raw_temp is not None:
            try:
                t = float(raw_temp)
                if t not in SENSOR_SENTINEL_VALUES and (VALID_TEMP_RANGE_C[0] <= t <= VALID_TEMP_RANGE_C[1]):
                    temp_val = t
                    provenance.measured["temperature"] = temp_val
                else:
                    validity.validation_warnings.append(f"Invalid/sentinel temperature: {raw_temp}")
            except (ValueError, TypeError):
                validity.validation_warnings.append(f"Non-numeric temperature: {raw_temp}")

        rh_val: Optional[float] = None
        if raw_rh is not None:
            try:
                rh = float(raw_rh)
                if rh not in SENSOR_SENTINEL_VALUES and (VALID_HUMIDITY_RANGE_PCT[0] <= rh <= VALID_HUMIDITY_RANGE_PCT[1]):
                    rh_val = rh
                    provenance.measured["relative_humidity"] = rh_val
                else:
                    validity.validation_warnings.append(f"Invalid/sentinel humidity: {raw_rh}")
            except (ValueError, TypeError):
                validity.validation_warnings.append(f"Non-numeric humidity: {raw_rh}")

        # Extract past espent_wh
        try:
            past_espent = max(0.0, float(raw_espent))
            provenance.measured["espent_wh"] = past_espent
        except (ValueError, TypeError):
            past_espent = 0.08
            provenance.assumed["espent_wh"] = past_espent

        # ML Forecast Execution
        ml_available = False
        cold_start = True
        predicted_next_energy_wh: Optional[float] = None
        ml_error: Optional[str] = None

        if temp_val is not None and rh_val is not None:
            artifacts_ok = self._ensure_artifacts_loaded()
            if not artifacts_ok:
                ml_error = f"Artifact loading failed: {self._load_error}"
                validity.validation_warnings.append(ml_error)
            else:
                feature_df = self._build_feature_row(
                    fan_id=fan_id,
                    current_dt=dt,
                    current_temp=temp_val,
                    current_rh=rh_val,
                    room_category=room_cat,
                )

                if feature_df is None:
                    cold_start = True
                    ml_available = False
                    # Auto-seed the current observation into history so cold start resolves naturally
                    self.update_history(fan_id, dt, temp_val, rh_val, current_speed, past_espent)
                    limitations.append(
                        f"Fan '{fan_id}' has insufficient lag history (< 2 packets); ML forecaster in cold-start mode."
                    )
                else:
                    cold_start = False
                    try:
                        X_proc = self._preprocessor.transform(feature_df)
                        raw_pred = float(self._ml_model.predict(X_proc)[0])
                        # Bounded prediction: energy cannot be negative
                        predicted_next_energy_wh = round(max(0.0, raw_pred), 4)
                        ml_available = True
                        provenance.predicted["target_next_eSpent_wh"] = predicted_next_energy_wh
                        provenance.predicted["ml_forecast_model"] = "LightGBM"
                        provenance.predicted["ml_forecast_r2"] = MODEL3_ML_BENCHMARK_R2
                        provenance.predicted["ml_forecast_mae"] = MODEL3_ML_BENCHMARK_MAE_WH
                        # Update history with current observation after generating forecast
                        self.update_history(fan_id, dt, temp_val, rh_val, current_speed, past_espent)
                    except Exception as e:
                        ml_error = f"Inference exception: {str(e)}"
                        validity.validation_warnings.append(ml_error)
                        ml_available = False
        else:
            missing_inputs.extend(["temperature", "humidity"])
            validity.validation_warnings.append("Temperature/humidity missing; ML forecaster cannot run.")

        # Fallback to physics engine calculation for current speed when ML is unavailable
        current_physics_energy = candidate_eval_table[current_speed]["energy_wh_physics"]

        is_degraded = (not ml_available) or freshness.is_stale
        degraded = DegradedState(
            is_degraded=is_degraded,
            degradation_reason="ML cold start or unavailable" if not ml_available else None,
            fallback_triggered=not ml_available,
            fallback_strategy="physics_engine_authoritative_fallback" if not ml_available else None,
        )

        ml_forecast_payload = {
            "predicted_next_energy_wh": predicted_next_energy_wh if ml_available else current_physics_energy,
            "model_used": "LightGBM" if ml_available else "physics_engine_fallback",
            "horizon": MODEL3_FORECAST_HORIZON,
            "r_squared_benchmark": MODEL3_ML_BENCHMARK_R2,
            "mae_benchmark_wh": MODEL3_ML_BENCHMARK_MAE_WH,
            "cold_start": cold_start,
            "ml_available": ml_available,
            "is_advisory": True,
            "cannot_command_speed": True,
        }

        normalized = {
            "adapter": self.ADAPTER_NAME,
            "mode": mode,
            "op_time_s_used": op_time_s,
            "candidate_evaluation_table": candidate_eval_table,
            "ml_forecast": ml_forecast_payload,
            "physics_engine_available": physics_engine_available,
            "fallback_triggered": not ml_available,
        }

        primary_prediction = (
            f"Advisory next packet energy: {predicted_next_energy_wh:.4f} Wh"
            if ml_available and predicted_next_energy_wh is not None
            else f"Physics engine fallback: {current_physics_energy:.4f} Wh"
        )

        return AdapterOutputEnvelope(
            model_id=self.MODEL_ID,
            adapter_name=self.ADAPTER_NAME,
            prediction=primary_prediction,
            prediction_unit="Watt-hours (Wh)",
            timestamp=str(raw_ts) if raw_ts else None,
            input_validity=validity,
            input_freshness=freshness,
            model_availability=ModelAvailabilityStatus(
                is_available=physics_engine_available,
                reason_if_unavailable=None
            ),
            missing_inputs=missing_inputs,
            value_provenance=provenance,
            limitations=limitations,
            degraded_state=degraded,
            raw_model_output={"ml_forecast": ml_forecast_payload, "candidate_table": candidate_eval_table},
            normalized_payload=normalized,
        )

