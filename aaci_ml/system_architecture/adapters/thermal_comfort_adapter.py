"""
AACI System Architecture – Thermal Comfort Adapter (Model 1)
============================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Wraps the frozen Model 1 thermal comfort inference interface (`predict_comfort` in `aaci_ml/predict.py`).
Enforces strict input validation, transparent value provenance, provisional airflow mapping disclosure,
and uncalibrated probability reporting without modifying Model 1.
"""

import os
import sys
from typing import Dict, Any, Optional, Union
import datetime

# Ensure project root is in sys.path to import predict_comfort without touching frozen files
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aaci_ml.predict import predict_comfort
from .contract import (
    AdapterOutputEnvelope,
    InputValidityStatus,
    InputFreshnessInfo,
    ModelAvailabilityStatus,
    ValueProvenance,
    DegradedState,
)
from .constants import (
    PROVISIONAL_SPEED_VELOCITY_MAP,
    COMFORT_UNCERTAINTY_THRESHOLD,
    MODEL1_DEFAULT_ASSUMPTIONS,
    VALID_TEMP_RANGE_C,
    VALID_HUMIDITY_RANGE_PCT,
    VALID_AIR_VELOCITY_RANGE_MS,
    VALID_FAN_SPEEDS,
    SENSOR_SENTINEL_VALUES,
    MAX_SENSOR_STALENESS_SECONDS,
)


class ThermalComfortAdapter:
    """
    Production adapter wrapping frozen Model 1 Thermal Comfort Intelligence.
    Translates raw telemetry or CSR inputs into Model 1's 9-parameter contract.
    """

    MODEL_ID = "AACI_Model_1_ThermalComfort_XGBoost_v3"
    ADAPTER_NAME = "ThermalComfortAdapter"

    def __init__(self, uncertainty_threshold: float = COMFORT_UNCERTAINTY_THRESHOLD):
        self.uncertainty_threshold = uncertainty_threshold

    @staticmethod
    def derive_season(timestamp: Optional[Union[str, datetime.datetime]]) -> str:
        """
        Derives an ASHRAE-compatible season string from a timestamp month.
        March–May: Summer
        June–September: Summer (Monsoon in India, mapped to Summer for ASHRAE model compatibility)
        October–November: Autumn
        December–February: Winter
        """
        if timestamp is None:
            return "Summer"

        if isinstance(timestamp, str):
            try:
                dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                try:
                    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    return "Summer"
        elif isinstance(timestamp, (datetime.date, datetime.datetime)):
            dt = timestamp
        else:
            return "Summer"

        month = dt.month
        if month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Summer"  # Indian Monsoon mapped to Summer for model compatibility
        elif month in [10, 11]:
            return "Autumn"
        else:
            return "Winter"

    def predict(self, inputs: Dict[str, Any]) -> AdapterOutputEnvelope:
        """
        Main adapter inference entry point.

        Accepts either flat dict or hierarchical Common State Representation (CSR):
        Flat keys:
          - 'air_temp_c' or 'temperature' (float, required)
          - 'relative_humidity' or 'humidity' (float, required)
          - 'air_velocity_ms' (float, optional if fan_speed provided)
          - 'fan_speed' or 'current_speed' (int 0..5, used to look up provisional air_velocity_ms)
          - 'timestamp' or 'datetime' (str or datetime, optional)
          - 'sensor_age_seconds' (float, optional)
          - 'outdoor_monthly_temp_c' (float, optional)
          - 'clo' (float, optional)
          - 'met' (float, optional)
          - 'season' (str, optional)
          - 'building_type' (str, optional)
          - 'cooling_strategy' (str, optional)
        """
        validity = InputValidityStatus()
        freshness = InputFreshnessInfo(max_staleness_threshold_s=MAX_SENSOR_STALENESS_SECONDS)
        provenance = ValueProvenance()
        limitations = []
        missing_inputs = []

        # 1. Extract inputs from flat or CSR dict
        env = inputs.get("environmental", {}) if isinstance(inputs.get("environmental"), dict) else inputs
        fan_state = inputs.get("fan_state", {}) if isinstance(inputs.get("fan_state"), dict) else inputs
        meta = inputs.get("metadata", {}) if isinstance(inputs.get("metadata"), dict) else inputs

        raw_temp = env.get("temperature_c", env.get("air_temp_c", inputs.get("temperature", inputs.get("air_temp_c"))))
        raw_rh = env.get("relative_humidity_pct", env.get("relative_humidity", inputs.get("humidity", inputs.get("relative_humidity"))))
        raw_vel = env.get("air_velocity_ms", inputs.get("air_velocity_ms"))
        raw_speed = fan_state.get("current_speed", inputs.get("fan_speed", inputs.get("current_speed")))
        raw_ts = meta.get("timestamp", inputs.get("timestamp", inputs.get("datetime")))
        sensor_age = meta.get("sensor_age_seconds", inputs.get("sensor_age_seconds"))

        # Check staleness
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

        # 2. Validate Temperature
        air_temp_c: Optional[float] = None
        if raw_temp is None:
            validity.is_valid = False
            validity.validation_errors.append("Missing required input: air_temp_c / temperature")
            missing_inputs.append("air_temp_c")
        else:
            try:
                temp_val = float(raw_temp)
                if temp_val in SENSOR_SENTINEL_VALUES:
                    validity.is_valid = False
                    validity.validation_errors.append(f"Hardware sensor sentinel error: temperature={temp_val}")
                elif temp_val == 0.0 and raw_rh == 0.0:
                    validity.is_valid = False
                    validity.validation_errors.append("Zero boot error: temperature and humidity are both exactly 0.0")
                elif not (VALID_TEMP_RANGE_C[0] <= temp_val <= VALID_TEMP_RANGE_C[1]):
                    validity.is_valid = False
                    validity.validation_errors.append(
                        f"Temperature out of valid range [{VALID_TEMP_RANGE_C[0]}, {VALID_TEMP_RANGE_C[1]}]: {temp_val} C"
                    )
                else:
                    air_temp_c = temp_val
                    provenance.measured["air_temp_c"] = air_temp_c
            except (ValueError, TypeError):
                validity.is_valid = False
                validity.validation_errors.append(f"Invalid non-numeric temperature: {raw_temp}")

        # 3. Validate Relative Humidity
        relative_humidity: Optional[float] = None
        if raw_rh is None:
            validity.is_valid = False
            validity.validation_errors.append("Missing required input: relative_humidity / humidity")
            missing_inputs.append("relative_humidity")
        else:
            try:
                rh_val = float(raw_rh)
                if rh_val in SENSOR_SENTINEL_VALUES:
                    validity.is_valid = False
                    validity.validation_errors.append(f"Hardware sensor sentinel error: relative_humidity={rh_val}")
                elif not (VALID_HUMIDITY_RANGE_PCT[0] <= rh_val <= VALID_HUMIDITY_RANGE_PCT[1]):
                    validity.is_valid = False
                    validity.validation_errors.append(
                        f"Relative humidity out of valid range [{VALID_HUMIDITY_RANGE_PCT[0]}, {VALID_HUMIDITY_RANGE_PCT[1]}]: {rh_val}%"
                    )
                else:
                    relative_humidity = rh_val
                    provenance.measured["relative_humidity"] = relative_humidity
            except (ValueError, TypeError):
                validity.is_valid = False
                validity.validation_errors.append(f"Invalid non-numeric relative humidity: {raw_rh}")

        # 4. Resolve and Validate Air Velocity
        # Rule: Do not substitute fan speed for air velocity. Air velocity must be in m/s.
        air_velocity_ms: Optional[float] = None
        provisional_airflow_used = False

        if raw_vel is not None:
            try:
                vel_val = float(raw_vel)
                if not (VALID_AIR_VELOCITY_RANGE_MS[0] <= vel_val <= VALID_AIR_VELOCITY_RANGE_MS[1]):
                    validity.is_valid = False
                    validity.validation_errors.append(
                        f"Air velocity out of valid range [{VALID_AIR_VELOCITY_RANGE_MS[0]}, {VALID_AIR_VELOCITY_RANGE_MS[1]}]: {vel_val} m/s"
                    )
                else:
                    air_velocity_ms = vel_val
                    provenance.measured["air_velocity_ms"] = air_velocity_ms
            except (ValueError, TypeError):
                validity.is_valid = False
                validity.validation_errors.append(f"Invalid non-numeric air velocity: {raw_vel}")
        elif raw_speed is not None:
            try:
                spd_int = int(raw_speed)
                if spd_int not in VALID_FAN_SPEEDS:
                    validity.is_valid = False
                    validity.validation_errors.append(f"Fan speed out of valid range (0-5): {raw_speed}")
                else:
                    air_velocity_ms = PROVISIONAL_SPEED_VELOCITY_MAP[spd_int]
                    provisional_airflow_used = True
                    provenance.measured["fan_speed"] = spd_int
                    provenance.estimated["air_velocity_ms"] = air_velocity_ms
                    limitations.append(
                        f"Air velocity ({air_velocity_ms} m/s) was estimated from fan speed {spd_int} "
                        "using the PROVISIONAL Dryad dataset mapping. This is NOT a validated Atomberg calibration."
                    )
            except (ValueError, TypeError):
                validity.is_valid = False
                validity.validation_errors.append(f"Invalid non-integer fan speed: {raw_speed}")
        else:
            validity.is_valid = False
            validity.validation_errors.append(
                "Missing air velocity: neither 'air_velocity_ms' nor 'fan_speed' was provided."
            )
            missing_inputs.append("air_velocity_ms")

        # 5. Extract or Assign Optional/Assumed Parameters
        # Outdoor monthly air temp
        raw_outdoor = env.get("outdoor_monthly_temp_c", inputs.get("outdoor_monthly_temp_c"))
        outdoor_monthly_temp_c = None
        if raw_outdoor is not None:
            try:
                outdoor_monthly_temp_c = float(raw_outdoor)
                provenance.measured["outdoor_monthly_temp_c"] = outdoor_monthly_temp_c
            except (ValueError, TypeError):
                validity.validation_warnings.append(f"Invalid outdoor_monthly_temp_c: {raw_outdoor}; defaulting to None")
                outdoor_monthly_temp_c = None
        if outdoor_monthly_temp_c is None:
            missing_inputs.append("outdoor_monthly_temp_c")
            provenance.assumed["outdoor_monthly_temp_c"] = None

        # Clo
        raw_clo = inputs.get("clo", MODEL1_DEFAULT_ASSUMPTIONS["clo"])
        try:
            clo = float(raw_clo)
            if "clo" in inputs:
                provenance.measured["clo"] = clo
            else:
                provenance.assumed["clo"] = clo
        except (ValueError, TypeError):
            clo = MODEL1_DEFAULT_ASSUMPTIONS["clo"]
            provenance.assumed["clo"] = clo

        # Met
        raw_met = inputs.get("met", MODEL1_DEFAULT_ASSUMPTIONS["met"])
        try:
            met = float(raw_met)
            if "met" in inputs:
                provenance.measured["met"] = met
            else:
                provenance.assumed["met"] = met
        except (ValueError, TypeError):
            met = MODEL1_DEFAULT_ASSUMPTIONS["met"]
            provenance.assumed["met"] = met

        # Season
        if "season" in inputs and inputs["season"] in ["Summer", "Winter", "Spring", "Autumn"]:
            season = str(inputs["season"])
            provenance.measured["season"] = season
        else:
            season = self.derive_season(raw_ts)
            provenance.estimated["season"] = season

        # Building type and Cooling strategy (domain mismatches acknowledged)
        building_type = str(inputs.get("building_type", MODEL1_DEFAULT_ASSUMPTIONS["building_type"]))
        provenance.assumed["building_type"] = building_type

        cooling_strategy = str(inputs.get("cooling_strategy", MODEL1_DEFAULT_ASSUMPTIONS["cooling_strategy"]))
        provenance.assumed["cooling_strategy"] = cooling_strategy

        limitations.append(MODEL1_DEFAULT_ASSUMPTIONS["domain_mismatch_note"])

        # 6. Check for validation failure before invoking frozen model
        if not validity.is_valid:
            degraded = DegradedState(
                is_degraded=True,
                degradation_reason="Input validation failed: " + "; ".join(validity.validation_errors),
                fallback_triggered=True,
                fallback_strategy="safe_neutral_fallback",
            )
            normalized = {
                "adapter": self.ADAPTER_NAME,
                "comfort_label": "Unknown",
                "comfort_class_id": -1,
                "comfort_probabilities": {"Cold": 0.0, "Neutral": 0.0, "Hot": 0.0},
                "max_probability": 0.0,
                "model_confidence": 0.0,
                "is_uncertain": True,
                "uncertainty_threshold": self.uncertainty_threshold,
                "air_velocity_used_ms": air_velocity_ms,
                "air_velocity_mapping": "PROVISIONAL_DRYAD" if provisional_airflow_used else "DIRECT_MEASUREMENT",
                "season_used": season,
                "building_type_used": building_type,
                "cooling_strategy_used": cooling_strategy,
                "clo_used": clo,
                "met_used": met,
                "domain_mismatch_logged": True,
                "domain_mismatch_note": MODEL1_DEFAULT_ASSUMPTIONS["domain_mismatch_note"],
                "model_available": False,
                "fallback_triggered": True,
            }
            return AdapterOutputEnvelope(
                model_id=self.MODEL_ID,
                adapter_name=self.ADAPTER_NAME,
                prediction="Unknown",
                prediction_unit="thermal_comfort_class",
                timestamp=str(raw_ts) if raw_ts else None,
                input_validity=validity,
                input_freshness=freshness,
                model_availability=ModelAvailabilityStatus(
                    is_available=False,
                    reason_if_unavailable="Input validation failed; model not invoked"
                ),
                missing_inputs=missing_inputs,
                value_provenance=provenance,
                limitations=limitations,
                degraded_state=degraded,
                raw_model_output=None,
                normalized_payload=normalized,
            )

        # 7. Invoke frozen Model 1 via public predict_comfort interface
        try:
            raw_res = predict_comfort(
                air_temp_c=air_temp_c,
                relative_humidity=relative_humidity,
                air_velocity_ms=air_velocity_ms,
                outdoor_monthly_temp_c=outdoor_monthly_temp_c,
                clo=clo,
                met=met,
                season=season,
                building_type=building_type,
                cooling_strategy=cooling_strategy,
            )
        except Exception as e:
            degraded = DegradedState(
                is_degraded=True,
                degradation_reason=f"Model 1 inference exception: {str(e)}",
                fallback_triggered=True,
                fallback_strategy="exception_fallback",
            )
            normalized = {
                "adapter": self.ADAPTER_NAME,
                "comfort_label": "Unknown",
                "comfort_class_id": -1,
                "comfort_probabilities": {"Cold": 0.0, "Neutral": 0.0, "Hot": 0.0},
                "max_probability": 0.0,
                "model_confidence": 0.0,
                "is_uncertain": True,
                "uncertainty_threshold": self.uncertainty_threshold,
                "air_velocity_used_ms": air_velocity_ms,
                "air_velocity_mapping": "PROVISIONAL_DRYAD" if provisional_airflow_used else "DIRECT_MEASUREMENT",
                "season_used": season,
                "building_type_used": building_type,
                "cooling_strategy_used": cooling_strategy,
                "clo_used": clo,
                "met_used": met,
                "domain_mismatch_logged": True,
                "domain_mismatch_note": MODEL1_DEFAULT_ASSUMPTIONS["domain_mismatch_note"],
                "model_available": False,
                "fallback_triggered": True,
            }
            return AdapterOutputEnvelope(
                model_id=self.MODEL_ID,
                adapter_name=self.ADAPTER_NAME,
                prediction="Unknown",
                prediction_unit="thermal_comfort_class",
                timestamp=str(raw_ts) if raw_ts else None,
                input_validity=validity,
                input_freshness=freshness,
                model_availability=ModelAvailabilityStatus(
                    is_available=False,
                    reason_if_unavailable=f"Exception in predict_comfort: {str(e)}"
                ),
                missing_inputs=missing_inputs,
                value_provenance=provenance,
                limitations=limitations,
                degraded_state=degraded,
                raw_model_output=None,
                normalized_payload=normalized,
            )

        # 8. Unpack and normalise raw Model 1 output
        comfort_label = raw_res["predicted_class"]
        class_id = raw_res["class_id"]
        probs = raw_res["probabilities"]
        raw_conf = raw_res["confidence"]
        max_prob = float(max(probs.values())) if probs else raw_conf

        # Uncertainty handling: winning probability < threshold triggers uncertainty
        is_uncertain = (max_prob < self.uncertainty_threshold) or freshness.is_stale

        provenance.predicted["comfort_class"] = comfort_label
        provenance.predicted["comfort_class_id"] = class_id
        provenance.predicted["probabilities"] = probs
        provenance.predicted["uncalibrated_winning_probability"] = raw_conf

        limitations.append(
            "Confidence is the raw uncalibrated XGBoost softmax probability of the winning class, "
            "NOT a calibrated posterior probability."
        )

        degraded = DegradedState(
            is_degraded=is_uncertain,
            degradation_reason="Confidence below threshold or sensor stale" if is_uncertain else None,
            fallback_triggered=False,
            fallback_strategy=None,
        )

        normalized = {
            "adapter": self.ADAPTER_NAME,
            "comfort_label": comfort_label,
            "comfort_class_id": class_id,
            "comfort_probabilities": probs,
            "max_probability": max_prob,
            "model_confidence": raw_conf,
            "is_uncertain": is_uncertain,
            "uncertainty_threshold": self.uncertainty_threshold,
            "air_velocity_used_ms": air_velocity_ms,
            "air_velocity_mapping": "PROVISIONAL_DRYAD" if provisional_airflow_used else "DIRECT_MEASUREMENT",
            "season_used": season,
            "building_type_used": building_type,
            "cooling_strategy_used": cooling_strategy,
            "clo_used": clo,
            "met_used": met,
            "domain_mismatch_logged": True,
            "domain_mismatch_note": MODEL1_DEFAULT_ASSUMPTIONS["domain_mismatch_note"],
            "model_available": True,
            "fallback_triggered": False,
        }

        return AdapterOutputEnvelope(
            model_id=self.MODEL_ID,
            adapter_name=self.ADAPTER_NAME,
            prediction=comfort_label,
            prediction_unit="thermal_comfort_class",
            timestamp=str(raw_ts) if raw_ts else None,
            input_validity=validity,
            input_freshness=freshness,
            model_availability=ModelAvailabilityStatus(is_available=True),
            missing_inputs=missing_inputs,
            value_provenance=provenance,
            limitations=limitations,
            degraded_state=degraded,
            raw_model_output=raw_res,
            normalized_payload=normalized,
        )

