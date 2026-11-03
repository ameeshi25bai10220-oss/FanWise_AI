"""
AACI System Architecture – Occupancy Adapter (Model 2)
======================================================
Phase 5 – Stage 1: Adapter Layer Implementation

Wraps frozen Model 2 Occupancy Intelligence interfaces:
  - Module 1: OccupancyPredictor (Binary occupancy classification & probability calibration)
  - Module 2: Module2SpatialPredictor (Continuous headcount regression & spatial load)

Enforces strict sensor availability inspection, the Phase 4.1 corrected feature-set fallback hierarchy,
zero sensor fabrication, and explicit distinction between binary occupancy and headcount estimation.
"""

import os
import sys
from typing import Dict, Any, Optional, Union, Tuple
import datetime
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure occupancy_intelligence/src is on path for data_loader and predictor
OCC_SRC_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "aaci_ml", "occupancy_intelligence", "src"))
if OCC_SRC_DIR not in sys.path:
    sys.path.insert(0, OCC_SRC_DIR)

# Ensure occupancy_intelligence/module2/src is on path
OCC_MOD2_SRC_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "aaci_ml", "occupancy_intelligence", "module2", "src"))
if OCC_MOD2_SRC_DIR not in sys.path:
    sys.path.insert(0, OCC_MOD2_SRC_DIR)

from data_loader import FEATURE_SETS
from predictor import OccupancyPredictor
from module2_predictor import Module2SpatialPredictor

from .contract import (
    AdapterOutputEnvelope,
    InputValidityStatus,
    InputFreshnessInfo,
    ModelAvailabilityStatus,
    ValueProvenance,
    DegradedState,
)
from .constants import (
    OCCUPANCY_UNCERTAINTY_BAND_LOW,
    OCCUPANCY_UNCERTAINTY_BAND_HIGH,
    DEFAULT_FAN_TO_ROOM_MAPPING,
    VALID_TEMP_RANGE_C,
    VALID_HUMIDITY_RANGE_PCT,
    VALID_LIGHT_RANGE_LUX,
    VALID_CO2_RANGE_PPM,
    SENSOR_SENTINEL_VALUES,
    MAX_SENSOR_STALENESS_SECONDS,
)


class OccupancyAdapter:
    """
    Production adapter wrapping frozen Model 2 Occupancy Intelligence.
    Handles the fan IoT Light/CO2 sensor gap via deterministic feature set selection.
    """

    MODEL_ID = "AACI_Model_2_Occupancy_Module1_RF_Module2_Spatial"
    ADAPTER_NAME = "OccupancyAdapter"

    def __init__(
        self,
        default_algorithm: str = "random_forest",
        use_calibrated: bool = True,
        enable_module2_spatial: bool = True,
    ):
        self.default_algorithm = default_algorithm
        self.use_calibrated = use_calibrated
        self.enable_module2_spatial = enable_module2_spatial
        self._predictor_cache: Dict[Tuple[str, str], OccupancyPredictor] = {}
        self._spatial_cache: Dict[Tuple[str, int], Module2SpatialPredictor] = {}

    def _get_module1_predictor(self, algorithm: str, feature_set: str) -> OccupancyPredictor:
        """Retrieves or creates a cached OccupancyPredictor instance."""
        key = (algorithm.lower(), feature_set.lower())
        if key not in self._predictor_cache:
            self._predictor_cache[key] = OccupancyPredictor(
                algorithm=algorithm,
                feature_set=feature_set,
                use_calibrated=self.use_calibrated,
            )
        return self._predictor_cache[key]

    def _get_module2_predictor(self, candidate_model_name: str, room_id: int) -> Module2SpatialPredictor:
        """Retrieves or creates a cached Module2SpatialPredictor instance."""
        key = (candidate_model_name, room_id)
        if key not in self._spatial_cache:
            self._spatial_cache[key] = Module2SpatialPredictor(
                candidate_model_name=candidate_model_name,
                room_id=room_id,
            )
        return self._spatial_cache[key]

    @staticmethod
    def _extract_hour(timestamp: Optional[Union[str, datetime.datetime]]) -> int:
        """Extracts integer hour (0-23) from timestamp."""
        if timestamp is None:
            return 12
        if isinstance(timestamp, str):
            try:
                dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return dt.hour
            except Exception:
                try:
                    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    return dt.hour
                except Exception:
                    return 12
        elif isinstance(timestamp, (datetime.date, datetime.datetime)):
            return timestamp.hour
        return 12

    def predict(self, inputs: Dict[str, Any]) -> AdapterOutputEnvelope:
        """
        Main occupancy inference entry point.

        Accepts either flat dict or hierarchical Common State Representation (CSR):
        Flat keys:
          - 'temperature' or 'temperature_c' (float, required)
          - 'humidity' or 'relative_humidity_pct' (float, required)
          - 'light' or 'light_lux' (float, optional; absence selects set_b)
          - 'co2' or 'co2_ppm' (float, optional; absence selects set_b)
          - 'hour' (int 0..23, optional; extracted from timestamp if omitted)
          - 'timestamp' or 'datetime' (str or datetime, optional)
          - 'sensor_age_seconds' (float, optional)
          - 'fan_id' or 'room_id' (str or int, optional; maps to integer room_id for Module 2)
        """
        validity = InputValidityStatus()
        freshness = InputFreshnessInfo(max_staleness_threshold_s=MAX_SENSOR_STALENESS_SECONDS)
        provenance = ValueProvenance()
        limitations = []
        missing_inputs = []

        # 1. Extract inputs from flat or CSR dict
        env = inputs.get("environmental", {}) if isinstance(inputs.get("environmental"), dict) else inputs
        occ_signals = inputs.get("occupancy_signals", {}) if isinstance(inputs.get("occupancy_signals"), dict) else inputs
        meta = inputs.get("metadata", {}) if isinstance(inputs.get("metadata"), dict) else inputs

        raw_temp = env.get("temperature_c", inputs.get("temperature", inputs.get("temperature_c")))
        raw_rh = env.get("relative_humidity_pct", inputs.get("humidity", inputs.get("relative_humidity_pct")))
        raw_light = occ_signals.get("light_lux", inputs.get("light", inputs.get("light_lux")))
        raw_co2 = occ_signals.get("co2_ppm", inputs.get("co2", inputs.get("co2_ppm")))
        raw_ts = meta.get("timestamp", inputs.get("timestamp", inputs.get("datetime")))
        sensor_age = meta.get("sensor_age_seconds", inputs.get("sensor_age_seconds"))
        raw_fan_id = meta.get("room_id", meta.get("fan_id", inputs.get("fan_id", inputs.get("room_id", "Bedroom_Fan_1"))))

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

        # 2. Validate Temperature (required for all feature sets)
        temp_val: Optional[float] = None
        if raw_temp is None:
            validity.is_valid = False
            validity.validation_errors.append("Missing required input: temperature / temperature_c")
            missing_inputs.append("temperature")
        else:
            try:
                t = float(raw_temp)
                if t in SENSOR_SENTINEL_VALUES:
                    validity.is_valid = False
                    validity.validation_errors.append(f"Hardware sensor sentinel error: temperature={t}")
                elif t == 0.0 and raw_rh == 0.0:
                    validity.is_valid = False
                    validity.validation_errors.append("Zero boot error: temperature and humidity are both exactly 0.0")
                elif not (VALID_TEMP_RANGE_C[0] <= t <= VALID_TEMP_RANGE_C[1]):
                    validity.is_valid = False
                    validity.validation_errors.append(
                        f"Temperature out of valid range [{VALID_TEMP_RANGE_C[0]}, {VALID_TEMP_RANGE_C[1]}]: {t} C"
                    )
                else:
                    temp_val = t
                    provenance.measured["temperature"] = temp_val
            except (ValueError, TypeError):
                validity.is_valid = False
                validity.validation_errors.append(f"Invalid non-numeric temperature: {raw_temp}")

        # 3. Validate Humidity (required for all feature sets)
        rh_val: Optional[float] = None
        if raw_rh is None:
            validity.is_valid = False
            validity.validation_errors.append("Missing required input: humidity / relative_humidity_pct")
            missing_inputs.append("humidity")
        else:
            try:
                rh = float(raw_rh)
                if rh in SENSOR_SENTINEL_VALUES:
                    validity.is_valid = False
                    validity.validation_errors.append(f"Hardware sensor sentinel error: humidity={rh}")
                elif not (VALID_HUMIDITY_RANGE_PCT[0] <= rh <= VALID_HUMIDITY_RANGE_PCT[1]):
                    validity.is_valid = False
                    validity.validation_errors.append(
                        f"Relative humidity out of valid range [{VALID_HUMIDITY_RANGE_PCT[0]}, {VALID_HUMIDITY_RANGE_PCT[1]}]: {rh}%"
                    )
                else:
                    rh_val = rh
                    provenance.measured["humidity"] = rh_val
            except (ValueError, TypeError):
                validity.is_valid = False
                validity.validation_errors.append(f"Invalid non-numeric relative humidity: {raw_rh}")

        # 4. Inspect Sensor Availability (Light & CO2)
        light_available = False
        light_val: Optional[float] = None
        if raw_light is not None:
            try:
                l = float(raw_light)
                if l not in SENSOR_SENTINEL_VALUES and (VALID_LIGHT_RANGE_LUX[0] <= l <= VALID_LIGHT_RANGE_LUX[1]):
                    light_val = l
                    light_available = True
                    provenance.measured["light_lux"] = light_val
                else:
                    validity.validation_warnings.append(f"Invalid or sentinel light reading: {raw_light}")
            except (ValueError, TypeError):
                validity.validation_warnings.append(f"Non-numeric light reading: {raw_light}")

        co2_available = False
        co2_val: Optional[float] = None
        if raw_co2 is not None:
            try:
                c = float(raw_co2)
                if c not in SENSOR_SENTINEL_VALUES and (VALID_CO2_RANGE_PPM[0] <= c <= VALID_CO2_RANGE_PPM[1]):
                    co2_val = c
                    co2_available = True
                    provenance.measured["co2_ppm"] = co2_val
                else:
                    validity.validation_warnings.append(f"Invalid or sentinel CO2 reading: {raw_co2}")
            except (ValueError, TypeError):
                validity.validation_warnings.append(f"Non-numeric CO2 reading: {raw_co2}")

        if not light_available:
            missing_inputs.append("light_lux")
        if not co2_available:
            missing_inputs.append("co2_ppm")

        # 5. Extract Hour
        hour_val = self._extract_hour(raw_ts)
        if "hour" in inputs:
            try:
                hour_val = int(inputs["hour"]) % 24
                provenance.measured["hour"] = hour_val
            except (ValueError, TypeError):
                provenance.estimated["hour"] = hour_val
        else:
            provenance.estimated["hour"] = hour_val

        # 6. Apply Feature Set Selection Hierarchy (DEFECT-02 CORRECTED)
        # CRITICAL RULE: Never fabricate Light or CO2 readings. Never pass incompatible vectors.
        selected_feature_set: Optional[str] = None
        sensor_gap_active = False
        fallback_strategy: Optional[str] = None

        if not validity.is_valid:
            selected_feature_set = None
        elif light_available and co2_available:
            # Full sensor context: try set_a first, then set_c, then set_b
            selected_feature_set = "set_a"
            sensor_gap_active = False
            fallback_strategy = "full_sensors_primary"
        else:
            # Fan IoT context: Light or CO2 missing.
            # set_a and set_c require Light and CO2 -> strictly incompatible.
            # ONLY set_b is compatible. Use set_b directly; do NOT attempt set_a or set_c.
            selected_feature_set = "set_b"
            sensor_gap_active = True
            fallback_strategy = "feature_set_downgrade"
            limitations.append(
                "Light and/or CO2 sensors are absent in fan IoT telemetry. "
                "Feature set downgraded to 'set_b' (Temperature, Humidity, HumidityRatio, Hour). "
                "No Light or CO2 values were fabricated."
            )

        # 7. Check if input is invalid before invoking model
        if not validity.is_valid or selected_feature_set is None:
            degraded = DegradedState(
                is_degraded=True,
                degradation_reason="Input validation failed: " + "; ".join(validity.validation_errors),
                fallback_triggered=True,
                fallback_strategy="sensor_failure_fallback",
            )
            normalized = {
                "adapter": self.ADAPTER_NAME,
                "occupancy_binary": 0,
                "occupancy_state": "Unknown",
                "occupancy_probability": 0.0,
                "occupancy_confidence": 0.0,
                "is_uncertain": True,
                "uncertainty_band_low": OCCUPANCY_UNCERTAINTY_BAND_LOW,
                "uncertainty_band_high": OCCUPANCY_UNCERTAINTY_BAND_HIGH,
                "light_available": light_available,
                "co2_available": co2_available,
                "sensor_gap_active": sensor_gap_active,
                "feature_set_used": None,
                "fallback_strategy": "sensor_failure_fallback",
                "headcount_estimate": None,
                "spatial_module_available": False,
                "model_available": False,
                "fallback_triggered": True,
            }
            return AdapterOutputEnvelope(
                model_id=self.MODEL_ID,
                adapter_name=self.ADAPTER_NAME,
                prediction="Unknown",
                prediction_unit="binary_occupancy_state",
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

        # 8. Build input dictionary for Module 1
        # HumidityRatio is automatically computed by OccupancyPredictor._validate_and_prepare_input
        # from Temperature and Humidity using the Magnus-Tetens equation.
        mod1_inputs: Dict[str, Any] = {
            "Temperature": temp_val,
            "Humidity": rh_val,
        }
        if selected_feature_set == "set_b":
            mod1_inputs["Hour"] = hour_val
        elif selected_feature_set in ["set_a", "set_c"]:
            mod1_inputs["Light"] = light_val
            mod1_inputs["CO2"] = co2_val

        # 9. Invoke Module 1 (OccupancyPredictor)
        raw_mod1_res: Optional[Dict[str, Any]] = None
        try:
            predictor = self._get_module1_predictor(self.default_algorithm, selected_feature_set)
            raw_mod1_res = predictor.predict(mod1_inputs)
        except Exception as e:
            degraded = DegradedState(
                is_degraded=True,
                degradation_reason=f"Module 1 inference exception: {str(e)}",
                fallback_triggered=True,
                fallback_strategy="exception_fallback",
            )
            normalized = {
                "adapter": self.ADAPTER_NAME,
                "occupancy_binary": 0,
                "occupancy_state": "Unknown",
                "occupancy_probability": 0.0,
                "occupancy_confidence": 0.0,
                "is_uncertain": True,
                "uncertainty_band_low": OCCUPANCY_UNCERTAINTY_BAND_LOW,
                "uncertainty_band_high": OCCUPANCY_UNCERTAINTY_BAND_HIGH,
                "light_available": light_available,
                "co2_available": co2_available,
                "sensor_gap_active": sensor_gap_active,
                "feature_set_used": selected_feature_set,
                "fallback_strategy": "exception_fallback",
                "headcount_estimate": None,
                "spatial_module_available": False,
                "model_available": False,
                "fallback_triggered": True,
            }
            return AdapterOutputEnvelope(
                model_id=self.MODEL_ID,
                adapter_name=self.ADAPTER_NAME,
                prediction="Unknown",
                prediction_unit="binary_occupancy_state",
                timestamp=str(raw_ts) if raw_ts else None,
                input_validity=validity,
                input_freshness=freshness,
                model_availability=ModelAvailabilityStatus(
                    is_available=False,
                    reason_if_unavailable=f"Exception in OccupancyPredictor: {str(e)}"
                ),
                missing_inputs=missing_inputs,
                value_provenance=provenance,
                limitations=limitations,
                degraded_state=degraded,
                raw_model_output=None,
                normalized_payload=normalized,
            )

        occ_binary = int(raw_mod1_res["occupied"])
        occ_prob = float(raw_mod1_res["probability"])
        occ_conf = float(raw_mod1_res["confidence"])
        occ_state = "Occupied" if occ_binary == 1 else "Unoccupied"

        # Determine uncertainty:
        # 1. Probability in intermediate band [0.35, 0.65] is uncertain
        # 2. Sensor gap active (Light/CO2 missing) triggers uncertainty
        # 3. Stale telemetry triggers uncertainty
        is_uncertain = (
            (OCCUPANCY_UNCERTAINTY_BAND_LOW <= occ_prob <= OCCUPANCY_UNCERTAINTY_BAND_HIGH)
            or sensor_gap_active
            or freshness.is_stale
        )

        provenance.predicted["occupancy_binary"] = occ_binary
        provenance.predicted["occupancy_probability"] = occ_prob
        provenance.predicted["occupancy_confidence"] = occ_conf
        provenance.predicted["feature_set_used"] = selected_feature_set

        # 10. Optional Module 2 Spatial Predictor (Headcount Regression)
        raw_mod2_res: Optional[Dict[str, Any]] = None
        spatial_available = False
        headcount_estimate = None
        headcount_rounded = None
        density_category = None
        uncertainty_interval = None

        if self.enable_module2_spatial:
            # Resolve integer room_id from raw_fan_id
            int_room_id = 1
            if isinstance(raw_fan_id, int):
                int_room_id = raw_fan_id
            elif isinstance(raw_fan_id, str):
                if raw_fan_id.isdigit():
                    int_room_id = int(raw_fan_id)
                else:
                    int_room_id = DEFAULT_FAN_TO_ROOM_MAPPING.get(raw_fan_id, 1)

            provenance.assumed["module2_room_id"] = int_room_id

            # Prepare Module 2 inputs
            mod2_inputs = {
                "air_temperature [Celsius]": temp_val,
                "indoor_relative_humidity [%]": rh_val,
                "hour": hour_val,
            }
            if co2_available and co2_val is not None:
                mod2_inputs["indoor_co2 [ppm]"] = co2_val
            if light_available and light_val is not None:
                mod2_inputs["illuminance [lux]"] = light_val

            try:
                spatial_pred = self._get_module2_predictor(
                    candidate_model_name="rf_multimodal",
                    room_id=int_room_id,
                )
                raw_mod2_res = spatial_pred.predict(mod2_inputs)
                spatial_available = True
                headcount_estimate = raw_mod2_res["estimated_headcount"]
                headcount_rounded = raw_mod2_res["headcount_rounded"]
                density_category = raw_mod2_res["density_category"]
                uncertainty_interval = raw_mod2_res["uncertainty_90pct_interval"]

                provenance.predicted["estimated_headcount"] = headcount_estimate
                provenance.predicted["headcount_rounded"] = headcount_rounded
                provenance.predicted["density_category"] = density_category
                provenance.estimated["uncertainty_interval_fixed_margin"] = uncertainty_interval

                limitations.append(
                    "Module 2 headcount estimate uncertainty interval uses a fixed empirical margin of +/- 1.5 persons; "
                    "it is not a statistically derived calibrated confidence interval."
                )
            except Exception as e:
                spatial_available = False
                validity.validation_warnings.append(f"Module 2 spatial prediction failed: {str(e)}")

        degraded = DegradedState(
            is_degraded=is_uncertain,
            degradation_reason="Sensor gap active or probability in uncertain band" if is_uncertain else None,
            fallback_triggered=sensor_gap_active,
            fallback_strategy=fallback_strategy,
        )

        normalized = {
            "adapter": self.ADAPTER_NAME,
            "occupancy_binary": occ_binary,
            "occupancy_state": occ_state,
            "occupancy_probability": occ_prob,
            "occupancy_confidence": occ_conf,
            "is_uncertain": is_uncertain,
            "uncertainty_band_low": OCCUPANCY_UNCERTAINTY_BAND_LOW,
            "uncertainty_band_high": OCCUPANCY_UNCERTAINTY_BAND_HIGH,
            "light_available": light_available,
            "co2_available": co2_available,
            "sensor_gap_active": sensor_gap_active,
            "feature_set_used": selected_feature_set,
            "fallback_strategy": fallback_strategy,
            "headcount_estimate": headcount_estimate,
            "headcount_rounded": headcount_rounded,
            "density_category": density_category,
            "spatial_module_available": spatial_available,
            "model_available": True,
            "fallback_triggered": sensor_gap_active,
        }

        raw_combined = {
            "module1": raw_mod1_res,
            "module2": raw_mod2_res,
        }

        return AdapterOutputEnvelope(
            model_id=self.MODEL_ID,
            adapter_name=self.ADAPTER_NAME,
            prediction=occ_state,
            prediction_unit="binary_occupancy_state",
            timestamp=str(raw_ts) if raw_ts else None,
            input_validity=validity,
            input_freshness=freshness,
            model_availability=ModelAvailabilityStatus(is_available=True),
            missing_inputs=missing_inputs,
            value_provenance=provenance,
            limitations=limitations,
            degraded_state=degraded,
            raw_model_output=raw_combined,
            normalized_payload=normalized,
        )

