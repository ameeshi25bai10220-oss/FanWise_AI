"""
AACI Common State Representation – Centralized Sensor Validator
==============================================================
Phase 5 – Stage 2: Common State Representation (CSR) Implementation

Performs comprehensive physical range verification, sentinel detection,
zero boot error interception, chronological ordering enforcement,
duplicate timestamp rejection, and negative duration invalidation.
Zero replacement sensor values are invented.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import datetime

from .constants import (
    PHYSICAL_TEMP_RANGE_C,
    PHYSICAL_HUMIDITY_RANGE_PCT,
    PHYSICAL_AIR_VELOCITY_RANGE_MS,
    PHYSICAL_LIGHT_RANGE_LUX,
    PHYSICAL_CO2_RANGE_PPM,
    SENSOR_SENTINEL_VALUES,
    VALID_FAN_SPEEDS,
    VALID_FAN_MODES,
    DEFAULT_STALENESS_THRESHOLD_SECONDS,
)


class SensorValidationError(Exception):
    """Raised when critical sensor validation fails and cannot be represented in state."""
    pass


class ValidationResult:
    """Detailed result of sensor validation checks."""
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.missing_fields: List[str] = []
        self.invalid_fields: List[str] = []
        self.stale_fields: List[str] = []
        self.is_stale: bool = False
        self.is_duplicate: bool = False
        self.is_out_of_order: bool = False
        self.sensor_age_s: Optional[float] = None
        self.parsed_dt: Optional[datetime.datetime] = None

    def add_error(self, field: str, message: str) -> None:
        self.is_valid = False
        self.invalid_fields.append(field)
        self.errors.append(f"[{field}] {message}")

    def add_warning(self, field: str, message: str) -> None:
        self.warnings.append(f"[{field}] {message}")


class CentralizedSensorValidator:
    """
    Centralized sensor and telemetry validator for the AACI system.
    Tracks chronological packet flow per fan to intercept duplicate and out-of-order packets.
    """

    def __init__(self, staleness_threshold_s: float = DEFAULT_STALENESS_THRESHOLD_SECONDS):
        self.staleness_threshold_s = staleness_threshold_s
        # Per-fan tracking of last seen timestamp: fan_id -> datetime
        self._last_seen_timestamps: Dict[str, datetime.datetime] = {}

    def reset_fan_tracking(self, fan_id: Optional[str] = None) -> None:
        """Resets chronological timestamp tracking for testing or reconnection."""
        if fan_id:
            self._last_seen_timestamps.pop(fan_id, None)
        else:
            self._last_seen_timestamps.clear()

    @staticmethod
    def parse_timestamp(raw_ts: Any) -> Tuple[Optional[datetime.datetime], Optional[str]]:
        """
        Parses various timestamp representations into a standard datetime and ISO 8601 string.
        Returns: (datetime_obj, iso_string) or (None, None) if parsing fails.
        """
        if raw_ts is None:
            return None, None
        if isinstance(raw_ts, datetime.datetime):
            return raw_ts, raw_ts.isoformat()
        if isinstance(raw_ts, datetime.date):
            dt = datetime.datetime(raw_ts.year, raw_ts.month, raw_ts.day, 12, 0, 0)
            return dt, dt.isoformat()
        if isinstance(raw_ts, str):
            clean_str = raw_ts.strip()
            # Try ISO format
            try:
                dt = datetime.datetime.fromisoformat(clean_str.replace("Z", "+00:00"))
                return dt, dt.isoformat()
            except Exception:
                pass
            # Try common database formats
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%d-%m-%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
            ):
                try:
                    dt = datetime.datetime.strptime(clean_str, fmt)
                    return dt, dt.isoformat()
                except Exception:
                    continue
        return None, None

    @staticmethod
    def _extract_first_not_none(*sources: Any) -> Any:
        """Extracts first non-None value from sources, preserving zero and false."""
        for s in sources:
            if s is not None:
                return s
        return None

    def validate_raw_telemetry(self, raw: Dict[str, Any]) -> ValidationResult:
        """
        Validates raw incoming telemetry dictionary.
        Accepts both flat dictionary and nested CSR hierarchy.
        """
        result = ValidationResult()

        if not isinstance(raw, dict):
            result.add_error("telemetry", f"Telemetry payload must be a dict, got {type(raw).__name__}")
            return result

        # 1. Unpack nested or flat structures
        env = raw.get("environmental", {}) if isinstance(raw.get("environmental"), dict) else raw
        fan_state = raw.get("fan_state", {}) if isinstance(raw.get("fan_state"), dict) else raw
        meta = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else raw
        occ = raw.get("occupancy_signals", {}) if isinstance(raw.get("occupancy_signals"), dict) else raw

        # 2. Fan Identifier Validation (Required)
        raw_fan_id = self._extract_first_not_none(
            meta.get("fan_id"), meta.get("room_id"),
            raw.get("fan_id"), raw.get("room_id")
        )
        fan_id: Optional[str] = None
        if raw_fan_id is None or str(raw_fan_id).strip() == "":
            result.missing_fields.append("fan_id")
            result.add_error("fan_id", "Missing required fan identifier ('fan_id' / 'room_id').")
        else:
            fan_id = str(raw_fan_id).strip()

        # 3. Timestamp & Chronological Order Validation
        raw_ts = self._extract_first_not_none(
            meta.get("timestamp"), raw.get("timestamp"),
            raw.get("datetime"), raw.get("Date")
        )
        parsed_dt, iso_ts = self.parse_timestamp(raw_ts)
        result.parsed_dt = parsed_dt

        if parsed_dt is None and raw_ts is not None:
            result.add_error("timestamp", f"Unparseable timestamp format: '{raw_ts}'")
        elif parsed_dt is not None and fan_id is not None:
            # Check chronological flow per fan
            if fan_id in self._last_seen_timestamps:
                last_dt = self._last_seen_timestamps[fan_id]
                if parsed_dt == last_dt:
                    # Duplicate timestamp policy: explicit rejection
                    result.is_duplicate = True
                    result.add_error(
                        "timestamp",
                        f"Duplicate telemetry packet received for fan '{fan_id}' at timestamp '{iso_ts}'. "
                        "Packet rejected per explicit deduplication policy; arbitrary record selection forbidden."
                    )
                elif parsed_dt < last_dt:
                    # Out of order packet
                    result.is_out_of_order = True
                    result.add_warning(
                        "timestamp",
                        f"Out-of-order telemetry packet for fan '{fan_id}': timestamp '{iso_ts}' "
                        f"precedes previously observed timestamp '{last_dt.isoformat()}'."
                    )
            # Update last seen timestamp only if packet is strictly forward or first seen
            if not result.is_duplicate:
                self._last_seen_timestamps[fan_id] = parsed_dt

        # 4. Sensor Freshness & Age Check
        raw_age = self._extract_first_not_none(
            meta.get("sensor_age_s"), meta.get("sensor_age_seconds"),
            raw.get("sensor_age_s"), raw.get("sensor_age_seconds")
        )
        if raw_age is not None:
            try:
                age_val = float(raw_age)
                if age_val < 0.0:
                    result.add_error("sensor_age_s", f"Negative sensor age is physically invalid: {age_val}s")
                else:
                    result.sensor_age_s = age_val
                    if age_val > self.staleness_threshold_s:
                        result.is_stale = True
                        result.stale_fields.append("sensor_data")
                        result.add_warning(
                            "sensor_age_s",
                            f"Sensor data is stale ({age_val:.1f}s > {self.staleness_threshold_s}s). "
                            "Value retained for historical context but invalid as current real-time reading."
                        )
            except (ValueError, TypeError):
                result.add_error("sensor_age_s", f"Non-numeric sensor age: {raw_age}")
        elif parsed_dt is not None:
            # If explicit sensor_age not provided, compute from current system time if in realistic range
            now = datetime.datetime.now(parsed_dt.tzinfo) if parsed_dt.tzinfo else datetime.datetime.now()
            # Only compute if difference is plausible (not historical replay from years ago)
            diff_s = (now - parsed_dt).total_seconds()
            if 0.0 <= diff_s <= 86400.0:
                result.sensor_age_s = diff_s
                if diff_s > self.staleness_threshold_s:
                    result.is_stale = True
                    result.stale_fields.append("sensor_data")
                    result.add_warning(
                        "timestamp",
                        f"Telemetry is stale based on wall-clock time ({diff_s:.1f}s > {self.staleness_threshold_s}s)."
                    )

        # 5. Temperature Validation
        raw_temp = self._extract_first_not_none(
            env.get("temperature_c"), env.get("air_temp_c"),
            raw.get("temperature"), raw.get("temperature_c"), raw.get("air_temp_c")
        )
        raw_rh = self._extract_first_not_none(
            env.get("relative_humidity_pct"), env.get("relative_humidity"),
            raw.get("humidity"), raw.get("relative_humidity_pct"), raw.get("relative_humidity")
        )

        temp_numeric: Optional[float] = None
        if raw_temp is None:
            result.missing_fields.append("temperature")
            result.add_error("temperature", "Missing required environmental reading: temperature.")
        else:
            try:
                t = float(raw_temp)
                if t in SENSOR_SENTINEL_VALUES:
                    result.add_error("temperature", f"Hardware sensor sentinel fault detected: temperature={t}")
                elif not (PHYSICAL_TEMP_RANGE_C[0] <= t <= PHYSICAL_TEMP_RANGE_C[1]):
                    result.add_error(
                        "temperature",
                        f"Temperature {t} C is outside physical limits [{PHYSICAL_TEMP_RANGE_C[0]}, {PHYSICAL_TEMP_RANGE_C[1]}] C."
                    )
                else:
                    temp_numeric = t
            except (ValueError, TypeError):
                result.add_error("temperature", f"Invalid non-numeric temperature value: {raw_temp}")

        # 6. Relative Humidity Validation
        rh_numeric: Optional[float] = None
        if raw_rh is None:
            result.missing_fields.append("relative_humidity")
            result.add_error("relative_humidity", "Missing required environmental reading: relative_humidity.")
        else:
            try:
                rh = float(raw_rh)
                if rh in SENSOR_SENTINEL_VALUES:
                    result.add_error("relative_humidity", f"Hardware sensor sentinel fault detected: humidity={rh}")
                elif not (PHYSICAL_HUMIDITY_RANGE_PCT[0] <= rh <= PHYSICAL_HUMIDITY_RANGE_PCT[1]):
                    result.add_error(
                        "relative_humidity",
                        f"Relative humidity {rh}% is outside physical limits [{PHYSICAL_HUMIDITY_RANGE_PCT[0]}, {PHYSICAL_HUMIDITY_RANGE_PCT[1]}]%."
                    )
                else:
                    rh_numeric = rh
            except (ValueError, TypeError):
                result.add_error("relative_humidity", f"Invalid non-numeric relative humidity value: {raw_rh}")

        # 7. Zero Boot Error Check (T=0.0 and RH=0.0 simultaneously)
        if temp_numeric == 0.0 and rh_numeric == 0.0:
            result.add_error(
                "environmental",
                "Zero boot error: temperature and humidity are both exactly 0.0. "
                "Hardware sensor failure detected upon boot."
            )

        # 8. Fan Speed Validation
        raw_speed = self._extract_first_not_none(
            fan_state.get("current_speed"),
            raw.get("current_speed"),
            raw.get("fan_speed"),
            raw.get("speed")
        )
        if raw_speed is not None:
            try:
                spd = int(raw_speed)
                if spd not in VALID_FAN_SPEEDS:
                    result.add_error("current_speed", f"Fan speed {raw_speed} is invalid; must be an integer in {VALID_FAN_SPEEDS}.")
            except (ValueError, TypeError):
                result.add_error("current_speed", f"Non-integer fan speed: {raw_speed}")

        # 9. Operating Duration (op_time) Validation
        # STRICT RULE: Negative operating durations must be rejected or marked invalid.
        # Do NOT silently clamp them to positive durations.
        raw_op_time = self._extract_first_not_none(
            fan_state.get("op_time_s"),
            raw.get("op_time_s"),
            raw.get("opTime")
        )
        if raw_op_time is not None:
            try:
                op_val = float(raw_op_time)
                if op_val < 0.0:
                    result.add_error(
                        "op_time_s",
                        f"Negative operating duration ({op_val}s) is physically invalid. "
                        "Explicitly rejected per Phase 5 Stage 2 validation rules (no silent positive clamping)."
                    )
            except (ValueError, TypeError):
                result.add_error("op_time_s", f"Non-numeric operating duration: {raw_op_time}")

        # 10. Measured Energy (eSpent) Validation
        raw_espent = self._extract_first_not_none(
            fan_state.get("espent_wh"),
            raw.get("espent_wh"),
            raw.get("eSpent")
        )
        if raw_espent is not None:
            try:
                esp_val = float(raw_espent)
                if esp_val < 0.0:
                    result.add_error("espent_wh", f"Negative electrical energy consumption ({esp_val} Wh) is unphysical.")
            except (ValueError, TypeError):
                result.add_error("espent_wh", f"Non-numeric energy consumption: {raw_espent}")

        # 11. Air Velocity Validation (if provided)
        raw_vel = self._extract_first_not_none(
            env.get("air_velocity_ms"),
            raw.get("air_velocity_ms")
        )
        if raw_vel is not None:
            try:
                vel = float(raw_vel)
                if not (PHYSICAL_AIR_VELOCITY_RANGE_MS[0] <= vel <= PHYSICAL_AIR_VELOCITY_RANGE_MS[1]):
                    result.add_error(
                        "air_velocity_ms",
                        f"Air velocity {vel} m/s is outside physical limits [{PHYSICAL_AIR_VELOCITY_RANGE_MS[0]}, {PHYSICAL_AIR_VELOCITY_RANGE_MS[1]}] m/s."
                    )
            except (ValueError, TypeError):
                result.add_error("air_velocity_ms", f"Non-numeric air velocity: {raw_vel}")

        # 12. Optional Light & CO2 Validation
        raw_light = self._extract_first_not_none(
            occ.get("light_lux"),
            raw.get("light_lux"),
            raw.get("Light")
        )
        if raw_light is not None:
            try:
                l = float(raw_light)
                if l < 0.0:
                    result.add_error("light_lux", f"Negative illuminance reading ({l} lux) is unphysical.")
                elif l > PHYSICAL_LIGHT_RANGE_LUX[1]:
                    result.add_warning("light_lux", f"Illuminance reading {l} lux exceeds standard indoor bounds.")
            except (ValueError, TypeError):
                result.add_error("light_lux", f"Non-numeric illuminance reading: {raw_light}")

        raw_co2 = self._extract_first_not_none(
            occ.get("co2_ppm"),
            raw.get("co2_ppm"),
            raw.get("CO2")
        )
        if raw_co2 is not None:
            try:
                c = float(raw_co2)
                if not (PHYSICAL_CO2_RANGE_PPM[0] <= c <= PHYSICAL_CO2_RANGE_PPM[1]):
                    result.add_error("co2_ppm", f"CO2 concentration {c} ppm is outside plausible atmospheric bounds [200, 10000] ppm.")
            except (ValueError, TypeError):
                result.add_error("co2_ppm", f"Non-numeric CO2 concentration: {raw_co2}")

        return result
