"""
AACI Model 3 – Ceiling Fan Energy Physics Engine
================================================
Deterministic physics, power constants, energy conversions, and aerodynamic
transfer functions for the Atomberg Adaptive Comfort Intelligence (AACI) Model 3.

Key Responsibilities:
1. Exact BLDC ceiling fan power ratings and deterministic energy equations.
2. Baseline induction motor ratings and modelled energy savings equations.
3. Unit conversions between Joules (firmware telemetry), Watt-hours (Wh), and kWh.
4. Aerodynamic speed-to-air-velocity transfer mapping:
   - STATUS: PROVISIONAL / APPROXIMATE AERODYNAMIC SURROGATE.
   - Values are whole-room spatial averages across 180 measurement grid points
     at 1.1m seated height in a controlled climatic chamber study (Liu et al., 2018;
     Dryad DOI: 10.6078/D1V67R) of a commercial 7-speed ceiling fan.
   - They are NOT direct factory anemometer measurements of Atomberg 5-speed BLDC fans.
   - Speed 0 is set to 0.050 m/s as an assumed ASHRAE Standard 55 indoor still-air
     baseline floor. It is NOT a measured fan output (the climatic chamber measured
     mean background air velocity with fan OFF was 0.012 m/s).
5. Ground-truth physical verification of recorded telemetry packets.

IMPORTANT NOTE ON TELEMETRY UNITS:
All energy values in the Mendeley IoT telemetry are reported in JOULES:
    eSpent [J] = P_BLDC(speed) * opTime [s]
    eSaved [J] = P_Saved(speed) * opTime [s]
"""

from typing import Dict, Union, Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Physical Power Constants (Watts)
# ─────────────────────────────────────────────────────────────────────────────

# Verified BLDC Motor Power Consumption by Speed (Watts)
# Atomberg Renesa / Efficio 28W-35W BLDC class ratings matching Mendeley telemetry
BLDC_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 10.0,
    2: 15.0,
    3: 20.0,
    4: 27.0,
    5: 48.0,
}

# Verified Baseline Conventional Induction Motor Power by Speed (Watts)
# Standard 75W-100W rated single-phase AC induction ceiling fan baseline
INDUCTION_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 24.0,
    2: 34.0,
    3: 55.0,
    4: 64.0,
    5: 96.0,
}

# Verified Modeled Power Saved by Speed (Watts)
# P_Saved = P_Induction - P_BLDC
SAVED_POWER_MAP: Dict[int, float] = {
    0: 0.0,
    1: 14.0,
    2: 19.0,
    3: 35.0,
    4: 37.0,
    5: 48.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Aerodynamic Transfer Function (Dryad Airflow Dataset)
# STATUS: PROVISIONAL SURROGATE MAPPING (NOT VALIDATED FOR DIRECT FAN DEPLOYMENT)
# ─────────────────────────────────────────────────────────────────────────────

# Empirical spatial mean air velocity (m/s) across a 15x12 grid (180 points)
# at 1.1m seated occupant level from Liu et al. (2018) / UC Berkeley CBE chamber
# testing of a 7-speed ceiling fan (Dryad DOI: 10.6078/D1V67R).
#
# CRITICAL SCIENTIFIC DISTINCTION:
# - Speed 0 (0.050 m/s): ASSUMED ASHRAE Standard 55 indoor still-air baseline floor.
#   It is NOT a measured fan output. The chamber background with fan OFF was 0.012 m/s.
# - Speeds 1 to 5 (0.112 - 0.527 m/s): Spatial average across 180 grid points at 1.1m
#   for speeds 1 to 5 of an experimental 7-speed fan in a 5.5m x 4.2m x 2.6m chamber.
#   Atomberg fans have 5 active speeds and differing blade profiles and motor RPMs.
#   Therefore, this mapping is explicitly designated as PROVISIONAL for simulation only.
SPEED_TO_AIR_VELOCITY_MAP: Dict[int, float] = {
    0: 0.050,  # Assumed still-air reference floor (ASHRAE 55), NOT measured fan output
    1: 0.112,  # Provisional: 180-point grid spatial mean at 1.1m height
    2: 0.239,  # Provisional: 180-point grid spatial mean at 1.1m height
    3: 0.293,  # Provisional: 180-point grid spatial mean at 1.1m height
    4: 0.410,  # Provisional: 180-point grid spatial mean at 1.1m height
    5: 0.527,  # Provisional: 180-point grid spatial mean at 1.1m height
}

# Multi-height provisional velocity profile (m/s) across 4 measurement planes:
# 0.1m (ankle), 0.6m (seated knee), 1.1m (seated chest/head), 1.7m (standing)
VERTICAL_VELOCITY_PROFILE: Dict[float, Dict[int, float]] = {
    0.1: {0: 0.050, 1: 0.134, 2: 0.274, 3: 0.344, 4: 0.457, 5: 0.570},
    0.6: {0: 0.050, 1: 0.116, 2: 0.239, 3: 0.306, 4: 0.428, 5: 0.548},
    1.1: {0: 0.050, 1: 0.112, 2: 0.239, 3: 0.293, 4: 0.410, 5: 0.527},
    1.7: {0: 0.050, 1: 0.108, 2: 0.234, 3: 0.277, 4: 0.387, 5: 0.493},
}

# Explicit flag documenting status of aerodynamic mapping
IS_AIR_VELOCITY_MAPPING_PROVISIONAL: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# 3. Deterministic Energy Calculations
# ─────────────────────────────────────────────────────────────────────────────

def get_bldc_power(speed: int) -> float:
    """Return electrical power consumption in Watts for a given BLDC fan speed (0-5)."""
    if speed not in BLDC_POWER_MAP:
        raise ValueError(f"Invalid fan speed {speed}. Expected integer in [0, 5].")
    return BLDC_POWER_MAP[speed]


def get_induction_power(speed: int) -> float:
    """Return baseline induction fan electrical power in Watts for a given speed (0-5)."""
    if speed not in INDUCTION_POWER_MAP:
        raise ValueError(f"Invalid fan speed {speed}. Expected integer in [0, 5].")
    return INDUCTION_POWER_MAP[speed]


def get_saved_power(speed: int) -> float:
    """Return power savings in Watts compared to baseline induction fan."""
    if speed not in SAVED_POWER_MAP:
        raise ValueError(f"Invalid fan speed {speed}. Expected integer in [0, 5].")
    return SAVED_POWER_MAP[speed]


def calculate_bldc_energy_joules(speed: int, op_time_sec: float) -> float:
    """
    Calculate deterministic BLDC electrical energy consumption in Joules.
    E [Joules] = Power [W] * Operating Time [s]
    """
    if op_time_sec < 0:
        raise ValueError(f"Operating time cannot be negative: {op_time_sec}")
    return get_bldc_power(speed) * float(op_time_sec)


def calculate_baseline_energy_joules(speed: int, op_time_sec: float) -> float:
    """Calculate baseline induction fan energy consumption in Joules."""
    if op_time_sec < 0:
        raise ValueError(f"Operating time cannot be negative: {op_time_sec}")
    return get_induction_power(speed) * float(op_time_sec)


def calculate_saved_energy_joules(speed: int, op_time_sec: float) -> float:
    """Calculate modeled energy savings in Joules over induction baseline."""
    if op_time_sec < 0:
        raise ValueError(f"Operating time cannot be negative: {op_time_sec}")
    return get_saved_power(speed) * float(op_time_sec)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Energy Unit Conversions
# ─────────────────────────────────────────────────────────────────────────────

def joules_to_wh(joules: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert energy from Joules to Watt-hours (Wh). 1 Wh = 3600 Joules."""
    return joules / 3600.0


def joules_to_kwh(joules: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert energy from Joules to Kilowatt-hours (kWh). 1 kWh = 3,600,000 Joules."""
    return joules / 3600000.0


def wh_to_joules(wh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert energy from Watt-hours (Wh) to Joules."""
    return wh * 3600.0


def kwh_to_joules(kwh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert energy from Kilowatt-hours (kWh) to Joules."""
    return kwh * 3600000.0


def calculate_bldc_energy_wh(speed: int, op_time_sec: float) -> float:
    """Calculate deterministic BLDC energy consumption directly in Watt-hours."""
    return joules_to_wh(calculate_bldc_energy_joules(speed, op_time_sec))


def calculate_bldc_energy_kwh(speed: int, op_time_sec: float) -> float:
    """Calculate deterministic BLDC energy consumption directly in Kilowatt-hours."""
    return joules_to_kwh(calculate_bldc_energy_joules(speed, op_time_sec))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aerodynamic Mapping Interface (Model 1 Comfort Bridge)
# ─────────────────────────────────────────────────────────────────────────────

def speed_to_air_velocity(speed: int, height_m: float = 1.1) -> float:
    """
    Map fan speed (0-5) to provisional aerodynamic air velocity (m/s).

    STATUS: PROVISIONAL SURROGATE FOR SIMULATION ONLY.

    Parameters
    ----------
    speed : int
        Ceiling fan speed setting (0 to 5).
    height_m : float, optional
        Occupant height plane (0.1, 0.6, 1.1, or 1.7 meters). Default is 1.1m (seated).

    Returns
    -------
    float
        Air velocity in meters per second (m/s).
    """
    if speed not in BLDC_POWER_MAP:
        raise ValueError(f"Invalid speed {speed}. Expected integer in [0, 5].")

    valid_heights = [0.1, 0.6, 1.1, 1.7]
    closest_height = min(valid_heights, key=lambda h: abs(h - height_m))

    return VERTICAL_VELOCITY_PROFILE[closest_height][speed]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Physical Verification Utilities
# ─────────────────────────────────────────────────────────────────────────────

def verify_energy_reading(
    speed: int,
    op_time_sec: float,
    espent_recorded: float,
    tolerance_joules: float = 1.0,
) -> bool:
    """
    Verify whether a recorded eSpent reading in telemetry matches the physical equation:
        eSpent_expected = P_BLDC(speed) * opTime
    """
    if speed not in BLDC_POWER_MAP:
        return False
    expected = calculate_bldc_energy_joules(speed, op_time_sec)
    return abs(float(espent_recorded) - expected) <= tolerance_joules
