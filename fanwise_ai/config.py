"""
FanWise AI - Configuration
Central configuration for all paths, parameters, and constants.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)  # Parent Atomberg folder
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Dataset paths
OCCUPANCY_CSV = os.path.join(DATA_DIR, "archive (2)", "Occupancy.csv")
ASHRAE_CSV = os.path.join(DATA_DIR, "archive (3)", "ashrae_db2.01.csv")
SINGLE_FAN_CSV = os.path.join(DATA_DIR, "doi_10_6078_D1V67R__v20180717", "Single_Fan.csv")
TWO_FAN_CSV = os.path.join(DATA_DIR, "doi_10_6078_D1V67R__v20180717", "Two_Fan.csv")
RDD_DIR = os.path.join(DATA_DIR, "RDD Datset")
ROBOD_DIR = os.path.join(DATA_DIR, "SupplementaryData")

# Model file paths
OCCUPANCY_MODEL_PATH = os.path.join(MODELS_DIR, "occupancy_model.joblib")
OCCUPANCY_SCALER_PATH = os.path.join(MODELS_DIR, "occupancy_scaler.joblib")
THERMAL_MODEL_PATH = os.path.join(MODELS_DIR, "thermal_comfort_model.joblib")
THERMAL_SCALER_PATH = os.path.join(MODELS_DIR, "thermal_comfort_scaler.joblib")
THERMAL_ENCODER_PATH = os.path.join(MODELS_DIR, "thermal_label_encoder.joblib")
FAN_OPTIMIZER_MODEL_PATH = os.path.join(MODELS_DIR, "fan_optimizer_model.joblib")
FAN_OPTIMIZER_SCALER_PATH = os.path.join(MODELS_DIR, "fan_optimizer_scaler.joblib")
AIRFLOW_LOOKUP_PATH = os.path.join(MODELS_DIR, "airflow_lookup.joblib")

# Fan specifications (Atomberg BLDC fans)
FAN_SPECS = {
    "Atomberg Renesa": {"rated_power": 28, "speeds": 5, "power_per_speed": [5, 10, 16, 22, 28]},
    "Atomberg Efficio": {"rated_power": 32, "speeds": 5, "power_per_speed": [6, 12, 18, 25, 32]},
    "Atomberg Studio+": {"rated_power": 28, "speeds": 5, "power_per_speed": [5, 10, 15, 21, 28]},
    "Atomberg Ikano": {"rated_power": 28, "speeds": 5, "power_per_speed": [5, 10, 16, 22, 28]},
    "Generic BLDC Fan": {"rated_power": 30, "speeds": 5, "power_per_speed": [6, 11, 17, 23, 30]},
    "Traditional Fan": {"rated_power": 75, "speeds": 5, "power_per_speed": [30, 40, 50, 65, 75]},
}

# Electricity tariff rates (INR per kWh)
TARIFF_RATES = {
    "Domestic (0-100 units)": 3.0,
    "Domestic (101-200 units)": 4.5,
    "Domestic (201-300 units)": 6.0,
    "Domestic (300+ units)": 7.5,
    "Commercial": 8.5,
    "Industrial": 7.0,
    "Custom": None,
}

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
