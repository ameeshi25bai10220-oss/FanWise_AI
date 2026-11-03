"""
AACI Model 3 – Energy Intelligence Pipeline Package
===================================================
Independent package for ceiling fan energy consumption modelling, physics,
and data pipelines for Atomberg Adaptive Comfort Intelligence (AACI).
"""

from .energy_physics import (
    BLDC_POWER_MAP,
    INDUCTION_POWER_MAP,
    SAVED_POWER_MAP,
    SPEED_TO_AIR_VELOCITY_MAP,
    VERTICAL_VELOCITY_PROFILE,
    IS_AIR_VELOCITY_MAPPING_PROVISIONAL,
    get_bldc_power,
    get_induction_power,
    get_saved_power,
    calculate_bldc_energy_joules,
    calculate_baseline_energy_joules,
    calculate_saved_energy_joules,
    joules_to_wh,
    joules_to_kwh,
    wh_to_joules,
    kwh_to_joules,
    calculate_bldc_energy_wh,
    calculate_bldc_energy_kwh,
    speed_to_air_velocity,
    verify_energy_reading,
)

from .data_loader import (
    AUTHORITATIVE_ROOMS,
    FAN_ROOM_MAP,
    TELEMETRY_COLUMNS,
    CONTENT_COLUMNS,
    get_room_category_for_fan,
    parse_and_sort_telemetry,
    apply_content_deduplication,
    tag_sensor_errors,
    filter_sensor_sentinels,
    tag_backlog_records,
    enrich_energy_units,
    load_fan_telemetry,
    load_all_mendeley_data,
)

from .preprocessing import (
    CausalSensorImputer,
    clean_and_impute_telemetry,
    EnergyPreprocessor,
)

from .feature_engineering import (
    add_cyclic_temporal_features,
    add_environmental_lag_features,
    add_operational_lag_features,
    build_supervised_energy_dataset,
    generate_candidate_speed_evaluations,
)

from .validation import (
    temporal_train_val_test_split,
    leave_one_fan_out_split,
    generate_all_lofo_splits,
    few_shot_fan_split,
    validate_no_temporal_leakage,
    validate_no_feature_leakage,
)

__all__ = [
    # Physics
    "BLDC_POWER_MAP",
    "INDUCTION_POWER_MAP",
    "SAVED_POWER_MAP",
    "SPEED_TO_AIR_VELOCITY_MAP",
    "VERTICAL_VELOCITY_PROFILE",
    "IS_AIR_VELOCITY_MAPPING_PROVISIONAL",
    "get_bldc_power",
    "get_induction_power",
    "get_saved_power",
    "calculate_bldc_energy_joules",
    "calculate_baseline_energy_joules",
    "calculate_saved_energy_joules",
    "joules_to_wh",
    "joules_to_kwh",
    "wh_to_joules",
    "kwh_to_joules",
    "calculate_bldc_energy_wh",
    "calculate_bldc_energy_kwh",
    "speed_to_air_velocity",
    "verify_energy_reading",
    # Data Loader
    "AUTHORITATIVE_ROOMS",
    "FAN_ROOM_MAP",
    "TELEMETRY_COLUMNS",
    "CONTENT_COLUMNS",
    "get_room_category_for_fan",
    "parse_and_sort_telemetry",
    "apply_content_deduplication",
    "tag_sensor_errors",
    "filter_sensor_sentinels",
    "tag_backlog_records",
    "enrich_energy_units",
    "load_fan_telemetry",
    "load_all_mendeley_data",
    # Preprocessing
    "CausalSensorImputer",
    "clean_and_impute_telemetry",
    "EnergyPreprocessor",
    # Feature Engineering
    "add_cyclic_temporal_features",
    "add_environmental_lag_features",
    "add_operational_lag_features",
    "build_supervised_energy_dataset",
    "generate_candidate_speed_evaluations",
    # Validation
    "temporal_train_val_test_split",
    "leave_one_fan_out_split",
    "generate_all_lofo_splits",
    "few_shot_fan_split",
    "validate_no_temporal_leakage",
    "validate_no_feature_leakage",
]
