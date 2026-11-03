# AACI Model 2 – Module 2: Spatial Headcount & Occupancy Pattern Intelligence

Part of **Atomberg Adaptive Comfort Intelligence (AACI)**  
Trained and validated on **ROBOD (Room-level Occupancy and Building Operation Dataset)** from NUS SDE4 ($N = 52,128$ records across 5 rooms).

---

## Architecture Overview

Module 2 is composed of two decoupled sub-modules:
- **Module 2A: Spatial Headcount Estimation**: Supervised regression models predicting continuous headcount ($y \in [0, 38]$) without using binary presence labels.
- **Module 2B: Occupancy Pattern Intelligence**: Descriptive analytics identifying utilization rates, peak hours, session durations, and typology profiles.

```
aaci_ml/occupancy_intelligence/module2/
├── src/
│   ├── robod_loader.py                 # Data loader, cyclical feature engineering, 5 feature groups
│   ├── train_headcount_regressor.py    # Training pipeline (per-room chronological + LORO)
│   ├── evaluate_headcount_regressor.py # Plotting and time-series evaluation
│   ├── occupancy_pattern_analytics.py  # Module 2B descriptive utilization & duration calculator
│   └── module2_predictor.py            # Unified runtime inference API
├── models/
│   ├── headcount_rf_multimodal.joblib  # Production multimodal Random Forest regressor (6.3 MB)
│   ├── headcount_rf_acoustic_wifi.joblib
│   ├── headcount_rf_env_only.joblib
│   ├── headcount_ridge_acoustic.joblib # Ultra-lightweight Ridge regressor (0.6 KB)
│   ├── headcount_xgb_multimodal.joblib # XGBoost regressor (278 KB)
│   └── headcount_metadata.json         # Model parameters, timing, and feature importances
├── evaluation/
│   ├── headcount_benchmark.csv         # Full 100-run per-room benchmark matrix
│   ├── loro_generalization_benchmark.csv # Leave-one-room-out cross-spatial transfer results
│   ├── occupancy_pattern_summary.csv   # Room utilization, peak hours, and session durations
│   ├── headcount_predictions_vs_actual.png # Time-series predictions vs ground truth
│   ├── headcount_error_distribution.png    # MAE breakdown across occupancy density brackets
│   ├── diurnal_occupancy_profiles.png      # 24-hour weekday vs weekend profiles across rooms
│   ├── spatial_utilization_comparison.png  # Utilization rate vs peak headcount across typologies
│   └── module2_evaluation_report.md       # Comprehensive evaluation engineering report
└── tests/
    └── test_module2_pipeline.py        # 10 automated unit & integration tests
```

---

## Quickstart & Reproducibility

### 1. Run Automated Test Suite
```bash
python -m pytest aaci_ml/occupancy_intelligence/module2/tests/test_module2_pipeline.py -v
```

### 2. Run Module 2B Descriptive Pattern Analytics
```bash
python aaci_ml/occupancy_intelligence/module2/src/occupancy_pattern_analytics.py
```

### 3. Retrain Module 2A Headcount Regressors & LORO Benchmarks
```bash
python aaci_ml/occupancy_intelligence/module2/src/train_headcount_regressor.py
```

### 4. Run Runtime Headcount Inference
```python
from aaci_ml.occupancy_intelligence.module2.src.module2_predictor import Module2SpatialPredictor

predictor = Module2SpatialPredictor(candidate_model_name="rf_multimodal", room_id=1)
reading = {
    "air_temperature [Celsius]": 24.5,
    "indoor_relative_humidity [%]": 62.0,
    "indoor_co2 [ppm]": 850.0,
    "sound_pressure_level [dba]": 62.0,
    "wifi_connected_devices [number]": 15.0,
    "illuminance [lux]": 350.0,
    "hour": 15
}
result = predictor.predict(reading)
print(result)
# Output:
# {
#   'estimated_headcount': 14.85,
#   'headcount_rounded': 15,
#   'occupancy_state': 'occupied',
#   'density_category': 'high (>10 persons)',
#   'uncertainty_90pct_interval': [13, 16],
#   'model_used': 'rf_multimodal',
#   'room_context': {'room_id': 1, 'room_type': 'Lecture Room 1'}
# }
```

