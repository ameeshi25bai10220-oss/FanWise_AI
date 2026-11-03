# AACI Model 2 – Module 3: Fan & Building Energy Intelligence

## Overview
Module 3 implements **Fan & Building Energy Intelligence** within the Atomberg Adaptive Comfort Intelligence (AACI) spatial intelligence architecture. It models smart ceiling fan energy consumption, operating duration, baseline induction fan savings, and human usage dynamics using **675,332 records of real Indian residential smart fan telemetry** (`RDD Datset/`).

---

## Directory Structure
```
aaci_ml/occupancy_intelligence/module3/
├── README.md                                # Module documentation (this file)
├── audit/
│   ├── audit_rdd_dataset.py                 # Comprehensive RDD dataset audit script
│   ├── rdd_data_quality_report.md           # Detailed data quality and semantics audit report
│   ├── rdd_dataset_audit_summary.csv        # Per-fan integrity and energy summary metrics
│   └── rdd_fan_speed_modes.csv              # Empirical vs theoretical power breakdown by speed
├── src/
│   ├── rdd_loader.py                        # Telemetry data loader with chronological sorting and cleaning
│   ├── fan_usage_analytics.py               # Diurnal usage, duty cycle, and session extraction
│   ├── train_energy_models.py               # Chronological training and evaluation pipeline
│   └── energy_predictor.py                  # Runtime inference API for energy and savings prediction
├── models/
│   └── *_rf_energy_pipeline.joblib          # Trained Random Forest energy and duration pipelines
├── evaluation/
│   ├── module3_energy_report.md             # Complete engineering and research report
│   ├── energy_model_benchmark.csv           # Full algorithm benchmark across all 9 fans
│   ├── fan_usage_patterns.csv               # Operational metrics and session statistics per fan
│   ├── hourly_diurnal_profiles.csv          # 24-hour diurnal probability and speed profiles
│   └── room_category_comparison.csv         # Aggregated Bedroom vs Drawing Room vs Lounge metrics
└── tests/
    └── test_module3_pipeline.py             # 10 automated unit tests for data, power math, and models
```

---

## Physical Power & Energy Semantics

Telemetry audit across all 9 fans verified that `eSpent` and `eSaved` are calculated via deterministic speed power constants:

$$\text{eSpent} = P_{\text{BLDC}}(\text{speed}) \times \text{opTime} \quad [\text{Joules}]$$
$$\text{eSaved} = P_{\text{Saved}}(\text{speed}) \times \text{opTime} \quad [\text{Joules}]$$

| Speed | BLDC Power ($P_{\text{BLDC}}$) | Induction Baseline ($P_{\text{Induction}}$) | Modeled Savings ($P_{\text{Saved}}$) | Savings % |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.0 W | 0.0 W | 0.0 W | 0.0% |
| 1 | 10.0 W | 24.0 W | 14.0 W | 58.3% |
| 2 | 15.0 W | 34.0 W | 19.0 W | 55.9% |
| 3 | 20.0 W | 55.0 W | 35.0 W | 63.6% |
| 4 | 27.0 W | 64.0 W | 37.0 W | 57.8% |
| 5 | 48.0 W | 96.0 W | 48.0 W | 50.0% |

- **`opTime`:** Interval duration in **seconds** (modal interval ~150s).
- **Units:** Energy values in Joules are converted to Watt-hours by dividing by $3,600$, and to kWh by dividing by $3,600,000$.

---

## Usage Quickstart

```python
from aaci_ml.occupancy_intelligence.module3.src.energy_predictor import EnergyIntelligencePredictor

# Initialize predictor for a specific fan
predictor = EnergyIntelligencePredictor("Bedroom_Fan_1")

# Predict next interval energy consumption and savings
result = predictor.predict_next_interval_energy(
    temperature=28.5,
    humidity=65.0,
    current_speed=3,
    current_op_time=150.0
)

print(f"Predicted Energy: {result['predicted_energy_Wh']} Wh")
print(f"Modeled Savings: {result['modeled_energy_saved_Wh']} Wh")
```

---

## Test Suite Execution
```bash
python -m pytest aaci_ml/occupancy_intelligence/module3/tests/
```
All 10 unit tests verify data integrity, chronological sorting, power law exact matching, and predictor inference.

