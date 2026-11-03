# AACI Model Adapter Specification

**Document Type:** Interface Contract Specification
**Phase:** 4 – Architecture & Adaptive Controller Design
**Status:** DESIGN ONLY – No implementation authorized

---

## Purpose

This document specifies the exact input/output contracts for the three model adapters in the AACI system. Each adapter wraps one frozen intelligence model and is solely responsible for translating the Common State Representation (CSR) into model-specific inputs and normalising model outputs into a common intermediate signal format.

> [!IMPORTANT]
> Adapters must **never** bypass the frozen model interfaces. The exact public inference methods are:
> - Model 1: `predict_comfort(air_temp_c, relative_humidity, air_velocity_ms, outdoor_monthly_temp_c, clo, met, season, building_type, cooling_strategy)` in `aaci_ml/predict.py`
> - Model 2 Module 1: `OccupancyPredictor(algorithm, feature_set, use_calibrated).predict(inputs_dict)` — instantiated per feature set; see §2 for details
> - Model 2 Module 2: `Module2SpatialPredictor(candidate_model_name, room_id).predict(inputs_dict)` — `room_id` is an **integer** set at instantiation, not a call parameter
> - Model 3 ML: LightGBM pipeline loaded from `aaci_ml/v3_energy_intelligence/models/aaci_energy_lightgbm_model.joblib`
> - Model 3 Physics: Functions in `aaci_ml/v3_energy_intelligence/src/energy_physics.py`

---

## 1. ThermalComfortAdapter (Model 1)

### 1.1 Purpose

Translates available sensor readings into the input format expected by Model 1, applies the provisional speed-to-air-velocity mapping, and returns a standardised comfort signal.

### 1.2 Provisional Speed-to-Air-Velocity Mapping

> [!WARNING]
> **This mapping is PROVISIONAL.** Values are sourced from the Dryad open-access dataset measured on a different fan model at 1.1 m measurement height. No Atomberg-specific hardware calibration has been performed. The mapping must be clearly labelled provisional in all logs and outputs.

| Fan Speed | Air Velocity (m/s) | Source Note |
|-----------|-------------------|------------|
| 0 | 0.050 | ASHRAE still-air reference — NOT a measured fan output |
| 1 | 0.112 | Dryad measurement at 1.1 m height (PROVISIONAL) |
| 2 | 0.239 | Dryad measurement at 1.1 m height (PROVISIONAL) |
| 3 | 0.293 | Dryad measurement at 1.1 m height (PROVISIONAL) |
| 4 | 0.410 | Dryad measurement at 1.1 m height (PROVISIONAL) |
| 5 | 0.527 | Dryad measurement at 1.1 m height (PROVISIONAL) |

### 1.3 Season Derivation

The `season` parameter required by Model 1 is derived from the CSR timestamp month:

| Month | Season |
|-------|--------|
| March–May | Summer |
| June–September | Monsoon |
| October–November | Post-Monsoon |
| December–February | Winter |

### 1.4 Input Contract

> [!IMPORTANT]
> **DEFECT-01 CORRECTED.** The complete `predict_comfort()` signature from `aaci_ml/predict.py` is documented below. The adapter must pass all parameters explicitly — it must not silently rely on defaults that embed commercial/office domain assumptions.

**Full `predict_comfort()` function signature (source: `aaci_ml/predict.py` lines 86–96):**

```python
def predict_comfort(
    air_temp_c: float,                           # REQUIRED — indoor air temperature (°C)
    relative_humidity: float,                    # REQUIRED — relative humidity (0–100 %)
    air_velocity_ms: float,                      # REQUIRED — air velocity at occupant level (m/s)
    outdoor_monthly_temp_c: Optional[float] = None, # OPTIONAL — outdoor monthly avg temp (°C)
    clo: float = 0.6,                            # OPTIONAL — clothing insulation (default: 0.6)
    met: float = 1.2,                            # OPTIONAL — metabolic rate (default: 1.2 = sedentary)
    season: str = "Summer",                      # OPTIONAL — one of: "Summer", "Winter", "Spring", "Autumn"
    building_type: str = "Office",               # OPTIONAL — one of: "Office", "Classroom", "Multifamily housing", "Others", "Senior center"
    cooling_strategy: str = "Air Conditioned",   # OPTIONAL — one of: "Air Conditioned", "Naturally Ventilated", "Mixed Mode", "NA"
) -> dict
```

**CSR fields consumed:**

```json
{
  "environmental": {
    "temperature_c": 28.4,
    "relative_humidity_pct": 62.0
  },
  "fan_state": {
    "current_speed": 3
  },
  "metadata": {
    "timestamp": "2024-03-15T14:32:00"
  }
}
```

**Derived adapter call with all parameters explicitly specified:**

```python
air_velocity_ms = PROVISIONAL_SPEED_VELOCITY_MAP[current_speed]   # = 0.293 (PROVISIONAL)
season = derive_season(timestamp_month)                            # = "Summer"

result = predict_comfort(
    air_temp_c=28.4,
    relative_humidity=62.0,
    air_velocity_ms=0.293,
    outdoor_monthly_temp_c=None,          # Not available in fan IoT — passed as None (model imputes)
    clo=0.6,                              # Default; residential occupant assumption (acknowledged mismatch)
    met=1.2,                              # Default; sedentary occupant assumption (acknowledged mismatch)
    season="Summer",                      # Derived from timestamp month
    building_type="Office",               # ⚠️ DOMAIN MISMATCH — model trained on commercial/office data;
                                          #    no residential building_type is available in ASHRAE DB II
    cooling_strategy="Air Conditioned",   # Default; acknowledged as unvalidated for fan-only scenarios
)
```

> [!WARNING]
> **Acknowledged domain mismatches — must be logged in every adapter call:**
> - `building_type="Office"`: Model 1 was trained on ASHRAE commercial/office data. No residential Indian building type is available. The adapter must log this mismatch and flag it in the output.
> - `cooling_strategy="Air Conditioned"`: The Mendeley telemetry is for ceiling-fan-only rooms, not air-conditioned spaces. This is a known training-domain gap.
> - `clo=0.6`, `met=1.2`: Sedentary office defaults. Actual residential occupant clothing and activity levels are unvalidated.
> - These defaults do **not** affect the frozen model; they affect the comfort prediction accuracy. The domain mismatch is inherited from Model 1's training data and cannot be resolved in the adapter layer.

**Adapter call note — per-candidate-speed invocation:**

The adapter must be called once per candidate fan speed (0–5) during controller scoring, substituting the provisional air velocity for that candidate speed. The function is not vectorised over speed; each call is independent.

### 1.5 Output Contract

**Complete raw Model 1 output (source: `aaci_ml/predict.py` lines 168–186):**

```json
{
  "predicted_class": "Neutral",
  "class_id": 1,
  "probabilities": {
    "Cold": 0.18,
    "Neutral": 0.61,
    "Hot": 0.21
  },
  "confidence": 0.61,
  "recommendation": "Occupant is thermally comfortable. Maintain current fan speed and temperature settings.",
  "model_name": "AACI_XGBoost_v3",
  "inputs_used": {
    "air_temp_c": 28.4,
    "relative_humidity": 62.0,
    "air_velocity_ms": 0.293,
    "outdoor_temp_c": null,
    "clo": 0.6,
    "met": 1.2,
    "season": "Summer",
    "building_type": "Office",
    "cooling_strategy": "Air Conditioned"
  }
}
```

> [!NOTE]
> The `confidence` field in the raw Model 1 output equals `probabilities[predicted_class]`. It is the uncalibrated XGBoost softmax output of the winning class, **not** a calibrated posterior probability. The adapter must propagate this distinction clearly.

**Normalised adapter output:**

**Normalised adapter output:**

```json
{
  "adapter": "ThermalComfortAdapter",
  "comfort_label": "Neutral",
  "comfort_class_id": 1,
  "comfort_probabilities": {
    "Cold": 0.18,
    "Neutral": 0.61,
    "Hot": 0.21
  },
  "max_probability": 0.61,
  "model_confidence": 0.61,
  "is_uncertain": false,
  "uncertainty_threshold": 0.60,
  "air_velocity_used_ms": 0.293,
  "air_velocity_mapping": "PROVISIONAL_DRYAD",
  "season_used": "Summer",
  "building_type_used": "Office",
  "cooling_strategy_used": "Air Conditioned",
  "clo_used": 0.6,
  "met_used": 1.2,
  "domain_mismatch_logged": true,
  "domain_mismatch_note": "Model trained on ASHRAE commercial/office data. building_type=Office is an acknowledged approximation for residential context.",
  "model_available": true,
  "fallback_triggered": false
}
```

> [!NOTE]
> `is_uncertain` is set to `true` when `max_probability < 0.60`. In this case, the controller treats comfort as uncertain and applies conservative scoring. See `safety_and_fallback_design.md`.

### 1.6 Failure Handling

| Failure | Fallback Behaviour |
|---------|--------------------|
| Model 1 raises exception | `model_available: false`, `comfort_label: "Unknown"`, `is_uncertain: true` |
| Temperature == 255 (sensor error) | Mark CSR as invalid; do not call adapter |
| Humidity == 255 (sensor error) | Mark CSR as invalid; do not call adapter |
| Invalid fan speed | Clamp to nearest valid speed (0 or 5) |

---

## 2. OccupancyAdapter (Model 2)

### 2.1 Purpose

Translates available sensor readings into the input format for Model 2 Module 1 and Module 2, handles the critical Light/CO₂ sensor gap, and returns a normalised occupancy signal.

### 2.2 Critical Sensor Gap

> [!WARNING]
> **Model 2 requires `Light` (lux) and `CO2` (ppm) that are NOT available in the Atomberg fan IoT telemetry.** The adapter must handle this gap explicitly. Prediction quality is reduced without these signals. This must be documented in all integration logs and user-facing explanations.

**Feature set definitions (source: `aaci_ml/occupancy_intelligence/src/data_loader.py` lines 32–36):**

> [!IMPORTANT]
> **DEFECT-02 CORRECTED.** The feature set definitions below are taken directly from `data_loader.py`. `set_a` is NOT "Temperature + Humidity only" — it requires Light, CO₂, and HumidityRatio. Only `set_b` avoids Light and CO₂.

| Feature Set | Required Features | Requires Light | Requires CO₂ | Requires HumidityRatio | Usable without Light/CO₂ |
|-------------|------------------|---------------|-------------|----------------------|--------------------------|
| `set_a` | Temperature, Humidity, **Light**, **CO₂**, HumidityRatio | ✅ Yes | ✅ Yes | ✅ Yes (auto-computed) | ❌ No |
| `set_b` | Temperature, Humidity, HumidityRatio, Hour | ❌ No | ❌ No | ✅ Yes (auto-computed) | ✅ **Yes — primary fan IoT fallback** |
| `set_c` | Temperature, Humidity, **Light**, **CO₂** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |

**HumidityRatio auto-computation:** `OccupancyPredictor._validate_and_prepare_input()` automatically computes `HumidityRatio` from Temperature and Humidity using the Magnus-Tetens psychrometric equation if it is not provided in the input dict. The adapter does not need to pre-compute it.

**Available vs. required signals:**

| Signal | `set_a` | `set_b` | `set_c` | Fan IoT Available | Gap |
|--------|---------|---------|---------|-------------------|-----|
| Temperature | ✅ | ✅ | ✅ | ✅ | None |
| Humidity | ✅ | ✅ | ✅ | ✅ | None |
| HumidityRatio | ✅ | ✅ | ❌ | ✅ (auto-computed) | None |
| Hour (0–23) | ❌ | ✅ | ❌ | ✅ (from timestamp) | None |
| Light (lux) | ✅ | ❌ | ✅ | ❌ | **MISSING** |
| CO₂ (ppm) | ✅ | ❌ | ✅ | ❌ | **MISSING** |

### 2.3 Feature Set Selection Logic and Fallback Strategy

> [!IMPORTANT]
> **DEFECT-02 CORRECTED.** The adapter must select the feature set by inspecting actual sensor availability. The previous specification described an incorrect downgrade order. The corrected logic is below.

**Corrected feature set selection — in priority order:**

```
1. IF Light available AND CO2 available:
     Try set_a first (most informative: Temperature, Humidity, Light, CO2, HumidityRatio)
     Then set_c (Temperature, Humidity, Light, CO2)
     Then set_b (Temperature, Humidity, HumidityRatio, Hour)

2. IF Light NOT available OR CO2 NOT available:
     ONLY set_b is compatible (Temperature, Humidity, HumidityRatio, Hour)
     set_a and set_c cannot be used — they require Light and/or CO2
     → Use set_b directly; do NOT attempt set_a or set_c

3. IF Temperature OR Humidity is missing/sentinel (255):
     No feature set is compatible
     → Return model_available: false, occupancy_state: "Unknown"
```

**Critical constraints — enforced by the adapter:**
1. **Never fabricate** missing Light or CO₂ readings. Conservative placeholders (0 lux, 400 ppm) must NOT be injected into the model input — they would silently produce a prediction under `set_a` or `set_c` with fabricated sensor values, masking the sensor gap.
2. **Never pass an incompatible feature vector** — if Light is missing, never instantiate a predictor with `set_a` or `set_c`.
3. **Always set `sensor_gap_active = true`** when Light or CO₂ are absent, regardless of which fallback is used.
4. **Explicitly mark `is_uncertain = true`** whenever sensor_gap_active is true (reduced prediction quality).

**OccupancyPredictor instantiation (Module 1):**

`OccupancyPredictor` is instantiated with a specific `(algorithm, feature_set)` pair. The adapter must instantiate the correct predictor for the selected feature set. Recommended default for fan IoT (no Light/CO₂):

```python
# Fan IoT context — Light and CO2 unavailable — use set_b
predictor = OccupancyPredictor(
    algorithm="random_forest",
    feature_set="set_b",
    use_calibrated=True
)
```

**When Light and CO₂ are available (extended sensor setup):**

```python
# Full sensor context — try set_a first
predictor = OccupancyPredictor(
    algorithm="random_forest",
    feature_set="set_a",
    use_calibrated=True
)
```

### 2.4 Input Contract

**CSR fields consumed:**

```json
{
  "environmental": {
    "temperature_c": 28.4,
    "relative_humidity_pct": 62.0
  },
  "occupancy_signals": {
    "light_lux": null,
    "co2_ppm": null,
    "light_available": false,
    "co2_available": false
  },
  "metadata": {
    "timestamp": "2024-03-15T14:32:00",
    "room_id": "Bedroom_Fan_1"
  }
}
```

**Derived adapter call (fan IoT context — Light/CO₂ unavailable → `set_b`):**

```python
# Step 1: Inspect sensor availability
light_available = csr["occupancy_signals"]["light_available"]   # False
co2_available   = csr["occupancy_signals"]["co2_available"]     # False

# Step 2: Select compatible feature set
# set_a requires Light, CO2 → incompatible
# set_c requires Light, CO2 → incompatible
# set_b requires only Temperature, Humidity, HumidityRatio, Hour → compatible
selected_feature_set = "set_b"

# Step 3: Instantiate predictor for selected feature set
predictor = OccupancyPredictor(
    algorithm="random_forest",
    feature_set="set_b",
    use_calibrated=True
)

# Step 4: Build input dict (HumidityRatio auto-computed by predictor)
inputs = {
    "Temperature": 28.4,
    "Humidity": 62.0,
    "Hour": 14,
    # HumidityRatio is NOT required in input — auto-computed by OccupancyPredictor
    # Light and CO2 are NOT included — feature set set_b does not require them
}

result = predictor.predict(inputs)
```

### 2.5 Output Contract

**Raw Module 1 output:**

```json
{
  "occupied": 1,
  "probability": 0.78,
  "threshold": 0.50,
  "confidence": 0.28,
  "algorithm": "random_forest",
  "feature_set": "set_b"
}
```

**Normalised adapter output:**

```json
{
  "adapter": "OccupancyAdapter",
  "occupancy_binary": 1,
  "occupancy_state": "Occupied",
  "occupancy_probability": 0.78,
  "occupancy_confidence": 0.28,
  "is_uncertain": false,
  "uncertainty_band_low": 0.35,
  "uncertainty_band_high": 0.65,
  "light_available": false,
  "co2_available": false,
  "sensor_gap_active": true,
  "feature_set_used": "set_b",
  "fallback_strategy": "feature_set_downgrade",
  "headcount_estimate": null,
  "spatial_module_available": false,
  "model_available": true,
  "fallback_triggered": false
}
```

> [!NOTE]
> `is_uncertain` is set to `true` when `occupancy_probability` is in the range [0.35, 0.65]. In this case the controller applies occupancy-uncertain scoring rather than hard gating.

### 2.6 Module 2 (Spatial Predictor) Integration

Module 2 provides headcount and density estimates. It is called when Module 1 indicates occupancy is likely.

> [!IMPORTANT]
> **MINOR-02 CORRECTED.** `Module2SpatialPredictor` takes `room_id` as an **integer constructor parameter**, not a string call parameter. The `predict()` method signature is `predict(inputs_dict) -> dict` with no `room_id` argument. The adapter must instantiate a separate predictor instance per integer room ID, or maintain a pool keyed by room ID.

**Actual `Module2SpatialPredictor` interface (source: `module2_predictor.py` lines 27–156):**

```python
# Instantiation — room_id is int (default 1), set at construction time
spatial_predictor = Module2SpatialPredictor(
    candidate_model_name="rf_multimodal",
    room_id=1                         # ← integer, NOT a string like "Bedroom_Fan_1"
)

# Call — no room_id parameter
spatial_result = spatial_predictor.predict(inputs_dict)
```

**Room ID mapping:** The adapter must maintain a mapping from string fan IDs (e.g., `"Bedroom_Fan_1"`) to integer room IDs expected by `Module2SpatialPredictor`. This mapping is deployment-specific and must be documented per environment.

**Module 2 actual return schema (source: `module2_predictor.py` lines 145–156):**

```json
{
  "estimated_headcount": 2.4,
  "headcount_rounded": 2,
  "occupancy_state": "occupied",
  "density_category": "low (1-3 persons)",
  "uncertainty_90pct_interval": [1, 4],
  "model_used": "rf_multimodal",
  "room_context": {
    "room_id": 1,
    "room_type": "Generic Room"
  }
}
```

> [!WARNING]
> The `uncertainty_90pct_interval` field uses a fixed empirical margin of ±1.5 persons. This is not a statistically derived confidence interval — it is a hardcoded default in `Module2SpatialPredictor.__init__()`. Do not represent it as a calibrated prediction interval.

Module 2 output is included in the adapter output as optional enrichment. The controller does not depend on Module 2 for the binary occupancy decision — only Module 1 drives occupancy gating.

### 2.7 Failure Handling

| Failure | Fallback Behaviour |
|---------|--------------------|
| Module 1 raises exception | `model_available: false`, `occupancy_state: "Unknown"`, `is_uncertain: true` |
| No valid feature set | `model_available: false`, `occupancy_state: "Unknown"` |
| Module 2 raises exception | `spatial_module_available: false`; Module 1 result still used |
| Sensor age > 300 s | Mark CSR as stale; adapter returns `sensor_stale: true` |

---

## 3. EnergyIntelligenceAdapter (Model 3)

### 3.1 Purpose

Wraps **both** the deterministic physics engine and the ML forecaster from Model 3 as separate capabilities. Generates a candidate speed evaluation table showing estimated power and predicted energy for each speed (0–5).

> [!IMPORTANT]
> The physics engine and the ML forecaster are **distinct and separate** capabilities. The adapter must never conflate them:
> - **Physics engine**: `E = P(s) × opTime` — used for known speed and duration.
> - **ML forecaster**: forecasts `target_next_eSpent_wh` from antecedent features — for unknown future speed/duration.
> These paths must remain independent in the adapter and in all downstream documentation.

### 3.2 Physics Engine Constants

```python
BLDC_POWER_MAP = {0: 0.0, 1: 10.0, 2: 15.0, 3: 20.0, 4: 27.0, 5: 48.0}      # Watts
INDUCTION_POWER_MAP = {0: 0.0, 1: 24.0, 2: 34.0, 3: 55.0, 4: 64.0, 5: 96.0}  # Watts
```

### 3.3 Input Contract

**CSR fields consumed:**

```json
{
  "fan_state": {
    "current_speed": 3,
    "mode": "BLDC",
    "espent_wh": 0.072,
    "op_time_s": 14400
  },
  "metadata": {
    "timestamp": "2024-03-15T14:32:00",
    "room_id": "Bedroom_Fan_1",
    "room_category": "Bedroom"
  }
}
```

**Physics engine call (per candidate speed):**

```python
for candidate_speed in range(6):
    power_w = get_power_for_speed(candidate_speed, mode="BLDC")
    energy_wh = joules_to_wh(calculate_bldc_energy_joules(candidate_speed, op_time_s))
```

**ML forecaster call (single call for current conditions):**

```python
# 26 causal features: lags, cyclic temporal, rolling energy momentum
# No contemporaneous speed, opTime, eSpent, or sensor values in features
ml_input = build_feature_vector(csr)
predicted_next_energy_wh = ml_model.predict(ml_input)[0]
```

> [!WARNING]
> The ML forecaster requires the **26 causal feature vector** built from historical lagged values. The adapter must maintain a rolling feature buffer per fan device. A cold-start condition (insufficient history) must be handled gracefully by falling back to the physics engine estimate.

### 3.4 Output Contract

**Normalised adapter output:**

> [!NOTE]
> **MINOR-03 CLARIFIED.** The `energy_wh_physics` values in the example below are computed for a **14.4-second illustrative operation time** (not the `op_time_s = 14400` s shown in the input CSR). The example uses a short duration to keep the values readable. In production, `energy_wh_physics = power_w × op_time_s / 3600`. For `op_time_s = 14400 s` at speed 1: `10 W × 14400 / 3600 = 40.0 Wh`. The table below uses `op_time_s = 14.4 s` for illustration only.

```json
{
  "adapter": "EnergyIntelligenceAdapter",
  "mode": "BLDC",
  "op_time_s_used": 14.4,
  "candidate_evaluation_table": [
    {"speed": 0, "power_w": 0.0,  "energy_wh_physics": 0.000, "source": "physics"},
    {"speed": 1, "power_w": 10.0, "energy_wh_physics": 0.040, "source": "physics"},
    {"speed": 2, "power_w": 15.0, "energy_wh_physics": 0.060, "source": "physics"},
    {"speed": 3, "power_w": 20.0, "energy_wh_physics": 0.080, "source": "physics"},
    {"speed": 4, "power_w": 27.0, "energy_wh_physics": 0.108, "source": "physics"},
    {"speed": 5, "power_w": 48.0, "energy_wh_physics": 0.192, "source": "physics"}
  ],
  "ml_forecast": {
    "predicted_next_energy_wh": 0.068,
    "model_used": "LightGBM",
    "horizon": "next_observed_packet_variable",
    "r_squared_benchmark": 0.1515,
    "mae_benchmark_wh": 1.1873,
    "cold_start": false,
    "ml_available": true
  },
  "physics_engine_available": true,
  "fallback_triggered": false
}
```

> [!NOTE]
> The ML forecast (`predicted_next_energy_wh`) represents the expected energy of the **next observed telemetry packet** under current operating conditions — not a fixed time window. It is provided as a reference signal, not as a per-candidate estimate (since the model cannot know the future speed). The physics engine provides per-candidate energy estimates using known power constants.

### 3.5 Failure Handling

| Failure | Fallback Behaviour |
|---------|--------------------|
| ML model unavailable | `ml_available: false`; physics engine used for scoring |
| Cold start (< required lag history) | `cold_start: true`, `ml_available: false`; physics engine fallback |
| Physics engine unavailable | `physics_engine_available: false`; adapter returns error |
| Fan mode unknown | Default to BLDC power map; log warning |
| sys.path issue loading preprocessor | Adapter must set `sys.path` before loading `.joblib` artifacts |

---

## 4. Adapter Isolation Requirements

| Requirement | Detail |
|-------------|--------|
| No direct model calls from controller | Controller receives only normalised adapter outputs |
| No shared state between adapters | Each adapter operates independently |
| No cross-adapter data flow | ThermalComfortAdapter result must not influence OccupancyAdapter inputs |
| Independent error handling | One adapter failure must not cascade to other adapters |
| Logging | Every adapter call must log: timestamp, input summary, output summary, fallback triggered |

