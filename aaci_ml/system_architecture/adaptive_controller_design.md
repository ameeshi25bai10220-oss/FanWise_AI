# AACI Adaptive Controller Design

**Document Type:** Controller Architecture & Mathematical Specification
**Phase:** 4 – Architecture & Adaptive Controller Design
**Status:** DESIGN ONLY – No implementation authorized

---

## 1. Purpose

The Adaptive Controller is the decision-making core of the AACI system. It receives normalised intermediate signals from all three model adapters and selects the optimal fan speed recommendation by solving a multi-objective optimisation problem at each decision cycle.

The controller does **not** modify or retrain any model. It does **not** interact with hardware directly. It produces a recommended speed and a structured rationale for every decision.

---

## 2. Objective Function

The controller selects the candidate fan speed $s^*$ from the set $\{0, 1, 2, 3, 4, 5\}$ that minimises a weighted multi-objective cost:

$$s^* = \arg\min_{s \in \{0, 1, 2, 3, 4, 5\}} \mathcal{C}(s)$$

Where the total cost function is:

$$\mathcal{C}(s) = \alpha \cdot \text{DiscomfortPenalty}(s) + \beta \cdot \text{EnergyScore}(s) + \gamma \cdot \Delta s_{\text{penalty}}(s) - \delta \cdot \text{OccupancyBonus}(s)$$

### 2.1 Term Definitions

#### DiscomfortPenalty(s)
Penalises speeds that are likely to produce thermal discomfort at the given environmental conditions:

$$\text{DiscomfortPenalty}(s) = \begin{cases}
P(\text{Cold}) & \text{if } s > s_{\text{current}} \\
P(\text{Hot}) & \text{if } s < s_{\text{current}} \\
1 - P(\text{Neutral}) & \text{otherwise}
\end{cases}$$

Where $P(\text{Cold})$, $P(\text{Neutral})$, $P(\text{Hot})$ are the (uncalibrated) comfort probabilities from the ThermalComfortAdapter, evaluated at the target speed using the provisional airflow mapping.

> [!WARNING]
> These probabilities are **uncalibrated XGBoost softmax outputs** from a model trained on commercial/office data (ASHRAE Database II). They are not validated Bayesian posteriors. A P(Neutral) > 0.50 threshold is **heuristic only** and has not been validated for Indian residential environments.

#### EnergyScore(s)
Normalises the physics-engine power estimate across all candidate speeds to produce a [0,1] relative energy cost:

$$\text{EnergyScore}(s) = \frac{P_{\text{physics}}(s) - P_{\min}}{P_{\max} - P_{\min}}$$

Where $P_{\text{physics}}(s)$ is the power in Watts from the BLDC or Induction power map for speed $s$, and $P_{\min}$, $P_{\max}$ are the minimum and maximum power values across all non-zero candidate speeds.

> [!NOTE]
> Speed 0 (fan off) is excluded from the normalisation denominator but remains a candidate. It receives EnergyScore = 0.0 but incurs full DiscomfortPenalty(0) for hot environments.

#### ΔsPenalty(s) – Speed Stability Penalty
Penalises large speed transitions to avoid rapid cycling and unnecessary wear:

$$\Delta s_{\text{penalty}}(s) = \frac{|s - s_{\text{current}}|}{5}$$

This penalty is normalised to [0, 1] where 5 represents the maximum possible speed change.

#### OccupancyBonus(s)
Provides a reward for energy-efficient speed selection when occupancy is uncertain or the room is likely unoccupied:

$$\text{OccupancyBonus}(s) = \begin{cases}
\text{OccupancyFactor} \cdot (1 - \text{EnergyScore}(s)) & \text{if occupied} \\
0 & \text{if unoccupied (hard gate applies)} \\
0.5 \cdot (1 - \text{EnergyScore}(s)) & \text{if occupancy uncertain}
\end{cases}$$

Where $\text{OccupancyFactor} = P(\text{occupied})$ from the OccupancyAdapter.

---

## 3. Weight Parameters

| Parameter | Symbol | Default Value | Physical Interpretation |
|-----------|--------|---------------|------------------------|
| Comfort weight | $\alpha$ | 1.5 | Thermal comfort is the primary objective |
| Energy weight | $\beta$ | 0.8 | Energy efficiency is secondary |
| Stability weight | $\gamma$ | 0.5 | Penalise unnecessary speed changes |
| Occupancy weight | $\delta$ | 1.0 | Full occupancy factor when room is occupied |

> [!NOTE]
> These default weights are **design estimates** and have not been validated against real-world user preference data. They should be treated as configurable hyperparameters requiring empirical tuning before production deployment.

**Interpretation of weight ratios:**

- $\alpha > \beta$: comfort is prioritised over energy savings in normal operation
- $\gamma = 0.5$: modest stability preference — controller can still make large changes when strongly justified
- $\delta = 1.0$: full occupancy bonus; reduces to 0 when room is unoccupied (overridden by hard gate)

---

## 4. Constraint Set

Before the objective function is evaluated, hard constraints are applied to filter invalid candidates.

### 4.1 Speed Bounds

$$s \in \{0, 1, 2, 3, 4, 5\}$$

No speed outside this range is ever recommended.

### 4.2 Rate Limiting

$$|s^* - s_{\text{current}}| \leq 2$$

Maximum speed change per decision cycle is ±2. Prevents abrupt transitions. Overridden only in emergency safety events.

### 4.3 Occupancy Hard Gate

$$\text{if } \hat{y}_{\text{occupied}} = 0 \text{ AND } P(\text{occupied}) < 0.35 \Rightarrow s^* = 0$$

If the room is confidently unoccupied (below the lower uncertainty band threshold), fan speed is forced to 0 regardless of the objective function, unless thermal safety overrides this (see §4.5).

### 4.4 Staleness Gate

$$\text{if } t_{\text{now}} - t_{\text{sensor}} > 300\text{s} \Rightarrow \text{maintain last known speed; do not update}$$

If sensor data is older than 300 seconds, no new recommendation is issued. The Digital Twin retains the last confirmed speed.

### 4.5 Thermal Safety Override

If temperature ≥ 35°C AND current speed = 0, override occupancy gate and set minimum speed = 1 to prevent heat accumulation. This thermal safety floor applies even when the room is assessed as unoccupied.

$$\text{if } T \geq 35°C \text{ AND } s_{\text{current}} = 0 \Rightarrow s_{\min} = 1$$

### 4.6 Uncertainty Constraints

| Condition | Effect on Objective |
|-----------|-------------------|
| Comfort max_probability < 0.60 | $\alpha$ reduced to 0.5; comfort penalty downweighted |
| Occupancy P in [0.35, 0.65] | Occupancy hard gate not applied; OccupancyBonus uses factor = 0.5 |
| ML forecaster cold-start | EnergyScore computed from physics engine only |
| Model 1 unavailable | DiscomfortPenalty = 0.33 for all speeds (uniform; no comfort signal) |
| Model 2 unavailable | OccupancyBonus = 0; occupancy gate not applied; controller proceeds conservatively |

---

## 5. Decision Algorithm – Step by Step

> [!IMPORTANT]
> **MINOR-06 CORRECTED.** Manual override is now checked at **Step 2**, before adapter invocation. This matches the `decision_logic.md` flowchart and the declared Priority P1 ordering. The previous specification incorrectly placed the override check at Step 9 (after full scoring), which contradicted the flowchart.

```
AACI_DECIDE(csr):

Step 1: Validate CSR
  - Check sensor timestamps → if stale (>300s), RETURN last_known_speed
  - Check for 255 sentinel values → mark fields invalid if present
  - Check all required fields present

Step 2: Check manual override (PRIORITY P1 — checked BEFORE adapter invocation)
  IF manual_override_active AND (now - override_start) < 600s:
    s* ← override_speed
    decision_mode ← "manual_override"
    UPDATE digital_twin with override decision
    RETURN decision output with decision_mode = "manual_override"
  # Note: Thermal safety minimum (speed ≥ 1 when T ≥ 35°C) still applies during override
  # See safety_and_fallback_design.md §8.4

Step 3: Invoke adapters (independently, in parallel — only reached if no override)
  - comfort_signal  ← ThermalComfortAdapter(csr)
  - occupancy_signal ← OccupancyAdapter(csr)
  - energy_signal   ← EnergyIntelligenceAdapter(csr)

Step 4: Apply occupancy hard gate
  - IF occupancy_signal.occupancy_binary == 0 AND
       occupancy_signal.occupancy_probability < 0.35:
      candidate_speeds ← {0}  (unoccupied)
    ELSE:
      candidate_speeds ← {0, 1, 2, 3, 4, 5}

Step 5: Apply thermal safety override
  - IF csr.temperature_c >= 35 AND 0 in candidate_speeds AND 0 == ONLY candidate:
      candidate_speeds ← {1}  (safety floor)

Step 6: Apply rate limit
  - candidate_speeds ← {s : |s - current_speed| ≤ 2} ∩ candidate_speeds

Step 7: Adjust weights for uncertainty
  - α_eff ← α * 0.33 if comfort_signal.is_uncertain else α
  - δ_eff ← δ * 0.50 if occupancy_signal.is_uncertain else δ

Step 8: Evaluate objective function for each candidate speed
  FOR s IN candidate_speeds:
    C(s) = α_eff * DiscomfortPenalty(s, comfort_signal)
          + β   * EnergyScore(s, energy_signal)
          + γ   * ΔsPenalty(s, current_speed)
          - δ_eff * OccupancyBonus(s, occupancy_signal)

Step 9: Select optimal speed
  s* ← argmin C(s)
  decision_mode ← "adaptive"

Step 10: Update Digital Twin
  digital_twin.update(s*, comfort_signal, occupancy_signal, energy_signal)

Step 11: Build decision output
  RETURN {
    proposed_speed: s*,
    current_speed: current_speed,
    speed_change: s* - current_speed,
    rationale: build_rationale(s*, C, comfort_signal, occupancy_signal),
    decision_mode: decision_mode,
    ...
  }
```

---

## 6. Multi-Speed Candidate Scoring Example

> [!NOTE]
> **MINOR-04 & MINOR-05 CLARIFIED.** This example uses comfort probabilities evaluated **per candidate speed** via separate adapter calls (one per speed), using the provisional air velocity for that speed. The rate limit (±2 from current speed 3) was applied; all candidates {1,2,3,4,5} satisfy |s−3| ≤ 2. Speed 0 is excluded by rate limit (|0−3|=3 > 2). The DiscomfortPenalty piecewise formula uses the branch corresponding to each candidate's position relative to `s_current=3`.

**Scenario:** Temperature 29°C, RH 65%, Current speed 3 (BLDC), Daytime, Room occupied (P=0.78)

**Rate limit check:** Current speed = 3. |s−3| ≤ 2 → Candidates after rate limit: {1, 2, 3, 4, 5}. Speed 0 excluded (|0−3|=3 > 2).

**DiscomfortPenalty per candidate** — adapter re-called at each candidate's provisional air velocity:

- Speed 1 (v=0.112 m/s): Model 1 returns P(Hot)=0.39 at low airflow → branch `s < s_current` → penalty = P(Hot) = **0.39**
- Speed 2 (v=0.239 m/s): P(Hot)=0.39 still → penalty = P(Hot) = **0.39**
- Speed 3 (v=0.293 m/s): `s == s_current` → penalty = `1 − P(Neutral) = 1 − 0.61` = **0.39** (neutral calc: `1 − P(Neutral)`)
- Speed 4 (v=0.410 m/s): P(Cold)=0.12 → branch `s > s_current` → penalty = P(Cold) = **0.12**
- Speed 5 (v=0.527 m/s): P(Cold)=0.12 → penalty = P(Cold) = **0.12**

**Candidate evaluation table:**

| Speed | Power (W) | EnergyScore | DiscomfortPenalty | ΔsPenalty | OccupancyBonus | **Total Cost** |
|-------|-----------|-------------|-------------------|-----------|----------------|---------------|
| 1 | 10.0 | 0.00 | 0.39 (Hot) | 0.40 | 0.78 × 1.00 = 0.78 | 1.5×0.39 + 0.8×0.00 + 0.5×0.40 − 1.0×0.78 = **0.025** |
| 2 | 15.0 | 0.13 | 0.39 (Hot) | 0.20 | 0.78 × 0.87 = 0.68 | 1.5×0.39 + 0.8×0.13 + 0.5×0.20 − 1.0×0.68 = **0.219** |
| 3 | 20.0 | 0.26 | 0.39 (neutral calc) | 0.00 | 0.78 × 0.74 = 0.58 | 1.5×0.39 + 0.8×0.26 + 0.5×0.00 − 1.0×0.58 = **0.433** |
| 4 | 27.0 | 0.45 | 0.12 (Cold) | 0.20 | 0.78 × 0.55 = 0.43 | 1.5×0.12 + 0.8×0.45 + 0.5×0.20 − 1.0×0.43 = **0.310** |
| 5 | 48.0 | 1.00 | 0.12 (Cold) | 0.40 | 0.78 × 0.00 = 0.00 | 1.5×0.12 + 0.8×1.00 + 0.5×0.40 − 1.0×0.00 = **1.380** |

> [!NOTE]
> This example illustrates the scoring logic conceptually. Exact DiscomfortPenalty values depend on Model 1's probability outputs at each candidate speed (via the adapter). In a real implementation, the adapter must be called once per candidate speed with the corresponding provisional air velocity.

---

## 7. Decision Cycle Frequency

The controller is designed to run at each incoming telemetry packet event, not on a fixed clock interval. This matches the variable-horizon nature of the fan telemetry dataset and the Model 3 ML forecaster target.

| Parameter | Design Value |
|-----------|-------------|
| Trigger | New telemetry packet received |
| Maximum decision age | 300 seconds (staleness threshold) |
| Minimum decision interval | No minimum; packet-driven |
| Manual override lockout | 600 seconds |

---

## 8. Output Schema

```json
{
  "proposed_speed": 3,
  "current_speed": 3,
  "speed_change": 0,
  "candidate_scores": {
    "0": 0.950,
    "1": 0.025,
    "2": 0.219,
    "3": 0.433,
    "4": 0.310,
    "5": 1.380
  },
  "rationale": "Speed 1 has lowest cost score (comfort Neutral at current speed, energy-efficient candidates preferred). Stability penalty prevents dropping below speed 1 without confirmation.",
  "comfort_signal": {
    "label": "Neutral",
    "max_probability": 0.61,
    "is_uncertain": false
  },
  "occupancy_signal": {
    "state": "Occupied",
    "probability": 0.78,
    "is_uncertain": false
  },
  "energy_signal": {
    "predicted_next_energy_wh": 0.068,
    "estimated_power_w_current": 20.0
  },
  "weights_used": {
    "alpha": 1.5,
    "beta": 0.8,
    "gamma": 0.5,
    "delta": 1.0
  },
  "decision_mode": "adaptive",
  "fallback_triggered": false,
  "timestamp": "2024-03-15T14:32:00"
}
```

