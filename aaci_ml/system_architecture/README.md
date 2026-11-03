# AACI System Architecture – Phase 4 Design Documentation

**Project:** Atomberg Adaptive Comfort Intelligence (AACI)
**Phase:** 4 – System-Level Architecture & Adaptive Controller Design
**Status:** DESIGN ONLY – Implementation not yet authorized
**Last Updated:** Phase 4 Architecture

---

## Overview

This directory contains the complete system-level architecture and design specifications for the AACI unified adaptive decision-making system. The system integrates three independently developed and frozen intelligence models into a single adaptive controller that intelligently balances thermal comfort, occupancy awareness, and energy efficiency for ceiling fan control.

> [!IMPORTANT]
> This is a **design-only phase**. No implementation code exists here. All three ML models (Model 1, 2, 3) are frozen and must not be modified, retrained, or overwritten. All integration must occur through the adapter and controller layer described in these documents.

---

## Three Frozen Intelligence Models

| Model | Name | Location | Status |
|-------|------|----------|--------|
| Model 1 | Thermal Comfort Intelligence | `aaci_ml/` | ✅ Frozen |
| Model 2 | Occupancy Intelligence | `aaci_ml/occupancy_intelligence/` | ✅ Frozen |
| Model 3 | Energy Intelligence | `aaci_ml/v3_energy_intelligence/` | ✅ Frozen |

---

## Design Documents

| Document | Purpose |
|----------|---------|
| [architecture_design.md](./architecture_design.md) | System-level block diagram, data flow, and unified architecture overview |
| [model_adapter_specification.md](./model_adapter_specification.md) | Exact input/output contracts for each model adapter |
| [adaptive_controller_design.md](./adaptive_controller_design.md) | Multi-objective decision framework and objective function |
| [decision_logic.md](./decision_logic.md) | Step-by-step decision flowchart and annotated decision trace |
| [fan_digital_twin_design.md](./fan_digital_twin_design.md) | Virtual fan state representation and Digital Twin schema |
| [safety_and_fallback_design.md](./safety_and_fallback_design.md) | Staleness handling, model failure fallbacks, and safe defaults |
| [integration_roadmap.md](./integration_roadmap.md) | 5-stage implementation and testing roadmap |

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Dataset Audit | ✅ PASS |
| Phase 1.5 | Critical Audit Review | ✅ PASS |
| Phase 2 | Data Pipeline Implementation | ✅ PASS |
| Phase 2.5 | Scientific Verification | ✅ PASS |
| Phase 2.6 | Forecasting Target & Evaluation Readiness | ✅ PASS |
| Phase 3 | Model Training & Benchmarking | ✅ PASS |
| Phase 3.1 | Post-Training Technical Audit | ✅ PASS (With Restrictions) |
| **Phase 4** | **System Architecture Design** | 🔄 In Progress |
| Phase 5 | Controller Implementation | ⏳ Not Yet Authorized |

---

## Key Architecture Constraints

- Dryad airflow mapping used in Model 1 adapter is **PROVISIONAL** – no Atomberg hardware calibration exists.
- Model 1 probabilities are **uncalibrated** XGBoost softmax outputs – not Bayesian posteriors.
- Model 2 requires `Light` (lux) and `CO2` (ppm) sensors that are **not available** in the Mendeley fan telemetry dataset. Documented fallback strategies are specified in the adapter specification.
- Model 3 forecasts **variable-horizon** next-packet energy, not a fixed time duration.
- The deterministic physics engine and the ML forecaster in Model 3 are **separate capabilities** and must never be conflated.

