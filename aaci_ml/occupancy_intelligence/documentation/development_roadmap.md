# AACI Occupancy Intelligence: Development Roadmap

## Phase Overview & Progress Tracking

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STATUS: Phase 1 to Phase 3 COMPLETED. Awaiting User Approval to start Phase 4 Training │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| Phase | Description | Deliverables / Artifacts | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: Dataset Audit** | Deep audit of Archive (2) (UCI), ROBOD (NUS SDE4), and RDD (Atomberg FanWise). | `archive2_audit.md`, `robod_audit.md`, `dataset_inventory.json` | **COMPLETE** |
| **Phase 2: Comparative Analysis & Hardware Feasibility** | 10-dimension comparison matrix, feature-by-feature mapping, BOM and hardware sensor audit. | `dataset_comparison.md`, `dataset_compatibility.csv`, `hardware_compatibility.md`, `data_quality_report.md` | **COMPLETE** |
| **Phase 3: Architecture & Evaluation Design** | 4-Module architecture specification, chronological non-shuffled partition design, rolling-origin CV plan. | `model2_architecture.md`, `evaluation_strategy.md`, `development_roadmap.md` | **COMPLETE** |
| **STOPPING GATE: USER APPROVAL REQUIRED** | Present complete audit, architecture design, and receive authorization to proceed. | AACI Phase 1–3 Comprehensive Report (Current Turn) | **ACTIVE GATE** |
| **Phase 4: Module 1 Training (UCI Archive 2)** | Train & validate 4 algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost) under Set A & Set B. | `src/train_module1.py`, `models/module1_*.joblib`, `evaluation/benchmark_summary.csv` | *Awaiting Approval* |
| **Phase 5: Evaluation & Threshold Optimization** | Forward-in-time (Test 2) vs Historical (Test 1) evaluation, calibration curves, threshold optimization. | `src/evaluate_module1.py`, `evaluation/calibration_curve.png`, `evaluation/threshold_tradeoff.png` | *Pending Phase 4* |
| **Phase 6: Module 2 & 3 Development (ROBOD/RDD)** | Headcount regression/classification (0–38 occupants), acoustic activity detection, and fan energy savings. | `src/train_module2_headcount.py`, `models/module2_headcount.joblib`, `evaluation/headcount_report.md` | *Pending Phase 5* |
| **Phase 7: Intelligence Fusion & Decision Gating** | Develop Module 4 fusion engine combining Model 1 (Thermal) + Model 2 (Occupancy) with hysteresis & sleep gating. | `src/occupancy_fusion_engine.py`, `tests/test_occupancy_fusion.py` | *Pending Phase 6* |
| **Phase 8: Comprehensive Verification & Packaging** | End-to-end integration tests, regression testing against V3.1 thermal comfort engine, final deployment report. | `tests/test_end_to_end.py`, `documentation/final_model2_report.md` | *Pending Phase 7* |

---

## Strict Development Safeguards
1. **Zero Impact on Thermal Comfort**: All files, code, tests, and models strictly reside within `aaci_ml/occupancy_intelligence/`. Thermal comfort models (V1, V2, V3, V3.1) in `aaci_ml/` are 100% untouched.
2. **Strict Chronological Evaluation**: No random shuffling permitted. All metrics reported on true forward-in-time unseen data (Test 2).
3. **No Fabricated Metrics**: All evaluation scores generated directly from reproducible script execution and logged to CSV/JSON artifacts.

