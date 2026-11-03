"""
AACI System Architecture – End-to-End Integration Package
==========================================================
Phase 5 – Stage 5: System Integration and Validation

Assembles Stages 1-4 into a single reproducible pipeline.

Package Contents:
    pipeline.py           – AACIPipeline end-to-end orchestrator
    execution_record.py   – IntegrationResult output contract (22 fields)
    energy_reconciliation – Documents INDUCTION_POWER_MAP discrepancy
    pipeline_demo.py      – 6 demonstration scenarios (run standalone)
    tests/                – 40 integration test scenarios

FROZEN COMPONENT BOUNDARY
--------------------------
This package imports from, but does NOT modify:
    aaci_ml/system_architecture/adapters/
    aaci_ml/system_architecture/common_state/
    aaci_ml/system_architecture/digital_twin/
    aaci_ml/system_architecture/controller/
    aaci_ml/predict.py  (Model 1)
    aaci_ml/occupancy_intelligence/  (Model 2)
    aaci_ml/v3_energy_intelligence/  (Model 3)
"""

from aaci_ml.system_architecture.integration.pipeline import AACIPipeline
from aaci_ml.system_architecture.integration.execution_record import IntegrationResult

__all__ = ["AACIPipeline", "IntegrationResult"]

