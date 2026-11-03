"""
AACI Integration – Energy Table Reconciliation
===============================================
Phase 5 – Stage 5: System Integration and Validation

Documents the confirmed discrepancy between the INDUCTION_POWER_MAP values
in Stage 1 (adapters/constants.py) and Stage 3 (digital_twin/state_managers.py).
No frozen component is modified. This module exposes the discrepancy as a
named, structured constant so the integration layer can carry it transparently
in every IntegrationResult.

BLDC Power Tables (consistent across both stages – no discrepancy):
    Speed 0: 0 W
    Speed 1: 10 W
    Speed 2: 15 W
    Speed 3: 20 W
    Speed 4: 27 W
    Speed 5: 48 W

INDUCTION Power Tables (DISCREPANCY – unresolved):
    Stage 1 adapters/constants.py INDUCTION_POWER_MAP:
        Speed 0:  0 W
        Speed 1: 24 W
        Speed 2: 34 W
        Speed 3: 55 W
        Speed 4: 64 W
        Speed 5: 96 W

    Stage 3 digital_twin/state_managers.py INDUCTION_POWER_TABLE_WATTS:
        Speed 0:  0 W
        Speed 1: 16 W
        Speed 2: 24 W
        Speed 3: 36 W
        Speed 4: 52 W
        Speed 5: 75 W

No single Atomberg-validated induction measurement exists in this codebase.
Both tables are labelled as nominal / research constants.
The integration layer does NOT silently select one table.
Integration outputs carry the UNRESOLVED_INDUCTION_BASELINE flag.
"""

from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# BLDC – consistent, no reconciliation required
# ─────────────────────────────────────────────────────────────────────────────

BLDC_POWER_MAP_STAGE1: Dict[int, float] = {0: 0.0, 1: 10.0, 2: 15.0, 3: 20.0, 4: 27.0, 5: 48.0}
BLDC_POWER_TABLE_STAGE3: Tuple[float, ...] = (0.0, 10.0, 15.0, 20.0, 27.0, 48.0)
BLDC_CONSISTENT: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# Induction – two diverging tables, unresolved
# ─────────────────────────────────────────────────────────────────────────────

INDUCTION_POWER_MAP_STAGE1: Dict[int, float] = {0: 0.0, 1: 24.0, 2: 34.0, 3: 55.0, 4: 64.0, 5: 96.0}
INDUCTION_POWER_TABLE_STAGE3: Tuple[float, ...] = (0.0, 16.0, 24.0, 36.0, 52.0, 75.0)
INDUCTION_CONSISTENT: bool = False

INDUCTION_DISCREPANCY_NOTE: str = (
    "UNRESOLVED_INDUCTION_BASELINE: Two non-equivalent nominal induction power tables "
    "exist in this codebase. Stage 1 adapters/constants.py INDUCTION_POWER_MAP uses "
    "[0, 24, 34, 55, 64, 96] W; Stage 3 digital_twin/state_managers.py "
    "INDUCTION_POWER_TABLE_WATTS uses [0, 16, 24, 36, 52, 75] W. Neither table is a "
    "validated Atomberg hardware measurement. No physical bench data is available to "
    "resolve this discrepancy. The integration layer does not select either table as "
    "authoritative. Any induction energy estimates in IntegrationResult outputs carry "
    "the UNRESOLVED_INDUCTION_BASELINE provenance flag."
)

# Flag string embedded in IntegrationResult.unresolved_limitations
UNRESOLVED_INDUCTION_FLAG: str = "UNRESOLVED_INDUCTION_BASELINE"


def get_reconciliation_report() -> Dict:
    """Returns a structured reconciliation report for embedding in reports."""
    return {
        "bldc_consistent": BLDC_CONSISTENT,
        "bldc_stage1_watts": list(BLDC_POWER_MAP_STAGE1.values()),
        "bldc_stage3_watts": list(BLDC_POWER_TABLE_STAGE3),
        "induction_consistent": INDUCTION_CONSISTENT,
        "induction_stage1_watts": list(INDUCTION_POWER_MAP_STAGE1.values()),
        "induction_stage3_watts": list(INDUCTION_POWER_TABLE_STAGE3),
        "induction_discrepancy_note": INDUCTION_DISCREPANCY_NOTE,
        "hardware_calibrated_data_available": False,
        "resolution_status": UNRESOLVED_INDUCTION_FLAG,
    }

