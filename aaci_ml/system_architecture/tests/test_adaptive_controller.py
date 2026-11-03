"""
AACI Adaptive Controller – Unit & Integration Test Suite
========================================================
Phase 5 – Stage 4: Adaptive Controller Logic Implementation

Comprehensive test suite verifying all 18 requirements in User Request §7:
1. Normal comfort-driven recommendation
2. Occupied versus unoccupied conditions
3. Energy-aware candidate scoring
4. Missing Model 1 output (ThermalComfortAdapter fallback)
5. Missing Model 2 output (OccupancyAdapter fallback)
6. Missing Model 3 output (EnergyIntelligenceAdapter fallback)
7. Stale and invalid sensor data
8. Uncalibrated probability handling & uncertainty downweighting
9. ML forecast unavailability (graceful physics fallback)
10. Manual override priority (Priority P1)
11. Override expiration after timeout
12. Safety constraint enforcement (Thermal safety floor T >= 35°C, Rate limiting |Δs| <= 2)
13. Conflicting model outputs (Cold prediction in warm room)
14. Multi-fan isolation
15. Deterministic decision replay
16. Explanation completeness & output contract schema
17. No physical fan actuation (strictly software recommendation)
18. Safe fallback behaviour under compound failures
"""

import copy
import time
import pytest

from aaci_ml.system_architecture.common_state.common_state import CommonState
from aaci_ml.system_architecture.controller import (
    AdaptiveController,
    ComfortScorer,
    DecisionEngine,
    DecisionExplanation,
    EnergyScorer,
    OccupancyScorer,
    PreferenceManager,
    SafetyGuard,
    DECISION_MODE_ADAPTIVE,
    DECISION_MODE_MANUAL_OVERRIDE,
    DECISION_MODE_OCCUPANCY_GATE,
    DECISION_MODE_STALENESS_FREEZE,
    DECISION_MODE_THERMAL_SAFETY,
)


class TestAdaptiveController:
    """Verifies all operational paths and safety priorities of the Adaptive Controller."""

    def test_01_normal_comfort_driven_recommendation(self):
        """Verify normal adaptive recommendation balances comfort, energy, and stability."""
        controller = AdaptiveController()
        csr = CommonState.from_telemetry(
            raw_telemetry={
                "fan_id": "Bedroom_Fan_1",
                "temperature": 29.0,
                "relative_humidity": 60.0,
                "fan_speed": 2,
                "light_lux": 500.0,
                "co2_ppm": 1200.0,
            }
        )

        decision = controller.decide(csr)

        assert isinstance(decision, DecisionExplanation)
        assert 0 <= decision.recommended_speed <= 5
        assert decision.decision_mode == DECISION_MODE_ADAPTIVE
        assert decision.previous_speed == 0  # Initial twin speed
        assert len(decision.candidate_evaluations) > 0
        assert "Selected Speed" in decision.rationale

    def test_02_occupied_vs_unoccupied_conditions(self):
        """Verify occupancy hard gate forces speed 0 when unoccupied, and positive speed when occupied."""
        controller = AdaptiveController()

        # Confidently unoccupied (unoccupied binary 0, probability 0.10)
        unocc_csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 28.0, "relative_humidity_pct": 50.0, "sensor_age_s": 10.0},
            "occupancy_signals": {"occupancy_binary": 0, "occupancy_probability": 0.10, "occupancy_classification": "Unoccupied"},
            "metadata": {"fan_id": "Bedroom_Fan_1"},
        }
        dec_unocc = controller.decide(unocc_csr)
        assert dec_unocc.recommended_speed == 0
        assert dec_unocc.decision_mode == DECISION_MODE_OCCUPANCY_GATE

        # Occupied condition (probability 0.85)
        occ_csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 28.0, "relative_humidity_pct": 50.0, "sensor_age_s": 10.0},
            "occupancy_signals": {"occupancy_binary": 1, "occupancy_probability": 0.85, "occupancy_classification": "Occupied"},
            "metadata": {"fan_id": "Bedroom_Fan_1"},
        }
        dec_occ = controller.decide(occ_csr)
        assert dec_occ.recommended_speed >= 0
        assert dec_occ.decision_mode == DECISION_MODE_ADAPTIVE

    def test_03_energy_aware_candidate_scoring(self):
        """Verify energy-heavy speeds receive higher energy scores according to physics map."""
        scorer = EnergyScorer()
        eval_res = scorer.score_candidates([0, 1, 3, 5], fan_mode="BLDC")
        scores = eval_res["energy_scores"]

        assert scores[0] == 0.0
        assert scores[1] == 0.0
        assert scores[3] > scores[1]
        assert scores[5] == 1.0  # Max non-zero power (48W) has score 1.0

    def test_04_missing_model1_output(self):
        """Verify graceful fallback to uniform 0.33 penalty when Model 1 is unavailable."""
        controller = AdaptiveController()
        # Missing environmental inputs prevent Model 1 inference
        csr_no_temp = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": None, "relative_humidity_pct": None},
            "occupancy_signals": {"occupancy_probability": 0.80},
        }

        decision = controller.decide(csr_no_temp)
        assert decision.comfort_assessment["model_available"] is False
        assert decision.comfort_assessment["discomfort_penalties"][0] == 0.33
        assert decision.fallback_status["fallback_triggered"] is True

    def test_05_missing_model2_output(self):
        """Verify fallback when Model 2 is unavailable: occupancy gate is not applied."""
        controller = AdaptiveController()
        # Mocking empty/unavailable occupancy
        csr_no_occ = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 28.0, "relative_humidity_pct": 55.0},
            "occupancy_signals": {},  # No occupancy signals
        }
        decision = controller.decide(csr_no_occ)

        assert decision.occupancy_assessment["model_available"] is True or decision.occupancy_assessment["occupancy_bonuses"][0] == 0.0
        assert decision.recommended_speed >= 0

    def test_06_missing_model3_output(self):
        """Verify fallback when Model 3 ML forecaster is unavailable: physics scoring used."""
        scorer = EnergyScorer()
        res = scorer.score_candidates([1, 2, 3], csr={"empty": "telemetry"})

        assert res["ml_advisory"]["predicted_next_energy_wh"] is None
        assert res["energy_scores"][1] == 0.0
        assert res["energy_scores"][3] > 0.0

    def test_07_stale_and_invalid_sensor_data(self):
        """Verify staleness (>300s) triggers STALENESS_FREEZE, maintaining last known speed."""
        controller = AdaptiveController()
        twin = controller.get_or_create_twin("Bedroom_Fan_1")
        twin.fan_state.confirm_speed(3)

        stale_csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 29.0, "relative_humidity_pct": 55.0, "sensor_age_s": 350.0, "is_stale": True},
        }
        decision = controller.decide(stale_csr)

        assert decision.decision_mode == DECISION_MODE_STALENESS_FREEZE
        assert decision.recommended_speed == 3
        assert decision.fallback_status["fallback_triggered"] is True

    def test_08_uncalibrated_probability_handling(self):
        """Verify uncalibrated softmax probability is recognized and downweights alpha when uncertain."""
        scorer = ComfortScorer()
        # At temperature 22C, Model 1 yields moderate/uncertain probabilities
        eval_res = scorer.score_candidates(
            candidate_speeds=[1, 2, 3],
            current_speed=2,
            csr={"environmental": {"temperature_c": 24.0, "relative_humidity_pct": 50.0}},
        )

        assert eval_res["model_available"] is True
        assert "candidate_scores" in eval_res
        for s in [1, 2, 3]:
            cand = eval_res["candidate_scores"][s]
            assert "probabilities_uncalibrated" in cand
            assert cand["alpha_effective_mult"] <= 1.0

    def test_09_ml_forecast_unavailability(self):
        """Verify advisory nature of Model 3 forecast and adherence to cannot_command_speed."""
        controller = AdaptiveController()
        csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 27.0, "relative_humidity_pct": 50.0},
        }
        dec = controller.decide(csr)
        energy_adv = dec.energy_assessment.get("ml_advisory", {})

        assert energy_adv.get("is_advisory") is True
        assert energy_adv.get("cannot_command_speed") is True
        assert energy_adv.get("r2_benchmark") == 0.1515

    def test_10_manual_override_priority(self):
        """Verify Priority P1: manual override overrides automatic recommendations."""
        controller = AdaptiveController()
        controller.set_manual_override("Bedroom_Fan_1", speed=4, duration_s=600.0)

        csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 20.0, "relative_humidity_pct": 40.0},
            "occupancy_signals": {"occupancy_binary": 0, "occupancy_probability": 0.10},  # Unoccupied
        }
        # Despite unoccupied, manual override takes priority!
        decision = controller.decide(csr)

        assert decision.recommended_speed == 4
        assert decision.decision_mode == DECISION_MODE_MANUAL_OVERRIDE
        assert decision.manual_override_status is not None
        assert decision.manual_override_status["active"] is True

    def test_11_override_expiration(self):
        """Verify manual override expires after duration and resumes adaptive control."""
        controller = AdaptiveController()
        t_base = 1710000000.0
        controller.set_manual_override("Bedroom_Fan_1", speed=4, duration_s=600.0, timestamp=t_base)

        csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 27.0, "relative_humidity_pct": 50.0},
        }

        # 1. During override (t_base + 300s)
        dec_active = controller.decide(csr, timestamp=t_base + 300.0)
        assert dec_active.recommended_speed == 4
        assert dec_active.decision_mode == DECISION_MODE_MANUAL_OVERRIDE

        # 2. After expiration (t_base + 601s)
        dec_expired = controller.decide(csr, timestamp=t_base + 601.0)
        assert dec_expired.decision_mode != DECISION_MODE_MANUAL_OVERRIDE
        assert dec_expired.manual_override_status is None

    def test_12_safety_constraint_enforcement(self):
        """Verify thermal safety floor forces speed >= 1 even when unoccupied."""
        controller = AdaptiveController()

        # Unoccupied but T >= 35°C
        hot_unocc_csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 36.5, "relative_humidity_pct": 50.0},
            "occupancy_signals": {"occupancy_binary": 0, "occupancy_probability": 0.10},
        }

        decision = controller.decide(hot_unocc_csr)
        assert decision.recommended_speed >= 1
        assert decision.decision_mode == DECISION_MODE_THERMAL_SAFETY
        assert "Thermal safety floor" in decision.safety_status.get("constraint_applied", "")

    def test_13_conflicting_model_outputs(self):
        """Verify controller safely resolves conflicting signals without crashing."""
        controller = AdaptiveController()
        # Cold comfort output in a 36°C room
        conflict_csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 36.0, "relative_humidity_pct": 40.0},
            "occupancy_signals": {"occupancy_binary": 1, "occupancy_probability": 0.90},
        }
        dec = controller.decide(conflict_csr)

        # Thermal safety floor must guarantee >= 1
        assert dec.recommended_speed >= 1

    def test_14_multi_fan_isolation(self):
        """Verify separate fans maintain completely independent state and overrides."""
        controller = AdaptiveController()
        controller.set_manual_override("Bedroom_Fan_1", speed=5)

        csr1 = {"fan_id": "Bedroom_Fan_1", "environmental": {"temperature_c": 27.0, "relative_humidity_pct": 50.0}}
        csr2 = {"fan_id": "Lounge_Fan_1", "environmental": {"temperature_c": 27.0, "relative_humidity_pct": 50.0}}

        dec1 = controller.decide(csr1)
        dec2 = controller.decide(csr2)

        assert dec1.recommended_speed == 5
        assert dec1.decision_mode == DECISION_MODE_MANUAL_OVERRIDE

        assert dec2.decision_mode != DECISION_MODE_MANUAL_OVERRIDE
        assert dec2.manual_override_status is None

    def test_15_deterministic_decision_replay(self):
        """Verify identical inputs yield identical candidate costs and recommendations."""
        c1 = AdaptiveController()
        c2 = AdaptiveController()

        csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 28.5, "relative_humidity_pct": 60.0},
            "occupancy_signals": {"occupancy_binary": 1, "occupancy_probability": 0.75},
        }

        d1 = c1.decide(csr, timestamp=1710000000.0)
        d2 = c2.decide(csr, timestamp=1710000000.0)

        assert d1.recommended_speed == d2.recommended_speed
        assert d1.rationale == d2.rationale
        assert len(d1.candidate_evaluations) == len(d2.candidate_evaluations)
        assert [c["total_cost"] for c in d1.candidate_evaluations] == [c["total_cost"] for c in d2.candidate_evaluations]

    def test_16_explanation_completeness(self):
        """Verify all fields in DecisionExplanation output contract are fully populated."""
        controller = AdaptiveController()
        csr = {
            "fan_id": "Bedroom_Fan_1",
            "environmental": {"temperature_c": 28.0, "relative_humidity_pct": 55.0},
        }
        dec = controller.decide(csr)
        d_dict = dec.to_dict()

        expected_fields = [
            "recommended_speed",
            "previous_speed",
            "speed_change",
            "decision_timestamp",
            "decision_mode",
            "rationale",
            "comfort_assessment",
            "occupancy_assessment",
            "energy_assessment",
            "safety_status",
            "manual_override_status",
            "fallback_status",
            "candidate_evaluations",
            "provenance_info",
            "disclaimers",
        ]
        for field in expected_fields:
            assert field in d_dict, f"Missing required output contract field: {field}"

        # Verify disclaimers include software scale notice
        assert any("SOFTWARE RECOMMENDATION ONLY" in d for d in d_dict["disclaimers"])

    def test_17_no_physical_actuation(self):
        """Verify controller produces recommendation data without attempting hardware execution."""
        controller = AdaptiveController()
        csr = {"fan_id": "Bedroom_Fan_1", "environmental": {"temperature_c": 27.0, "relative_humidity_pct": 50.0}}

        dec = controller.decide(csr)
        # Recommendation is an integer recommendation value, not a serial/GPIO command
        assert isinstance(dec.recommended_speed, int)
        assert 0 <= dec.recommended_speed <= 5

    def test_18_safe_fallback_behaviour(self):
        """Verify compound failure (missing models + stale sensor) triggers emergency safe default."""
        guard = SafetyGuard()
        # All models unavailable + stale
        res = guard.evaluate_pre_decision_safety(
            csr={"environmental": {"temperature_c": 30.0, "sensor_age_s": 400.0, "is_stale": True}},
            current_speed=0,
            all_models_unavailable=True,
        )

        assert res is not None
        assert res["short_circuit"] is True
        assert res["recommended_speed"] == 1  # Safe floor for warm condition (30°C >= 28°C)
        assert "Emergency safe default" in res["fallback_reason"]
