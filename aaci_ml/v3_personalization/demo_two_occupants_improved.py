"""
AACI V3.1 – Rigorous Two-Occupant Personalization & Persistence Demonstration
=============================================================================
Atomberg Adaptive Comfort Intelligence (AACI)

NOTE: This is a controlled synthetic engineering demonstration designed to verify
multi-occupant logit divergence, controller safety, and SQLite persistence.
It is NOT a clinical human-subject field trial.

Demonstration Workflow:
  1. Starts both occupants with identical factory profiles (unpersonalized).
  2. Evaluates identical baseline conditions using the GENUINE trained V2 XGBoost model.
  3. Applies distinct realistic thermal feedback:
       - Occupant A ("Warm-Sensitive / Runs Hot"): reports Hot on mild temps (26.2C, 25.8C).
       - Occupant B ("Chill-Sensitive / Draft-Sensitive"): reports Cold on drafts (25.5C, 25.0C).
  4. Updates profiles independently and saves directly to persistent SQLite database.
  5. SIMULATES APPLICATION RESTART: Destroys engine and re-instantiates from SQLite disk.
  6. Evaluates both occupants across an identical 24-hour diurnal environmental stream
     (Ta: 21.5°C to 30.5°C, RH: 44% to 80%) with genuine V2 XGBoost model inference.
  7. Contrasts probabilities before vs after personalization and logs divergent fan speed policies.

Outputs:
  - demo_two_occupants_improved.csv
  - demo_two_occupants_improved.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V3_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(V3_DIR))
V2_DIR = os.path.join(ROOT_DIR, "aaci_ml", "v2_improved")

if V3_DIR not in sys.path:
    sys.path.insert(0, V3_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from personalization_database import PersonalizationDatabase
from personalization_engine import PersonalizationEngine
from fan_controller_v3 import AACIFanControllerV3
from predict_v2 import predict_comfort_v2

def main():
    print("=" * 88)
    print("AACI V3.1: CONTROLLED TWO-OCCUPANT DEMONSTRATION & PERSISTENCE VERIFICATION")
    print("=" * 88)
    print("DISCLAIMER: Controlled synthetic demonstration using trained V2 XGBoost inference.")
    print("            Not a real human-subject clinical experiment.\n")

    # Use dedicated demo SQLite DB
    demo_db_path = os.path.join(V3_DIR, "demo_occupant_profiles.db")
    if os.path.exists(demo_db_path):
        try:
            os.remove(demo_db_path)
        except Exception:
            pass

    # 1. Initialize engine with database persistence
    engine = PersonalizationEngine(db_path=demo_db_path)
    ctrl_a = AACIFanControllerV3(fan_model="Atomberg Renesa (Occupant A)")
    ctrl_b = AACIFanControllerV3(fan_model="Atomberg Renesa (Occupant B)")

    # 2. Environmental profile across 24h cycle
    hours = np.linspace(0, 24, 25)
    temp_profile = 26.0 + 4.5 * np.sin((hours - 9.0) * np.pi / 12.0)
    rh_profile   = 62.0 - 18.0 * np.sin((hours - 9.0) * np.pi / 12.0)
    tout_profile = temp_profile + 3.0

    # 3. Genuine V2 Model Inference
    print("[Step 1] Running genuine V2 XGBoost inference across 24h environmental profile...")
    generic_steps = []
    for h, ta, rh, tout in zip(hours, temp_profile, rh_profile, tout_profile):
        v2_res = predict_comfort_v2(
            air_temp_c=ta,
            relative_humidity=rh,
            air_velocity_ms=0.25,
            outdoor_monthly_temp_c=tout,
            current_fan_speed=2,
        )
        generic_steps.append({
            "hour": h,
            "ta": ta,
            "rh": rh,
            "tout": tout,
            "gen_class": v2_res["predicted_class"],
            "gen_probs": v2_res["probabilities"],
        })

    # Verify identical baseline before personalization
    print("\n[Step 2] Verifying identical unpersonalized baseline for both occupants...")
    mid_sample = generic_steps[12] # Hour 12 (Ta = 29.2°C, RH = 49.3%)
    print(f"At Hour 12.0 (Ta = {mid_sample['ta']:.1f}°C, RH = {mid_sample['rh']:.1f}%):")
    print(f"  Generic V2 Output: {mid_sample['gen_class']} (Cold: {mid_sample['gen_probs']['Cold']:.1%}, Neu: {mid_sample['gen_probs']['Neutral']:.1%}, Hot: {mid_sample['gen_probs']['Hot']:.1%})")
    
    dec_a_init = ctrl_a.evaluate_decision(mid_sample["gen_class"], mid_sample["gen_probs"], mid_sample["ta"], measured_fan_speed=2)
    dec_b_init = ctrl_b.evaluate_decision(mid_sample["gen_class"], mid_sample["gen_probs"], mid_sample["ta"], measured_fan_speed=2)
    print(f"  Occ A Decision: Speed {dec_a_init['target_speed']} ({dec_a_init['action']})")
    print(f"  Occ B Decision: Speed {dec_b_init['target_speed']} ({dec_b_init['action']})")
    assert dec_a_init["target_speed"] == dec_b_init["target_speed"], "Initial speeds must be identical!"

    # 4. Apply distinct feedback to occupants
    print("\n[Step 3] Applying distinct feedback to Occupants A and B...")
    # Occupant A: Warm-sensitive
    engine.record_feedback("user_A", "FEELING_HOT", 26.2, 60.0, 2)
    engine.record_feedback("user_A", "MANUAL_SPEED_CHANGE", 26.5, 62.0, 2, reason="TOO_HOT", manual_new_speed=4)
    engine.record_feedback("user_A", "FEELING_HOT", 25.8, 58.0, 3)

    # Occupant B: Chill-sensitive
    engine.record_feedback("user_B", "FEELING_COLD", 25.5, 60.0, 2)
    engine.record_feedback("user_B", "MANUAL_SPEED_CHANGE", 25.0, 55.0, 2, reason="ROOM_DRAFT", manual_new_speed=1)
    engine.record_feedback("user_B", "FEELING_COMFORTABLE", 26.5, 60.0, 1)

    # 5. SIMULATE RESTART: Destroy in-memory engine and reload from SQLite
    print("\n[Step 4] Simulating application restart: destroying in-memory engine and reloading from SQLite...")
    del engine
    restarted_engine = PersonalizationEngine(db_path=demo_db_path)
    prof_a = restarted_engine.get_or_create_profile("user_A")
    prof_b = restarted_engine.get_or_create_profile("user_B")
    print(f"  Recovered Occupant A from Disk: N={prof_a.feedback_count}, Setpoint={prof_a.setpoint_shift_c:+.2f}°C, Logits={np.round(prof_a.logit_offsets, 3)}")
    print(f"  Recovered Occupant B from Disk: N={prof_b.feedback_count}, Setpoint={prof_b.setpoint_shift_c:+.2f}°C, Logits={np.round(prof_b.logit_offsets, 3)}")
    assert prof_a.feedback_count == 3 and prof_b.feedback_count == 3, "Profile feedback counts must match after recovery!"

    # 6. Simulate 24-hour response
    print("\n[Step 5] Simulating 24-hour side-by-side response under identical conditions...")
    ctrl_a.commanded_speed = 2
    ctrl_b.commanded_speed = 2
    ctrl_a.last_adjustment_time = 0.0
    ctrl_b.last_adjustment_time = 0.0

    records = []
    print(f"{'Hour':<5} | {'Temp':<6} | {'RH':<5} | {'Generic Class':<14} | {'Occ A (Hot)':<18} | {'Occ B (Chill)':<18} | {'Spd A':<6} | {'Spd B':<6}")
    print("-" * 96)

    for step in generic_steps:
        h = step["hour"]
        ta = step["ta"]
        rh = step["rh"]
        gen_cls = step["gen_class"]
        gen_p = step["gen_probs"]
        sim_time = 1700000000.0 + (h * 3600.0)

        # Personalize for Occupant A (reloaded from SQLite)
        res_a = restarted_engine.personalize_probabilities(gen_p, "user_A", air_temp_c=ta)
        dec_a = ctrl_a.evaluate_decision(
            predicted_class=res_a["personalized_class"],
            probabilities=res_a["probabilities"],
            air_temp_c=ta,
            relative_humidity=rh,
            measured_fan_speed=ctrl_a.commanded_speed,
            current_timestamp=sim_time,
        )

        # Personalize for Occupant B (reloaded from SQLite)
        res_b = restarted_engine.personalize_probabilities(gen_p, "user_B", air_temp_c=ta)
        dec_b = ctrl_b.evaluate_decision(
            predicted_class=res_b["personalized_class"],
            probabilities=res_b["probabilities"],
            air_temp_c=ta,
            relative_humidity=rh,
            measured_fan_speed=ctrl_b.commanded_speed,
            current_timestamp=sim_time,
        )

        records.append({
            "hour": round(float(h), 1),
            "air_temp_c": round(float(ta), 2),
            "relative_humidity": round(float(rh), 1),
            "gen_class": gen_cls,
            "gen_p_hot": gen_p["Hot"],
            "gen_p_cold": gen_p["Cold"],
            "occ_a_class": res_a["personalized_class"],
            "occ_a_p_hot": res_a["probabilities"]["Hot"],
            "occ_a_speed": dec_a["target_speed"],
            "occ_a_action": dec_a["action"],
            "occ_b_class": res_b["personalized_class"],
            "occ_b_p_cold": res_b["probabilities"]["Cold"],
            "occ_b_speed": dec_b["target_speed"],
            "occ_b_action": dec_b["action"],
        })

        if int(h) % 3 == 0:
            str_a = f"{res_a['personalized_class']} ({res_a['probabilities']['Hot']:.0%})"
            str_b = f"{res_b['personalized_class']} ({res_b['probabilities']['Cold']:.0%})"
            print(f"{h:4.1f}h | {ta:5.1f}C | {rh:4.1f}% | {gen_cls:<14} | {str_a:<18} | {str_b:<18} | Spd {dec_a['target_speed']:<2} | Spd {dec_b['target_speed']:<2}")

    print("=" * 96)

    # Save CSV
    df_res = pd.DataFrame(records)
    csv_path = os.path.join(V3_DIR, "demo_two_occupants_improved.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved demonstration results to: {csv_path}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Ambient Temperature
    axes[0].plot(df_res["hour"], df_res["air_temp_c"], color="#d62728", linewidth=2.5, label="Indoor Room Temp (°C)")
    axes[0].set_ylabel("Air Temp (°C)", color="#d62728", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("Identical Environmental Conditions Across 24h Cycle", fontsize=12, fontweight="bold")

    # Plot 2: Probabilities Before vs After Personalization
    axes[1].plot(df_res["hour"], df_res["gen_p_hot"], color="#7f7f7f", linestyle=":", linewidth=1.8, label="Generic Baseline P(Hot)")
    axes[1].plot(df_res["hour"], df_res["occ_a_p_hot"], color="#e74c3c", linewidth=2.2, label="Occupant A: P(Hot) [Warm-Sensitive, Persisted]")
    axes[1].plot(df_res["hour"], df_res["occ_b_p_cold"], color="#3498db", linewidth=2.2, label="Occupant B: P(Cold) [Chill-Sensitive, Persisted]")
    axes[1].axhline(0.52, color="#e74c3c", linestyle="--", alpha=0.6, label="Hot Decision Threshold (0.52)")
    axes[1].axhline(0.45, color="#3498db", linestyle="--", alpha=0.6, label="Cold Decision Threshold (0.45)")
    axes[1].set_ylabel("Probability", fontsize=11, fontweight="bold")
    axes[1].set_title("Probability Divergence (Genuine V2 XGBoost + Persisted SQLite Profiles)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9)

    # Plot 3: Evolved Fan Speeds
    axes[2].step(df_res["hour"], df_res["occ_a_speed"], where="post", color="#c0392b", linewidth=3.0, label="Occupant A Fan Speed (Renesa)")
    axes[2].step(df_res["hour"], df_res["occ_b_speed"], where="post", color="#2980b9", linewidth=3.0, linestyle="-.", label="Occupant B Fan Speed (Renesa)")
    axes[2].set_yticks([0, 1, 2, 3, 4, 5])
    axes[2].set_ylabel("Fan Speed (0-5)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Hour of Day", fontsize=11, fontweight="bold")
    axes[2].set_title("Controller Motor Action: Differentiated Comfort Policies", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(V3_DIR, "demo_two_occupants_improved.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved visualization figure to: {plot_path}")

if __name__ == "__main__":
    main()
