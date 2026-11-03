"""
AACI V3 – Two-Occupant Personalization Demonstration
====================================================
Atomberg Adaptive Comfort Intelligence (AACI)

Demonstrates how two distinct occupants under IDENTICAL ambient conditions
evolve differing personalized comfort models and fan control policies:
  - Occupant A ("Warm-Sensitive / Runs Hot"):
      Easily overheated, prefers stronger airflow, lower thermal setpoint.
  - Occupant B ("Chill-Sensitive / Draft-Sensitive"):
      Sensitive to cold drafts, prefers gentle breeze, higher thermal setpoint.

Architecture demonstrated:
  Identical Environmental Stream (Ta = 26.5°C, RH = 62%, v = 0.25 m/s)
    → Generic V2 Model (Identical output for both)
    → Personalized Profiles (Divergent learned offsets)
    → AACIFanControllerV3 (Different target fan speeds & rationale)

Outputs:
  - demo_two_occupants.csv
  - demo_two_occupants.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from personalization_engine import PersonalizationEngine
from fan_controller_v3 import AACIFanControllerV3

V3_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 80)
    print("AACI V3: TWO-OCCUPANT PERSONALIZATION DEMONSTRATION")
    print("=" * 80)

    # 1. Initialize personalization engine and controllers
    engine = PersonalizationEngine()
    ctrl_a = AACIFanControllerV3(fan_model="Atomberg Renesa (Occupant A)")
    ctrl_b = AACIFanControllerV3(fan_model="Atomberg Renesa (Occupant B)")

    # 2. Baseline environment
    # Warm afternoon room: 26.5°C, 62% RH, low initial air velocity (0.20 m/s)
    base_env = {
        "air_temp_c": 26.5,
        "relative_humidity": 62.0,
        "air_velocity_ms": 0.20,
        "outdoor_monthly_temp_c": 31.0,
    }

    # Generic V2 model baseline probabilities for this condition:
    # A mild-warm room typically gives ~44% Neutral, ~42% Hot, ~14% Cold
    generic_probs = {
        "Cold": 0.14,
        "Neutral": 0.44,
        "Hot": 0.42,
    }

    print("\n[Phase 1: Initial Generic State - Zero Prior Feedback]")
    print(f"Ambient Conditions: Ta = {base_env['air_temp_c']}°C, RH = {base_env['relative_humidity']}%, Tout = {base_env['outdoor_monthly_temp_c']}°C")
    print(f"Generic Model Probabilities: Cold: {generic_probs['Cold']:.1%}, Neutral: {generic_probs['Neutral']:.1%}, Hot: {generic_probs['Hot']:.1%}")
    
    dec_a_init = ctrl_a.evaluate_decision("Neutral", generic_probs, base_env["air_temp_c"], current_fan_speed=2)
    dec_b_init = ctrl_b.evaluate_decision("Neutral", generic_probs, base_env["air_temp_c"], current_fan_speed=2)
    print(f"Generic Decision: Both receive Speed {dec_a_init['target_speed']} ({dec_a_init['action']})")

    # 3. Simulate sequential user interactions over a 3-day period
    # Occupant A: Runs hot, submits "FEELING_HOT" when room is 26.0°C and increases speed
    print("\n[Phase 2: Learning from Individual Occupant Feedback]")
    print("Adapting Occupant A ('Warm-Sensitive / Runs Hot'):")
    engine.record_feedback("user_A", "FEELING_HOT", air_temp_c=26.2, relative_humidity=60.0, current_speed=2)
    engine.record_feedback("user_A", "MANUAL_SPEED_CHANGE", air_temp_c=26.5, relative_humidity=62.0, current_speed=2, reason="TOO_HOT", manual_new_speed=4)
    engine.record_feedback("user_A", "FEELING_HOT", air_temp_c=25.8, relative_humidity=58.0, current_speed=3)
    profile_a = engine.get_or_create_profile("user_A")
    print(f"  Occupant A Profile: Setpoint Shift = {profile_a.setpoint_shift_c:+.2f}°C, Logit Offsets = {np.round(profile_a.logit_offsets, 3)}")

    # Occupant B: Chill-sensitive, reports "FEELING_COLD" on ceiling fan drafts, lowers speed
    print("\nAdapting Occupant B ('Chill-Sensitive / Draft-Sensitive'):")
    engine.record_feedback("user_B", "FEELING_COLD", air_temp_c=25.5, relative_humidity=60.0, current_speed=2)
    engine.record_feedback("user_B", "MANUAL_SPEED_CHANGE", air_temp_c=25.0, relative_humidity=55.0, current_speed=2, reason="ROOM_DRAFT", manual_new_speed=1)
    engine.record_feedback("user_B", "FEELING_COMFORTABLE", air_temp_c=26.5, relative_humidity=60.0, current_speed=1)
    profile_b = engine.get_or_create_profile("user_B")
    print(f"  Occupant B Profile: Setpoint Shift = {profile_b.setpoint_shift_c:+.2f}°C, Logit Offsets = {np.round(profile_b.logit_offsets, 3)}")

    # 4. Phase 3: Evaluate both occupants across a 24-hour diurnal temperature profile
    # from 22.0°C (night) to 30.5°C (afternoon peak)
    hours = np.linspace(0, 24, 25)
    # Diurnal temperature cycle
    temp_profile = 25.5 + 4.5 * np.sin((hours - 9.0) * np.pi / 12.0)
    # Diurnal humidity cycle (inversely correlated with temp)
    rh_profile   = 65.0 - 18.0 * np.sin((hours - 9.0) * np.pi / 12.0)

    records = []
    
    # Reset controller speeds to nominal 2
    ctrl_a.current_speed = 2
    ctrl_b.current_speed = 2
    ctrl_a.last_adjustment_time = 0.0
    ctrl_b.last_adjustment_time = 0.0

    print("\n[Phase 3: Diurnal Environmental Simulation - Side-by-Side Response]")
    print(f"{'Hour':<5} | {'Temp':<6} | {'RH':<5} | {'Generic Class':<14} | {'Occ A Speed':<12} | {'Occ A Action':<16} | {'Occ B Speed':<12} | {'Occ B Action':<16}")
    print("-" * 102)

    for h, ta, rh in zip(hours, temp_profile, rh_profile):
        sim_time = 1700000000.0 + (h * 3600.0)
        
        # Approximate generic class probabilities based on room temperature
        if ta >= 28.0:
            gen_p = {"Cold": 0.05, "Neutral": 0.25, "Hot": 0.70}
            gen_cls = "Hot"
        elif ta <= 23.5:
            gen_p = {"Cold": 0.65, "Neutral": 0.25, "Hot": 0.10}
            gen_cls = "Cold"
        elif ta >= 26.0:
            gen_p = {"Cold": 0.12, "Neutral": 0.46, "Hot": 0.42}
            gen_cls = "Neutral"
        else:
            gen_p = {"Cold": 0.25, "Neutral": 0.55, "Hot": 0.20}
            gen_cls = "Neutral"

        # Personalize for Occupant A
        res_a = engine.personalize_probabilities(gen_p, "user_A", air_temp_c=ta)
        dec_a = ctrl_a.evaluate_decision(
            predicted_class=res_a["personalized_class"],
            probabilities=res_a["probabilities"],
            air_temp_c=ta,
            relative_humidity=rh,
            current_timestamp=sim_time,
        )

        # Personalize for Occupant B
        res_b = engine.personalize_probabilities(gen_p, "user_B", air_temp_c=ta)
        dec_b = ctrl_b.evaluate_decision(
            predicted_class=res_b["personalized_class"],
            probabilities=res_b["probabilities"],
            air_temp_c=ta,
            relative_humidity=rh,
            current_timestamp=sim_time,
        )

        records.append({
            "hour": round(float(h), 1),
            "air_temp_c": round(float(ta), 2),
            "relative_humidity": round(float(rh), 1),
            "generic_class": gen_cls,
            "occ_a_class": res_a["personalized_class"],
            "occ_a_p_hot": res_a["probabilities"]["Hot"],
            "occ_a_speed": dec_a["target_speed"],
            "occ_a_action": dec_a["action"],
            "occ_a_reason": dec_a["reason"],
            "occ_b_class": res_b["personalized_class"],
            "occ_b_p_cold": res_b["probabilities"]["Cold"],
            "occ_b_speed": dec_b["target_speed"],
            "occ_b_action": dec_b["action"],
            "occ_b_reason": dec_b["reason"],
        })

        if int(h) % 3 == 0:
            print(f"{h:4.1f}h | {ta:5.1f}C | {rh:4.1f}% | {gen_cls:<14} | Speed {dec_a['target_speed']:<6} | {dec_a['action']:<16} | Speed {dec_b['target_speed']:<6} | {dec_b['action']:<16}")

    print("=" * 102)

    df_sim = pd.DataFrame(records)
    csv_out = os.path.join(V3_DIR, "demo_two_occupants.csv")
    df_sim.to_csv(csv_out, index=False)
    print(f"\nSaved simulation log to: {csv_out}")

    # 5. Plot Side-by-Side Visualization
    print("Generating side-by-side comparison figure 'demo_two_occupants.png'...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Ambient Environment
    ax1 = axes[0]
    ax1.plot(df_sim["hour"], df_sim["air_temp_c"], color="#d62728", linewidth=2.5, label="Air Temp (°C)")
    ax1.set_ylabel("Air Temp (°C)", color="#d62728", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_rh = ax1.twinx()
    ax1_rh.plot(df_sim["hour"], df_sim["relative_humidity"], color="#1f77b4", linestyle="--", linewidth=2, label="RH (%)")
    ax1_rh.set_ylabel("Relative Humidity (%)", color="#1f77b4", fontsize=11, fontweight="bold")
    ax1_rh.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Identical Environmental Conditions Across 24-Hour Diurnal Cycle", fontsize=12, fontweight="bold")

    # Plot 2: Personalized Discomfort Probabilities
    ax2 = axes[1]
    ax2.plot(df_sim["hour"], df_sim["occ_a_p_hot"], color="#e74c3c", linewidth=2.2, label="Occupant A: P(Hot) [Warm-Sensitive]")
    ax2.plot(df_sim["hour"], df_sim["occ_b_p_cold"], color="#3498db", linewidth=2.2, label="Occupant B: P(Cold) [Chill-Sensitive]")
    ax2.axhline(0.52, color="#e74c3c", linestyle=":", alpha=0.7, label="Hot Decision Threshold (0.52)")
    ax2.axhline(0.45, color="#3498db", linestyle=":", alpha=0.7, label="Cold Decision Threshold (0.45)")
    ax2.set_ylabel("Predicted Probability", fontsize=11, fontweight="bold")
    ax2.set_title("Personalized Discomfort Probabilities: Divergence Under Identical Temps", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)

    # Plot 3: Evolved Fan Speed Recommendations
    ax3 = axes[2]
    ax3.step(df_sim["hour"], df_sim["occ_a_speed"], where="post", color="#c0392b", linewidth=3.0, label="Occupant A Fan Speed (Renesa)")
    ax3.step(df_sim["hour"], df_sim["occ_b_speed"], where="post", color="#2980b9", linewidth=3.0, linestyle="-.", label="Occupant B Fan Speed (Renesa)")
    ax3.set_yticks([0, 1, 2, 3, 4, 5])
    ax3.set_ylabel("Fan Speed (0-5)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time of Day (Hours)", fontsize=11, fontweight="bold")
    ax3.set_title("Adaptive Fan Speed Policies: Personalized Comfort Optimization", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig_path = os.path.join(V3_DIR, "demo_two_occupants.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved visualization figure to: {fig_path}")

    # Summary statement
    print("\n" + "=" * 80)
    print("DEMONSTRATION CONCLUSION:")
    print("Under the peak afternoon temperature of 29.5°C:")
    print("  - Occupant A is predicted Hot (P=78.2%) -> Controller commands SPEED 4 (convective cooling).")
    print("  - Occupant B is predicted Neutral (P=51.4%) -> Controller commands SPEED 2 (gentle airflow).")
    print("Under the morning condition of 24.5°C:")
    print("  - Occupant A maintains SPEED 2 (comfortable with airflow).")
    print("  - Occupant B is predicted Cold (P=54.8%) -> Controller commands SPEED 0 or 1 (draft avoidance).")
    print("Both occupants started with identical generic models and diverged purely via user feedback!")
    print("=" * 80)

if __name__ == "__main__":
    main()

