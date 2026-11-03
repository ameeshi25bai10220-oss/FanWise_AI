"""
AACI V3 – Interactive Streamlit Dashboard Component
===================================================
Atomberg Adaptive Comfort Intelligence (AACI)

Interactive web dashboard demonstrating:
  - Real-time environmental sensing & biometeorological feature engine
  - Generic V2 XGBoost inference vs Individual Personalization
  - Multi-occupant profiles (Generic, Warm-Sensitive, Chill-Sensitive, Custom)
  - Interactive feedback learning (Hot, Cold, Neutral, Manual Speed + Reason)
  - AACIFanControllerV3 decision engine with fault injection and safety guards
  - Atomberg BLDC power savings vs traditional induction fans

Run locally:
    streamlit run aaci_ml/v3_personalization/streamlit_aaci_component.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st

# Ensure v3 directory is on path
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from personalization_engine import PersonalizationEngine
from fan_controller_v3 import AACIFanControllerV3

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atomberg AACI V3 – Adaptive Comfort Intelligence",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = PersonalizationEngine()
    # Pre-seed preset profiles
    st.session_state.engine.record_feedback("user_warm", "FEELING_HOT", 26.5, 60.0, 2)
    st.session_state.engine.record_feedback("user_warm", "MANUAL_SPEED_CHANGE", 26.5, 60.0, 2, reason="TOO_HOT", manual_new_speed=4)
    st.session_state.engine.record_feedback("user_chill", "FEELING_COLD", 25.0, 55.0, 2)
    st.session_state.engine.record_feedback("user_chill", "MANUAL_SPEED_CHANGE", 25.0, 55.0, 2, reason="ROOM_DRAFT", manual_new_speed=1)

if "controller" not in st.session_state:
    st.session_state.controller = AACIFanControllerV3(fan_model="Atomberg Renesa BLDC")
    st.session_state.current_speed = 2

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://atomberg.com/wp-content/themes/atomberg/assets/images/logo.png", width=180) if False else None
st.sidebar.title("🌀 AACI Control Station")
st.sidebar.markdown("**Atomberg Adaptive Comfort Intelligence (V3)**")

st.sidebar.subheader("1. Active Occupant Profile")
occupant_choice = st.sidebar.selectbox(
    "Select Occupant Profile:",
    [
        "Generic Model (Zero Personalization)",
        "Occupant A: Warm-Sensitive ('Runs Hot')",
        "Occupant B: Chill-Sensitive ('Draft-Sensitive')",
        "Custom User (Interactive Feedback Mode)",
    ]
)

user_id_map = {
    "Generic Model (Zero Personalization)": "user_generic",
    "Occupant A: Warm-Sensitive ('Runs Hot')": "user_warm",
    "Occupant B: Chill-Sensitive ('Draft-Sensitive')": "user_chill",
    "Custom User (Interactive Feedback Mode)": "user_custom",
}
active_user_id = user_id_map[occupant_choice]

st.sidebar.subheader("2. Ambient Sensors")
temp_c = st.sidebar.slider("Room Temperature (°C)", min_value=16.0, max_value=38.0, value=27.5, step=0.1)
rh = st.sidebar.slider("Relative Humidity (%)", min_value=20.0, max_value=95.0, value=60.0, step=1.0)
air_v = st.sidebar.slider("Current Air Velocity (m/s)", min_value=0.05, max_value=1.50, value=0.25, step=0.05)
t_out = st.sidebar.slider("Outdoor Temperature (°C)", min_value=15.0, max_value=45.0, value=32.0, step=0.5)

st.sidebar.subheader("3. Fault Injection Testing")
inject_fault = st.sidebar.selectbox(
    "Simulate Sensor / Telemetry Fault:",
    ["None (Normal Operation)", "Invalid Speed (Speed = 99)", "Missing Temperature Sensor", "Extreme Impossible Temp (65°C)", "Stale Telemetry (>180s)"]
)

# ─────────────────────────────────────────────────────────────────────────────
# CORE PREDICTION COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
# Generic baseline model approximation based on verified V2 psychrometric response
if temp_c >= 28.5:
    p_cold, p_neu, p_hot = 0.05, 0.22, 0.73
elif temp_c >= 27.0:
    p_cold, p_neu, p_hot = 0.10, 0.42, 0.48
elif temp_c <= 22.0:
    p_cold, p_neu, p_hot = 0.72, 0.22, 0.06
elif temp_c <= 24.5:
    p_cold, p_neu, p_hot = 0.48, 0.40, 0.12
else:
    p_cold, p_neu, p_hot = 0.20, 0.58, 0.22

generic_probs = {"Cold": p_cold, "Neutral": p_neu, "Hot": p_hot}

# Apply Personalization Layer
if active_user_id == "user_generic":
    personalized_probs = generic_probs
    predicted_class = max(generic_probs, key=generic_probs.get)
    confidence = generic_probs[predicted_class]
    active_profile = None
else:
    pers_res = st.session_state.engine.personalize_probabilities(generic_probs, active_user_id, temp_c)
    personalized_probs = pers_res["probabilities"]
    predicted_class = pers_res["personalized_class"]
    confidence = pers_res["confidence"]
    active_profile = st.session_state.engine.get_or_create_profile(active_user_id)

# Apply Fault Injection overrides
sim_speed = st.session_state.current_speed
sim_temp = temp_c
sim_timestamp = time.time()

if inject_fault == "Invalid Speed (Speed = 99)":
    sim_speed = 99
elif inject_fault == "Missing Temperature Sensor":
    sim_temp = None
elif inject_fault == "Extreme Impossible Temp (65°C)":
    sim_temp = 65.0
elif inject_fault == "Stale Telemetry (>180s)":
    sim_timestamp = time.time() - 250.0

# Evaluate Fan Controller Decision
decision = st.session_state.controller.evaluate_decision(
    predicted_class=predicted_class,
    probabilities=personalized_probs,
    air_temp_c=sim_temp,
    relative_humidity=rh,
    current_fan_speed=sim_speed,
    sensor_timestamp=sim_timestamp,
    current_timestamp=time.time(),
)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌀 Atomberg AACI: Adaptive Comfort Intelligence")
st.markdown("""
*Production AI-Assisted Smart Ceiling Fan Control Layer with Multi-Occupant Personalization, 
Dynamic Confidence Auditing, and BLDC Energy Optimization.*
""")

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Predicted Comfort State", predicted_class, delta=f"{confidence:.1%} conf")
with kpi2:
    st.metric("Current Fan Speed", f"Speed {st.session_state.current_speed}", delta=f"{decision['speed_delta']:+d} step" if decision['speed_delta'] != 0 else "Holding")
with kpi3:
    st.metric("Target Recommended Speed", f"Speed {decision['target_speed']}", delta=decision["action"])
with kpi4:
    cooling_c = decision.get("estimated_cooling_c", 0.0)
    st.metric("Convective Wind Chill", f"-{cooling_c:.2f} °C", delta=f"{cooling_c*1.8:.1f} °F equivalent")

st.divider()

# Layout: 2 Columns (Decision & Motor Action vs Probabilities & Profile)
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("🛡️ Controller Action & Motor Execution")
    
    # Action Banner
    action = decision["action"]
    if "FAULT" in action:
        st.error(f"⚠️ **ACTION: {action}** — {decision['reason']}")
    elif "SAFETY" in action:
        st.warning(f"🛡️ **SAFETY OVERRIDE: {action}** — {decision['reason']}")
    elif "INCREASE" in action:
        st.success(f"🔼 **ACTION: {action} (Speed {decision['target_speed']})** — {decision['reason']}")
    elif "DECREASE" in action:
        st.info(f"🔽 **ACTION: {action} (Speed {decision['target_speed']})** — {decision['reason']}")
    else:
        st.markdown(f"⏸️ **ACTION: {action} (Speed {decision['target_speed']})** — {decision['reason']}")

    # Apply button
    if decision["speed_delta"] != 0 and "FAULT" not in action:
        if st.button("⚡ Apply Target Fan Speed to Atomberg Renesa Motor"):
            st.session_state.current_speed = decision["target_speed"]
            st.session_state.controller.current_speed = decision["target_speed"]
            st.session_state.controller.last_adjustment_time = time.time()
            st.rerun()

    # Energy Comparison Card
    st.markdown("#### ⚡ Real-Time Power & Energy Optimization")
    power_map_bldc = {0: 0, 1: 5, 2: 10, 3: 16, 4: 22, 5: 28}
    power_map_ind = {0: 0, 1: 30, 2: 40, 3: 50, 4: 65, 5: 75}
    
    bldc_watt = power_map_bldc.get(decision['target_speed'], 16)
    ind_watt = power_map_ind.get(decision['target_speed'], 50)
    savings_pct = int(((ind_watt - bldc_watt) / max(1, ind_watt)) * 100) if ind_watt > 0 else 0
    
    st.write(f"• **Atomberg BLDC Motor (Catalog Spec):** `{bldc_watt} W` (Speed {decision['target_speed']})")
    st.write(f"• **Traditional Induction Motor (Baseline):** `{ind_watt} W`")
    st.write(f"• **Hardware BLDC Motor Reduction:** `{savings_pct}% lower power` vs induction motor")
    st.caption("ℹ️ *AACI software achieves additional energy optimization via autonomous DECREASE_ECO speed trims during thermally neutral periods.*")

with col_right:
    st.subheader("📊 Probability Breakdown & Confidence")
    
    prob_df = pd.DataFrame({
        "Thermal State": ["Cold", "Neutral", "Hot"],
        "Personalized": [personalized_probs["Cold"], personalized_probs["Neutral"], personalized_probs["Hot"]],
        "Generic Baseline": [generic_probs["Cold"], generic_probs["Neutral"], generic_probs["Hot"]],
    })
    st.bar_chart(prob_df.set_index("Thermal State"), height=220)

    if active_profile:
        st.markdown("#### 👤 Active Personalization Profile")
        p_dict = active_profile.to_dict()
        st.write(f"• **User ID:** `{p_dict['user_id']}` ({p_dict['feedback_count']} feedback interactions)")
        st.write(f"• **Thermal Setpoint Shift:** `{p_dict['setpoint_shift_c']:+.2f} °C` from population mean")
        st.write(f"• **Logit Offsets [Cold, Neu, Hot]:** `{p_dict['logit_offsets']}`")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE FEEDBACK & MANUAL OVERRIDE SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💬 Interactive Occupant Feedback & Learning")
st.markdown("Provide real-time feedback to train the active personalization layer online:")

fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)

with fb_col1:
    if st.button("🔥 I Feel Too Hot (+Hot)"):
        st.session_state.engine.record_feedback(active_user_id, "FEELING_HOT", temp_c, rh, st.session_state.current_speed)
        st.success("Learned! Boosted warm-sensitivity profile.")
        st.rerun()

with fb_col2:
    if st.button("❄️ I Feel Too Cold (+Cold)"):
        st.session_state.engine.record_feedback(active_user_id, "FEELING_COLD", temp_c, rh, st.session_state.current_speed)
        st.success("Learned! Boosted draft-sensitivity profile.")
        st.rerun()

with fb_col3:
    if st.button("👌 I Feel Comfortable (Neutral)"):
        st.session_state.engine.record_feedback(active_user_id, "FEELING_COMFORTABLE", temp_c, rh, st.session_state.current_speed)
        st.success("Learned! Reinforced neutral comfort.")
        st.rerun()

with fb_col4:
    if st.button("🔄 Reset Profile to Factory"):
        st.session_state.engine.reset_profile(active_user_id)
        st.warning("Profile reset to generic baseline.")
        st.rerun()

# Manual Override Drawer
with st.expander("🛠️ Set Manual Override (Remote / App Speed Command)"):
    ov_c1, ov_c2, ov_c3 = st.columns(3)
    with ov_c1:
        manual_spd = st.selectbox("Select Target Fan Speed:", [0, 1, 2, 3, 4, 5], index=st.session_state.current_speed)
    with ov_c2:
        reason_choice = st.selectbox("Reason for Override:", [
            "TOO_HOT (Thermal discomfort)",
            "TOO_COLD (Draft avoidance)",
            "WHITE_NOISE (Acoustic background for sleep)",
            "STUFFY_AIR (Air circulation)",
            "ROOM_DRAFT (Direct air disturbance)",
        ])
    with ov_c3:
        duration_h = st.slider("Hold Duration (Hours):", 1, 8, 2)

    ov_btn_c1, ov_btn_c2 = st.columns(2)
    with ov_btn_c1:
        if st.button("📌 Lock In Manual Override"):
            parsed_reason = reason_choice.split(" ")[0]
            st.session_state.controller.set_manual_override(
                speed=manual_spd,
                duration_mins=duration_h * 60,
                reason=parsed_reason,
            )
            # Record in personalization engine to learn non-thermal vs thermal reasons
            st.session_state.engine.record_feedback(
                user_id=active_user_id,
                feedback_type="MANUAL_SPEED_CHANGE",
                air_temp_c=temp_c,
                relative_humidity=rh,
                current_speed=st.session_state.current_speed,
                reason=parsed_reason,
                manual_new_speed=manual_spd,
            )
            st.session_state.current_speed = manual_spd
            st.info(f"Manual override locked at Speed {manual_spd} for {duration_h}h. Reason: {parsed_reason}")
            st.rerun()
    with ov_btn_c2:
        if st.button("🔓 Clear Manual Override (Resume Auto)"):
            st.session_state.controller.clear_manual_override()
            st.success("Automated AI comfort control resumed.")
            st.rerun()

st.caption("AACI V3 Intelligent Smart Ceiling Fan Engine • Atomberg Technologies Hackathon 2026")

