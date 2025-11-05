# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO
from RL import HVACChillerEnv

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "artifacts", "rl", "ppo_hvac_chiller")

if not os.path.exists(MODEL_PATH):
    alt_path = os.path.join(os.path.dirname(BASE), "artifacts", "rl", "ppo_hvac_chiller.zip")
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path

st.write(f"🔍 Using model from: {MODEL_PATH}")

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    return PPO.load(MODEL_PATH)

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="HVAC RL Demo", layout="centered")
st.title("🏢 HVAC Chiller Optimization with RL")

st.sidebar.header("Building Conditions")
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 0, 45, 32)
occupancy = st.sidebar.slider("Occupancy Fraction", 0.0, 1.0, 0.85, step=0.05)
solar_gain = st.sidebar.number_input("Solar Gain (kW)", value=120.0, step=10.0)
internal_gain = st.sidebar.number_input("Internal Gain (kW)", value=75.0, step=5.0)
heat_load = st.sidebar.number_input("Building Heat Load (kW)", value=250.0, step=10.0)
return_air_temp = st.sidebar.slider("Return Air Temperature (°C)", 18, 30, 26)

if st.sidebar.button("Run Optimization"):
    # ----------------------------
    # Run RL inference
    # ----------------------------
    env = HVACChillerEnv(episode_length_hours=1)
    obs, info = env.reset()

    # overwrite state with new inputs
    new_conditions = {
        "ambient_temp_C": ambient_temp,
        "occupancy_frac": occupancy,
        "solar_gain_kW": solar_gain,
        "building_internal_gain_kW": internal_gain,
        "building_heat_load_kW": heat_load,
        "return_air_temp_C": return_air_temp,
    }
    for k, v in new_conditions.items():
        if k in env.state_series.index:
            env.state_series[k] = v

    obs = env._build_obs_from_series(env.state_series)

    # RL action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # ----------------------------
    # Baseline (no control changes)
    # ----------------------------
    env_base = HVACChillerEnv(episode_length_hours=1)
    obs_base, info_base = env_base.reset()
    for k, v in new_conditions.items():
        if k in env_base.state_series.index:
            env_base.state_series[k] = v
    obs_base = env_base._build_obs_from_series(env_base.state_series)
    zero_action = np.zeros(env_base.action_space.shape, dtype=float)
    _, _, _, _, info_base = env_base.step(zero_action)

    # ----------------------------
    # Display results
    # ----------------------------
    st.subheader("🔧 RL Recommendations")
    st.write(f"**Supply Air Setpoint:** {env.state_series['setpoint_supply_air_temp_C']:.2f} °C")
    st.write(f"**CHW Supply Setpoint:** {env.state_series['setpoint_chilled_water_supply_C']:.2f} °C")

    st.subheader("⚡ Predicted Performance")
    col1, col2 = st.columns(2)
    col1.metric("RL Predicted Power (kW)", f"{info['predicted_total_power_kW']:.1f}")
    col2.metric("Comfort Deviation (°C)", f"{info['comfort_deviation_C']:.2f}")

    st.subheader("📊 Comparison vs Baseline")
    savings = info_base["predicted_total_power_kW"] - info["predicted_total_power_kW"]
    st.write(f"Baseline Power: {info_base['predicted_total_power_kW']:.1f} kW")
    st.write(f"RL Power: {info['predicted_total_power_kW']:.1f} kW")
    st.write(f"**Power Savings: {savings:.1f} kW ({(savings/info_base['predicted_total_power_kW']*100):.1f}%)**")

    if info["comfort_deviation_C"] > 1.0:
        st.error("⚠️ Comfort violation risk detected!")
    else:
        st.success("✅ Comfort maintained within limits.")
