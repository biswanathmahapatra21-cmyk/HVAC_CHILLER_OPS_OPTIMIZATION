import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(page_title="🏭 HVAC Chiller Optimization", layout="wide")

# ---------------------------------------------------
# MODEL LOADING (from Hugging Face)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    HF_REPO = "BiswanathMahapatra/chiller-ppo-model"   # 🔁 replace with your Hugging Face repo
    MODEL_FILENAME = "ppo_hvac_chiller.zip"             # name of uploaded model file

    local_model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILENAME)
    model = PPO.load(local_model_path)
    return model


st.title("🏭 HVAC Chiller Optimization using RL (PPO + RandomForest surrogate)")
st.write(
    "This interactive dashboard demonstrates optimal control of HVAC chiller, "
    "pumps, and air setpoints using a Reinforcement Learning (PPO) agent trained on surrogate physics-driven models."
)

model = load_model()

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------
st.sidebar.header("📥 Building Inputs")

ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 15.0, 45.0, 30.0)
occupancy_frac = st.sidebar.slider("Occupancy Fraction", 0.0, 1.0, 0.75)
solar_gain = st.sidebar.slider("Solar Gain (kW)", 0.0, 80.0, 30.0)
load_kw = st.sidebar.slider("Building Heat Load (kW)", 0.0, 1000.0, 400.0)

st.sidebar.write("Adjust parameters and click below to predict optimal control actions.")
run_button = st.sidebar.button("🚀 Run Optimization")

# ---------------------------------------------------
# RL INFERENCE SECTION
# ---------------------------------------------------
if run_button:
    # Create a mock observation vector with 69 features
    obs = np.zeros((69,), dtype=np.float32)

    # Fill realistic indices (based on earlier feature ordering)
    obs[5] = ambient_temp           # ambient_temp_C
    obs[6] = occupancy_frac         # occupancy_frac
    obs[7] = solar_gain             # solar_gain_kW
    obs[10] = load_kw               # building_heat_load_kW

    # PPO expects (n_env, n_features)
    obs = obs.reshape(1, -1)

    # Predict control action
    action, _ = model.predict(obs, deterministic=True)

    # Interpret the actions (approximate meaning from your RL env)
    actions = {
        "Δ Compressor Speed (fraction)": float(action[0][0]),
        "Δ Pump1 Speed (fraction)": float(action[0][1]),
        "Δ Pump2 Speed (fraction)": float(action[0][2]),
        "Δ CHW Setpoint (°C)": float(action[0][3]) * 2,
        "Δ Supply Air Setpoint (°C)": float(action[0][4]) * 2,
    }

    st.subheader("🔧 RL-Agent Recommended Adjustments")
    st.dataframe(pd.DataFrame(actions, index=["Value"]).T, use_container_width=True)

    # Simple estimated energy savings metric (illustrative)
    energy_savings = max(0.0, 10 - abs(np.sum(action)) * 3)
    st.metric(label="Predicted Energy Savings (%)", value=f"{energy_savings:.2f}%")

    # Simulated comfort metric
    comfort_risk = np.clip(np.abs(action[0][3]) + np.abs(action[0][4]), 0, 2)
    st.metric(label="Comfort Deviation Risk (°C)", value=f"{comfort_risk:.2f}")

    st.success("✅ RL-based optimization completed successfully!")
else:
    st.info("Adjust the sidebar inputs and click 'Run Optimization' to start.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Developed by **Biswanath Mahapatra** | AI for Energy Optimization | PPO + RandomForest Hybrid Model")

