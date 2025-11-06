import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="HVAC Chiller Optimization", layout="wide")

# -------------------------------
# Load PPO model from Hugging Face
# -------------------------------
@st.cache_resource
def load_model():
    HF_REPO = "BiswanathMahapatra/chiller-ppo-model"  # Your HF repo
    MODEL_FILENAME = "ppo_hvac_chiller.zip"            # Uploaded model file

    # Download model file from Hugging Face
    local_model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILENAME)
    model = PPO.load(local_model_path)
    return model

model = load_model()

st.title("🏭 HVAC Chiller Optimization (RL Agent)")

# ----------------------------------
# Input Section
# ----------------------------------
st.sidebar.header("Building Conditions")
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 20.0, 40.0, 30.0)
occupancy_frac = st.sidebar.slider("Occupancy Fraction", 0.0, 1.0, 0.7)
solar_gain = st.sidebar.slider("Solar Gain (kW)", 0.0, 50.0, 20.0)
load_kw = st.sidebar.slider("Building Heat Load (kW)", 0.0, 500.0, 250.0)

# Dummy normalized feature vector (match RL input scale)
X = np.array([[ambient_temp, occupancy_frac, solar_gain, load_kw]])

# Simulate the model policy inference
action, _ = model.predict(X, deterministic=True)

# Interpret the actions
actions = {
    "Compressor ΔSpeed": action[0][0],
    "Pump1 ΔSpeed": action[0][1],
    "Pump2 ΔSpeed": action[0][2],
    "CHW ΔSetpoint (°C)": action[0][3] * 2,
    "Supply Air ΔSetpoint (°C)": action[0][4] * 2
}

st.subheader("🔧 RL-Agent Recommended Adjustments")
st.write(pd.DataFrame(actions, index=["Adjustment Value"]))

# Predicted improvement estimate (simple illustrative logic)
predicted_savings = np.clip(10 - np.abs(action).sum() * 2, 0, 10)
st.metric("Predicted Energy Savings (%)", f"{predicted_savings:.2f}%")

st.success("RL-based optimization suggestion generated successfully!")
