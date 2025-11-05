# scripts/infer_rl.py
"""
Run trained RL agent in inference mode.
Accepts new building conditions and outputs optimal setpoints + predicted savings.
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from RL import HVACChillerEnv

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "artifacts", "rl")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hvac_chiller.zip")

# --------------------------
# Example building conditions
# --------------------------
new_conditions = {
    "hour": 14,
    "dayofweek": 2,          # Wednesday
    "dayofyear": 150,
    "month": 6,              # June
    "is_weekend": 0,
    "ambient_temp_C": 32.0,
    "occupancy_frac": 0.85,
    "solar_gain_kW": 120.0,
    "building_internal_gain_kW": 75.0,
    "building_heat_load_kW": 250.0,
    "return_air_temp_C": 26.0,
    # You can add more fields if your feature list needs them
}

# --------------------------
# Load model + env
# --------------------------
print("Loading RL agent from:", MODEL_PATH)
model = PPO.load(MODEL_PATH)
env = HVACChillerEnv(episode_length_hours=1)  # single-step inference

# Sample initial state, then overwrite with new_conditions
obs, info = env.reset()
for k, v in new_conditions.items():
    if k in env.state_series.index:
        env.state_series[k] = v

# Build obs vector
obs = env._build_obs_from_series(env.state_series)

# --------------------------
# Predict optimal action
# --------------------------
action, _ = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)

# --------------------------
# Report results
# --------------------------
print("\n=== RL Inference Result ===")
print("Input Conditions:", new_conditions)
print("Optimal action vector:", action)
print("Updated setpoints:")
print(" - Supply Air Temp SP (°C):", env.state_series["setpoint_supply_air_temp_C"])
print(" - CHW Supply Temp SP (°C):", env.state_series["setpoint_chilled_water_supply_C"])
print("Predicted Power (kW):", info["predicted_total_power_kW"])
print("Comfort Deviation (°C):", info["comfort_deviation_C"])
print("Reward (energy-efficiency score):", reward)
