# scripts/compare_baseline_rl.py
"""
Compare trained RL agent vs baselines (random, zero-action).
Saves:
 - artifacts/rl/compare_summary.csv
 - artifacts/rl/compare_barplot.png
 - artifacts/rl/compare_timeseries.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from RL import HVACChillerEnv

# -----------------------
# Paths & settings
# -----------------------
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "artifacts", "rl")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hvac_chiller.zip")   # adjust if different filename
N_EPISODES = 10
EP_LENGTH = 24   # hours per episode (env default)

# -----------------------
# Helper: run single episode
# -----------------------
def run_episode(env, policy="random", model=None):
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    power_vals = []
    comfort_vals = []
    actions = []

    while not (terminated or truncated):
        if policy == "rl":
            action, _ = model.predict(obs, deterministic=True)
        elif policy == "random":
            action = env.action_space.sample()
        elif policy == "zero":
            action = np.zeros(env.action_space.shape, dtype=float)
        else:
            raise ValueError("Unknown policy")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        power_vals.append(info.get("predicted_total_power_kW", np.nan))
        comfort_vals.append(info.get("comfort_deviation_C", np.nan))
        actions.append(action)

    return {
        "total_reward": float(total_reward),
        "mean_power": float(np.nanmean(power_vals)),
        "mean_comfort_dev": float(np.nanmean(comfort_vals)),
        "power_ts": power_vals,
        "comfort_ts": comfort_vals,
        "actions": actions
    }

# -----------------------
# Load model if exists
# -----------------------
trained_model = None
if os.path.exists(MODEL_PATH):
    try:
        trained_model = PPO.load(MODEL_PATH)
        print("Loaded trained RL model from:", MODEL_PATH)
    except Exception as e:
        print("Could not load RL model:", e)
else:
    print("Trained RL model not found at", MODEL_PATH, "- RL policy will be skipped.")

# -----------------------
# Run evaluations
# -----------------------
results = {"policy": [], "episode": [], "total_reward": [], "mean_power": [], "mean_comfort_dev": []}
timeseries_example = {}

policies = ["zero", "random"]
if trained_model is not None:
    policies.append("rl")

for policy in policies:
    print(f"\nEvaluating policy: {policy}")
    # use a fresh env each time
    env = HVACChillerEnv(episode_length_hours=EP_LENGTH)
    for ep in range(N_EPISODES):
        res = run_episode(env, policy=policy, model=trained_model)
        results["policy"].append(policy)
        results["episode"].append(ep)
        results["total_reward"].append(res["total_reward"])
        results["mean_power"].append(res["mean_power"])
        results["mean_comfort_dev"].append(res["mean_comfort_dev"])
        # store the first episode time series for plotting
        if ep == 0:
            timeseries_example[policy] = {
                "power": res["power_ts"],
                "comfort": res["comfort_ts"],
            }
    env.close()

# -----------------------
# Save summary CSV + JSON
# -----------------------
df_res = pd.DataFrame(results)
csv_path = os.path.join(MODEL_DIR, "compare_summary.csv")
df_res.to_csv(csv_path, index=False)
print("\nSaved comparison CSV:", csv_path)

json_path = os.path.join(MODEL_DIR, "compare_summary.json")
with open(json_path, "w") as f:
    json.dump({
        "n_episodes": N_EPISODES,
        "policies": policies,
        "aggregates": df_res.groupby("policy")[["total_reward","mean_power","mean_comfort_dev"]].mean().to_dict()
    }, f, indent=2)
print("Saved comparison JSON:", json_path)

# -----------------------
# Plot: aggregated bar chart
# -----------------------
agg = df_res.groupby("policy").mean().reset_index()
fig, axes = plt.subplots(1,3, figsize=(15,4))
axes[0].bar(agg["policy"], agg["mean_power"])
axes[0].set_title("Avg Predicted Power (kW)")
axes[0].set_ylabel("kW")

axes[1].bar(agg["policy"], agg["mean_comfort_dev"])
axes[1].set_title("Avg Comfort Deviation (°C)")
axes[1].set_ylabel("°C")

axes[2].bar(agg["policy"], agg["total_reward"])
axes[2].set_title("Avg Episode Reward")
axes[2].set_ylabel("Reward")

plt.tight_layout()
barplot_path = os.path.join(MODEL_DIR, "compare_barplot.png")
plt.savefig(barplot_path)
plt.close()
print("Saved barplot:", barplot_path)

# -----------------------
# Plot: timeseries for first episode of each policy
# -----------------------
fig, axes = plt.subplots(2,1, figsize=(10,6), sharex=True)
for policy, ts in timeseries_example.items():
    axes[0].plot(ts["power"], label=policy)
axes[0].set_ylabel("Predicted Power (kW)")
axes[0].legend()
axes[0].set_title("Power time series (first episode)")

for policy, ts in timeseries_example.items():
    axes[1].plot(ts["comfort"], label=policy)
axes[1].set_ylabel("Comfort deviation (°C)")
axes[1].legend()
axes[1].set_title("Comfort deviation time series (first episode)")
plt.xlabel("Timestep (hour)")
plt.tight_layout()
ts_path = os.path.join(MODEL_DIR, "compare_timeseries.png")
plt.savefig(ts_path)
plt.close()
print("Saved timeseries plot:", ts_path)

print("\nDone. Open the CSV and PNGs in", MODEL_DIR)
