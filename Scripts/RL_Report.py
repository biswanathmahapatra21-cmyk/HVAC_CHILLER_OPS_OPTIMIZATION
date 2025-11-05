# scripts/generate_report.py
"""
Generate RL performance report (HTML + optional PDF).
Includes:
- Episode reward trend
- Power vs setpoints
- Comfort deviation histogram
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from RL import HVACChillerEnv
from jinja2 import Template

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "artifacts", "rl")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hvac_chiller.zip")
REPORT_DIR = os.path.join(MODEL_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ----------------------------
# Run evaluation
# ----------------------------
env = HVACChillerEnv(episode_length_hours=24)
model = PPO.load(MODEL_PATH)

n_episodes = 20
all_rewards, all_power, all_comfort = [], [], []
power_vs_setpoints = []

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    power_ts, comfort_ts, setpoint_sa, setpoint_chw = [], [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        power_ts.append(info["predicted_total_power_kW"])
        comfort_ts.append(info["comfort_deviation_C"])
        setpoint_sa.append(env.state_series.get("setpoint_supply_air_temp_C", np.nan))
        setpoint_chw.append(env.state_series.get("setpoint_chilled_water_supply_C", np.nan))
    all_rewards.append(total_reward)
    all_power.extend(power_ts)
    all_comfort.extend(comfort_ts)
    if ep == 0:  # store one episode's power vs setpoints
        power_vs_setpoints = pd.DataFrame({
            "t": list(range(len(power_ts))),
            "power": power_ts,
            "sa_setpoint": setpoint_sa,
            "chw_setpoint": setpoint_chw
        })

# ----------------------------
# Plots
# ----------------------------

# 1. Episode reward trend
plt.figure(figsize=(6,4))
plt.plot(all_rewards, marker="o")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Episode reward trend")
reward_plot = os.path.join(REPORT_DIR, "reward_trend.png")
plt.tight_layout(); plt.savefig(reward_plot); plt.close()

# 2. Power vs setpoints (first episode)
plt.figure(figsize=(8,4))
plt.plot(power_vs_setpoints["t"], power_vs_setpoints["power"], label="Power (kW)")
plt.plot(power_vs_setpoints["t"], power_vs_setpoints["sa_setpoint"], label="SA setpoint (°C)")
plt.plot(power_vs_setpoints["t"], power_vs_setpoints["chw_setpoint"], label="CHW setpoint (°C)")
plt.xlabel("Timestep (hour)")
plt.legend()
plt.title("Power vs setpoints (1st episode)")
pv_plot = os.path.join(REPORT_DIR, "power_vs_setpoints.png")
plt.tight_layout(); plt.savefig(pv_plot); plt.close()

# 3. Comfort deviation histogram
plt.figure(figsize=(6,4))
plt.hist(all_comfort, bins=20, edgecolor="black")
plt.xlabel("Comfort deviation (°C)")
plt.ylabel("Count")
plt.title("Comfort deviation histogram")
cv_plot = os.path.join(REPORT_DIR, "comfort_hist.png")
plt.tight_layout(); plt.savefig(cv_plot); plt.close()

# ----------------------------
# HTML report with Jinja2
# ----------------------------
html_template = """
<html>
<head><title>HVAC RL Report</title></head>
<body>
<h1>HVAC RL Optimization Report</h1>
<p><b>Episodes evaluated:</b> {{ n_episodes }}</p>
<p><b>Average reward:</b> {{ avg_reward|round(2) }}</p>
<p><b>Average power (kW):</b> {{ avg_power|round(2) }}</p>
<p><b>Average comfort deviation (°C):</b> {{ avg_comfort|round(2) }}</p>

<h2>Episode Reward Trend</h2>
<img src="{{ reward_plot }}" width="600">

<h2>Power vs Setpoints (Episode 1)</h2>
<img src="{{ pv_plot }}" width="700">

<h2>Comfort Deviation Histogram</h2>
<img src="{{ cv_plot }}" width="600">

</body>
</html>
"""

template = Template(html_template)
html_content = template.render(
    n_episodes=n_episodes,
    avg_reward=np.mean(all_rewards),
    avg_power=np.mean(all_power),
    avg_comfort=np.mean(all_comfort),
    reward_plot=os.path.basename(reward_plot),
    pv_plot=os.path.basename(pv_plot),
    cv_plot=os.path.basename(cv_plot),
)

html_path = os.path.join(REPORT_DIR, "rl_report.html")
with open(html_path, "w") as f:
    f.write(html_content)

print("✅ Report generated:", html_path)
print("Open it in your browser. For PDF, you can use Chrome 'Print to PDF'.")
