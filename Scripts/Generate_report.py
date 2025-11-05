# scripts/generate_report.py
"""
Visualization & Reporting for HVAC RL Agent
Generates plots + HTML report (convertible to PDF).
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

# -------------------------------
# Load trained agent
# -------------------------------
model = PPO.load(MODEL_PATH)
env = HVACChillerEnv(episode_length_hours=24)

n_eval_episodes = 20
all_rewards, all_power, all_comfort = [], [], []
power_vs_setpoints = None

for ep in range(n_eval_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    power_ts, comfort_ts, sa_setpoints, chw_setpoints = [], [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        power_ts.append(info["predicted_total_power_kW"])
        comfort_ts.append(info["comfort_deviation_C"])
        sa_setpoints.append(env.state_series.get("setpoint_supply_air_temp_C", np.nan))
        chw_setpoints.append(env.state_series.get("setpoint_chilled_water_supply_C", np.nan))

    all_rewards.append(total_reward)
    all_power.extend(power_ts)
    all_comfort.extend(comfort_ts)

    if ep == 0:  # store first episode for plotting
        power_vs_setpoints = pd.DataFrame({
            "t": np.arange(len(power_ts)),
            "power": power_ts,
            "sa_setpoint": sa_setpoints,
            "chw_setpoint": chw_setpoints,
            "comfort": comfort_ts
        })

# -------------------------------
# Plot 1: Episode reward trend
# -------------------------------
plt.figure(figsize=(6,4))
plt.plot(all_rewards, marker="o")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Episode reward trend (evaluation)")
reward_plot = os.path.join(REPORT_DIR, "reward_trend.png")
plt.tight_layout(); plt.savefig(reward_plot); plt.close()

# -------------------------------
# Plot 2: Power vs setpoints
# -------------------------------
plt.figure(figsize=(8,4))
plt.plot(power_vs_setpoints["t"], power_vs_setpoints["power"], label="Power (kW)")
plt.plot(power_vs_setpoints["t"], power_vs_setpoints["sa_setpoint"], label="SA setpoint (°C)")
plt.plot(power_vs_setpoints["t"], power_vs_setpoints["chw_setpoint"], label="CHW setpoint (°C)")
plt.xlabel("Timestep (hour)")
plt.title("Power vs Setpoints (Episode 1)")
plt.legend()
pv_plot = os.path.join(REPORT_DIR, "power_vs_setpoints.png")
plt.tight_layout(); plt.savefig(pv_plot); plt.close()

# -------------------------------
# Plot 3: Comfort deviation histogram
# -------------------------------
plt.figure(figsize=(6,4))
plt.hist(all_comfort, bins=20, edgecolor="black")
plt.xlabel("Comfort deviation (°C)")
plt.ylabel("Count")
plt.title("Comfort deviation histogram")
cv_plot = os.path.join(REPORT_DIR, "comfort_hist.png")
plt.tight_layout(); plt.savefig(cv_plot); plt.close()

# -------------------------------
# Generate HTML report
# -------------------------------
html_template = """
<html>
<head><title>HVAC RL Report</title></head>
<body>
<h1>HVAC RL Optimization Report</h1>
<p><b>Episodes evaluated:</b> {{ n_eval }}</p>
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
    n_eval=n_eval_episodes,
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
print("Open it in your browser. For PDF: open in Chrome → Print → Save as PDF")
