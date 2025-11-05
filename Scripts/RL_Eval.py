import os
import numpy as np
from stable_baselines3 import PPO
from RL import HVACChillerEnv

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "artifacts", "rl")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hvac_chiller.zip")

# Load trained agent
model = PPO.load(MODEL_PATH)

# Make eval env
env = HVACChillerEnv(episode_length_hours=24)

n_episodes = 10
episode_rewards = []
episode_power = []
episode_comfort = []

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    power_vals = []
    comfort_vals = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        power_vals.append(info["predicted_total_power_kW"])
        comfort_vals.append(info["comfort_deviation_C"])

    episode_rewards.append(total_reward)
    episode_power.append(np.mean(power_vals))
    episode_comfort.append(np.mean(comfort_vals))

print("==== Evaluation Results ====")
print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
print(f"Average predicted power: {np.mean(episode_power):.2f} kW")
print(f"Average comfort deviation: {np.mean(episode_comfort):.2f} °C")
