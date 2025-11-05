# scripts/train_rl.py
"""
Train RL agent (PPO) on the HVACChillerEnv surrogate environment.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from RL import HVACChillerEnv

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "artifacts", "rl")
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env():
    return HVACChillerEnv(episode_length_hours=24,
                          chw_sp_range=2.0,
                          sa_sp_range=2.0,
                          max_speed_change_frac=0.12,
                          comfort_deadband=1.0,
                          reward_power_weight=1.0,
                          reward_comfort_weight=50.0)

env = DummyVecEnv([make_env])

# callbacks
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=MODEL_DIR, name_prefix="rl_chkpt")
# optional evaluation callback - requires environment for eval
eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,log_path=MODEL_DIR, eval_freq=5000,deterministic=True, render=False)

model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4)

# Train
TIMESTEPS = 200_000  # adjust as you like
model.learn(total_timesteps=TIMESTEPS, callback=[checkpoint_callback, eval_callback])

# Save final model
model.save(os.path.join(MODEL_DIR, "ppo_hvac_chiller"))

print("RL training complete. Models saved to", MODEL_DIR)
