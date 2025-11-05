# scripts/rl_env.py
"""
Gymnasium environment for HVAC chiller control using a surrogate RandomForest model.

Action space (continuous, each in [-1,1]):
    0: compressor_speed_frac_delta
    1: pump1_speed_frac_delta
    2: pump2_speed_frac_delta
    3: chw_setpoint_delta_C  (scaled +/-2 °C)
    4: supply_air_setpoint_delta_C (scaled +/-2 °C)

Observation:
    Numeric feature vector from engineered dataset.

Reward:
    - total_power_kW (from surrogate RF)
    - comfort penalty if supply_air_temp deviates from setpoint > deadband
"""

import os, joblib, numpy as np, pandas as pd
import gymnasium as gym
from gymnasium import spaces

BASE = os.path.dirname(os.path.dirname(__file__))
FE_OUT = os.path.join(BASE, "feature_engineering", "fe_outputs")
TUNED_MODEL_PATH = os.path.join(BASE, "artifacts", "tuned_models", "best_random_forest.joblib")


class HVACChillerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 dataset_csv=os.path.join(FE_OUT, "processed_features.csv"),
                 surrogate_model_path=TUNED_MODEL_PATH,
                 episode_length_hours=24,
                 chw_sp_range=2.0,
                 sa_sp_range=2.0,
                 max_speed_change_frac=0.15,
                 comfort_deadband=1.0,
                 reward_power_weight=1.0,
                 reward_comfort_weight=5.0,
                 seed=42):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.df = pd.read_csv(dataset_csv, parse_dates=["timestamp"])
        self.surrogate = joblib.load(surrogate_model_path)

        # Features expected by surrogate
        self.feature_cols = list(pd.read_csv(os.path.join(FE_OUT, "features_list.txt"),
                                             header=None).squeeze())

        # Action space: 5 continuous actions in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Observation space: min/max from dataset
        # Observation space: min/max from dataset
        obs_min, obs_max = [], []
        for c in self.feature_cols:
           if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c]):
               obs_min.append(float(self.df[c].min()))
               obs_max.append(float(self.df[c].max()))
           else:
                  # fallback for non-numeric or missing
                   obs_min.append(-1e6)
                   obs_max.append(1e6)

        self.observation_space = spaces.Box(low=np.array(obs_min, dtype=np.float32),
                                    high=np.array(obs_max, dtype=np.float32),
                                    dtype=np.float32)


        # Params
        self.episode_length = episode_length_hours
        self.chw_sp_range = chw_sp_range
        self.sa_sp_range = sa_sp_range
        self.max_speed_change_frac = max_speed_change_frac
        self.comfort_deadband = comfort_deadband
        self.reward_power_weight = reward_power_weight
        self.reward_comfort_weight = reward_comfort_weight

        self.current_step = 0
        self.state_series = None
        self._last_pred_power = None

    def sample_initial_state(self):
        return self.df.sample(n=1, random_state=self.rng.randint(0, 1_000_000)).iloc[0]

    #def _build_obs_from_series(self, s: pd.Series):
        #return np.array([s.get(c, 0.0) for c in self.feature_cols], dtype=np.float32)
    
    
    def _build_obs_from_series(self, s: pd.Series):
         vals = []
         for c in self.feature_cols:
             if c in s.index and pd.api.types.is_numeric_dtype(type(s[c])):
               try:
                   vals.append(float(s[c]))
               except Exception:
                   vals.append(0.0)
             else:
                vals.append(0.0)
         return np.array(vals, dtype=np.float32)

    

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state_series = self.sample_initial_state().fillna(0)
        obs = self._build_obs_from_series(self.state_series)
        self._last_pred_power = float(self.surrogate.predict(obs.reshape(1, -1))[0])
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        comp_delta, p1_delta, p2_delta, chw_sp_delta_norm, sa_sp_delta_norm = action

        # Current normalized speeds
        comp_norm = float(self.state_series.get("compressor_speed_proxy_norm", 1.0))
        p1_norm = float(self.state_series.get("pump1_speed_proxy_norm", 1.0))
        p2_norm = float(self.state_series.get("pump2_speed_proxy_norm", 1.0))

        # Apply deltas
        comp_norm_new = np.clip(comp_norm + comp_delta * self.max_speed_change_frac, 0.0, 2.0)
        p1_norm_new = np.clip(p1_norm + p1_delta * self.max_speed_change_frac, 0.0, 2.0)
        p2_norm_new = np.clip(p2_norm + p2_delta * self.max_speed_change_frac, 0.0, 2.0)

        self.state_series["compressor_speed_proxy_norm"] = comp_norm_new
        self.state_series["pump1_speed_proxy_norm"] = p1_norm_new
        self.state_series["pump2_speed_proxy_norm"] = p2_norm_new

        # Setpoints
        self.state_series["setpoint_chilled_water_supply_C"] = \
            float(self.state_series.get("setpoint_chilled_water_supply_C", 6.0)) + chw_sp_delta_norm * self.chw_sp_range
        self.state_series["setpoint_supply_air_temp_C"] = \
            float(self.state_series.get("setpoint_supply_air_temp_C", 14.0)) + sa_sp_delta_norm * self.sa_sp_range

        # Surrogate prediction
        obs_vec = self._build_obs_from_series(self.state_series)
        predicted_power = float(self.surrogate.predict(obs_vec.reshape(1, -1))[0])
        self._last_pred_power = predicted_power

        # Comfort penalty
        supply = float(self.state_series.get("supply_air_temp_C", 14.0))
        sp = float(self.state_series.get("setpoint_supply_air_temp_C", 14.0))
        dev = abs(supply - sp)
        comfort_pen = max(0.0, dev - self.comfort_deadband)

        # Reward
        reward = - self.reward_power_weight * predicted_power - self.reward_comfort_weight * comfort_pen

        # Episode termination
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        obs = self._build_obs_from_series(self.state_series)
        info = {"predicted_total_power_kW": predicted_power,
                "comfort_deviation_C": dev,
                "comfort_penalty": comfort_pen}

        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(f"step {self.current_step} | power {self._last_pred_power:.2f} kW")

if __name__ == "__main__":
    env = HVACChillerEnv()
    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, power={info['predicted_total_power_kW']:.2f} kW")
        if terminated or truncated:
            obs, info = env.reset()