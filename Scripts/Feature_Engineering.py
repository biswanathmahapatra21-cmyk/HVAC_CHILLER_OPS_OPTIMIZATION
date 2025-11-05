# scripts/fe_step2.py
"""
Physics-aware Feature Engineering for HVAC chiller dataset.
Now includes compressor & pump speed features (real or proxy).
"""

import os, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------
# Paths
# -------------------
BASE = os.path.dirname(os.path.dirname(__file__))   # project root
IN_PATH = os.path.join(BASE, "feature_engineering", "processed_hvac_chiller_dataset_extended.csv")
OUT_DIR = os.path.join(BASE, "feature_engineering", "fe_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading:", IN_PATH)
df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

LEAKAGE_COLS = ['COP', 'system_efficiency_kW_per_ton','inst_COP', 'kW_per_ton', 'energy_intensity']

# -------------------
# Time-based features
# -------------------
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)

# -------------------
# Air-side physics
# -------------------
if {"supply_air_temp_C","return_air_temp_C","air_flow_m3s"} <= set(df.columns):
    df["air_deltaT_C"] = df["return_air_temp_C"] - df["supply_air_temp_C"]
    df["coil_capacity_proxy_kW"] = df["air_deltaT_C"] * df["air_flow_m3s"] * 4186 / 1000
if "setpoint_supply_air_temp_C" in df.columns:
    df["air_setpoint_dev_C"] = df["supply_air_temp_C"] - df["setpoint_supply_air_temp_C"]

# -------------------
# Chilled-water side
# -------------------
if {"chilled_water_supply_temp_C","chilled_water_return_temp_C","evap_flow_m3s"} <= set(df.columns):
    df["chw_deltaT_C"] = df["chilled_water_return_temp_C"] - df["chilled_water_supply_temp_C"]
    df["evap_capacity_proxy_kW"] = df["chw_deltaT_C"] * df["evap_flow_m3s"] * 4186 / 1000
if "setpoint_chilled_water_supply_C" in df.columns:
    df["chw_setpoint_dev_C"] = df["chilled_water_supply_temp_C"] - df["setpoint_chilled_water_supply_C"]

# -------------------
# Condenser side
# -------------------
if {"condenser_water_supply_temp_C","condenser_water_return_temp_C","cond_flow_m3s"} <= set(df.columns):
    df["cond_deltaT_C"] = df["condenser_water_return_temp_C"] - df["condenser_water_supply_temp_C"]
    df["cond_capacity_proxy_kW"] = df["cond_deltaT_C"] * df["cond_flow_m3s"] * 4186 / 1000

# -------------------
# Pumps
# -------------------
if "evap_flow_m3s" in df.columns:
    df["evap_flow_cubed"] = df["evap_flow_m3s"]**3
if "cond_flow_m3s" in df.columns:
    df["cond_flow_cubed"] = df["cond_flow_m3s"]**3
if {"pump1_power_kW","evap_flow_m3s"} <= set(df.columns):
    df["pump1_eff_proxy"] = df["pump1_power_kW"] / (df["evap_flow_m3s"]+1e-6)
if {"pump2_power_kW","cond_flow_m3s"} <= set(df.columns):
    df["pump2_eff_proxy"] = df["pump2_power_kW"] / (df["cond_flow_m3s"]+1e-6)

# -------------------
# Chiller performance
# -------------------
if {"cooling_capacity_kW","total_power_kW"} <= set(df.columns):
    df["inst_COP"] = df["cooling_capacity_kW"] / (df["total_power_kW"]+1e-6)
if {"building_heat_load_kW","cooling_capacity_kW"} <= set(df.columns):
    df["PLR_proxy"] = df["building_heat_load_kW"] / (df["cooling_capacity_kW"]+1e-6)

# -------------------
# Energy KPIs
# -------------------
if {"total_power_kW","cooling_capacity_kW"} <= set(df.columns):
    df["kW_per_ton"] = df["total_power_kW"] / ((df["cooling_capacity_kW"]/3.517)+1e-6)
    df["energy_intensity"] = df["total_power_kW"] / (df["building_heat_load_kW"]+1e-6)

# -------------------
# Compressor & Pump Speeds (real or proxy)
# -------------------
NOMINAL_COMPRESSOR_SPEED_RPM = 3600.0
NOMINAL_CHILLER_POWER_KW = max(df.get("chiller_power_kW", pd.Series([500])).max(), 1.0)

NOMINAL_PUMP_SPEED_RPM = 1450.0
NOMINAL_PUMP1_POWER_KW = max(df.get("pump1_power_kW", pd.Series([30])).max(), 1.0)
NOMINAL_PUMP2_POWER_KW = max(df.get("pump2_power_kW", pd.Series([25])).max(), 1.0)

# Compressor
if "compressor_speed_rpm" in df.columns:
    df["compressor_speed_rpm"] = pd.to_numeric(df["compressor_speed_rpm"], errors="coerce").fillna(0)
else:
    if "chiller_power_kW" in df.columns:
        chp = df["chiller_power_kW"].clip(lower=0.0)
        frac = (chp / (NOMINAL_CHILLER_POWER_KW+1e-9)).clip(0, None)
        df["compressor_speed_proxy_rpm"] = NOMINAL_COMPRESSOR_SPEED_RPM * (frac ** (1/3))
        df["compressor_speed_proxy_norm"] = df["compressor_speed_proxy_rpm"] / NOMINAL_COMPRESSOR_SPEED_RPM

# Pump1
if "pump1_speed_rpm" in df.columns:
    df["pump1_speed_rpm"] = pd.to_numeric(df["pump1_speed_rpm"], errors="coerce").fillna(0)
else:
    if "pump1_power_kW" in df.columns:
        p1 = df["pump1_power_kW"].clip(lower=0.0)
        frac = (p1 / (NOMINAL_PUMP1_POWER_KW+1e-9)).clip(0, None)
        df["pump1_speed_proxy_rpm"] = NOMINAL_PUMP_SPEED_RPM * (frac ** (1/3))
        df["pump1_speed_proxy_norm"] = df["pump1_speed_proxy_rpm"] / NOMINAL_PUMP_SPEED_RPM

# Pump2
if "pump2_speed_rpm" in df.columns:
    df["pump2_speed_rpm"] = pd.to_numeric(df["pump2_speed_rpm"], errors="coerce").fillna(0)
else:
    if "pump2_power_kW" in df.columns:
        p2 = df["pump2_power_kW"].clip(lower=0.0)
        frac = (p2 / (NOMINAL_PUMP2_POWER_KW+1e-9)).clip(0, None)
        df["pump2_speed_proxy_rpm"] = NOMINAL_PUMP_SPEED_RPM * (frac ** (1/3))
        df["pump2_speed_proxy_norm"] = df["pump2_speed_proxy_rpm"] / NOMINAL_PUMP_SPEED_RPM

# Cubed features for affinity laws
if "compressor_speed_proxy_rpm" in df.columns:
    df["compressor_speed_cubed"] = df["compressor_speed_proxy_rpm"]**3
if "pump1_speed_proxy_rpm" in df.columns:
    df["pump1_speed_cubed"] = df["pump1_speed_proxy_rpm"]**3
if "pump2_speed_proxy_rpm" in df.columns:
    df["pump2_speed_cubed"] = df["pump2_speed_proxy_rpm"]**3

# -------------------
# Lag and rolling
# -------------------
for v in ["building_heat_load_kW","total_power_kW","ambient_temp_C"]:
    if v in df.columns:
        for l in [1,3,6,24]:
            df[f"{v}_lag{l}h"] = df[v].shift(l)
        for w in [6,24]:
            df[f"{v}_roll_mean_{w}h"] = df[v].rolling(w, min_periods=1).mean()

# -------------------
# Daily profile clustering
# -------------------
if "total_power_kW" in df.columns:
    daily = df.pivot_table(index=df["timestamp"].dt.date, columns=df["hour"], values="total_power_kW")
    daily_norm = daily.div(daily.mean(axis=1), axis=0).fillna(1.0)
    daily_pca = PCA(n_components=4).fit_transform(daily_norm.fillna(0).values)
    km = KMeans(n_clusters=4, random_state=42).fit(daily_pca)
    daily_labels = pd.Series(km.labels_, index=daily.index, name="daily_cluster")
    df["daily_cluster"] = df["timestamp"].dt.date.map(daily_labels)

# -------------------
# Final cleaning
# -------------------
df_final = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

# -------------------
# Train/test split
# -------------------
TARGET = "total_power_kW"
n = len(df_final)
split_idx = int(0.8 * n)
train_df = df_final.iloc[:split_idx].reset_index(drop=True)
test_df = df_final.iloc[split_idx:].reset_index(drop=True)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[[TARGET]]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[[TARGET]]

# -------------------
# Save outputs
# -------------------
df_final.to_csv(os.path.join(OUT_DIR, "processed_features.csv"), index=False)
X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

#with open(os.path.join(OUT_DIR, "features_list.txt"), "w") as f:
    #f.write("\n".join(X_train.columns.tolist()))

#numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [c for c in numeric_features if c not in LEAKAGE_COLS]


with open(os.path.join(OUT_DIR, "features_list.txt"), "w") as f:
    f.write("\n".join(numeric_features))

print("✅ Saved features_list.txt with", len(numeric_features), "features")


report = {
    "n_rows_total": int(n),
    "n_features": len(X_train.columns),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "physics_features_included": [
        "air_deltaT_C","coil_capacity_proxy_kW","chw_deltaT_C","evap_capacity_proxy_kW",
        "cond_deltaT_C","cond_capacity_proxy_kW","inst_COP","PLR_proxy","kW_per_ton",
        "compressor_speed_proxy_rpm","pump1_speed_proxy_rpm","pump2_speed_proxy_rpm"
    ]
}
with open(os.path.join(OUT_DIR, "fe_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("✅ Feature engineering complete with compressor/pump speeds.")
print("Processed features saved in:", OUT_DIR)

