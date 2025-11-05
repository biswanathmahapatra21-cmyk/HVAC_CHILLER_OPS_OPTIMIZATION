# scripts/eda_step1.py
"""
Exhaustive EDA for HVAC chiller dataset (extended version).
- Loads hvac_chiller_dataset_extended.csv
- Produces plots, tables, JSON summaries
- Saves cleaned dataset for ML pipeline
Run:
    python scripts/eda_step1.py
"""

import os, json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# -------------------
# Paths
BASE = os.path.dirname(os.path.dirname(__file__))

# Input dataset (raw)
DATA_PATH = os.path.join(BASE, "Data", "hvac_chiller_dataset_extended.csv")

# EDA outputs
OUT_DIR = os.path.join(BASE, "artifacts", "eda")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
TABLES_DIR = os.path.join(OUT_DIR, "tables")

# Cleaned dataset (for feature engineering)
CLEAN_PATH = os.path.join(BASE, "feature_engineering", "processed_hvac_chiller_dataset_extended.csv")
# -------------------


os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH,parse_dates=["timestamp"])
print("Shape:", df.shape)

# -------------------
# Ensure time features
# -------------------
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["dayofyear"] = df["timestamp"].dt.dayofyear
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

# -------------------
# Numeric / categorical
# -------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------
# Missing + duplicates
# -------------------
missing = df.isnull().sum().sort_values(ascending=False)
missing.to_csv(os.path.join(TABLES_DIR, "missing_values.csv"))
dup_count = int(df.duplicated().sum())

# -------------------
# Descriptive stats
# -------------------
desc = df[num_cols].describe().T
desc["skew"] = df[num_cols].skew().values
desc["kurtosis"] = df[num_cols].kurtosis().values
desc.to_csv(os.path.join(TABLES_DIR, "descriptive_stats.csv"))

# -------------------
# Distributions + boxplots
# -------------------
for col in num_cols:
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(df[col].dropna(), bins=60, kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution: {col}")
    sns.boxplot(x=df[col].dropna(), ax=axes[1])
    axes[1].set_title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"dist_box_{col}.png"))
    plt.close(fig)

# -------------------
# Time-series plots
# -------------------
for var in ["building_heat_load_kW","total_power_kW","chiller_power_kW","COP"]:
    if var in df.columns:
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(df["timestamp"], df[var], linewidth=0.6)
        ax.set_title(f"Time series: {var} (full year)")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"ts_full_{var}.png")); plt.close(fig)

# -------------------
# Hourly & monthly profiles
# -------------------
if "total_power_kW" in df.columns:
    hourly = df.groupby("hour")["total_power_kW"].agg(["mean","std"]).reset_index()
    hourly.to_csv(os.path.join(TABLES_DIR, "hourly_total_power_stats.csv"), index=False)
    x, y, s = hourly["hour"], hourly["mean"], hourly["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y-s, y+s, alpha=0.2)
    ax.set_title("Average hourly total_power_kW (+/-1 std)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "hourly_profile_total_power.png")); plt.close(fig)

    monthly = df.groupby("month")["total_power_kW"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(monthly["month"], monthly["total_power_kW"], marker="o")
    ax.set_title("Monthly mean total_power_kW")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "monthly_mean_total_power.png")); plt.close(fig)

# -------------------
# Correlations
# -------------------
corr = df[num_cols].corr()
corr.to_csv(os.path.join(TABLES_DIR, "correlation_matrix.csv"))
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png")); plt.close(fig)

target = "total_power_kW"
top_corr = corr[target].abs().sort_values(ascending=False)
top_corr.to_csv(os.path.join(TABLES_DIR, "top_correlations_with_target.csv"))

# -------------------
# PCA
# -------------------
evr = []
if len(num_cols) >= 3:
    Xs = StandardScaler().fit_transform(df[num_cols].fillna(0))
    pca = PCA(n_components=6).fit(Xs)
    evr = pca.explained_variance_ratio_
    with open(os.path.join(TABLES_DIR, "pca_evr.json"), "w") as f:
        json.dump({"explained_variance_ratio": evr.tolist()}, f, indent=2)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(range(1,len(evr)+1), evr, alpha=0.7)
    ax.step(range(1,len(evr)+1), np.cumsum(evr), where="mid", color="red")
    ax.set_title("PCA Scree Plot")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "pca_scree.png")); plt.close(fig)

# -------------------
# RandomForest importance
# -------------------
feats = [c for c in num_cols if c != target]
sample_df = df.dropna(subset=feats+[target]).sample(min(5000, len(df)), random_state=42)
X = sample_df[feats]; y = sample_df[target]
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X,y)
importances = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
importances.to_csv(os.path.join(TABLES_DIR, "rf_feature_importances.csv"))
fig, ax = plt.subplots(figsize=(8,6))
importances.head(20).plot(kind="barh", ax=ax); ax.invert_yaxis()
ax.set_title("RandomForest importances")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "rf_feature_importances.png")); plt.close(fig)

# -------------------
# Comfort violations
# -------------------
if "comfort_violation_flag" in df.columns:
    cv_counts = df["comfort_violation_flag"].value_counts().to_dict()
    with open(os.path.join(TABLES_DIR, "comfort_violation_summary.json"), "w") as f:
        json.dump(cv_counts, f, indent=2)

# -------------------
# Daily profile clustering
# -------------------
if "total_power_kW" in df.columns:
    daily = df.pivot_table(index=df["timestamp"].dt.date, columns=df["hour"], values="total_power_kW")
    daily_norm = daily.div(daily.mean(axis=1), axis=0).fillna(1.0)
    daily_pca = PCA(n_components=4).fit_transform(daily_norm.fillna(0).values)
    km = KMeans(n_clusters=4, random_state=42).fit(daily_pca)
    daily_labels = pd.Series(km.labels_, index=daily.index, name="daily_cluster")
    df = df.merge(daily_labels.rename("daily_cluster"), left_on=df["timestamp"].dt.date, right_index=True, how="left")
    cluster_means = daily_norm.groupby(daily_labels).mean()
    cluster_means.to_csv(os.path.join(TABLES_DIR, "daily_cluster_profiles.csv"))
    fig, ax = plt.subplots(figsize=(10,4))
    for c in cluster_means.index:
        ax.plot(cluster_means.columns, cluster_means.loc[c], label=f"cluster {c}")
    ax.set_title("Daily load shape clusters")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "daily_clusters.png")); plt.close(fig)

# -------------------
# HVAC domain-specific EDA
# -------------------
print("Running HVAC-specific EDA...")

# 1. Air-side: Supply vs Return Temp
if {"supply_air_temp_C","return_air_temp_C"} <= set(df.columns):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="return_air_temp_C", y="supply_air_temp_C", 
                    data=df.sample(min(2000,len(df)), random_state=1), alpha=0.5)
    plt.title("Supply vs Return Air Temperature")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "air_supply_vs_return.png")); plt.close()

# 2. Water-side: Chilled Water ΔT vs Load
if {"chilled_water_deltaT_C","building_heat_load_kW"} <= set(df.columns):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="chilled_water_deltaT_C", y="building_heat_load_kW", 
                    data=df.sample(min(2000,len(df)), random_state=2), alpha=0.5)
    plt.title("CHW ΔT vs Building Heat Load")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "chw_deltaT_vs_load.png")); plt.close()

# 3. Condenser Supply Temp vs COP
if {"condenser_water_supply_temp_C","COP"} <= set(df.columns):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="condenser_water_supply_temp_C", y="COP", 
                    data=df.sample(min(2000,len(df)), random_state=3), alpha=0.5)
    plt.title("Condenser Water Supply Temp vs COP")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "condenser_supply_vs_COP.png")); plt.close()

# 4. Pump performance: Pump Power vs Flow
for flow_col, pump_col in [("evap_flow_m3s","pump1_power_kW"),("cond_flow_m3s","pump2_power_kW")]:
    if {flow_col,pump_col} <= set(df.columns):
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=flow_col, y=pump_col, 
                        data=df.sample(min(2000,len(df)), random_state=4), alpha=0.5)
        plt.title(f"{pump_col} vs {flow_col}")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{pump_col}_vs_{flow_col}.png")); plt.close()

# 5. Comfort: Supply Air Temp vs Setpoint
if {"supply_air_temp_C","setpoint_supply_air_temp_C"} <= set(df.columns):
    df["supply_minus_setpoint"] = df["supply_air_temp_C"] - df["setpoint_supply_air_temp_C"]
    plt.figure(figsize=(6,4))
    sns.histplot(df["supply_minus_setpoint"], bins=50, kde=True)
    plt.axvline(0, color="r", linestyle="--")
    plt.title("Supply Air Temp - Setpoint Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "supply_minus_setpoint_dist.png")); plt.close()
    df["supply_minus_setpoint"].describe().to_csv(os.path.join(TABLES_DIR, "supply_minus_setpoint_stats.csv"))

# 6. System Efficiency vs Load
if {"system_efficiency_kW_per_ton","building_heat_load_kW"} <= set(df.columns):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="building_heat_load_kW", y="system_efficiency_kW_per_ton", 
                    data=df.sample(min(2000,len(df)), random_state=5), alpha=0.5)
    plt.title("System Efficiency (kW/ton) vs Heat Load")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "efficiency_vs_load.png")); plt.close()

# -------------------
# Save cleaned dataset
# -------------------
df_clean = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
df_clean.to_csv(CLEAN_PATH, index=False)

# -------------------
# Save summary JSON
# -------------------
summary = {
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "duplicates": dup_count,
    "top_correlations": top_corr.head(10).to_dict(),
    "pca_first3": evr[:3].tolist() if len(evr) else [],
    "rf_top_features": importances.head(10).to_dict(),
}
with open(os.path.join(OUT_DIR, "eda_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("EDA complete.")
print("Plots saved to:", PLOTS_DIR)
print("Tables saved to:", TABLES_DIR)
print("Cleaned dataset saved to:", CLEAN_PATH)
