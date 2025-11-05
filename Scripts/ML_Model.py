# scripts/train_ml.py
"""
Train ML models (Linear Regression, Random Forest, XGBoost)
on HVAC chiller engineered dataset.
"""

import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed. Run `pip install xgboost` to enable it.")

# -------------------
# Paths
# -------------------
BASE = os.path.dirname(os.path.dirname(__file__))
FE_DIR = os.path.join(BASE, "feature_engineering", "fe_outputs")
MODEL_DIR = os.path.join(BASE, "artifacts", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------
# Load data
# -------------------
X_train = pd.read_csv(os.path.join(FE_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(FE_DIR, "y_train.csv"))
X_test = pd.read_csv(os.path.join(FE_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(FE_DIR, "y_test.csv"))

non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
if len(non_numeric) > 0:
    print("Dropping non-numeric columns:", list(non_numeric))
    X_train = X_train.drop(columns=non_numeric)
    X_test = X_test.drop(columns=non_numeric)

leakage_cols = [c for c in X_train.columns if "total_power_kW" in c.lower()
                or "cop" in c.lower()
                or "kw_per_ton" in c.lower()
                or "energy_intensity" in c.lower()]


target = "total_power_kW"

print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# -------------------
# Evaluation helper
# -------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

results = {}
best_model = None
best_score = float("inf")

# -------------------
# 1. Linear Regression
# -------------------
linreg = LinearRegression()
linreg.fit(X_train, y_train)
results["LinearRegression"] = evaluate(linreg, X_test, y_test)
joblib.dump(linreg, os.path.join(MODEL_DIR, "linear_regression.joblib"))

# -------------------
# 2. Random Forest
# -------------------
rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train.values.ravel())
results["RandomForest"] = evaluate(rf, X_test, y_test)
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))

# -------------------
# 3. XGBoost (if available)
# -------------------
if HAS_XGB:
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train.values.ravel())
    results["XGBoost"] = evaluate(xgb, X_test, y_test)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.joblib"))

# -------------------
# Save results
# -------------------
with open(os.path.join(MODEL_DIR, "model_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Print results
print("\n✅ Training complete. Results:")
for model, metrics in results.items():
    print(f"\n{model}:")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}")
    if metrics["RMSE"] < best_score:
        best_score = metrics["RMSE"]
        best_model = model

print(f"\n🏆 Best model: {best_model} (RMSE={best_score:.4f})")
