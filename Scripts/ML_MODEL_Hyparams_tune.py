# scripts/tune_models.py
"""
Hyperparameter tuning for RandomForest and XGBoost on HVAC dataset
"""

import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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
MODEL_DIR = os.path.join(BASE, "artifacts", "tuned_models")
os.makedirs(MODEL_DIR, exist_ok=True)

LEAKAGE_COLS = ['COP', 'system_efficiency_kW_per_ton',
                'inst_COP', 'kW_per_ton', 'energy_intensity']

# -------------------
# Load data
# -------------------
X_train = pd.read_csv(os.path.join(FE_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(FE_DIR, "y_train.csv"))
X_test = pd.read_csv(os.path.join(FE_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(FE_DIR, "y_test.csv"))

# Drop non-numeric (timestamp etc.)
non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
if len(non_numeric) > 0:
    print("Dropping non-numeric cols:", list(non_numeric))
    X_train = X_train.drop(columns=non_numeric)
    X_test = X_test.drop(columns=non_numeric)

# Drop leakage features
leakage_cols = [c for c in X_train.columns if "total_power_kW" in c.lower()
                or "cop" in c.lower()
                or "kw_per_ton" in c.lower()
                or "energy_intensity" in c.lower()]
if leakage_cols:
    print("Dropping leakage cols:", leakage_cols)
    X_train = X_train.drop(columns=leakage_cols)
    X_test = X_test.drop(columns=leakage_cols)

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

# -------------------
# Random Forest tuning
# -------------------
print("\n🔍 Tuning Random Forest...")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rf_param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_grid,
    n_iter=15,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=42
)

rf_search.fit(X_train, y_train.values.ravel())
best_rf = rf_search.best_estimator_
results["RandomForest"] = {
    "best_params": rf_search.best_params_,
    "cv_score": -rf_search.best_score_,
    "test_score": evaluate(best_rf, X_test, y_test)
}

# Save best RF model
import joblib
joblib.dump(best_rf, os.path.join(MODEL_DIR, "best_random_forest.joblib"))

# -------------------
# XGBoost tuning
# -------------------
if HAS_XGB:
    print("\n🔍 Tuning XGBoost...")
    xgb = XGBRegressor(random_state=42, n_jobs=-1)

    xgb_param_grid = {
        "n_estimators": [200, 500, 800],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_param_grid,
        n_iter=15,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    xgb_search.fit(X_train, y_train.values.ravel())
    best_xgb = xgb_search.best_estimator_
    results["XGBoost"] = {
        "best_params": xgb_search.best_params_,
        "cv_score": -xgb_search.best_score_,
        "test_score": evaluate(best_xgb, X_test, y_test)
    }

    # Save best XGB model
    joblib.dump(best_xgb, os.path.join(MODEL_DIR, "best_xgboost.joblib"))

# -------------------
# Save results
# -------------------
with open(os.path.join(MODEL_DIR, "tuning_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Tuning complete. Results:")
print(json.dumps(results, indent=2))

# Pick best model
best_model = min(results.items(), key=lambda kv: kv[1]["test_score"]["RMSE"])
print(f"\n🏆 Best model: {best_model[0]} (RMSE={best_model[1]['test_score']['RMSE']:.4f})")


feat_summary = {}

# -------------------
# Random Forest Feature Importance
# -------------------
if "RandomForest" in results:
    importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
    top_feats = importances.sort_values(ascending=False).head(20)

    # Save plot
    plt.figure(figsize=(8,6))
    sns.barplot(x=top_feats.values, y=top_feats.index)
    plt.title("Random Forest Feature Importances (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "rf_feature_importances.png"))
    plt.close()

    # Save top 10 as JSON/CSV
    feat_summary["RandomForest"] = top_feats.head(10).to_dict()
    top_feats.head(10).to_csv(os.path.join(MODEL_DIR, "rf_top10_features.csv"))

# -------------------
# XGBoost Feature Importance
# -------------------
if "XGBoost" in results:
    importances = pd.Series(best_xgb.feature_importances_, index=X_train.columns)
    top_feats = importances.sort_values(ascending=False).head(20)

    # Save plot
    plt.figure(figsize=(8,6))
    sns.barplot(x=top_feats.values, y=top_feats.index)
    plt.title("XGBoost Feature Importances (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "xgb_feature_importances.png"))
    plt.close()

    # Save top 10 as JSON/CSV
    feat_summary["XGBoost"] = top_feats.head(10).to_dict()
    top_feats.head(10).to_csv(os.path.join(MODEL_DIR, "xgb_top10_features.csv"))

# -------------------
# Save feature summary JSON
# -------------------
with open(os.path.join(MODEL_DIR, "top_features_summary.json"), "w") as f:
    json.dump(feat_summary, f, indent=2)

print("\n📊 Top 10 features saved for each model in artifacts/tuned_models/")
