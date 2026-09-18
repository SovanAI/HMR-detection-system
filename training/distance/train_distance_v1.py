import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.expanduser("~/bas-hmr")

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset_collector",
    "data_v2",
    "distance_dataset_v2.csv",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "distance",
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "distance",
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    f"feature_{i:02d}"
    for i in range(1, 35)
]

TARGET_COLUMN = "distance_cm"


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BAS-HMR — SUPERVISED DISTANCE MODEL TRAINING V1")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"[OK] Dataset loaded.")
print(f"[OK] Samples: {len(df)}")
print(f"[OK] Columns: {len(df.columns)}")


# ============================================================
# VERIFY FEATURES
# ============================================================

print("\n[2] Checking feature schema...")

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

print(f"[OK] Found all {len(FEATURE_COLUMNS)} features.")
print(f"[OK] Target: {TARGET_COLUMN}")


# ============================================================
# CREATE X AND y
# ============================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


# ============================================================
# NUMERIC VALIDATION
# ============================================================

print("\n[3] Validating numerical data...")

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

if X.isna().any().any():
    raise ValueError("NaN values found in feature matrix.")

if y.isna().any():
    raise ValueError("NaN values found in target.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Inf values found in feature matrix.")

if not np.isfinite(y.to_numpy()).all():
    raise ValueError("Inf values found in target.")

print("[OK] No missing values.")
print("[OK] No NaN values.")
print("[OK] No Inf values.")

print(f"[OK] X shape: {X.shape}")
print(f"[OK] y shape: {y.shape}")

print(
    f"[OK] Distance range: "
    f"{y.min():.1f} cm → {y.max():.1f} cm"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Creating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

print(f"[OK] Training samples: {len(X_train)}")
print(f"[OK] Testing samples : {len(X_test)}")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "ridge": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            Ridge(alpha=10.0)
        ),
    ]),

    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "extra_trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        loss="huber",
    ),
}


# ============================================================
# TRAIN MODELS
# ============================================================

print("\n[5] Training models...")
print("-" * 70)

results = []
predictions = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    predictions[name] = pred

    results.append({
        "model": name,
        "MAE_cm": mae,
        "RMSE_cm": rmse,
        "R2": r2,
    })

    print(f"  MAE  : {mae:.3f} cm")
    print(f"  RMSE : {rmse:.3f} cm")
    print(f"  R²   : {r2:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE_cm",
    ascending=True
).reset_index(drop=True)

print("\n")
print("=" * 70)
print("PILOT MODEL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SELECT MODEL BASED ON TEST MAE
# ============================================================

best_model_name = results_df.iloc[0]["model"]

best_model = models[best_model_name]

print("\n" + "=" * 70)
print("PILOT MODEL SELECTED")
print("=" * 70)

print(f"Model: {best_model_name}")
print(
    f"Test MAE: "
    f"{results_df.iloc[0]['MAE_cm']:.3f} cm"
)
print(
    f"Test RMSE: "
    f"{results_df.iloc[0]['RMSE_cm']:.3f} cm"
)
print(
    f"Test R²: "
    f"{results_df.iloc[0]['R2']:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "distance_model_v1.joblib",
)

model_package = {
    "model": best_model,
    "feature_names": FEATURE_COLUMNS,
    "target": TARGET_COLUMN,
    "model_name": best_model_name,
    "version": "v1",
    "dataset": DATASET_PATH,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "random_state": RANDOM_STATE,
}

joblib.dump(
    model_package,
    model_path,
)

print(f"\n[OK] Model saved:")
print(model_path)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULT_DIR,
    "distance_model_v1_metrics.csv",
)

results_df.to_csv(
    metrics_path,
    index=False,
)

print(f"[OK] Metrics saved:")
print(metrics_path)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "actual_distance_cm": y_test.to_numpy(),
})

for name, pred in predictions.items():
    prediction_df[
        f"{name}_prediction_cm"
    ] = pred

prediction_path = os.path.join(
    RESULT_DIR,
    "distance_model_v1_predictions.csv",
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)

print(f"[OK] Predictions saved:")
print(prediction_path)


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

info = {
    "version": "v1",
    "dataset": DATASET_PATH,
    "samples": int(len(df)),
    "features": int(len(FEATURE_COLUMNS)),
    "feature_names": FEATURE_COLUMNS,
    "target": TARGET_COLUMN,
    "distance_min_cm": float(y.min()),
    "distance_max_cm": float(y.max()),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "models_tested": list(models.keys()),
    "selected_model": best_model_name,
}

info_path = os.path.join(
    RESULT_DIR,
    "distance_model_v1_info.json",
)

with open(
    info_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        info,
        f,
        indent=2
    )

print(f"[OK] Training information saved:")
print(info_path)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(f"  Model:")
print(f"    {model_path}")

print(f"\n  Metrics:")
print(f"    {metrics_path}")

print(f"\n  Predictions:")
print(f"    {prediction_path}")

print(f"\n  Training info:")
print(f"    {info_path}")

print("\n[IMPORTANT]")
print("This is a PILOT model.")
print("The test metrics do not yet establish production accuracy.")
print("=" * 70)
