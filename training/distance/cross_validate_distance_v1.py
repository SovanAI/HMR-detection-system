import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "distance",
)

RANDOM_STATE = 42
N_SPLITS = 5

FEATURE_COLUMNS = [
    f"feature_{i:02d}"
    for i in range(1, 35)
]

TARGET_COLUMN = "distance_cm"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BAS-HMR — 5-FOLD CROSS-VALIDATION V1")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1] Loading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

print(f"[OK] Samples : {len(df)}")
print(f"[OK] Features: {len(FEATURE_COLUMNS)}")
print(f"[OK] X shape : {X.shape}")
print(f"[OK] y shape : {y.shape}")


# ============================================================
# VALIDATION
# ============================================================

print("\n[2] Validating dataset...")

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

if X.isna().any().any():
    raise ValueError("NaN found in X.")

if y.isna().any():
    raise ValueError("NaN found in y.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Inf found in X.")

if not np.isfinite(y.to_numpy()).all():
    raise ValueError("Inf found in y.")

print("[OK] Dataset contains no NaN or Inf.")


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
# CROSS VALIDATION
# ============================================================

print("\n[3] Starting 5-fold cross-validation...")
print("-" * 70)

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

all_results = []


for name, model in models.items():

    print(f"\nEvaluating: {name}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
        return_train_score=False,
    )

    mae = -scores["test_mae"]
    rmse = -scores["test_rmse"]
    r2 = scores["test_r2"]

    print("\nFold results:")

    for i in range(N_SPLITS):

        print(
            f"  Fold {i + 1}: "
            f"MAE={mae[i]:.3f} cm | "
            f"RMSE={rmse[i]:.3f} cm | "
            f"R²={r2[i]:.4f}"
        )

    print("\nMean ± Std:")

    print(
        f"  MAE : "
        f"{mae.mean():.3f} ± {mae.std():.3f} cm"
    )

    print(
        f"  RMSE: "
        f"{rmse.mean():.3f} ± {rmse.std():.3f} cm"
    )

    print(
        f"  R²  : "
        f"{r2.mean():.4f} ± {r2.std():.4f}"
    )

    all_results.append({
        "model": name,

        "MAE_mean_cm": mae.mean(),
        "MAE_std_cm": mae.std(),

        "RMSE_mean_cm": rmse.mean(),
        "RMSE_std_cm": rmse.std(),

        "R2_mean": r2.mean(),
        "R2_std": r2.std(),
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="MAE_mean_cm",
    ascending=True,
).reset_index(drop=True)


print("\n")
print("=" * 70)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(
    OUTPUT_DIR,
    "distance_model_v1_cross_validation.csv",
)

results_df.to_csv(
    output_path,
    index=False,
)

print("\n[OK] Cross-validation results saved:")
print(output_path)


# ============================================================
# BEST MODEL BY MEAN MAE
# ============================================================

best = results_df.iloc[0]

print("\n")
print("=" * 70)
print("CROSS-VALIDATION RESULT")
print("=" * 70)

print(
    f"Model: {best['model']}"
)

print(
    f"Mean MAE: "
    f"{best['MAE_mean_cm']:.3f} ± "
    f"{best['MAE_std_cm']:.3f} cm"
)

print(
    f"Mean RMSE: "
    f"{best['RMSE_mean_cm']:.3f} ± "
    f"{best['RMSE_std_cm']:.3f} cm"
)

print(
    f"Mean R²: "
    f"{best['R2_mean']:.4f} ± "
    f"{best['R2_std']:.4f}"
)

print("\n[IMPORTANT]")
print("This is still a pilot validation.")
print("The dataset has limited environmental and pose diversity.")

print("=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)
