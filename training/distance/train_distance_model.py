import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "data/validation/"
    "yolo26_distance_grid/distance_dataset_v2.csv"
)

MODEL_DIR = "training/distance/models"

TEST_SIZE = 0.25
RANDOM_STATE = 42


# ============================================================
# METADATA COLUMNS
# These must NOT be used as ML features.
# ============================================================

METADATA_COLUMNS = [
    "sample_id",
    "timestamp",
    "frame_id",
    "person_id",
    "chair_id",

    "ground_truth_distance_cm",

    "person_bbox_x1",
    "person_bbox_y1",
    "person_bbox_x2",
    "person_bbox_y2",

    "chair_bbox_x1",
    "chair_bbox_y1",
    "chair_bbox_x2",
    "chair_bbox_y2",
]


TARGET_COLUMN = "ground_truth_distance_cm"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("BAS-HMR DISTANCE REGRESSION TRAINING")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset: {DATASET_PATH}")
print(f"Samples: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(df) < 10:
    raise RuntimeError(
        f"Only {len(df)} samples available. "
        "At least 10 samples are required for this prototype."
    )

if TARGET_COLUMN not in df.columns:
    raise RuntimeError(
        f"Missing target column: {TARGET_COLUMN}"
    )


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

before = len(df)

df = df.dropna(
    subset=[TARGET_COLUMN]
)

after = len(df)

if before != after:
    print(
        f"\nRemoved {before - after} rows "
        "with invalid target values."
    )


# ============================================================
# SELECT FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in METADATA_COLUMNS
]

print("\nFeature columns:")
for i, column in enumerate(feature_columns, start=1):
    print(f"{i:02d}. {column}")

print(f"\nNumber of ML features: {len(feature_columns)}")


if len(feature_columns) != 34:
    print(
        "\nWARNING:"
        f" Expected 34 features, found {len(feature_columns)}."
    )


X = df[feature_columns].copy()
y = df[TARGET_COLUMN].astype(float)


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for column in X.columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# REMOVE FEATURES THAT ARE COMPLETELY EMPTY
# ============================================================

empty_features = [
    column
    for column in X.columns
    if X[column].isna().all()
]

if empty_features:
    print("\nRemoving completely empty features:")

    for column in empty_features:
        print(" -", column)

    X = X.drop(columns=empty_features)

    feature_columns = [
        column
        for column in feature_columns
        if column not in empty_features
    ]


# ============================================================
# DATA SUMMARY
# ============================================================

print("\n" + "-" * 80)
print("TARGET DISTRIBUTION")
print("-" * 80)

print(
    y.value_counts()
    .sort_index()
    .to_string()
)

print("\nTarget range:")
print(f"Minimum: {y.min():.2f} cm")
print(f"Maximum: {y.max():.2f} cm")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "-" * 80)
print("TRAIN / TEST SPLIT")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Ridge": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(alpha=10.0)
        ),
    ]),

    "RandomForest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    ]),

    "GradientBoosting": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
            )
        ),
    ]),
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "model": name,
        "MAE_cm": mae,
        "RMSE_cm": rmse,
        "R2": r2,
    })

    trained_models[name] = model

    print(f"MAE  : {mae:.3f} cm")
    print(f"RMSE : {rmse:.3f} cm")
    print(f"R²   : {r2:.4f}")


# ============================================================
# BASELINE COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("RAW GEOMETRIC BASELINES")
print("=" * 80)

baseline_columns = [
    "raw_3d_distance",
    "raw_ground_distance",
]

for column in baseline_columns:

    if column not in df.columns:
        continue

    baseline = df.loc[
        X_test.index,
        column
    ].astype(float) * 100.0

    valid = baseline.notna()

    if valid.sum() == 0:
        continue

    baseline_y = y_test[valid]
    baseline_pred = baseline[valid]

    mae = mean_absolute_error(
        baseline_y,
        baseline_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            baseline_y,
            baseline_pred
        )
    )

    r2 = r2_score(
        baseline_y,
        baseline_pred
    )

    print(f"\n{column}")
    print(f"MAE  : {mae:.3f} cm")
    print(f"RMSE : {rmse:.3f} cm")
    print(f"R²   : {r2:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "MAE_cm"
)

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_name
]

best_mae = results_df.iloc[0]["MAE_cm"]

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(f"Model: {best_name}")
print(f"Test MAE: {best_mae:.3f} cm")


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model_path = os.path.join(
    MODEL_DIR,
    "best_distance_model.joblib"
)

artifact = {
    "model": best_model,
    "model_name": best_name,
    "feature_columns": feature_columns,
    "target": TARGET_COLUMN,
    "dataset": DATASET_PATH,
    "random_state": RANDOM_STATE,
}

joblib.dump(
    artifact,
    model_path
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame({
    "ground_truth_cm": y_test.values,
})

for name, model in trained_models.items():

    test_predictions[
        f"{name}_prediction_cm"
    ] = model.predict(X_test)


predictions_path = os.path.join(
    MODEL_DIR,
    "test_predictions.csv"
)

test_predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\nSaved files:")

print(
    f"Model comparison : {results_path}"
)

print(
    f"Test predictions : {predictions_path}"
)

print(
    f"Best model       : {model_path}"
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
