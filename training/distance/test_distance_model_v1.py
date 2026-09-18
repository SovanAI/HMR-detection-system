import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = os.path.expanduser("~/bas-hmr")

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset_collector",
    "data_v2",
    "distance_dataset_v2.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "distance",
    "distance_model_v1.joblib"
)


FEATURE_COLUMNS = [
    f"feature_{i:02d}"
    for i in range(1, 35)
]

TARGET_COLUMN = "distance_cm"

RANDOM_STATE = 42
TEST_SIZE = 0.20


print("=" * 70)
print("BAS-HMR — DISTANCE MODEL V1 TEST")
print("=" * 70)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\n[1] Loading dataset...")

df = pd.read_csv(DATASET_PATH)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ------------------------------------------------------------
# Recreate exact test split
# ------------------------------------------------------------

print("[2] Creating test set...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("[3] Loading trained model...")

package = joblib.load(MODEL_PATH)

model = package["model"]
feature_names = package["feature_names"]

print("[OK] Model loaded")
print("[OK] Model:", package["model_name"])
print("[OK] Version:", package["version"])


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print("\n[4] Running predictions...")

predictions = model.predict(X_test)

predictions = np.asarray(predictions)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

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


print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.3f} cm")
print(f"RMSE : {rmse:.3f} cm")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------
# Individual predictions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INDIVIDUAL TEST PREDICTIONS")
print("=" * 70)

print(
    f"{'Actual':>12} "
    f"{'Predicted':>12} "
    f"{'Error':>12}"
)

print("-" * 40)

for actual, predicted in zip(
    y_test.to_numpy(),
    predictions
):

    error = predicted - actual

    print(
        f"{actual:>10.1f} cm "
        f"{predicted:>10.1f} cm "
        f"{error:>+10.1f} cm"
    )


# ------------------------------------------------------------
# Error statistics
# ------------------------------------------------------------

absolute_errors = np.abs(
    predictions - y_test.to_numpy()
)

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print(
    f"Mean absolute error : "
    f"{absolute_errors.mean():.3f} cm"
)

print(
    f"Median error        : "
    f"{np.median(absolute_errors):.3f} cm"
)

print(
    f"Maximum error       : "
    f"{absolute_errors.max():.3f} cm"
)

print(
    f"Within ±5 cm        : "
    f"{np.mean(absolute_errors <= 5) * 100:.1f}%"
)

print(
    f"Within ±10 cm       : "
    f"{np.mean(absolute_errors <= 10) * 100:.1f}%"
)

print(
    f"Within ±20 cm       : "
    f"{np.mean(absolute_errors <= 20) * 100:.1f}%"
)


# ------------------------------------------------------------
# Save detailed predictions
# ------------------------------------------------------------

result = pd.DataFrame({
    "actual_distance_cm": y_test.to_numpy(),
    "predicted_distance_cm": predictions,
    "error_cm": predictions - y_test.to_numpy(),
    "absolute_error_cm": absolute_errors
})

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "training",
    "distance",
    "distance_model_v1_test_results.csv"
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n[OK] Detailed results saved:")
print(OUTPUT_PATH)


print("\n" + "=" * 70)
print("MODEL TEST COMPLETE")
print("=" * 70)
