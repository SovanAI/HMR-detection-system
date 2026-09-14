"""
Depth Anything ↔ HMR2 Calibration Analysis

Reads:
    test_results/depth_hmr_calibration/calibration_samples.json

Fits:
    HMR_Z = scale * relative_depth + offset

Calculates:
    R²
    RMSE
    MAE

Outputs:
    test_results/depth_hmr_calibration/calibration_parameters.json
    test_results/depth_hmr_calibration/calibration_plot.png

Run:
    python -m human.analyze_depth_hmr_calibration
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "test_results/depth_hmr_calibration/"
    "calibration_samples.json"
)

OUTPUT_DIR = (
    "test_results/depth_hmr_calibration"
)

PARAMETERS_FILE = os.path.join(
    OUTPUT_DIR,
    "calibration_parameters.json"
)

PLOT_FILE = os.path.join(
    OUTPUT_DIR,
    "calibration_plot.png"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_samples():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Calibration file not found:\n"
            f"{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r"
    ) as f:

        data = json.load(f)

    samples = data.get(
        "samples",
        []
    )

    if len(samples) < 2:

        raise ValueError(
            "At least 2 calibration samples "
            "are required."
        )

    return samples


# ============================================================
# EXTRACT VALUES
# ============================================================

def extract_values(samples):

    relative_depth = []

    hmr_z = []

    sample_ids = []

    for sample in samples:

        depth = sample.get(
            "relative_depth"
        )

        translation = sample.get(
            "hmr_camera_translation"
        )

        if depth is None:
            continue

        if not translation:
            continue

        z = translation.get("z")

        if z is None:
            continue

        relative_depth.append(
            float(depth)
        )

        hmr_z.append(
            float(z)
        )

        sample_ids.append(
            sample.get(
                "sample_id",
                len(sample_ids) + 1
            )
        )

    if len(relative_depth) < 2:

        raise ValueError(
            "Not enough valid calibration samples."
        )

    return (
        np.asarray(relative_depth),
        np.asarray(hmr_z),
        sample_ids,
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

def fit_linear_model(
    x,
    y
):

    # y = scale*x + offset

    scale, offset = np.polyfit(
        x,
        y,
        1
    )

    predictions = (
        scale * x
        + offset
    )

    return (
        float(scale),
        float(offset),
        predictions
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    errors = (
        actual
        - predicted
    )

    squared_errors = (
        errors ** 2
    )

    absolute_errors = (
        np.abs(errors)
    )

    mse = np.mean(
        squared_errors
    )

    rmse = np.sqrt(
        mse
    )

    mae = np.mean(
        absolute_errors
    )

    ss_res = np.sum(
        squared_errors
    )

    ss_tot = np.sum(
        (
            actual
            - np.mean(actual)
        ) ** 2
    )

    if ss_tot < 1e-12:

        r2 = 0.0

    else:

        r2 = (
            1.0
            - ss_res / ss_tot
        )

    correlation = np.corrcoef(
        actual,
        predicted
    )[0, 1]

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(
            correlation
        ),
    }


# ============================================================
# PRINT SAMPLE TABLE
# ============================================================

def print_sample_table(
    sample_ids,
    x,
    actual,
    predicted
):

    print()

    print(
        "=" * 90
    )

    print(
        "CALIBRATION SAMPLE ANALYSIS"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Sample':>8} | "
        f"{'Depth':>12} | "
        f"{'Actual HMR Z':>15} | "
        f"{'Predicted Z':>15} | "
        f"{'Error':>12}"
    )

    print(
        "-" * 90
    )

    for sid, depth, actual_z, predicted_z in zip(
        sample_ids,
        x,
        actual,
        predicted
    ):

        error = (
            actual_z
            - predicted_z
        )

        print(
            f"{sid:>8} | "
            f"{depth:>12.6f} | "
            f"{actual_z:>15.6f} | "
            f"{predicted_z:>15.6f} | "
            f"{error:>12.6f}"
        )

    print(
        "=" * 90
    )


# ============================================================
# SAVE PARAMETERS
# ============================================================

def save_parameters(
    scale,
    offset,
    metrics,
    sample_count
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    data = {

        "model": (
            "linear_depth_to_hmr_z"
        ),

        "equation": (
            "hmr_z = "
            "scale * relative_depth "
            "+ offset"
        ),

        "scale": scale,

        "offset": offset,

        "sample_count": sample_count,

        "metrics": metrics,

        "units": {
            "relative_depth": (
                "Depth Anything relative value"
            ),

            "hmr_z": (
                "HMR2 coordinate-system value"
            ),

            "distance": (
                "NOT physical metres"
            ),
        },

        "warning": (
            "This calibration maps "
            "Depth Anything relative depth "
            "to HMR2 Z. It does not establish "
            "metric distance in metres."
        ),
    }

    with open(
        PARAMETERS_FILE,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


# ============================================================
# CREATE PLOT
# ============================================================

def create_plot(
    x,
    actual,
    predicted,
    scale,
    offset,
    metrics
):

    plt.figure(
        figsize=(10, 7)
    )

    # Actual measurements
    plt.scatter(
        x,
        actual,
        label="Measured HMR2 Z"
    )

    # Regression line
    x_line = np.linspace(
        np.min(x) - 0.03,
        np.max(x) + 0.03,
        200
    )

    y_line = (
        scale * x_line
        + offset
    )

    plt.plot(
        x_line,
        y_line,
        label="Linear calibration"
    )

    plt.xlabel(
        "Depth Anything Relative Depth"
    )

    plt.ylabel(
        "HMR2 Camera Translation Z"
    )

    plt.title(
        "Depth Anything ↔ HMR2 Z Calibration"
    )

    equation = (
        f"HMR Z = "
        f"{scale:.4f} × Depth "
        f"+ {offset:.4f}"
    )

    metric_text = (
        f"{equation}\n"
        f"R² = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}"
    )

    plt.text(
        0.03,
        0.97,
        metric_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.8
        }
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=150
    )

    plt.close()


# ============================================================
# RECOMMENDATION
# ============================================================

def print_recommendation(
    metrics
):

    r2 = metrics["r2"]
    rmse = metrics["rmse"]

    print()

    print(
        "=" * 90
    )

    print(
        "CALIBRATION ASSESSMENT"
    )

    print(
        "=" * 90
    )

    print(
        f"R²         : {r2:.4f}"
    )

    print(
        f"RMSE       : {rmse:.4f}"
    )

    print(
        f"MAE        : {metrics['mae']:.4f}"
    )

    print(
        f"Correlation: {metrics['correlation']:.4f}"
    )

    print()

    if r2 >= 0.95:

        print(
            "RESULT: Strong relationship detected."
        )

        print(
            "The calibration is promising for "
            "relative spatial reasoning."
        )

    elif r2 >= 0.85:

        print(
            "RESULT: Moderate/strong relationship."
        )

        print(
            "Additional calibration samples "
            "are recommended."
        )

    else:

        print(
            "RESULT: Weak relationship."
        )

        print(
            "Do NOT use this calibration for "
            "spatial distance yet."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "HMR Z is NOT guaranteed to be metres."
    )

    print(
        "Depth Anything output is relative depth."
    )

    print(
        "This calibration currently establishes "
        "a coordinate mapping, not metric distance."
    )

    print(
        "=" * 90
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 90
    )

    print(
        "BAS HMR - DEPTH ↔ HMR2 CALIBRATION ANALYSIS"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    samples = load_samples()

    print(
        f"Loaded {len(samples)} calibration samples."
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    (
        x,
        actual,
        sample_ids
    ) = extract_values(
        samples
    )

    print(
        f"Valid samples: {len(x)}"
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    (
        scale,
        offset,
        predicted
    ) = fit_linear_model(
        x,
        actual
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        actual,
        predicted
    )

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    print_sample_table(
        sample_ids,
        x,
        actual,
        predicted
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()

    print(
        "CALIBRATION EQUATION"
    )

    print(
        "-" * 50
    )

    print(
        f"HMR Z = "
        f"{scale:.6f} × RelativeDepth "
        f"+ {offset:.6f}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print()

    print(
        "METRICS"
    )

    print(
        "-" * 50
    )

    print(
        f"R²          = "
        f"{metrics['r2']:.6f}"
    )

    print(
        f"RMSE        = "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"MAE         = "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"Correlation = "
        f"{metrics['correlation']:.6f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_parameters(
        scale,
        offset,
        metrics,
        len(x)
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    create_plot(
        x,
        actual,
        predicted,
        scale,
        offset,
        metrics
    )

    print()

    print(
        f"Parameters saved:"
    )

    print(
        PARAMETERS_FILE
    )

    print()

    print(
        f"Plot saved:"
    )

    print(
        PLOT_FILE
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    print_recommendation(
        metrics
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
