import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "yolo26n-depth.pt"

CAMERA_DEVICE = "/dev/video0"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 10

IMAGE_SIZE = 640

# Number of valid depth readings collected per test
READINGS_PER_TEST = 20

# Seconds to wait before collecting measurements
COUNTDOWN_SECONDS = 3

# Delay between measurements
MEASUREMENT_INTERVAL = 0.25

# Output directory
OUTPUT_DIR = "data/validation/yolo26_depth"

CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "yolo26_depth_measurements.csv"
)


# ============================================================
# DEPTH UTILITIES
# ============================================================

def get_depth_at_pixel(
    depth_map,
    x,
    y,
    radius=5
):
    """
    Extract robust depth around a pixel.

    Returns:
        depth in metres
    """

    if depth_map is None:
        return None

    h, w = depth_map.shape

    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)

    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    region = depth_map[
        y1:y2,
        x1:x2
    ]

    valid = region[
        np.isfinite(region) &
        (region > 0)
    ]

    if len(valid) == 0:
        return None

    return float(np.median(valid))


# ============================================================
# EXTRACT DEPTH MAP
# ============================================================

def extract_depth(result, frame_shape):
    """
    Extract depth map from YOLO26 result.
    """

    if not hasattr(result, "depth"):
        return None

    if result.depth is None:
        return None

    depth_map = (
        result.depth.data
        .detach()
        .cpu()
        .numpy()
    )

    depth_map = np.asarray(
        depth_map,
        dtype=np.float32
    )

    depth_map = np.squeeze(depth_map)

    if depth_map.ndim != 2:
        return None

    target_h = frame_shape[0]
    target_w = frame_shape[1]

    if depth_map.shape != (
        target_h,
        target_w
    ):

        depth_map = cv2.resize(
            depth_map,
            (
                target_w,
                target_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    return depth_map


# ============================================================
# YOLO26 DEPTH INFERENCE
# ============================================================

def get_frame_depth(
    model,
    frame
):
    """
    Run YOLO26 depth inference.

    Returns:
        depth_map,
        inference_time
    """

    start = time.perf_counter()

    results = model(
        frame,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )

    inference_time = (
        time.perf_counter() - start
    )

    if not results:
        return None, inference_time

    result = results[0]

    depth_map = extract_depth(
        result,
        frame.shape
    )

    return depth_map, inference_time


# ============================================================
# CAMERA SETUP
# ============================================================

def open_camera():

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"{CAMERA_DEVICE}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    return cap


# ============================================================
# CSV SETUP
# ============================================================

def prepare_csv():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    file_exists = os.path.exists(
        CSV_FILE
    )

    if not file_exists:

        with open(
            CSV_FILE,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "timestamp",
                    "test_id",
                    "actual_distance_cm",
                    "reading_number",
                    "depth_m",
                    "depth_cm",
                    "inference_time_s"
                ]
            )


# ============================================================
# SAVE READING
# ============================================================

def save_reading(
    test_id,
    actual_distance_cm,
    reading_number,
    depth_m,
    inference_time
):

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                test_id,
                actual_distance_cm,
                reading_number,
                f"{depth_m:.6f}",
                f"{depth_m * 100:.3f}",
                f"{inference_time:.6f}"
            ]
        )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    readings
):

    values = np.asarray(
        readings,
        dtype=np.float64
    )

    return {
        "count": len(values),
        "mean_m": float(
            np.mean(values)
        ),
        "median_m": float(
            np.median(values)
        ),
        "min_m": float(
            np.min(values)
        ),
        "max_m": float(
            np.max(values)
        ),
        "std_m": float(
            np.std(values)
        )
    }


# ============================================================
# PRINT TEST RESULT
# ============================================================

def print_test_result(
    test_id,
    actual_distance_cm,
    readings,
    inference_times
):

    stats = calculate_statistics(
        readings
    )

    actual_m = (
        actual_distance_cm / 100.0
    )

    mean_error_m = (
        stats["mean_m"] -
        actual_m
    )

    median_error_m = (
        stats["median_m"] -
        actual_m
    )

    mean_abs_error_cm = abs(
        mean_error_m
    ) * 100

    median_abs_error_cm = abs(
        median_error_m
    ) * 100

    if actual_m > 0:

        mean_percentage_error = (
            abs(mean_error_m) /
            actual_m
        ) * 100

    else:

        mean_percentage_error = 0.0

    avg_inference = np.mean(
        inference_times
    )

    print()
    print("=" * 70)
    print(
        f"TEST {test_id} RESULT"
    )
    print("=" * 70)

    print(
        f"Actual distance       : "
        f"{actual_distance_cm:.1f} cm"
    )

    print(
        f"Mean YOLO26 depth     : "
        f"{stats['mean_m']:.3f} m "
        f"({stats['mean_m'] * 100:.1f} cm)"
    )

    print(
        f"Median YOLO26 depth   : "
        f"{stats['median_m']:.3f} m "
        f"({stats['median_m'] * 100:.1f} cm)"
    )

    print(
        f"Minimum               : "
        f"{stats['min_m']:.3f} m"
    )

    print(
        f"Maximum               : "
        f"{stats['max_m']:.3f} m"
    )

    print(
        f"Std deviation         : "
        f"{stats['std_m']:.3f} m"
    )

    print(
        f"Mean error            : "
        f"{mean_error_m * 100:+.1f} cm"
    )

    print(
        f"Median error          : "
        f"{median_error_m * 100:+.1f} cm"
    )

    print(
        f"Mean absolute error   : "
        f"{mean_abs_error_cm:.1f} cm"
    )

    print(
        f"Mean percentage error : "
        f"{mean_percentage_error:.2f}%"
    )

    print(
        f"Average inference     : "
        f"{avg_inference:.3f} s"
    )

    print("=" * 70)


# ============================================================
# SINGLE TEST
# ============================================================

def run_test(
    model,
    cap,
    test_id,
    actual_distance_cm
):

    print()
    print()
    print("#" * 70)
    print(
        f"PREPARING TEST {test_id}"
    )
    print(
        f"ACTUAL DISTANCE: "
        f"{actual_distance_cm} cm"
    )
    print("#" * 70)

    print()
    print(
        "Place the test object at the "
        "EXACT measured distance."
    )

    print(
        "Keep the camera and object stationary."
    )

    input(
        "Press ENTER when ready..."
    )

    # --------------------------------------------------------
    # COUNTDOWN
    # --------------------------------------------------------

    for remaining in range(
        COUNTDOWN_SECONDS,
        0,
        -1
    ):

        print(
            f"Starting in {remaining}..."
        )

        time.sleep(1)

    print()
    print(
        "COLLECTING DEPTH READINGS..."
    )

    readings = []
    inference_times = []

    reading_number = 0

    while len(readings) < READINGS_PER_TEST:

        success, frame = cap.read()

        if not success:

            print(
                "[CAMERA] Frame read failed."
            )

            continue

        depth_map, inference_time = (
            get_frame_depth(
                model,
                frame
            )
        )

        if depth_map is None:

            print(
                "[DEPTH] No depth map."
            )

            continue

        # ----------------------------------------------------
        # CENTER PIXEL
        # ----------------------------------------------------

        center_x = (
            frame.shape[1] // 2
        )

        center_y = (
            frame.shape[0] // 2
        )

        depth_m = get_depth_at_pixel(
            depth_map,
            center_x,
            center_y,
            radius=5
        )

        if depth_m is None:

            print(
                "[DEPTH] Invalid depth."
            )

            continue

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not np.isfinite(depth_m):

            continue

        if depth_m <= 0:

            continue

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        reading_number += 1

        readings.append(
            depth_m
        )

        inference_times.append(
            inference_time
        )

        save_reading(
            test_id,
            actual_distance_cm,
            reading_number,
            depth_m,
            inference_time
        )

        print(
            f"Reading "
            f"{reading_number:02d}/"
            f"{READINGS_PER_TEST}: "
            f"{depth_m:.3f} m "
            f"({depth_m * 100:.1f} cm)"
        )

        time.sleep(
            MEASUREMENT_INTERVAL
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print_test_result(
        test_id,
        actual_distance_cm,
        readings,
        inference_times
    )

    return readings


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("YOLO26n-Depth Controlled Distance Validation")
    print("=" * 70)
    print()

    print(
        f"Model          : {MODEL_NAME}"
    )

    print(
        f"Camera         : {CAMERA_DEVICE}"
    )

    print(
        f"Resolution     : "
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
    )

    print(
        f"Device         : CPU"
    )

    print(
        f"Readings/test  : "
        f"{READINGS_PER_TEST}"
    )

    print(
        f"Output CSV     : {CSV_FILE}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # PREPARE CSV
    # --------------------------------------------------------

    prepare_csv()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print(
        "[MODEL] Loading YOLO26n-Depth..."
    )

    model = YOLO(
        MODEL_NAME
    )

    print(
        "[MODEL] Model loaded successfully."
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    print(
        "[CAMERA] Opening camera..."
    )

    cap = open_camera()

    print(
        "[CAMERA] Camera opened successfully."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The object must be placed at the "
        "specified tape-measured distance."
    )

    print(
        "Keep the object near the CENTER "
        "of the camera image."
    )

    print()

    # --------------------------------------------------------
    # TEST DISTANCES
    # --------------------------------------------------------

    test_distances = [
        50,
        100,
        150,
        200,
        250
    ]

    try:

        for test_id, distance_cm in enumerate(
            test_distances,
            start=1
        ):

            run_test(
                model,
                cap,
                test_id,
                distance_cm
            )

    except KeyboardInterrupt:

        print()
        print(
            "[STOP] Experiment interrupted."
        )

    finally:

        cap.release()

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT FINISHED")
    print("=" * 70)

    print()
    print(
        f"Measurements saved to:"
    )

    print(
        f"  {CSV_FILE}"
    )

    print()

    print(
        "Send me the CSV file after completing "
        "the experiment."
    )

    print()


if __name__ == "__main__":

    main()