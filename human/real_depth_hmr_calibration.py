from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

from hmr.hmr_processor import HMRProcessor


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

YOLO_MODEL = ROOT / "yolo11n.pt"

OUTPUT_DIR = ROOT / "test_results" / "depth_hmr_calibration"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_JSON = OUTPUT_DIR / "calibration_samples.json"


# ============================================================
# CAMERA
# ============================================================

CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10

IMG_SIZE = 416

PERSON_CLASS = 0
CONFIDENCE = 0.35


# ============================================================
# CALIBRATION SETTINGS
# ============================================================

TARGET_SAMPLES = 10

# Minimum number of samples before fitting.
MIN_SAMPLES = 5


# ============================================================
# DEPTH
# ============================================================

print("=" * 70)
print("BAS-HMR REAL DEPTH ↔ HMR2 CALIBRATION")
print("=" * 70)

print("\nLoading YOLO11n...")

yolo = YOLO(
    str(YOLO_MODEL)
)

print("YOLO loaded.")


print("\nLoading Depth Anything V2 Small...")

depth_pipeline = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=-1,
)

print("Depth model loaded.")


# ============================================================
# HMR2
# ============================================================

print("\nLoading HMR2...")

hmr_processor = HMRProcessor()

print("HMR2 loaded.")


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2,
)

if not camera.isOpened():

    raise RuntimeError(
        "Could not open camera."
    )


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH,
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT,
)

camera.set(
    cv2.CAP_PROP_FPS,
    FPS,
)

camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        *"MJPG"
    ),
)


# ============================================================
# HELPERS
# ============================================================

def get_person_boxes(frame):

    results = yolo.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    persons = []

    if result.boxes is None:
        return persons

    for box in result.boxes:

        cls = int(
            box.cls[0].item()
        )

        if cls != PERSON_CLASS:
            continue

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        persons.append(
            {
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "confidence": confidence,
            }
        )

    return persons


def calculate_depth_map(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(
        rgb
    )

    result = depth_pipeline(
        image
    )

    depth = np.array(
        result["depth"],
        dtype=np.float32,
    )

    depth = cv2.resize(
        depth,
        (
            frame.shape[1],
            frame.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return depth


def sample_depth_at_bbox(
    depth_map,
    bbox,
):

    x1 = bbox["x1"]
    y1 = bbox["y1"]
    x2 = bbox["x2"]
    y2 = bbox["y2"]

    height, width = depth_map.shape

    x1 = max(
        0,
        min(width - 1, x1)
    )

    x2 = max(
        0,
        min(width, x2)
    )

    y1 = max(
        0,
        min(height - 1, y1)
    )

    y2 = max(
        0,
        min(height, y2)
    )

    if x2 <= x1 or y2 <= y1:

        return None

    # Use central region to reduce background contamination.
    margin_x = int(
        (x2 - x1) * 0.30
    )

    margin_y = int(
        (y2 - y1) * 0.20
    )

    sx1 = x1 + margin_x
    sx2 = x2 - margin_x

    sy1 = y1 + margin_y
    sy2 = y2 - margin_y

    region = depth_map[
        sy1:sy2,
        sx1:sx2,
    ]

    if region.size == 0:

        return None

    return float(
        np.median(region)
    )


def normalize_depth(
    value,
    depth_map,
):

    minimum = float(
        np.min(depth_map)
    )

    maximum = float(
        np.max(depth_map)
    )

    denominator = (
        maximum - minimum
    )

    if denominator < 1e-6:

        return None

    normalized = (
        value - minimum
    ) / denominator

    return float(
        np.clip(
            normalized,
            0.0,
            1.0,
        )
    )


def choose_person(persons):

    if not persons:
        return None

    # Use the largest person.
    return max(
        persons,
        key=lambda p:
            (
                p["bbox"]["x2"]
                - p["bbox"]["x1"]
            )
            *
            (
                p["bbox"]["y2"]
                - p["bbox"]["y1"]
            )
    )


# ============================================================
# WARM-UP
# ============================================================

print("\nWarming up camera...")

for _ in range(10):

    success, frame = camera.read()

    if not success:
        continue

    time.sleep(0.05)


print("\nCamera ready.")


# ============================================================
# CALIBRATION STORAGE
# ============================================================

samples = []

frame_id = 0


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 70)
print("CALIBRATION COLLECTION STARTED")
print("=" * 70)

print()
print(
    "The system will collect real Depth Anything ↔ HMR2 samples."
)

print()
print(
    "IMPORTANT:"
)

print(
    "Keep one person clearly visible."
)

print(
    "Move slowly toward/away from the camera."
)

print(
    "The system will automatically collect samples."
)

print()
print(
    f"Target samples: {TARGET_SAMPLES}"
)

print()
print(
    "Press Ctrl+C to stop."
)

print("=" * 70)


try:

    while len(samples) < TARGET_SAMPLES:

        success, frame = camera.read()

        if not success:

            time.sleep(0.1)
            continue

        frame_id += 1

        # ----------------------------------------------------
        # PERSON DETECTION
        # ----------------------------------------------------

        persons = get_person_boxes(
            frame
        )

        person = choose_person(
            persons
        )

        if person is None:

            print(
                "No person detected."
            )

            time.sleep(0.2)
            continue

        bbox = person["bbox"]

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------

        print(
            f"\nFrame {frame_id}: "
            "estimating depth..."
        )

        depth_map = calculate_depth_map(
            frame
        )

        depth_raw = sample_depth_at_bbox(
            depth_map,
            bbox,
        )

        if depth_raw is None:

            print(
                "Could not sample depth."
            )

            continue

        depth_relative = normalize_depth(
            depth_raw,
            depth_map,
        )

        if depth_relative is None:

            print(
                "Invalid relative depth."
            )

            continue

        # ----------------------------------------------------
        # HMR2
        # ----------------------------------------------------

        print(
            f"Frame {frame_id}: "
            "running HMR2..."
        )

        hmr_result = hmr_processor.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=time.time(),
            boxes=[bbox],
        )

        hmr_persons = hmr_result.get(
            "persons",
            [],
        )

        if not hmr_persons:

            print(
                "HMR2 did not return a person."
            )

            continue

        hmr_person = hmr_persons[0]

        camera_translation = hmr_person.get(
            "camera_translation"
        )

        if not camera_translation:

            print(
                "Missing HMR2 camera translation."
            )

            continue

        hmr_z = float(
            camera_translation["z"]
        )

        # ----------------------------------------------------
        # STORE SAMPLE
        # ----------------------------------------------------

        sample = {
            "sample_id": len(samples),

            "frame_id": frame_id,

            "timestamp": time.time(),

            "person_bbox": bbox,

            "person_detection_confidence":
                person["confidence"],

            "depth_raw": depth_raw,

            "depth_relative":
                depth_relative,

            "hmr_camera_translation": {
                "x": float(
                    camera_translation["x"]
                ),
                "y": float(
                    camera_translation["y"]
                ),
                "z": hmr_z,
            },

            "hmr_z": hmr_z,
        }

        samples.append(
            sample
        )

        print()
        print(
            "CALIBRATION SAMPLE"
        )

        print(
            f"Sample: "
            f"{len(samples)}/{TARGET_SAMPLES}"
        )

        print(
            f"Depth relative: "
            f"{depth_relative:.5f}"
        )

        print(
            f"HMR Z: "
            f"{hmr_z:.5f}"
        )

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        with open(
            OUTPUT_JSON,
            "w"
        ) as f:

            json.dump(
                {
                    "samples": samples,
                    "metadata": {
                        "camera": CAMERA,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "model_depth":
                            "Depth-Anything-V2-Small",
                        "model_hmr":
                            "4D-Humans HMR2",
                    },
                },
                f,
                indent=2,
            )

        # Give CPU time to settle.
        time.sleep(0.5)


except KeyboardInterrupt:

    print(
        "\n\nCalibration interrupted."
    )


finally:

    camera.release()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("CALIBRATION COLLECTION COMPLETE")
print("=" * 70)

print(
    f"Samples collected: {len(samples)}"
)

print(
    f"Saved to: {OUTPUT_JSON}"
)

print("=" * 70)
