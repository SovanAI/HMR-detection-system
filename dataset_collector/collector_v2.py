import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import csv
import json
import threading
from datetime import datetime

import cv2
import numpy as np

from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
)

from ultralytics import YOLO


# ============================================================
# BAS-HMR DATASET COLLECTOR V2
#
# YOLO11 + HMR2 + Metric Depth + 3D Pair Features
#
# HMR2 and depth are executed ONLY when CAPTURE is pressed.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    BASE_DIR
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images_v2"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data_v2"
)

SESSION_DIR = os.path.join(
    BASE_DIR,
    "sessions_v2"
)

CSV_FILE = os.path.join(
    DATA_DIR,
    "distance_dataset_v2.csv"
)


os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    SESSION_DIR,
    exist_ok=True
)


# ============================================================
# CAMERA
# ============================================================

CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10


# ============================================================
# YOLO
# ============================================================

YOLO_MODEL = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt"
)

YOLO_CONFIDENCE = 0.25

PERSON_CLASS = 0
CHAIR_CLASS = 56


# ============================================================
# CAMERA INTRINSICS
#
# DEVELOPMENT VALUES ONLY.
# ============================================================

FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0


# ============================================================
# FLASK
# ============================================================

HOST = "0.0.0.0"
PORT = 5011


# ============================================================
# CSV
# ============================================================

BASE_COLUMNS = [
    "sample_id",
    "timestamp",

    "distance_cm",

    "person_id",
    "chair_id",

    "pose",
    "position",
    "chair_type",
    "lighting",

    "person_confidence",
    "chair_confidence",

    "person_x1",
    "person_y1",
    "person_x2",
    "person_y2",

    "chair_x1",
    "chair_y1",
    "chair_x2",
    "chair_y2",

    "hmr_success",
    "hmr_joint_count",

    "image_path",
]


# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

frame_lock = threading.Lock()

latest_frame = None

latest_persons = []

latest_chairs = []

frame_number = 0

sample_counter = 0

processing_capture = False


state = {
    "distance_cm": "",
    "pose": "standing",
    "position": "front",
    "chair_type": "unknown",
    "lighting": "normal",
}


# ============================================================
# OPTIONS
# ============================================================

POSES = [
    "standing",
    "sitting",
    "walking",
    "leaning",
    "front_facing",
    "back_facing",
    "side_profile",
]

POSITIONS = [
    "front",
    "left",
    "right",
    "diagonal",
]

CHAIR_TYPES = [
    "unknown",
    "office",
    "plastic",
    "wooden",
    "other",
]

LIGHTING_TYPES = [
    "normal",
    "low",
    "bright",
]


# ============================================================
# SESSION
# ============================================================

session_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

session_file = os.path.join(
    SESSION_DIR,
    f"session_{session_id}.json"
)


session_data = {
    "session_id": session_id,
    "created": datetime.now().isoformat(),
    "samples": [],
}


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("BAS-HMR DATASET COLLECTOR V2")
print("=" * 70)

print()
print("[1] Loading YOLO11n...")

yolo_model = YOLO(
    YOLO_MODEL
)

print("[OK] YOLO11n loaded.")
print("[OK] Device: CPU")


# ============================================================
# LOAD HMR2
# ============================================================

print()
print("[2] Loading HMR2...")

from human.hmr.hmr_processor import HMRProcessor

hmr_processor = HMRProcessor()

print("[OK] HMR2 loaded.")


# ============================================================
# LOAD METRIC DEPTH
# ============================================================

print()
print("[3] Loading metric depth...")

from human.depth.metric_depth import MetricDepthEstimator

depth_estimator = MetricDepthEstimator()

print("[OK] Metric depth loaded.")


# ============================================================
# LOAD 3D FEATURE MODULES
# ============================================================

print()
print("[4] Loading 3D feature modules...")

from human.distance.grid.depth_to_3d import (
    DepthTo3D,
    CameraIntrinsics,
)

from human.distance.grid.object_grid import (
    ObjectGridExtractor,
    BoundingBox,
)

from human.distance.grid.feature_extractor import (
    FeatureExtractor,
)

from human.distance.grid.pair_feature_extractor import (
    PairFeatureExtractor,
)


INTRINSICS = CameraIntrinsics(
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY,
)


depth_converter = DepthTo3D(
    intrinsics=INTRINSICS,
    min_depth=0.05,
    max_depth=20.0,
)


object_grid_extractor = ObjectGridExtractor(
    depth_converter=depth_converter,
    rows=5,
    cols=5,
    border_ratio=0.1,
    depth_radius=2,
)


feature_extractor = FeatureExtractor()

pair_feature_extractor = PairFeatureExtractor()


print("[OK] 3D feature modules loaded.")


# ============================================================
# CSV HEADER
#
# The PairFeatureExtractor vector is appended dynamically.
# ============================================================

PAIR_FEATURE_NAMES = []

try:

    # Create a temporary zero feature vector only to discover
    # the expected vector length after a real pair extraction.
    PAIR_FEATURE_NAMES = []

except Exception:
    pass


# ============================================================
# CREATE CAMERA
# ============================================================

print()
print("[5] Opening camera...")

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2,
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )


cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG"),
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT,
)

cap.set(
    cv2.CAP_PROP_FPS,
    FPS,
)


print("[OK] Camera opened.")


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv():

    if os.path.exists(CSV_FILE):
        return

    # We know PairFeatureExtractor.to_vector()
    # returns the feature vector in the correct order.
    #
    # The actual names are generated as feature_01...
    # feature_34 after the first valid extraction.

    return


initialize_csv()


# ============================================================
# DETECTION
# ============================================================

def detect_objects(frame):

    results = yolo_model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        imgsz=640,
        classes=[
            PERSON_CLASS,
            CHAIR_CLASS,
        ],
        device="cpu",
        verbose=False,
    )

    result = results[0]

    persons = []

    chairs = []

    if result.boxes is None:

        return persons, chairs


    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        detection = {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": confidence,
        }


        if class_id == PERSON_CLASS:

            persons.append(
                detection
            )


        elif class_id == CHAIR_CLASS:

            chairs.append(
                detection
            )


    return persons, chairs


# ============================================================
# DRAW FRAME
# ============================================================

def draw_frame(
    frame,
    persons,
    chairs,
):

    display = frame.copy()


    # --------------------------------------------------------
    # PERSONS
    # --------------------------------------------------------

    for i, person in enumerate(persons):

        x1 = int(person["x1"])
        y1 = int(person["y1"])
        x2 = int(person["x2"])
        y2 = int(person["y2"])

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"PERSON {i + 1} "
            f"{person['confidence']:.2f}",
            (
                x1,
                max(20, y1 - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )


    # --------------------------------------------------------
    # CHAIRS
    # --------------------------------------------------------

    for i, chair in enumerate(chairs):

        x1 = int(chair["x1"])
        y1 = int(chair["y1"])
        x2 = int(chair["x2"])
        y2 = int(chair["y2"])

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            display,
            f"CHAIR {i + 1} "
            f"{chair['confidence']:.2f}",
            (
                x1,
                max(20, y1 - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )


    # --------------------------------------------------------
    # INFO PANEL
    # --------------------------------------------------------

    cv2.rectangle(
        display,
        (0, 0),
        (640, 130),
        (0, 0, 0),
        -1,
    )


    distance = state["distance_cm"]

    if not distance:

        distance = "NOT SET"


    cv2.putText(
        display,
        f"Persons: {len(persons)}   "
        f"Chairs: {len(chairs)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


    cv2.putText(
        display,
        f"Distance: {distance} cm",
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


    cv2.putText(
        display,
        f"Pose: {state['pose']}",
        (10, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


    cv2.putText(
        display,
        f"Position: {state['position']}",
        (10, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


    cv2.putText(
        display,
        f"Samples: {sample_counter}",
        (10, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


    return display


# ============================================================
# HMR2
# ============================================================

def run_hmr(
    frame,
    person,
):

    bbox = [
        person["x1"],
        person["y1"],
        person["x2"],
        person["y2"],
    ]


    timestamp = datetime.now().timestamp()


    print()
    print("[HMR2] Processing person...")


    result = hmr_processor.process_frame(
        frame,
        frame_number,
        timestamp,
        [bbox],
    )


    if result is None:

        raise RuntimeError(
            "HMR2 returned None."
        )


    persons = result.get(
        "persons",
        [],
    )


    if len(persons) == 0:

        raise RuntimeError(
            "HMR2 returned zero persons."
        )


    hmr_person = persons[0]


    pose = hmr_person.get(
        "pose",
        {},
    )


    joints = pose.get(
        "joints_3d",
        [],
    )


    return {
        "raw": result,
        "joint_count": len(joints),
        "joints": joints,
        "camera_translation":
            hmr_person.get(
                "camera_translation"
            ),
    }


# ============================================================
# EXTRACT 3D FEATURES
# ============================================================

def extract_pair_features(
    depth_map,
    person,
    chair,
):

    person_bbox = BoundingBox(
        x1=person["x1"],
        y1=person["y1"],
        x2=person["x2"],
        y2=person["y2"],
    )


    chair_bbox = BoundingBox(
        x1=chair["x1"],
        y1=chair["y1"],
        x2=chair["x2"],
        y2=chair["y2"],
    )


    print(
        "[3D] Extracting person grid..."
    )


    person_grid = (
        object_grid_extractor.extract(
            depth_map,
            person_bbox,
            object_id=1,
            object_class="person",
        )
    )


    if person_grid is None:

        raise RuntimeError(
            "Could not create person 3D grid."
        )


    print(
        "[3D] Extracting chair grid..."
    )


    chair_grid = (
        object_grid_extractor.extract(
            depth_map,
            chair_bbox,
            object_id=1,
            object_class="chair",
        )
    )


    if chair_grid is None:

        raise RuntimeError(
            "Could not create chair 3D grid."
        )


    print(
        "[FEATURE] Extracting pair features..."
    )


    person_features = (
        feature_extractor.extract(
            person_grid
        )
    )


    chair_features = (
        feature_extractor.extract(
            chair_grid
        )
    )


    pair_features = (
        pair_feature_extractor.extract(
            person_grid,
            chair_grid,
        )
    )


    vector = (
        pair_feature_extractor.to_vector(
            pair_features
        )
    )


    vector = np.asarray(
        vector,
        dtype=np.float32,
    )


    if vector.ndim != 1:

        vector = vector.reshape(-1)


    if len(vector) == 0:

        raise RuntimeError(
            "Pair feature vector is empty."
        )


    if not np.all(
        np.isfinite(vector)
    ):

        raise RuntimeError(
            "Pair feature vector contains "
            "NaN or Inf."
        )


    return {
        "person_features":
            person_features,

        "chair_features":
            chair_features,

        "pair_features":
            pair_features,

        "vector":
            vector,

        "person_grid":
            person_grid,

        "chair_grid":
            chair_grid,
    }


# ============================================================
# SAVE DATASET ROW
# ============================================================

def save_dataset_row(
    sample_id,
    timestamp,
    distance,
    person,
    chair,
    hmr_result,
    feature_result,
    image_filename,
):

    vector = feature_result["vector"]


    # --------------------------------------------------------
    # CSV HEADER
    # --------------------------------------------------------

    feature_columns = [
        f"feature_{i + 1:02d}"
        for i in range(len(vector))
    ]


    columns = (
        BASE_COLUMNS
        + feature_columns
    )


    # --------------------------------------------------------
    # CREATE CSV IF NEEDED
    # --------------------------------------------------------

    file_exists = os.path.exists(
        CSV_FILE
    )


    row = {

        "sample_id":
            sample_id,

        "timestamp":
            timestamp,

        "distance_cm":
            distance,

        "person_id":
            1,

        "chair_id":
            1,

        "pose":
            state["pose"],

        "position":
            state["position"],

        "chair_type":
            state["chair_type"],

        "lighting":
            state["lighting"],

        "person_confidence":
            person["confidence"],

        "chair_confidence":
            chair["confidence"],

        "person_x1":
            person["x1"],

        "person_y1":
            person["y1"],

        "person_x2":
            person["x2"],

        "person_y2":
            person["y2"],

        "chair_x1":
            chair["x1"],

        "chair_y1":
            chair["y1"],

        "chair_x2":
            chair["x2"],

        "chair_y2":
            chair["y2"],

        "hmr_success":
            True,

        "hmr_joint_count":
            hmr_result["joint_count"],

        "image_path":
            f"images_v2/{image_filename}",
    }


    for i, value in enumerate(vector):

        row[
            f"feature_{i + 1:02d}"
        ] = float(value)


    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    with open(
        CSV_FILE,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )


        if not file_exists:

            writer.writeheader()


        writer.writerow(row)


    return len(vector)


# ============================================================
# SAVE SAMPLE
# ============================================================

def capture_sample():

    global sample_counter
    global processing_capture


    if processing_capture:

        return (
            False,
            "Another capture is already processing."
        )


    processing_capture = True


    try:

        # ----------------------------------------------------
        # COPY CURRENT FRAME
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:

                return (
                    False,
                    "Camera frame unavailable."
                )


            frame = latest_frame.copy()

            persons = list(
                latest_persons
            )

            chairs = list(
                latest_chairs
            )


        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        if len(persons) == 0:

            return (
                False,
                "No person detected."
            )


        if len(chairs) == 0:

            return (
                False,
                "No chair detected."
            )


        if state["distance_cm"] == "":

            return (
                False,
                "Enter ground-truth distance first."
            )


        try:

            distance = float(
                state["distance_cm"]
            )

        except ValueError:

            return (
                False,
                "Invalid distance."
            )


        if distance <= 0:

            return (
                False,
                "Distance must be greater than zero."
            )


        # ----------------------------------------------------
        # CURRENT V1 STRATEGY:
        # FIRST PERSON + FIRST CHAIR
        # ----------------------------------------------------

        person = persons[0]

        chair = chairs[0]


        print()
        print("=" * 70)
        print("PROCESSING DATASET SAMPLE")
        print("=" * 70)

        print(
            f"Ground truth distance: "
            f"{distance} cm"
        )


        # ----------------------------------------------------
        # HMR2
        # ----------------------------------------------------

        try:

            hmr_result = run_hmr(
                frame,
                person,
            )

            print(
                "[OK] HMR2 completed."
            )

            print(
                f"[OK] Joints: "
                f"{hmr_result['joint_count']}"
            )

        except Exception as exc:

            print(
                f"[ERROR] HMR2 failed: {exc}"
            )

            return (
                False,
                f"HMR2 failed: {exc}"
            )


        # ----------------------------------------------------
        # METRIC DEPTH
        # ----------------------------------------------------

        try:

            print(
                "[DEPTH] Estimating metric depth..."
            )


            depth_map = (
                depth_estimator.estimate(
                    frame
                )
            )


            if depth_map is None:

                raise RuntimeError(
                    "Depth map is None."
                )


            if depth_map.size == 0:

                raise RuntimeError(
                    "Depth map is empty."
                )


            if not np.all(
                np.isfinite(depth_map)
            ):

                raise RuntimeError(
                    "Depth map contains NaN/Inf."
                )


            print(
                "[OK] Metric depth completed."
            )


            print(
                f"[OK] Depth shape: "
                f"{depth_map.shape}"
            )


            print(
                f"[OK] Depth range: "
                f"{depth_map.min():.3f} - "
                f"{depth_map.max():.3f}"
            )


        except Exception as exc:

            print(
                f"[ERROR] Depth failed: {exc}"
            )

            return (
                False,
                f"Metric depth failed: {exc}"
            )


        # ----------------------------------------------------
        # 3D FEATURES
        # ----------------------------------------------------

        try:

            feature_result = (
                extract_pair_features(
                    depth_map,
                    person,
                    chair,
                )
            )


            vector = (
                feature_result["vector"]
            )


            print(
                "[OK] Pair features extracted."
            )


            print(
                f"[OK] Feature count: "
                f"{len(vector)}"
            )


        except Exception as exc:

            print(
                f"[ERROR] Feature extraction "
                f"failed: {exc}"
            )

            return (
                False,
                f"Feature extraction failed: {exc}"
            )


        # ----------------------------------------------------
        # SAMPLE ID
        # ----------------------------------------------------

        sample_counter += 1


        sample_id = (
            f"sample_{sample_counter:06d}"
        )


        timestamp = (
            datetime.now().isoformat()
        )


        image_filename = (
            f"{sample_id}.jpg"
        )


        image_path = os.path.join(
            IMAGE_DIR,
            image_filename,
        )


        # ----------------------------------------------------
        # SAVE ORIGINAL IMAGE
        # ----------------------------------------------------

        if not cv2.imwrite(
            image_path,
            frame,
        ):

            return (
                False,
                "Could not save image."
            )


        # ----------------------------------------------------
        # SAVE HMR JSON
        # ----------------------------------------------------

        hmr_filename = (
            f"{sample_id}_hmr.json"
        )


        hmr_path = os.path.join(
            DATA_DIR,
            hmr_filename,
        )


        with open(
            hmr_path,
            "w",
        ) as f:

            json.dump(
                hmr_result["raw"],
                f,
                indent=2,
            )


        # ----------------------------------------------------
        # SAVE DEPTH NUMPY FILE
        # ----------------------------------------------------

        depth_filename = (
            f"{sample_id}_depth.npy"
        )


        depth_path = os.path.join(
            DATA_DIR,
            depth_filename,
        )


        np.save(
            depth_path,
            depth_map,
        )


        # ----------------------------------------------------
        # SAVE CSV
        # ----------------------------------------------------

        feature_count = (
            save_dataset_row(
                sample_id,
                timestamp,
                distance,
                person,
                chair,
                hmr_result,
                feature_result,
                image_filename,
            )
        )


        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = {

            "sample_id":
                sample_id,

            "timestamp":
                timestamp,

            "ground_truth": {

                "distance_cm":
                    distance,

                "definition":
                    "Manual physical person-chair "
                    "distance label.",
            },


            "labels": {

                "pose":
                    state["pose"],

                "position":
                    state["position"],

                "chair_type":
                    state["chair_type"],

                "lighting":
                    state["lighting"],
            },


            "detections": {

                "person":
                    person,

                "chair":
                    chair,
            },


            "hmr": {

                "success":
                    True,

                "joint_count":
                    hmr_result[
                        "joint_count"
                    ],

                "camera_translation":
                    hmr_result[
                        "camera_translation"
                    ],
            },


            "features": {

                "count":
                    feature_count,

                "values":
                    vector.tolist(),
            },


            "files": {

                "image":
                    f"images_v2/{image_filename}",

                "hmr":
                    f"data_v2/{hmr_filename}",

                "depth":
                    f"data_v2/{depth_filename}",
            },
        }


        metadata_file = os.path.join(
            DATA_DIR,
            f"{sample_id}.json",
        )


        with open(
            metadata_file,
            "w",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
            )


        # ----------------------------------------------------
        # SESSION
        # ----------------------------------------------------

        session_data[
            "samples"
        ].append(
            metadata
        )


        with open(
            session_file,
            "w",
        ) as f:

            json.dump(
                session_data,
                f,
                indent=2,
            )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("[SAVED] DATASET SAMPLE")
        print("=" * 70)

        print(
            f"Sample       : {sample_id}"
        )

        print(
            f"Distance     : {distance} cm"
        )

        print(
            f"Pose         : {state['pose']}"
        )

        print(
            f"Position     : {state['position']}"
        )

        print(
            f"Chair        : {state['chair_type']}"
        )

        print(
            f"Lighting     : {state['lighting']}"
        )

        print(
            f"HMR joints   : "
            f"{hmr_result['joint_count']}"
        )

        print(
            f"ML features  : "
            f"{feature_count}"
        )

        print(
            f"Image        : {image_path}"
        )

        print("=" * 70)


        return (
            True,
            f"SAVED {sample_id} "
            f"with {feature_count} features."
        )


    finally:

        processing_capture = False


# ============================================================
# VIDEO GENERATOR
# ============================================================

def generate_frames():

    global latest_frame
    global latest_persons
    global latest_chairs
    global frame_number


    while True:

        success, frame = cap.read()


        if not success:

            continue


        frame_number += 1


        try:

            persons, chairs = (
                detect_objects(frame)
            )

        except Exception as exc:

            print(
                f"[WARNING] YOLO error: {exc}"
            )

            continue


        with frame_lock:

            latest_frame = frame.copy()

            latest_persons = persons

            latest_chairs = chairs


        display = draw_frame(
            frame,
            persons,
            chairs,
        )


        success, buffer = cv2.imencode(
            ".jpg",
            display,
        )


        if not success:

            continue


        frame_bytes = buffer.tobytes()


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>BAS-HMR Dataset Collector V2</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

body {
    margin: 0;
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 1150px;
    margin: auto;
    padding: 20px;
}

h1 {
    text-align: center;
}

.layout {
    display: flex;
    gap: 20px;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
}

.camera {
    width: 640px;
}

.camera img {
    width: 640px;
    max-width: 100%;
    border: 2px solid #444;
}

.controls {
    width: 300px;
    background: #1d1d1d;
    padding: 20px;
    border-radius: 10px;
}

label {
    display: block;
    margin-top: 12px;
    margin-bottom: 5px;
}

input,
select,
button {
    width: 100%;
    box-sizing: border-box;
    padding: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
    border: none;
}

button {
    margin-top: 15px;
    cursor: pointer;
    font-size: 17px;
}

.capture {
    font-weight: bold;
}

.status {
    margin-top: 15px;
    padding: 12px;
    background: #222;
    border-radius: 5px;
}

.warning {
    margin-top: 15px;
    padding: 10px;
    background: #332800;
    border-radius: 5px;
    font-size: 13px;
}

</style>

</head>


<body>

<div class="container">

<h1>BAS-HMR Dataset Collector V2</h1>


<div class="layout">


<div class="camera">

<img src="/video">

</div>


<div class="controls">

<h2>Sample Labels</h2>


<label>
Ground Truth Distance (cm)
</label>

<input
id="distance"
type="number"
min="1"
step="1"
placeholder="Example: 87"
>


<label>
Pose
</label>

<select id="pose">

<option>standing</option>
<option>sitting</option>
<option>walking</option>
<option>leaning</option>
<option>front_facing</option>
<option>back_facing</option>
<option>side_profile</option>

</select>


<label>
Person–Chair Position
</label>

<select id="position">

<option>front</option>
<option>left</option>
<option>right</option>
<option>diagonal</option>

</select>


<label>
Chair Type
</label>

<select id="chair_type">

<option>unknown</option>
<option>office</option>
<option>plastic</option>
<option>wooden</option>
<option>other</option>

</select>


<label>
Lighting
</label>

<select id="lighting">

<option>normal</option>
<option>low</option>
<option>bright</option>

</select>


<button
class="capture"
onclick="captureSample()"
>
CAPTURE SAMPLE
</button>


<div class="status" id="status">
Ready.
</div>


<div class="warning">

HMR2 and metric depth run only when
CAPTURE is pressed. Processing may take
several seconds on CPU.

</div>


</div>

</div>

</div>


<script>

async function captureSample() {

    const status =
        document.getElementById(
            "status"
        );

    status.innerText =
        "Processing HMR2 + depth + 3D features...";


    const distance =
        document.getElementById(
            "distance"
        ).value;


    const pose =
        document.getElementById(
            "pose"
        ).value;


    const position =
        document.getElementById(
            "position"
        ).value;


    const chair_type =
        document.getElementById(
            "chair_type"
        ).value;


    const lighting =
        document.getElementById(
            "lighting"
        ).value;


    if (!distance) {

        status.innerText =
            "Enter distance first.";

        return;
    }


    try {

        const response =
            await fetch(
                "/capture",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        distance_cm:
                            distance,

                        pose:
                            pose,

                        position:
                            position,

                        chair_type:
                            chair_type,

                        lighting:
                            lighting
                    })
                }
            );


        const data =
            await response.json();


        status.innerText =
            data.message;


    } catch (error) {

        status.innerText =
            "Capture request failed: "
            + error;

    }

}

</script>


</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame",
    )


@app.route(
    "/capture",
    methods=["POST"],
)
def capture():

    data = request.get_json()


    state["distance_cm"] = str(
        data.get(
            "distance_cm",
            "",
        )
    )


    state["pose"] = data.get(
        "pose",
        "standing",
    )


    state["position"] = data.get(
        "position",
        "front",
    )


    state["chair_type"] = data.get(
        "chair_type",
        "unknown",
    )


    state["lighting"] = data.get(
        "lighting",
        "normal",
    )


    success, message = (
        capture_sample()
    )


    return jsonify({

        "success":
            success,

        "message":
            message,

        "sample_count":
            sample_counter,
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)

    print(
        f"BAS-HMR Dataset Collector V2"
    )

    print(
        f"Open browser: "
        f"http://localhost:{PORT}"
    )

    print("=" * 70)


    try:

        app.run(
            host=HOST,
            port=PORT,
            threaded=True,
            debug=False,
        )

    finally:

        cap.release()

        print(
            "[OK] Camera released."
        )

        print(
            f"[OK] Samples collected: "
            f"{sample_counter}"
        )