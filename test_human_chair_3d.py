import time
import json
import threading
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from ultralytics import YOLO

from human.hmr.hmr_processor import HMRProcessor
from human.distance.grid.depth_to_3d import (
    DepthTo3D,
    CameraIntrinsics,
)
from human.distance.grid.object_grid import (
    ObjectGridExtractor,
    BoundingBox,
)
from human.distance.grid.feature_extractor import FeatureExtractor


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_DEVICE = "/dev/video0"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

YOLO_PERSON_CONF = 0.25
YOLO_CHAIR_CONF = 0.25

YOLO_PERSON_CLASS = 0
YOLO_CHAIR_CLASS = 56

YOLO_PERSON_MODEL = "yolo11n.pt"
YOLO_DEPTH_MODEL = "yolo26n-depth.pt"

FLASK_PORT = 5016

# Development intrinsics already used in the project.
# These are NOT checkerboard-calibrated values.
INTRINSICS = CameraIntrinsics(
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
)


# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

latest_result = {
    "status": "starting",
    "frame_id": 0,
    "persons": [],
    "chairs": [],
    "pairs": [],
}

latest_frame = None
frame_lock = threading.Lock()

running = True


# ============================================================
# MODELS
# ============================================================

print("=" * 80)
print("BAS-HMR HUMAN + CHAIR 3D INTEGRATION TEST")
print("=" * 80)

print("\n[1/5] Loading YOLO11n...")

person_model = YOLO(YOLO_PERSON_MODEL)

print("[OK] YOLO11n loaded.")
print("[OK] Device: CPU")


print("\n[2/5] Loading YOLO26 Depth model...")

depth_model = YOLO(YOLO_DEPTH_MODEL)

print("[OK] YOLO26 Depth loaded.")
print("[OK] Device: CPU")


print("\n[3/5] Loading HMR2...")

hmr = HMRProcessor()

print("[OK] HMR2 loaded.")
print("[OK] Device: CPU")


print("\n[4/5] Creating DepthTo3D...")

depth_converter = DepthTo3D(
    intrinsics=INTRINSICS,
    min_depth=0.05,
    max_depth=20.0,
)

print("[OK] DepthTo3D created.")


print("\n[5/5] Creating 3D grid extractors...")

grid_extractor = ObjectGridExtractor(
    depth_converter=depth_converter,
    rows=5,
    cols=5,
    border_ratio=0.1,
    depth_radius=2,
)

feature_extractor = FeatureExtractor()

print("[OK] ObjectGridExtractor created.")
print("[OK] FeatureExtractor created.")


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    print(f"\n[CAMERA] Opening: {CAMERA_DEVICE}")

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return None

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"[CAMERA] Actual: "
        f"{actual_width} x {actual_height} "
        f"@ {actual_fps:.1f} FPS"
    )

    # Warm-up
    print("[CAMERA] Warming up...")

    for attempt in range(10):

        ret, frame = cap.read()

        if ret and frame is not None:

            print(
                f"[CAMERA] Warm-up successful "
                f"on attempt {attempt + 1}"
            )

            print(
                f"[CAMERA] Frame shape: {frame.shape}"
            )

            return cap

        time.sleep(0.2)

    print("[ERROR] Camera warm-up failed.")

    cap.release()

    return None


# ============================================================
# HMR JOINT CONVERSION
# ============================================================

def joints_to_numpy(joints):

    points = []

    for joint in joints:

        if not isinstance(joint, dict):
            continue

        try:

            x = float(joint["x"])
            y = float(joint["y"])
            z = float(joint["z"])

            points.append([x, y, z])

        except Exception:
            continue

    if not points:
        return None

    return np.asarray(
        points,
        dtype=np.float32
    )


# ============================================================
# HUMAN REFERENCE POINTS
# ============================================================

def calculate_human_reference(joints_xyz):

    if joints_xyz is None or len(joints_xyz) == 0:
        return None

    centroid = np.mean(
        joints_xyz,
        axis=0
    )

    minimum = np.min(
        joints_xyz,
        axis=0
    )

    maximum = np.max(
        joints_xyz,
        axis=0
    )

    # Lowest and highest based on Y.
    lowest_index = np.argmax(
        joints_xyz[:, 1]
    )

    highest_index = np.argmin(
        joints_xyz[:, 1]
    )

    lowest = joints_xyz[
        lowest_index
    ]

    highest = joints_xyz[
        highest_index
    ]

    return {
        "centroid": centroid.tolist(),
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "lowest_joint": lowest.tolist(),
        "highest_joint": highest.tolist(),
        "joint_count": int(len(joints_xyz)),
    }


# ============================================================
# GRID REFERENCE POINT
# ============================================================

def calculate_grid_reference(grid):

    if grid is None:
        return None

    # Try to obtain the 3D points from the Object3DGrid.
    points = None

    candidate_names = [
        "points_3d",
        "points",
        "xyz_points",
        "point_cloud",
    ]

    for name in candidate_names:

        if hasattr(grid, name):

            candidate = getattr(
                grid,
                name
            )

            if candidate is not None:

                try:

                    arr = np.asarray(
                        candidate,
                        dtype=np.float32
                    )

                    if (
                        arr.ndim == 2
                        and arr.shape[1] == 3
                    ):
                        points = arr
                        break

                except Exception:
                    pass

    if points is None:

        # Some implementations expose grid data
        # through nested cell objects.
        for name in [
            "grid",
            "cells",
            "points_3d_grid",
        ]:

            if not hasattr(grid, name):
                continue

            candidate = getattr(
                grid,
                name
            )

            try:

                arr = np.asarray(
                    candidate,
                    dtype=np.float32
                )

                if (
                    arr.ndim == 2
                    and arr.shape[1] == 3
                ):

                    points = arr
                    break

                if (
                    arr.ndim == 3
                    and arr.shape[-1] == 3
                ):

                    points = arr.reshape(
                        -1,
                        3
                    )

                    break

            except Exception:
                pass

    if points is None:

        return {
            "available": False,
            "reason": (
                "Object3DGrid 3D points "
                "could not be extracted "
                "by the diagnostic adapter."
            ),
        }

    valid = np.isfinite(
        points
    ).all(axis=1)

    points = points[valid]

    if len(points) == 0:

        return {
            "available": False,
            "reason": "No valid 3D grid points.",
        }

    centroid = np.mean(
        points,
        axis=0
    )

    minimum = np.min(
        points,
        axis=0
    )

    maximum = np.max(
        points,
        axis=0
    )

    return {
        "available": True,
        "point_count": int(len(points)),
        "centroid": centroid.tolist(),
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
    }


# ============================================================
# SAFE 3D DISTANCE
# ============================================================

def euclidean_distance(a, b):

    if a is None or b is None:
        return None

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    if a.shape != (3,) or b.shape != (3,):
        return None

    distance = np.linalg.norm(
        a - b
    )

    return float(distance)


# ============================================================
# DEPTH EXTRACTION
# ============================================================

def extract_depth_map(frame):

    results = depth_model.predict(
        source=frame,
        verbose=False,
        device="cpu",
    )

    if not results:
        return None

    result = results[0]

    depth_map = None

    # Try common depth attributes.
    if hasattr(result, "depth"):

        depth = result.depth

        if depth is not None:

            try:

                if hasattr(depth, "data"):
                    depth = depth.data

                depth_map = np.asarray(
                    depth,
                    dtype=np.float32
                )

            except Exception:
                pass

    # Ultralytics depth output may be exposed
    # through result.depth.data.
    if depth_map is None:

        try:

            depth_map = np.asarray(
                result.depth.data,
                dtype=np.float32
            )

        except Exception:
            pass

    if depth_map is None:
        return None

    depth_map = np.squeeze(
        depth_map
    )

    if depth_map.ndim != 2:
        return None

    return depth_map


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

def processing_loop():

    global latest_frame
    global latest_result
    global running

    cap = open_camera()

    if cap is None:

        latest_result = {
            "status": "camera_error"
        }

        return

    frame_id = 0

    consecutive_failures = 0

    print("\n[OK] Camera ready.")

    print("\nOpen browser:")
    print(
        f"http://localhost:{FLASK_PORT}"
    )

    print("\nRaw API:")
    print(
        f"http://localhost:{FLASK_PORT}/api/3d"
    )

    print("\nHealth:")
    print(
        f"http://localhost:{FLASK_PORT}/api/health"
    )

    print("\nPress CTRL+C to stop.\n")

    while running:

        ret, frame = cap.read()

        if not ret or frame is None:

            consecutive_failures += 1

            print(
                f"[CAMERA] Read failure "
                f"{consecutive_failures}/5"
            )

            if consecutive_failures >= 5:

                print(
                    "[CAMERA] Reopening..."
                )

                try:
                    cap.release()
                except Exception:
                    pass

                time.sleep(1)

                cap = open_camera()

                if cap is None:
                    time.sleep(2)
                    continue

                consecutive_failures = 0

            continue

        consecutive_failures = 0

        frame_id += 1

        # ----------------------------------------------------
        # YOLO PERSON + CHAIR
        # ----------------------------------------------------

        yolo_results = person_model.predict(
            source=frame,
            classes=[
                YOLO_PERSON_CLASS,
                YOLO_CHAIR_CLASS,
            ],
            conf=YOLO_PERSON_CONF,
            imgsz=640,
            verbose=False,
            device="cpu",
        )

        if not yolo_results:
            continue

        yolo_result = yolo_results[0]

        persons = []
        chairs = []

        if yolo_result.boxes is not None:

            boxes = yolo_result.boxes

            for i in range(len(boxes)):

                cls = int(
                    boxes.cls[i].item()
                )

                conf = float(
                    boxes.conf[i].item()
                )

                xyxy = boxes.xyxy[
                    i
                ].cpu().numpy()

                x1, y1, x2, y2 = map(
                    float,
                    xyxy
                )

                detection = {
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "confidence": conf,
                    "class_id": cls,
                }

                if cls == YOLO_PERSON_CLASS:

                    persons.append(
                        detection
                    )

                elif cls == YOLO_CHAIR_CLASS:

                    chairs.append(
                        detection
                    )

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------

        depth_map = None

        if chairs:

            try:

                depth_map = extract_depth_map(
                    frame
                )

            except Exception as exc:

                print(
                    "[DEPTH ERROR]",
                    exc
                )

        # ----------------------------------------------------
        # CHAIR 3D
        # ----------------------------------------------------

        chair_outputs = []

        if depth_map is not None:

            for chair_index, chair in enumerate(chairs):

                b = chair["bbox"]

                bbox = BoundingBox(
                    x1=b["x1"],
                    y1=b["y1"],
                    x2=b["x2"],
                    y2=b["y2"],
                )

                try:

                    grid = grid_extractor.extract(
                        depth_map,
                        bbox,
                        object_id=chair_index,
                        object_class="chair",
                    )

                except Exception as exc:

                    print(
                        "[GRID ERROR]",
                        exc
                    )

                    grid = None

                reference = (
                    calculate_grid_reference(
                        grid
                    )
                )

                features = None

                if grid is not None:

                    try:

                        features = (
                            feature_extractor.extract(
                                grid
                            )
                        )

                    except Exception as exc:

                        print(
                            "[FEATURE ERROR]",
                            exc
                        )

                chair_outputs.append({
                    "chair_id": chair_index,
                    "bbox": chair["bbox"],
                    "confidence": chair["confidence"],
                    "grid_available": grid is not None,
                    "grid_reference": reference,
                    "feature_count": (
                        len(features)
                        if features is not None
                        else 0
                    ),
                })

        # ----------------------------------------------------
        # HMR2 HUMAN 3D
        # ----------------------------------------------------

        hmr_outputs = []

        if persons:

            numeric_boxes = []

            for person in persons:

                b = person["bbox"]

                numeric_boxes.append([
                    b["x1"],
                    b["y1"],
                    b["x2"],
                    b["y2"],
                ])

            print(
                f"\n[FRAME {frame_id}] "
                f"persons={len(persons)} "
                f"chairs={len(chairs)}"
            )

            try:

                hmr_result = hmr.process_frame(
                    frame,
                    frame_id,
                    time.time(),
                    np.asarray(
                        numeric_boxes,
                        dtype=np.float32
                    ),
                )

            except Exception as exc:

                print(
                    "[HMR ERROR]",
                    exc
                )

                hmr_result = {
                    "persons": []
                }

            returned_persons = (
                hmr_result.get(
                    "persons",
                    []
                )
            )

            for person_data in returned_persons:

                pose = person_data.get(
                    "pose",
                    {}
                )

                joints = pose.get(
                    "joints_3d",
                    []
                )

                joints_xyz = joints_to_numpy(
                    joints
                )

                human_reference = (
                    calculate_human_reference(
                        joints_xyz
                    )
                )

                camera_translation = (
                    person_data.get(
                        "camera_translation"
                    )
                )

                camera_frame_centroid = None

                if (
                    joints_xyz is not None
                    and camera_translation is not None
                ):

                    translation = np.array([
                        float(
                            camera_translation["x"]
                        ),
                        float(
                            camera_translation["y"]
                        ),
                        float(
                            camera_translation["z"]
                        ),
                    ], dtype=np.float32)

                    camera_frame_joints = (
                        joints_xyz + translation
                    )

                    camera_frame_centroid = (
                        np.mean(
                            camera_frame_joints,
                            axis=0
                        ).tolist()
                    )

                hmr_outputs.append({

                    "person_id": person_data.get(
                        "person_id"
                    ),

                    "bbox": person_data.get(
                        "bbox"
                    ),

                    "joint_count": (
                        len(joints_xyz)
                        if joints_xyz is not None
                        else 0
                    ),

                    "camera_translation":
                        camera_translation,

                    "human_reference":
                        human_reference,

                    "camera_frame_centroid":
                        camera_frame_centroid,
                })

        # ----------------------------------------------------
        # HUMAN ↔ CHAIR GEOMETRY
        # ----------------------------------------------------

        pairs = []

        for person_data in hmr_outputs:

            human_reference = (
                person_data.get(
                    "human_reference"
                )
            )

            if not human_reference:
                continue

            human_centroid = (
                human_reference[
                    "centroid"
                ]
            )

            for chair_data in chair_outputs:

                chair_reference = (
                    chair_data.get(
                        "grid_reference"
                    )
                )

                if not chair_reference:
                    continue

                if not chair_reference.get(
                    "available",
                    False
                ):
                    continue

                chair_centroid = (
                    chair_reference[
                        "centroid"
                    ]
                )

                # IMPORTANT:
                # This is only a diagnostic subtraction.
                # HMR coordinates and depth coordinates
                # have NOT yet been proven to share the
                # same metric coordinate system.

                delta = (
                    np.asarray(
                        human_centroid,
                        dtype=np.float32
                    )
                    -
                    np.asarray(
                        chair_centroid,
                        dtype=np.float32
                    )
                )

                separation = float(
                    np.linalg.norm(
                        delta
                    )
                )

                pairs.append({

                    "person_id":
                        person_data.get(
                            "person_id"
                        ),

                    "chair_id":
                        chair_data.get(
                            "chair_id"
                        ),

                    "diagnostic_delta":
                        delta.tolist(),

                    "diagnostic_3d_separation":
                        separation,

                    "WARNING":
                        "Not yet metric physical distance. "
                        "Coordinate alignment must be validated."
                })

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        latest_result = {

            "status": "running",

            "frame_id": frame_id,

            "persons": hmr_outputs,

            "chairs": chair_outputs,

            "pairs": pairs,

        }

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        display = frame.copy()

        # Draw YOLO detections.
        for person in persons:

            b = person["bbox"]

            cv2.rectangle(
                display,
                (
                    int(b["x1"]),
                    int(b["y1"])
                ),
                (
                    int(b["x2"]),
                    int(b["y2"])
                ),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                "PERSON",
                (
                    int(b["x1"]),
                    max(
                        20,
                        int(b["y1"]) - 8
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        for chair in chairs:

            b = chair["bbox"]

            cv2.rectangle(
                display,
                (
                    int(b["x1"]),
                    int(b["y1"])
                ),
                (
                    int(b["x2"]),
                    int(b["y2"])
                ),
                (255, 180, 0),
                2,
            )

            cv2.putText(
                display,
                "CHAIR",
                (
                    int(b["x1"]),
                    max(
                        20,
                        int(b["y1"]) - 8
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2,
            )

        cv2.putText(
            display,
            f"Frame: {frame_id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"Persons: {len(persons)}  "
            f"Chairs: {len(chairs)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        with frame_lock:

            latest_frame = display.copy()


# ============================================================
# FLASK
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BAS-HMR 3D Integration Test</title>

        <style>
            body {
                background: #111;
                color: #eee;
                font-family: Arial, sans-serif;
                margin: 20px;
            }

            img {
                width: 640px;
                max-width: 100%;
                border: 2px solid #555;
            }

            pre {
                background: #222;
                padding: 15px;
                overflow-x: auto;
            }
        </style>
    </head>

    <body>

        <h1>BAS-HMR Human + Chair 3D Integration</h1>

        <img src="/video">

        <h2>Latest 3D Result</h2>

        <pre id="result">
Loading...
        </pre>

        <script>

        async function update() {

            try {

                const response =
                    await fetch("/api/3d");

                const data =
                    await response.json();

                document.getElementById(
                    "result"
                ).textContent =
                    JSON.stringify(
                        data,
                        null,
                        2
                    );

            } catch (error) {

                document.getElementById(
                    "result"
                ).textContent =
                    error.toString();

            }
        }

        setInterval(
            update,
            2000
        );

        update();

        </script>

    </body>
    </html>
    """


@app.route("/api/3d")
def api_3d():

    return jsonify(
        latest_result
    )


@app.route("/api/health")
def health():

    return jsonify({
        "status": latest_result.get(
            "status",
            "unknown"
        ),
        "frame_id": latest_result.get(
            "frame_id",
            0
        ),
    })


def generate_video():

    global latest_frame

    while True:

        with frame_lock:

            frame = (
                None
                if latest_frame is None
                else latest_frame.copy()
            )

        if frame is None:

            time.sleep(0.05)

            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                80,
            ],
        )

        if not success:

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


@app.route("/video")
def video():

    return Response(
        generate_video(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    processing_thread = threading.Thread(
        target=processing_loop,
        daemon=True,
    )

    processing_thread.start()

    try:

        app.run(
            host="0.0.0.0",
            port=FLASK_PORT,
            debug=False,
            threaded=True,
        )

    except KeyboardInterrupt:

        running = False

        print(
            "\n[STOP] CTRL+C received."
        )

    finally:

        running = False

        print(
            "[OK] BAS-HMR integration test stopped."
        )
