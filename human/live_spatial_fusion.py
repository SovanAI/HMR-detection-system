"""
BAS LIVE SPATIAL FUSION

Pipeline:

Camera
    ↓
YOLO11 Person + Chair Detection
    ↓
Chair Tracking
    ↓
Depth Anything V2
    ↓
Latest-frame HMR2
    ↓
TrackedChair → Fusion Dictionary Adapter
    ↓
Person ↔ Chair Spatial Fusion
    ↓
Temporal Spatial Tracking
    ↓
Flask MJPEG Viewer

CPU implementation.
"""


import os
import time
import threading

import cv2
import numpy as np
import torch

from PIL import Image
from flask import Flask, Response, jsonify
from transformers import pipeline
from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FILES
# ============================================================

YOLO_MODEL = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt",
)

CALIBRATION_FILE = os.path.join(
    PROJECT_ROOT,
    "test_results",
    "depth_hmr_calibration",
    "calibration_samples.json",
)


# ============================================================
# CAMERA
# ============================================================

CAMERA_DEVICE = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10


# ============================================================
# YOLO
# ============================================================

PERSON_CLASS = 0
CHAIR_CLASS = 56

YOLO_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 416


# ============================================================
# DEPTH
# ============================================================

DEPTH_MODEL = (
    "depth-anything/Depth-Anything-V2-Small-hf"
)

CHAIR_DEPTH_CROP_RATIO = 0.60


# ============================================================
# SPATIAL
# ============================================================

NEAR_THRESHOLD = 2.0


# ============================================================
# TEMPORAL
# ============================================================

MOVEMENT_THRESHOLD = 0.05
APPROACHING_THRESHOLD = -0.05
MOVING_AWAY_THRESHOLD = 0.05
SMOOTHING_ALPHA = 0.5


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from hmr.hmr_processor import HMRProcessor

from human.chair_tracker import (
    ChairTracker,
    ChairDetection,
)

from human.person_chair_spatial_fusion import (
    PersonChairFusion,
    PersonChairSpatialEngine,
    DepthHMRCalibration,
)

from human.spatial_temporal_tracker import (
    SpatialTemporalTracker,
)


# ============================================================
# GLOBAL STATE
# ============================================================

running = True


latest_frame = None
latest_frame_id = 0

latest_display_frame = None

latest_hmr_result = None
latest_fused_result = None


frame_lock = threading.Lock()
hmr_lock = threading.Lock()
display_lock = threading.Lock()
fusion_lock = threading.Lock()


# ============================================================
# MODEL GLOBALS
# ============================================================

yolo_model = None
hmr_processor = None


# ============================================================
# CAMERA READER
# ============================================================

def camera_reader():

    global latest_frame
    global latest_frame_id
    global running

    print()
    print("=" * 70)
    print("OPENING CAMERA")
    print("=" * 70)

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
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

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        running = False

        return

    print(
        "Camera opened successfully."
    )

    while running:

        success, frame = cap.read()

        if not success:

            time.sleep(0.05)

            continue

        with frame_lock:

            latest_frame = frame.copy()

            latest_frame_id += 1

    cap.release()

    print(
        "Camera released."
    )


# ============================================================
# LATEST FRAME
# ============================================================

def get_latest_frame():

    with frame_lock:

        if latest_frame is None:

            return None, None

        return (
            latest_frame.copy(),
            latest_frame_id,
        )


# ============================================================
# HMR RESULT
# ============================================================

def get_hmr_result():

    with hmr_lock:

        if latest_hmr_result is None:

            return None

        return latest_hmr_result


# ============================================================
# FUSED RESULT
# ============================================================

def get_fused_result():

    with fusion_lock:

        if latest_fused_result is None:

            return None

        return latest_fused_result


# ============================================================
# DISPLAY FRAME
# ============================================================

def get_display_frame():

    with display_lock:

        if latest_display_frame is None:

            frame, _ = (
                get_latest_frame()
            )

            return frame

        return latest_display_frame.copy()


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration():

    print()
    print("=" * 70)
    print("LOADING DEPTH ↔ HMR CALIBRATION")
    print("=" * 70)

    try:

        calibration = (
            DepthHMRCalibration.from_file(
                CALIBRATION_FILE
            )
        )

        print(
            "Calibration loaded successfully."
        )

        return calibration

    except Exception as exc:

        print(
            "WARNING: Could not load calibration:"
        )

        print(
            repr(exc)
        )

        print(
            "Using fallback calibration."
        )

        return DepthHMRCalibration(
            scale=-40.219,
            offset=49.882,
        )


# ============================================================
# DEPTH MODEL
# ============================================================

def load_depth_model():

    print()
    print("=" * 70)
    print("LOADING DEPTH ANYTHING V2")
    print("=" * 70)

    estimator = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=-1,
    )

    print(
        "Depth Anything loaded."
    )

    return estimator


# ============================================================
# DEPTH MAP
# ============================================================

def compute_depth_map(
    depth_estimator,
    frame,
):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(
        rgb
    )

    result = depth_estimator(
        image
    )

    depth = result["depth"]

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    depth = cv2.resize(
        depth,
        (
            frame.shape[1],
            frame.shape[0],
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    return depth


# ============================================================
# NORMALIZE DEPTH
# ============================================================

def normalize_depth(depth):

    min_value = float(
        np.min(depth)
    )

    max_value = float(
        np.max(depth)
    )

    if (
        max_value - min_value
        < 1e-8
    ):

        return np.zeros_like(
            depth,
            dtype=np.float32,
        )

    normalized = (
        depth - min_value
    ) / (
        max_value - min_value
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# CHAIR DEPTH
# ============================================================

def get_chair_depth(
    normalized_depth,
    bbox,
):

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        int(x1),
    )

    y1 = max(
        0,
        int(y1),
    )

    x2 = min(
        normalized_depth.shape[1],
        int(x2),
    )

    y2 = min(
        normalized_depth.shape[0],
        int(y2),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return 0.5

    width = x2 - x1
    height = y2 - y1

    margin_x = int(
        width
        * (
            1.0
            - CHAIR_DEPTH_CROP_RATIO
        )
        / 2.0
    )

    margin_y = int(
        height
        * (
            1.0
            - CHAIR_DEPTH_CROP_RATIO
        )
        / 2.0
    )

    cx1 = x1 + margin_x
    cy1 = y1 + margin_y

    cx2 = x2 - margin_x
    cy2 = y2 - margin_y

    crop = normalized_depth[
        cy1:cy2,
        cx1:cx2,
    ]

    if crop.size == 0:

        return 0.5

    return float(
        np.median(crop)
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_objects(
    frame,
):

    results = yolo_model.predict(
        source=frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        classes=[
            PERSON_CLASS,
            CHAIR_CLASS,
        ],
        device="cpu",
        verbose=False,
    )

    persons = []
    chairs = []

    if not results:

        return persons, chairs

    result = results[0]

    if result.boxes is None:

        return persons, chairs

    for box in result.boxes:

        cls = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = map(
            float,
            xyxy,
        )

        bbox = (
            x1,
            y1,
            x2,
            y2,
        )

        if cls == PERSON_CLASS:

            persons.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

        elif cls == CHAIR_CLASS:

            chairs.append(
                ChairDetection(
                    bbox=bbox,
                    confidence=confidence,
                    depth=0.5,
                )
            )

    return persons, chairs


# ============================================================
# HMR PERSON DETECTION
# ============================================================

def detect_person_boxes(
    frame,
):

    results = yolo_model.predict(
        source=frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        classes=[PERSON_CLASS],
        device="cpu",
        verbose=False,
    )

    boxes = []

    if not results:

        return boxes

    result = results[0]

    if result.boxes is None:

        return boxes

    for box in result.boxes:

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = map(
            float,
            xyxy,
        )

        boxes.append(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

    return boxes


# ============================================================
# HMR WORKER
# ============================================================

def hmr_worker():

    global latest_hmr_result
    global running

    last_frame_id = -1

    print()
    print("=" * 70)
    print("HMR WORKER STARTED")
    print("=" * 70)

    while running:

        frame, frame_id = (
            get_latest_frame()
        )

        if frame is None:

            time.sleep(0.05)

            continue

        if frame_id == last_frame_id:

            time.sleep(0.02)

            continue

        last_frame_id = frame_id

        try:

            person_boxes = (
                detect_person_boxes(
                    frame
                )
            )

            print(
                f"[HMR] Processing frame "
                f"{frame_id} "
                f"| persons="
                f"{len(person_boxes)}"
            )

            if not person_boxes:

                result = {
                    "frame_id": frame_id,
                    "timestamp": time.time(),
                    "model": "HMR2",
                    "device": "cpu",
                    "persons": [],
                }

            else:

                result = (
                    hmr_processor.process_frame(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=time.time(),
                        boxes=person_boxes,
                    )
                )

            with hmr_lock:

                latest_hmr_result = result

        except Exception as exc:

            print(
                "[HMR] ERROR:",
                repr(exc),
            )

            time.sleep(0.1)


# ============================================================
# CRITICAL ADAPTER
# ============================================================

def tracked_chair_to_fusion_dict(
    chair,
):
    """
    Convert the ChairTracker's TrackedChair
    object into the dictionary format required
    by ChairSpatialExtractor.

    Tracker format:

        chair.bbox
        chair.depth
        chair.chair_id
        chair.confidence

    Fusion format:

        {
            "chair_id": ...,
            "bbox": {
                "x1": ...,
                "y1": ...,
                "x2": ...,
                "y2": ...
            },
            "relative_depth": ...,
            "confidence": ...
        }
    """

    x1, y1, x2, y2 = (
        chair.bbox
    )

    return {
        "chair_id": int(
            chair.chair_id
        ),

        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },

        "relative_depth": float(
            chair.depth
        ),

        "confidence": float(
            chair.confidence
        ),
    }


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.50,
    thickness=1,
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# PROCESSING LOOP
# ============================================================

def processing_loop():

    global latest_display_frame
    global latest_fused_result
    global running

    print()
    print("=" * 70)
    print("STARTING PROCESSING LOOP")
    print("=" * 70)

    chair_tracker = ChairTracker()

    temporal_tracker = (
        SpatialTemporalTracker(
            movement_threshold=(
                MOVEMENT_THRESHOLD
            ),

            approaching_threshold=(
                APPROACHING_THRESHOLD
            ),

            moving_away_threshold=(
                MOVING_AWAY_THRESHOLD
            ),

            smoothing_alpha=(
                SMOOTHING_ALPHA
            ),
        )
    )

    depth_estimator = (
        load_depth_model()
    )

    calibration = (
        load_calibration()
    )

    print(
        f"HMR_Z = "
        f"{calibration.scale:.3f} * "
        f"RelativeDepth + "
        f"{calibration.offset:.3f}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only pass arguments supported by the actual
    # PersonChairSpatialEngine constructor.
    # --------------------------------------------------------

    spatial_engine = (
        PersonChairSpatialEngine(
            near_threshold=NEAR_THRESHOLD,
        )
    )

    fusion = PersonChairFusion(
        calibration=calibration,
        spatial_engine=spatial_engine,
    )

    last_frame_id = -1

    while running:

        frame, frame_id = (
            get_latest_frame()
        )

        if frame is None:

            time.sleep(0.05)

            continue

        if frame_id == last_frame_id:

            time.sleep(0.02)

            continue

        last_frame_id = frame_id

        display = frame.copy()

        try:

            # =================================================
            # YOLO
            # =================================================

            persons, chairs = (
                detect_objects(
                    frame
                )
            )

            # =================================================
            # DEPTH
            # =================================================

            depth = (
                compute_depth_map(
                    depth_estimator,
                    frame,
                )
            )

            normalized_depth = (
                normalize_depth(
                    depth
                )
            )

            # =================================================
            # CHAIR DEPTH
            # =================================================

            for chair in chairs:

                chair.depth = (
                    get_chair_depth(
                        normalized_depth,
                        chair.bbox,
                    )
                )

            # =================================================
            # CHAIR TRACKING
            # =================================================

            tracked_chairs = (
                chair_tracker.update(
                    chairs
                )
            )

            # =================================================
            # DRAW PERSONS
            # =================================================

            for person in persons:

                x1, y1, x2, y2 = map(
                    int,
                    person["bbox"],
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                draw_text(
                    display,
                    (
                        f"PERSON "
                        f"{person['confidence']:.2f}"
                    ),
                    (
                        x1,
                        max(
                            20,
                            y1 - 5,
                        ),
                    ),
                )

            # =================================================
            # DRAW CHAIRS
            # =================================================

            for chair in tracked_chairs:

                x1, y1, x2, y2 = map(
                    int,
                    chair.bbox,
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

                cv2.circle(
                    display,
                    (
                        center_x,
                        center_y,
                    ),
                    5,
                    (255, 0, 0),
                    -1,
                )

                draw_text(
                    display,
                    (
                        f"CHAIR "
                        f"{chair.chair_id} "
                        f"D="
                        f"{chair.depth:.3f}"
                    ),
                    (
                        x1,
                        max(
                            20,
                            y1 - 5,
                        ),
                    ),
                )

            # =================================================
            # HMR
            # =================================================

            hmr_result = (
                get_hmr_result()
            )

            if (
                hmr_result is not None
                and tracked_chairs
            ):

                hmr_persons = (
                    hmr_result.get(
                        "persons",
                        [],
                    )
                )

                if hmr_persons:

                    # =========================================
                    # ADAPT TRACKED CHAIRS
                    # =========================================

                    fusion_chairs = []

                    for chair in (
                        tracked_chairs
                    ):

                        fusion_chair = (
                            tracked_chair_to_fusion_dict(
                                chair
                            )
                        )

                        fusion_chairs.append(
                            fusion_chair
                        )

                    # =========================================
                    # SPATIAL FUSION
                    # =========================================

                    try:

                        fusion_output = (
                            fusion.fuse(
                                hmr_persons=(
                                    hmr_persons
                                ),

                                chairs=(
                                    fusion_chairs
                                ),

                                frame_width=WIDTH,

                                frame_height=HEIGHT,
                            )
                        )

                        # =====================================
                        # PersonChairFusion returns:
                        #
                        # (
                        #     persons,
                        #     chair_objects,
                        #     relationships
                        # )
                        # =====================================

                        fused_persons = (
                            fusion_output[0]
                        )

                        fused_chairs = (
                            fusion_output[1]
                        )

                        relationships = (
                            fusion_output[2]
                        )

                        # =====================================
                        # TEMPORAL TRACKING
                        # =====================================

                        temporal_states = []

                        for relationship in (
                            relationships
                        ):

                            try:

                                state = (
                                    temporal_tracker.update(
                                        person_id=(
                                            relationship.person_id
                                        ),

                                        chair_id=(
                                            relationship.chair_id
                                        ),

                                        distance=(
                                            relationship.distance
                                        ),

                                        relationship=(
                                            relationship.relationship
                                        ),

                                        timestamp=time.time(),
                                    )
                                )

                                temporal_states.append(
                                    state
                                )

                            except Exception as exc:

                                print(
                                    "[TEMPORAL] ERROR:",
                                    repr(exc),
                                )

                        # =====================================
                        # JSON RESULT
                        # =====================================

                        relationship_json = []

                        for relationship in (
                            relationships
                        ):

                            relationship_json.append(
                                {
                                    "person_id": int(
                                        relationship.person_id
                                    ),

                                    "chair_id": int(
                                        relationship.chair_id
                                    ),

                                    "distance": float(
                                        relationship.distance
                                    ),

                                    "relationship": (
                                        relationship.relationship
                                    ),
                                }
                            )

                        result_json = {
                            "frame_id": int(
                                frame_id
                            ),

                            "hmr_frame_id": int(
                                hmr_result.get(
                                    "frame_id",
                                    -1,
                                )
                            ),

                            "persons": len(
                                fused_persons
                            ),

                            "chairs": len(
                                fused_chairs
                            ),

                            "relationships": (
                                relationship_json
                            ),
                        }

                        with fusion_lock:

                            latest_fused_result = (
                                result_json
                            )

                        # =====================================
                        # DRAW SPATIAL RELATIONSHIPS
                        # =====================================

                        text_y = 30

                        for state in (
                            temporal_states
                        ):

                            text = (
                                f"P{state.person_id}"
                                f" -> "
                                f"C{state.chair_id}"
                                f"  "
                                f"D="
                                f"{state.current_distance:.2f}"
                                f"  "
                                f"{state.relationship}"
                                f"  "
                                f"{state.motion_state}"
                            )

                            draw_text(
                                display,
                                text,
                                (
                                    10,
                                    text_y,
                                ),
                                scale=0.48,
                                thickness=1,
                            )

                            text_y += 22

                    except Exception as exc:

                        print(
                            "Spatial fusion error:",
                            repr(exc),
                        )

            # =================================================
            # HMR STATUS
            # =================================================

            if hmr_result is not None:

                hmr_frame = (
                    hmr_result.get(
                        "frame_id",
                        -1,
                    )
                )

                hmr_count = len(
                    hmr_result.get(
                        "persons",
                        [],
                    )
                )

                draw_text(
                    display,
                    (
                        f"HMR frame="
                        f"{hmr_frame} "
                        f"persons="
                        f"{hmr_count}"
                    ),
                    (
                        WIDTH - 220,
                        25,
                    ),
                    scale=0.45,
                )

            # =================================================
            # HEADER
            # =================================================

            draw_text(
                display,
                (
                    "BAS SPATIAL FUSION | "
                    "YOLO + DEPTH + HMR2"
                ),
                (
                    10,
                    HEIGHT - 40,
                ),
                scale=0.52,
                thickness=2,
            )

            draw_text(
                display,
                (
                    f"Frame: "
                    f"{frame_id}"
                ),
                (
                    10,
                    HEIGHT - 18,
                ),
                scale=0.48,
            )

            # =================================================
            # SAVE DISPLAY
            # =================================================

            with display_lock:

                latest_display_frame = (
                    display.copy()
                )

        except Exception as exc:

            print(
                "[PROCESSING] ERROR:",
                repr(exc),
            )

        time.sleep(0.01)


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():

    while running:

        frame = (
            get_display_frame()
        )

        if frame is None:

            time.sleep(0.05)

            continue

        success, encoded = (
            cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80,
                ],
            )
        )

        if not success:

            time.sleep(0.02)

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )

        time.sleep(
            1.0 / FPS
        )


# ============================================================
# WEB PAGE
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>BAS Spatial Fusion</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    margin: 20px;
}

h1 {
    margin-bottom: 5px;
}

img {
    width: 640px;
    height: 480px;
    object-fit: contain;
    border: 2px solid #444;
}

pre {
    background: #222;
    padding: 15px;
    width: 610px;
    overflow-x: auto;
}

</style>

</head>

<body>

<h1>BAS Live Spatial Fusion</h1>

<p>
YOLO11 → Chair Tracking → Depth Anything → HMR2
→ Spatial Fusion → Temporal Tracking
</p>

<img src="/video">

<h2>Fusion Status</h2>

<pre id="status">
Loading...
</pre>

<script>

async function updateStatus() {

    try {

        const response =
            await fetch("/status");

        const data =
            await response.json();

        document.getElementById(
            "status"
        ).textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    } catch (error) {

        document.getElementById(
            "status"
        ).textContent =
            "Status unavailable";

    }

}

setInterval(
    updateStatus,
    1000
);

updateStatus();

</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return HTML_PAGE


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


@app.route("/status")
def status():

    _, frame_id = (
        get_latest_frame()
    )

    hmr_result = (
        get_hmr_result()
    )

    fused_result = (
        get_fused_result()
    )

    return jsonify(
        {
            "running": running,

            "camera": {
                "device": CAMERA_DEVICE,
                "frame_id": frame_id,
                "resolution": (
                    f"{WIDTH}x{HEIGHT}"
                ),
            },

            "hmr": (
                None
                if hmr_result is None
                else {
                    "frame_id": (
                        hmr_result.get(
                            "frame_id"
                        )
                    ),

                    "persons": len(
                        hmr_result.get(
                            "persons",
                            [],
                        )
                    ),

                    "model": (
                        hmr_result.get(
                            "model"
                        )
                    ),

                    "device": (
                        hmr_result.get(
                            "device"
                        )
                    ),
                }
            ),

            "spatial_fusion": (
                fused_result
            ),
        }
    )


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("=" * 70)
print("LOADING YOLO11")
print("=" * 70)

yolo_model = YOLO(
    YOLO_MODEL
)

print(
    "YOLO loaded."
)


# ============================================================
# LOAD HMR
# ============================================================

print()
print("=" * 70)
print("LOADING HMR2")
print("=" * 70)

hmr_processor = (
    HMRProcessor()
)

print(
    "HMR2 loaded."
)


# ============================================================
# START
# ============================================================

def start_threads():

    camera_thread = threading.Thread(
        target=camera_reader,
        name="CameraReader",
        daemon=True,
    )

    camera_thread.start()

    time.sleep(1.0)

    hmr_thread = threading.Thread(
        target=hmr_worker,
        name="HMRWorker",
        daemon=True,
    )

    hmr_thread.start()

    processing_thread = threading.Thread(
        target=processing_loop,
        name="ProcessingLoop",
        daemon=True,
    )

    processing_thread.start()

    return (
        camera_thread,
        hmr_thread,
        processing_thread,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("BAS LIVE SPATIAL FUSION")
    print("=" * 70)

    print(
        "Pipeline:"
    )

    print(
        "YOLO11 → Chair Tracking → Depth → HMR2 "
        "→ Spatial Fusion → Temporal Tracking"
    )

    print()
    print(
        "Open in Windows Chrome:"
    )

    print(
        "http://localhost:5000"
    )

    print()
    print(
        "Press CTRL+C to stop."
    )

    threads = start_threads()

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping BAS spatial fusion..."
        )

    finally:

        running = False

        time.sleep(1.0)

        print(
            "BAS spatial fusion stopped."
        )