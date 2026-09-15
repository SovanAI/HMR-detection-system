"""
BAS-HMR
3D Grid Distance Dataset Collector

Pipeline
--------
Webcam
   |
   +--> YOLO11
   |      |
   |      +--> Person
   |      +--> Chair
   |
   +--> YOLO26 Depth
          |
          +--> Dense metric depth map
                    |
                    v
              Object 3D Grid
                    |
                    v
             Pair Features
                    |
                    v
          Ground Truth Distance
                 (cm)
                    |
                    v
                CSV Dataset

IMPORTANT
---------
The manually measured distance is the ground-truth target.

The raw YOLO26-derived 3D distance is stored only as a
feature. It is NOT treated as ground truth.

YOLO26 depth values are expected in metres.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

from human.distance.grid.depth_to_3d import (
    CameraIntrinsics,
    DepthTo3D,
)
from human.distance.grid.object_grid import (
    BoundingBox,
    ObjectGridExtractor,
)
from human.distance.grid.pair_feature_extractor import (
    PairFeatureExtractor,
)


# ============================================================
# Configuration
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 10

YOLO_MODEL = "yolo11n.pt"
YOLO26_DEPTH_MODEL = "yolo26n-depth.pt"

YOLO_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 416

# Run YOLO26 depth every N frames and reuse the latest map.
DEPTH_INTERVAL = 3
DEPTH_IMAGE_SIZE = 640

# Object-centric 3D grid.
GRID_ROWS = 5
GRID_COLS = 5

GRID_BORDER_RATIO = 0.10
DEPTH_SAMPLE_RADIUS = 2

# Development intrinsics.
#
# IMPORTANT:
# These are NOT final calibrated camera intrinsics.
# They are only being used for the current development
# dataset pipeline.
FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0

# Flask.
HOST = "0.0.0.0"
PORT = 5006

# Dataset.
OUTPUT_DIR = (
    "data/validation/"
    "yolo26_distance_grid"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "distance_dataset_v2.csv",
)


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Global state
# ============================================================

camera: Optional[cv2.VideoCapture] = None

yolo_model: Optional[YOLO] = None
depth_model: Optional[YOLO] = None

depth_converter: Optional[DepthTo3D] = None
grid_extractor: Optional[ObjectGridExtractor] = None
pair_feature_extractor: Optional[
    PairFeatureExtractor
] = None

frame_id = 0

latest_frame: Optional[np.ndarray] = None
latest_result: Optional[Dict] = None

last_depth_map: Optional[np.ndarray] = None
last_depth_frame = -1
last_depth_inference_seconds = 0.0

actual_distance_cm: Optional[float] = None

sample_counter = 0


# ============================================================
# Simple centroid tracker
# ============================================================

class SimpleTracker:
    """
    Lightweight centroid tracker.

    This tracker is intentionally simple because the purpose
    here is dataset collection, not final production tracking.
    """

    def __init__(
        self,
        max_center_distance: float = 120.0,
        max_missing_frames: int = 8,
    ):
        self.max_center_distance = (
            max_center_distance
        )

        self.max_missing_frames = (
            max_missing_frames
        )

        self.next_id = 0

        self.objects: Dict[int, Dict] = {}

    @staticmethod
    def center(
        bbox: BoundingBox,
    ) -> Tuple[float, float]:

        return (
            bbox.center_x,
            bbox.center_y,
        )

    @staticmethod
    def distance(
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:

        return float(
            np.hypot(
                a[0] - b[0],
                a[1] - b[1],
            )
        )

    def update(
        self,
        detections: List[
            Tuple[BoundingBox, float]
        ],
    ) -> List[Dict]:

        # ----------------------------------------------------
        # Nothing detected.
        # ----------------------------------------------------

        if not detections:

            for object_id in list(
                self.objects.keys()
            ):

                self.objects[
                    object_id
                ]["missing"] += 1

            self._remove_old_objects()

            return []

        # ----------------------------------------------------
        # Convert detections.
        # ----------------------------------------------------

        candidates = []

        for bbox, confidence in detections:

            cx, cy = self.center(
                bbox
            )

            candidates.append(
                {
                    "bbox": bbox,
                    "confidence": float(
                        confidence
                    ),
                    "center": (
                        cx,
                        cy,
                    ),
                    "matched": False,
                }
            )

        # ----------------------------------------------------
        # Build possible matches.
        # ----------------------------------------------------

        matches = []

        for object_id, obj in (
            self.objects.items()
        ):

            old_center = obj[
                "center"
            ]

            for index, detection in enumerate(
                candidates
            ):

                if detection["matched"]:
                    continue

                distance = self.distance(
                    old_center,
                    detection["center"],
                )

                if (
                    distance
                    <= self.max_center_distance
                ):

                    matches.append(
                        (
                            distance,
                            object_id,
                            index,
                        )
                    )

        # Closest matches first.
        matches.sort(
            key=lambda item: item[0]
        )

        matched_objects = set()
        matched_detections = set()

        for (
            distance,
            object_id,
            index,
        ) in matches:

            if object_id in matched_objects:
                continue

            if index in matched_detections:
                continue

            detection = candidates[
                index
            ]

            obj = self.objects[
                object_id
            ]

            obj["bbox"] = detection[
                "bbox"
            ]

            obj["confidence"] = detection[
                "confidence"
            ]

            obj["center"] = detection[
                "center"
            ]

            obj["missing"] = 0

            detection["matched"] = True

            matched_objects.add(
                object_id
            )

            matched_detections.add(
                index
            )

        # ----------------------------------------------------
        # Create IDs for unmatched detections.
        # ----------------------------------------------------

        for index, detection in enumerate(
            candidates
        ):

            if index in matched_detections:
                continue

            object_id = self.next_id
            self.next_id += 1

            self.objects[
                object_id
            ] = {
                "bbox": detection[
                    "bbox"
                ],
                "confidence": detection[
                    "confidence"
                ],
                "center": detection[
                    "center"
                ],
                "missing": 0,
            }

        # ----------------------------------------------------
        # Increase missing count for unmatched objects.
        # ----------------------------------------------------

        for object_id in list(
            self.objects.keys()
        ):

            if object_id not in matched_objects:

                # Newly created objects are not
                # considered missing.
                center = self.objects[
                    object_id
                ]["center"]

                is_current_detection = any(
                    self.distance(
                        center,
                        detection[
                            "center"
                        ],
                    )
                    < 1e-6
                    for detection in candidates
                )

                if not is_current_detection:

                    self.objects[
                        object_id
                    ]["missing"] += 1

        self._remove_old_objects()

        # ----------------------------------------------------
        # Return only currently visible objects.
        # ----------------------------------------------------

        results = []

        for object_id, obj in (
            self.objects.items()
        ):

            if obj["missing"] == 0:

                results.append(
                    {
                        "id": object_id,
                        "bbox": obj["bbox"],
                        "confidence": obj[
                            "confidence"
                        ],
                    }
                )

        return results

    def _remove_old_objects(self):

        for object_id in list(
            self.objects.keys()
        ):

            if (
                self.objects[
                    object_id
                ]["missing"]
                > self.max_missing_frames
            ):

                del self.objects[
                    object_id
                ]


person_tracker = SimpleTracker(
    max_center_distance=120.0,
    max_missing_frames=8,
)

chair_tracker = SimpleTracker(
    max_center_distance=120.0,
    max_missing_frames=8,
)


# ============================================================
# CSV definition
# ============================================================

PAIR_FEATURE_NAMES = (
    PairFeatureExtractor.feature_names()
)


CSV_FIELDS = [
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

CSV_FIELDS.extend(
    PAIR_FEATURE_NAMES
)


def ensure_csv():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
            )
            writer.writeheader()
        return

    # Validate an existing CSV before appending. This prevents
    # accidentally mixing datasets with incompatible schemas.
    with open(CSV_PATH, "r", newline="") as file:
        reader = csv.reader(file)
        existing_header = next(reader, [])

    if existing_header != CSV_FIELDS:
        raise RuntimeError(
            "CSV schema mismatch.\n"
            f"Expected {len(CSV_FIELDS)} columns, "
            f"found {len(existing_header)} columns.\n"
            f"CSV: {CSV_PATH}"
        )


def get_existing_sample_count() -> int:

    if not os.path.exists(
        CSV_PATH
    ):
        return 0

    try:

        with open(
            CSV_PATH,
            "r",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            count = 0

            for _ in reader:
                count += 1

            return count

    except Exception:

        return 0


# ============================================================
# Model initialization
# ============================================================

def initialize():

    global camera
    global yolo_model
    global depth_model

    global depth_converter
    global grid_extractor
    global pair_feature_extractor

    global sample_counter

    print("=" * 70)
    print("BAS-HMR 3D GRID DISTANCE DATASET COLLECTOR")
    print("=" * 70)

    # --------------------------------------------------------
    # YOLO11
    # --------------------------------------------------------

    print()
    print("[1/5] Loading YOLO11...")

    yolo_model = YOLO(
        YOLO_MODEL
    )

    print(
        "[OK] YOLO11 loaded"
    )

    # --------------------------------------------------------
    # YOLO26 depth
    # --------------------------------------------------------

    print()
    print(
        "[2/5] Loading YOLO26 depth..."
    )

    depth_model = YOLO(
        YOLO26_DEPTH_MODEL
    )

    print(
        "[OK] YOLO26 depth loaded"
    )

    # --------------------------------------------------------
    # 3D components
    # --------------------------------------------------------

    print()
    print(
        "[3/5] Creating 3D grid components..."
    )

    intrinsics = CameraIntrinsics(
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )

    depth_converter = DepthTo3D(
        intrinsics=intrinsics
    )

    grid_extractor = ObjectGridExtractor(
        depth_converter=depth_converter,
        rows=GRID_ROWS,
        cols=GRID_COLS,
        border_ratio=GRID_BORDER_RATIO,
        depth_radius=DEPTH_SAMPLE_RADIUS,
    )

    pair_feature_extractor = (
        PairFeatureExtractor()
    )

    print(
        "[OK] 3D grid components ready"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    print()
    print(
        "[4/5] Preparing dataset..."
    )

    ensure_csv()

    sample_counter = (
        get_existing_sample_count()
    )

    print(
        f"[OK] Existing samples: "
        f"{sample_counter}"
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    print()
    print(
        "[5/5] Opening webcam..."
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS,
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    print(
        "[OK] Webcam opened"
    )

    print()
    print("=" * 70)
    print("COLLECTOR READY")
    print("=" * 70)

    print()
    print(
        f"CSV: {CSV_PATH}"
    )

    print()
    print(
        "Browser:"
    )

    print(
        f"http://127.0.0.1:{PORT}"
    )

    print()
    print(
        "Workflow:"
    )

    print(
        "1. Position the person and chair."
    )

    print(
        "2. Measure the physical distance with a tape."
    )

    print(
        "3. Enter that distance in centimetres."
    )

    print(
        "4. Set the ground truth."
    )

    print(
        "5. Capture a sample."
    )

    print()
    print("=" * 70)


# ============================================================
# YOLO11 detection
# ============================================================

def detect_objects(
    frame: np.ndarray,
):

    if yolo_model is None:

        raise RuntimeError(
            "YOLO11 is not initialized."
        )

    results = yolo_model.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    if not results:

        return [], []

    result = results[0]

    person_detections = []
    chair_detections = []

    if result.boxes is None:

        return (
            person_detections,
            chair_detections,
        )

    for box in result.boxes:

        cls = int(
            box.cls.item()
        )

        confidence = float(
            box.conf.item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        bbox = BoundingBox(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
        )

        # COCO:
        # 0  = person
        # 56 = chair

        if cls == 0:

            person_detections.append(
                (
                    bbox,
                    confidence,
                )
            )

        elif cls == 56:

            chair_detections.append(
                (
                    bbox,
                    confidence,
                )
            )

    return (
        person_detections,
        chair_detections,
    )


# ============================================================
# YOLO26 depth
# ============================================================

def compute_depth(
    frame: np.ndarray,
) -> np.ndarray:

    global last_depth_map
    global last_depth_frame
    global last_depth_inference_seconds

    # Reuse recent depth map.
    if (
        last_depth_map is not None
        and
        (
            frame_id
            - last_depth_frame
        )
        < DEPTH_INTERVAL
    ):

        return last_depth_map

    if depth_model is None:

        raise RuntimeError(
            "YOLO26 depth model is not initialized."
        )

    start = time.time()

    results = depth_model.predict(
        frame,
        imgsz=DEPTH_IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    if not results:

        raise RuntimeError(
            "YOLO26 returned no result."
        )

    result = results[0]

    # --------------------------------------------------------
    # Official Ultralytics API:
    #
    # result.depth.data
    #
    # returns H x W depth values in metres.
    # --------------------------------------------------------

    if not hasattr(
        result,
        "depth",
    ):

        raise RuntimeError(
            "YOLO26 result does not contain "
            "'depth'."
        )

    depth_object = result.depth

    if not hasattr(
        depth_object,
        "data",
    ):

        raise RuntimeError(
            "YOLO26 depth object does not "
            "contain 'data'."
        )

    depth = (
        depth_object.data
        .detach()
        .cpu()
        .numpy()
    )

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    depth = np.squeeze(
        depth
    )

    if depth.ndim != 2:

        raise RuntimeError(
            "Unexpected YOLO26 depth shape: "
            f"{depth.shape}"
        )

    # YOLO26 depth output is aligned with
    # the original input image according to
    # the current Ultralytics API, but resize
    # defensively to our camera frame.
    if (
        depth.shape[1] != FRAME_WIDTH
        or
        depth.shape[0] != FRAME_HEIGHT
    ):

        depth = cv2.resize(
            depth,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    # --------------------------------------------------------
    # Clean invalid values.
    # --------------------------------------------------------

    depth[
        ~np.isfinite(depth)
    ] = 0.0

    depth[
        depth <= 0
    ] = 0.0

    # Our current project uses the metric range
    # 0.05m -> 20m.
    depth[
        depth > 20.0
    ] = 0.0

    last_depth_map = depth
    last_depth_frame = frame_id

    last_depth_inference_seconds = (
        time.time() - start
    )

    return depth


# ============================================================
# Drawing
# ============================================================

def draw_box(
    frame: np.ndarray,
    bbox: BoundingBox,
    label: str,
    object_id: int,
):

    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"{label} ID:{object_id}",
        (
            x1,
            max(
                20,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
    )


# ============================================================
# Process frame
# ============================================================

def process_frame(
    frame: np.ndarray,
) -> np.ndarray:

    global latest_result

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    (
        person_detections,
        chair_detections,
    ) = detect_objects(
        frame
    )

    persons = person_tracker.update(
        person_detections
    )

    chairs = chair_tracker.update(
        chair_detections
    )

    # --------------------------------------------------------
    # No person/chair
    # --------------------------------------------------------

    if (
        not persons
        or not chairs
    ):

        latest_result = {
            "frame_id": frame_id,
            "persons": len(persons),
            "chairs": len(chairs),
            "pairs": [],
            "actual_distance_cm":
                actual_distance_cm,
        }

        draw_text(
            frame,
            f"Frame: {frame_id}",
            10,
            25,
        )

        draw_text(
            frame,
            f"Persons: {len(persons)}",
            10,
            50,
        )

        draw_text(
            frame,
            f"Chairs: {len(chairs)}",
            10,
            75,
        )

        draw_text(
            frame,
            (
                f"Actual: "
                f"{actual_distance_cm:.1f} cm"
                if actual_distance_cm
                is not None
                else
                "Actual: NOT SET"
            ),
            10,
            100,
        )

        return frame

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    try:

        depth_map = compute_depth(
            frame
        )

    except Exception as error:

        print(
            f"[DEPTH ERROR] {error}"
        )

        latest_result = {
            "frame_id": frame_id,
            "persons": len(persons),
            "chairs": len(chairs),
            "pairs": [],
            "actual_distance_cm":
                actual_distance_cm,
            "error": str(error),
        }

        return frame

    # --------------------------------------------------------
    # Pair generation
    # --------------------------------------------------------

    pair_results = []

    for person in persons:

        person_grid = (
            grid_extractor.extract(
                depth_map,
                person["bbox"],
                object_id=person["id"],
                object_class="person",
            )
        )

        if person_grid is None:
            continue

        for chair in chairs:

            chair_grid = (
                grid_extractor.extract(
                    depth_map,
                    chair["bbox"],
                    object_id=chair["id"],
                    object_class="chair",
                )
            )

            if chair_grid is None:
                continue

            features = (
                pair_feature_extractor.extract(
                    person_grid,
                    chair_grid,
                )
            )

            raw_distance_cm = (
                features[
                    "raw_3d_distance"
                ]
                * 100.0
            )

            ground_distance_cm = (
                features[
                    "raw_ground_distance"
                ]
                * 100.0
            )

            pair_results.append(
                {
                    "person_id":
                        person["id"],

                    "chair_id":
                        chair["id"],

                    "raw_distance_cm":
                        float(
                            raw_distance_cm
                        ),

                    "ground_distance_cm":
                        float(
                            ground_distance_cm
                        ),

                    "features":
                        features,
                }
            )

            # Draw person.
            draw_box(
                frame,
                person["bbox"],
                "PERSON",
                person["id"],
            )

            # Draw chair.
            draw_box(
                frame,
                chair["bbox"],
                "CHAIR",
                chair["id"],
            )

            # Distance label.
            x = int(
                (
                    person["bbox"].center_x
                    + chair["bbox"].center_x
                )
                / 2
            )

            y = int(
                (
                    person["bbox"].center_y
                    + chair["bbox"].center_y
                )
                / 2
            )

            draw_text(
                frame,
                (
                    f"Raw 3D: "
                    f"{raw_distance_cm:.1f} cm"
                ),
                x,
                y,
            )

    # --------------------------------------------------------
    # Save latest result.
    # --------------------------------------------------------

    latest_result = {
        "frame_id": frame_id,
        "persons": len(persons),
        "chairs": len(chairs),
        "pairs": pair_results,
        "actual_distance_cm":
            actual_distance_cm,
    }

    # --------------------------------------------------------
    # Overlay.
    # --------------------------------------------------------

    draw_text(
        frame,
        f"Frame: {frame_id}",
        10,
        25,
    )

    draw_text(
        frame,
        f"Persons: {len(persons)}",
        10,
        50,
    )

    draw_text(
        frame,
        f"Chairs: {len(chairs)}",
        10,
        75,
    )

    draw_text(
        frame,
        (
            f"Actual: "
            f"{actual_distance_cm:.1f} cm"
            if actual_distance_cm
            is not None
            else
            "Actual: NOT SET"
        ),
        10,
        100,
    )

    if pair_results:

        draw_text(
            frame,
            (
                f"Pairs: "
                f"{len(pair_results)}"
            ),
            10,
            125,
        )

    else:

        draw_text(
            frame,
            "Pairs: NONE",
            10,
            125,
        )

    return frame


# ============================================================
# Camera frame generator
# ============================================================

def generate_frames():

    global latest_frame
    global frame_id

    if camera is None:

        raise RuntimeError(
            "Camera is not initialized."
        )

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "[CAMERA] Failed to read frame."
            )

            time.sleep(0.1)
            continue

        frame_id += 1

        try:

            processed = process_frame(
                frame
            )

        except Exception as error:

            print(
                f"[FRAME ERROR] {error}"
            )

            processed = frame

        latest_frame = processed.copy()

        success, encoded = cv2.imencode(
            ".jpg",
            processed,
            [
                cv2.IMWRITE_JPEG_QUALITY,
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


# ============================================================
# Web interface
# ============================================================

@app.route("/")
def index():

    return """
<!DOCTYPE html>

<html>

<head>

<title>BAS-HMR 3D Grid Dataset Collector</title>

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<style>

body {
    font-family: Arial, sans-serif;
    margin: 20px;
    background: #f5f5f5;
}

.container {
    max-width: 1000px;
    margin: auto;
}

.card {
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 10px;
}

img {
    width: 640px;
    max-width: 100%;
    border: 2px solid #222;
    border-radius: 6px;
}

input {
    padding: 10px;
    font-size: 16px;
    width: 180px;
}

button {
    padding: 10px 16px;
    margin: 5px;
    font-size: 15px;
    cursor: pointer;
}

.capture {
    font-weight: bold;
}

pre {
    background: #111;
    color: #eee;
    padding: 15px;
    overflow-x: auto;
}

.warning {
    padding: 10px;
    background: #fff3cd;
    border: 1px solid #ffeeba;
    border-radius: 5px;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>BAS-HMR 3D Grid Dataset Collector</h1>

<p class="warning">
The displayed Raw 3D distance is an input feature,
not the ground-truth distance.
</p>

<img src="/video_feed">

</div>

<div class="card">

<h2>Ground Truth Distance</h2>

<p>
Measure the physical person-chair distance with a
tape measure and enter it in centimetres.
</p>

<input
    id="distance"
    type="number"
    min="1"
    step="0.1"
    placeholder="Distance (cm)"
>

<br>

<button onclick="setActual()">
    Set Actual Distance
</button>

<button
    class="capture"
    onclick="captureSample()"
>
    Capture Sample
</button>

</div>

<div class="card">

<h2>Status</h2>

<pre id="status">
Loading...
</pre>

</div>

</div>

<script>

async function setActual() {

    const input =
        document.getElementById(
            "distance"
        );

    const value =
        parseFloat(input.value);

    if (!Number.isFinite(value) ||
        value <= 0) {

        alert(
            "Enter a valid distance in cm."
        );

        return;
    }

    try {

        const response =
            await fetch(
                "/api/set_actual",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        distance_cm:
                            value
                    })
                }
            );

        const data =
            await response.json();

        alert(
            data.message ||
            data.error ||
            "Updated."
        );

        updateStatus();

    } catch (error) {

        alert(
            "Request failed: " +
            error
        );
    }
}


async function captureSample() {

    try {

        const response =
            await fetch(
                "/api/capture",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        alert(
            data.message ||
            data.error ||
            "Capture completed."
        );

        updateStatus();

    } catch (error) {

        alert(
            "Capture failed: " +
            error
        );
    }
}


async function updateStatus() {

    try {

        const response =
            await fetch(
                "/api/status"
            );

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
            "Status error: " +
            error;
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
# Video feed
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


# ============================================================
# Set ground truth
# ============================================================

@app.route(
    "/api/set_actual",
    methods=["POST"],
)
def set_actual():

    global actual_distance_cm

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict,
    ):

        return jsonify(
            {
                "error":
                    "JSON body required."
            }
        ), 400

    value = data.get(
        "distance_cm"
    )

    if value is None:

        return jsonify(
            {
                "error":
                    "distance_cm is required."
            }
        ), 400

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return jsonify(
            {
                "error":
                    "distance_cm must be numeric."
            }
        ), 400

    if not np.isfinite(
        value
    ):

        return jsonify(
            {
                "error":
                    "Distance must be finite."
            }
        ), 400

    if value <= 0:

        return jsonify(
            {
                "error":
                    "Distance must be greater than 0."
            }
        ), 400

    actual_distance_cm = value

    print(
        "[GROUND TRUTH] "
        f"{actual_distance_cm:.2f} cm"
    )

    return jsonify(
        {
            "message":
                "Ground-truth distance updated.",

            "distance_cm":
                actual_distance_cm,
        }
    )


# ============================================================
# Capture sample
# ============================================================

@app.route(
    "/api/capture",
    methods=["POST"],
)
def capture():

    global sample_counter

    # --------------------------------------------------------
    # Ground truth check.
    # --------------------------------------------------------

    if actual_distance_cm is None:

        return jsonify(
            {
                "error":
                    "Set the measured distance first."
            }
        ), 400

    # --------------------------------------------------------
    # Pair check.
    # --------------------------------------------------------

    if (
        latest_result is None
        or not latest_result.get(
            "pairs"
        )
    ):

        return jsonify(
            {
                "error":
                    "No valid person-chair pair detected."
            }
        ), 400

    ensure_csv()

    saved = 0

    try:

        with open(
            CSV_PATH,
            "a",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
                extrasaction="ignore",
            )

            for pair in latest_result[
                "pairs"
            ]:

                sample_counter += 1

                person_id = pair[
                    "person_id"
                ]

                chair_id = pair[
                    "chair_id"
                ]

                row = {
                    "sample_id":
                        sample_counter,

                    "timestamp":
                        datetime.now()
                        .isoformat(),

                    "frame_id":
                        latest_result[
                            "frame_id"
                        ],

                    "person_id":
                        person_id,

                    "chair_id":
                        chair_id,

                    "ground_truth_distance_cm":
                        actual_distance_cm,
                }

                # ------------------------------------------------
                # Person bbox.
                # ------------------------------------------------

                person_obj = (
                    person_tracker.objects.get(
                        person_id
                    )
                )

                if person_obj is not None:

                    bbox = person_obj[
                        "bbox"
                    ]

                    row.update(
                        {
                            "person_bbox_x1":
                                bbox.x1,

                            "person_bbox_y1":
                                bbox.y1,

                            "person_bbox_x2":
                                bbox.x2,

                            "person_bbox_y2":
                                bbox.y2,
                        }
                    )

                # ------------------------------------------------
                # Chair bbox.
                # ------------------------------------------------

                chair_obj = (
                    chair_tracker.objects.get(
                        chair_id
                    )
                )

                if chair_obj is not None:

                    bbox = chair_obj[
                        "bbox"
                    ]

                    row.update(
                        {
                            "chair_bbox_x1":
                                bbox.x1,

                            "chair_bbox_y1":
                                bbox.y1,

                            "chair_bbox_x2":
                                bbox.x2,

                            "chair_bbox_y2":
                                bbox.y2,
                        }
                    )

                # ------------------------------------------------
                # Pair features.
                # ------------------------------------------------

                row.update(
                    pair["features"]
                )

                writer.writerow(
                    row
                )

                saved += 1

    except Exception as error:

        return jsonify(
            {
                "error":
                    f"Could not save dataset: "
                    f"{error}"
            }
        ), 500

    print(
        "[DATASET] Saved "
        f"{saved} sample(s) "
        f"| Ground truth = "
        f"{actual_distance_cm:.2f} cm"
    )

    return jsonify(
        {
            "message":
                f"Saved {saved} sample(s).",

            "samples_saved":
                saved,

            "ground_truth_cm":
                actual_distance_cm,

            "total_samples":
                sample_counter,

            "csv":
                CSV_PATH,
        }
    )


# ============================================================
# Status API
# ============================================================

@app.route("/api/status")
def status():

    pairs = []

    if latest_result is not None:

        for pair in latest_result.get(
            "pairs",
            [],
        ):

            pairs.append(
                {
                    "person_id":
                        pair[
                            "person_id"
                        ],

                    "chair_id":
                        pair[
                            "chair_id"
                        ],

                    "raw_3d_distance_cm":
                        round(
                            pair[
                                "raw_distance_cm"
                            ],
                            2,
                        ),

                    "raw_ground_distance_cm":
                        round(
                            pair[
                                "ground_distance_cm"
                            ],
                            2,
                        ),
                }
            )

    return jsonify(
        {
            "frame_id":
                frame_id,

            "persons":
                (
                    latest_result.get(
                        "persons",
                        0,
                    )
                    if latest_result
                    else 0
                ),

            "chairs":
                (
                    latest_result.get(
                        "chairs",
                        0,
                    )
                    if latest_result
                    else 0
                ),

            "pairs":
                len(pairs),

            "pair_details":
                pairs,

            "actual_distance_cm":
                actual_distance_cm,

            "sample_count":
                sample_counter,

            "csv_path":
                CSV_PATH,

            "depth_interval":
                DEPTH_INTERVAL,

            "depth_inference_seconds":
                round(
                    last_depth_inference_seconds,
                    3,
                ),
        }
    )


# ============================================================
# Health check
# ============================================================

@app.route("/api/health")
def health():

    return jsonify(
        {
            "status": "ok",

            "camera":
                camera is not None
                and camera.isOpened(),

            "yolo":
                yolo_model is not None,

            "yolo26_depth":
                depth_model is not None,

            "grid":
                grid_extractor is not None,

            "pair_features":
                pair_feature_extractor
                is not None,

            "csv":
                CSV_PATH,
        }
    )


# ============================================================
# Cleanup
# ============================================================

def cleanup():

    global camera

    print()
    print(
        "[SHUTDOWN] Cleaning up..."
    )

    if camera is not None:

        try:
            camera.release()
        except Exception:
            pass

        camera = None

    print(
        "[SHUTDOWN] Camera released."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    try:

        initialize()

        app.run(
            host=HOST,
            port=PORT,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    except KeyboardInterrupt:

        print(
            "\n[SHUTDOWN] Keyboard interrupt."
        )

    except Exception as error:

        print(
            f"\n[FATAL ERROR] {error}"
        )

        raise

    finally:

        cleanup()