"""
BAS-HMR
Realtime ML Person-Chair Distance System

Fixed architecture:

Camera
  -> YOLO11n (person=0, chair=56)
  -> independent stable trackers
  -> YOLO26n metric depth
  -> 5x5 Object3DGrid for person/chair
  -> PairFeatureExtractor (34 features)
  -> trained Ridge model
  -> temporal smoothing
  -> Flask MJPEG

Important fixes:
1. YOLO11 is explicitly restricted to classes 0 and 56.
2. Person and chair trackers are completely independent.
3. Detection boxes are displayed BEFORE any depth/ML processing.
4. ObjectGridExtractor receives (depth_map, bbox).
5. FeatureExtractor receives Object3DGrid, never dict.
6. PairFeatureExtractor receives Object3DGrid, never feature dictionaries.
7. The exact 34 feature order from PairFeatureExtractor is used.
8. A failed depth/grid/model frame does not hide detections.
9. Depth is refreshed independently and reused between depth frames.
10. Multiple people and chairs are supported.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify
from ultralytics import YOLO

from human.distance.grid.depth_to_3d import CameraIntrinsics, DepthTo3D
from human.distance.grid.object_grid import (
    BoundingBox,
    Object3DGrid,
    ObjectGridExtractor,
)
from human.distance.grid.feature_extractor import FeatureExtractor
from human.distance.grid.pair_feature_extractor import PairFeatureExtractor


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

YOLO_MODEL = "yolo11n.pt"
DEPTH_MODEL = "yolo26n-depth.pt"

# Lower detection threshold slightly so COCO chairs are less likely
# to disappear. We still explicitly keep only classes 0 and 56.
YOLO_CONF = 0.25
YOLO_IMGSZ = 640

DEPTH_IMGSZ = 640
DEPTH_INTERVAL = 3

PERSON_CLASS = 0
CHAIR_CLASS = 56

GRID_ROWS = 5
GRID_COLS = 5
GRID_BORDER = 0.10
DEPTH_RADIUS = 2

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5007

MODEL_PATH = ROOT / "training" / "distance" / "models" / "best_distance_model.joblib"

# Development camera intrinsics.
INTRINSICS = CameraIntrinsics(
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
)

MIN_DEPTH = 0.05
MAX_DEPTH = 20.0

# Independent tracker settings.
PERSON_MAX_CENTER_DISTANCE = 180.0
CHAIR_MAX_CENTER_DISTANCE = 180.0
MAX_MISSING = 10

# Distance temporal smoothing.
SMOOTHING_WINDOW = 7
EMA_ALPHA = 0.35

# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

camera = None
yolo_model = None
depth_model = None
distance_model = None

feature_columns = []
artifact = None

depth_to_3d = None
grid_extractor = None
feature_extractor = None
pair_extractor = None

latest_frame = None
latest_result = None

frame_lock = threading.Lock()
running = False


# ============================================================
# SIMPLE CLASS-SPECIFIC TRACKER
# ============================================================

class SimpleTracker:
    """
    Independent tracker for ONE object class.

    This is deliberately class-specific. A person can NEVER be
    matched to a chair.
    """

    def __init__(
        self,
        max_center_distance: float = 180.0,
        max_missing: int = 10,
    ):
        self.max_center_distance = float(max_center_distance)
        self.max_missing = int(max_missing)

        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def center(bbox):
        return (
            (bbox["x1"] + bbox["x2"]) / 2.0,
            (bbox["y1"] + bbox["y2"]) / 2.0,
        )

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = (
            a["x1"], a["y1"], a["x2"], a["y2"]
        )
        bx1, by1, bx2, by2 = (
            b["x1"], b["y1"], b["x2"], b["y2"]
        )

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def update(self, detections):
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["missing"] += 1

                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]

            return []

        current = []

        for detection in detections:
            d = dict(detection)
            d["bbox"] = dict(detection["bbox"])
            current.append(d)

        # ----------------------------------------------------
        # Build all possible matches.
        # Score = center distance with IoU as secondary cue.
        # ----------------------------------------------------

        candidates = []

        for di, detection in enumerate(current):
            center = self.center(detection["bbox"])

            for tid, track in self.tracks.items():
                old_center = track["center"]

                distance = math.hypot(
                    center[0] - old_center[0],
                    center[1] - old_center[1],
                )

                if distance <= self.max_center_distance:
                    overlap = self.iou(
                        detection["bbox"],
                        track["bbox"],
                    )

                    candidates.append(
                        (
                            distance,
                            -overlap,
                            di,
                            tid,
                        )
                    )

        candidates.sort()

        used_detections = set()
        used_tracks = set()

        results = []

        # ----------------------------------------------------
        # Existing tracks
        # ----------------------------------------------------

        for _, _, di, tid in candidates:
            if di in used_detections or tid in used_tracks:
                continue

            detection = current[di]
            track = self.tracks[tid]

            # Light bbox smoothing.
            old = track["bbox"]
            new = detection["bbox"]

            alpha = 0.70

            smoothed_bbox = {
                "x1": alpha * new["x1"] + (1 - alpha) * old["x1"],
                "y1": alpha * new["y1"] + (1 - alpha) * old["y1"],
                "x2": alpha * new["x2"] + (1 - alpha) * old["x2"],
                "y2": alpha * new["y2"] + (1 - alpha) * old["y2"],
            }

            track["bbox"] = smoothed_bbox
            track["center"] = self.center(smoothed_bbox)
            track["confidence"] = detection["confidence"]
            track["missing"] = 0

            detection["object_id"] = tid
            detection["bbox"] = smoothed_bbox

            used_detections.add(di)
            used_tracks.add(tid)

            results.append(detection)

        # ----------------------------------------------------
        # New detections
        # ----------------------------------------------------

        for di, detection in enumerate(current):
            if di in used_detections:
                continue

            tid = self.next_id
            self.next_id += 1

            bbox = detection["bbox"]
            center = self.center(bbox)

            self.tracks[tid] = {
                "bbox": bbox,
                "center": center,
                "confidence": detection["confidence"],
                "missing": 0,
            }

            detection["object_id"] = tid

            results.append(detection)

        # ----------------------------------------------------
        # Missing tracks
        # ----------------------------------------------------

        for tid in list(self.tracks):
            if tid not in used_tracks:
                # A newly created track is already represented in results.
                if any(
                    item["object_id"] == tid
                    for item in results
                ):
                    continue

                self.tracks[tid]["missing"] += 1

                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]

        results.sort(key=lambda d: d["object_id"])

        return results


person_tracker = SimpleTracker(
    max_center_distance=PERSON_MAX_CENTER_DISTANCE,
    max_missing=MAX_MISSING,
)

chair_tracker = SimpleTracker(
    max_center_distance=CHAIR_MAX_CENTER_DISTANCE,
    max_missing=MAX_MISSING,
)


# ============================================================
# DISTANCE SMOOTHER
# ============================================================

class DistanceSmoother:

    def __init__(
        self,
        window=SMOOTHING_WINDOW,
        alpha=EMA_ALPHA,
    ):
        self.window = int(window)
        self.alpha = float(alpha)

        self.history = defaultdict(
            lambda: deque(maxlen=self.window)
        )

        self.ema = {}

    def update(self, key, value):
        if not np.isfinite(value):
            return None

        value = float(np.clip(value, 0.0, 2000.0))

        history = self.history[key]

        if history:
            median = float(np.median(np.asarray(history)))

            # Very large isolated jumps are ignored.
            if abs(value - median) > 500.0:
                value = median

        history.append(value)

        median = float(
            np.median(
                np.asarray(history)
            )
        )

        if key not in self.ema:
            self.ema[key] = median
        else:
            self.ema[key] = (
                self.alpha * median
                + (1.0 - self.alpha) * self.ema[key]
            )

        return float(self.ema[key])


distance_smoother = DistanceSmoother()


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():

    global yolo_model
    global depth_model
    global distance_model
    global feature_columns
    global artifact
    global depth_to_3d
    global grid_extractor
    global feature_extractor
    global pair_extractor

    print("=" * 70)
    print("BAS-HMR REALTIME ML DISTANCE SYSTEM")
    print("=" * 70)

    # --------------------------------------------------------
    # YOLO11
    # --------------------------------------------------------

    print("\n[1/5] Loading YOLO11...")

    yolo_model = YOLO(YOLO_MODEL)

    print("YOLO11 loaded.")
    print(
        f"Detection classes: "
        f"PERSON={PERSON_CLASS}, CHAIR={CHAIR_CLASS}"
    )
    print(f"Confidence threshold: {YOLO_CONF}")
    print(f"Image size: {YOLO_IMGSZ}")

    # --------------------------------------------------------
    # YOLO26 depth
    # --------------------------------------------------------

    print("\n[2/5] Loading YOLO26 depth model...")

    depth_model = YOLO(DEPTH_MODEL)

    print("YOLO26 depth model loaded.")

    # --------------------------------------------------------
    # Distance model
    # --------------------------------------------------------

    print("\n[3/5] Loading trained distance model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Distance model not found:\n{MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    distance_model = artifact["model"]
    feature_columns = list(
        artifact["feature_columns"]
    )

    print(
        "Distance model:",
        artifact.get(
            "model_name",
            type(distance_model).__name__,
        ),
    )

    print(
        "Features:",
        len(feature_columns),
    )

    if len(feature_columns) != 34:
        raise RuntimeError(
            f"Expected 34 model features, got "
            f"{len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    print("\n[4/5] Initializing 3D grid...")

    depth_to_3d = DepthTo3D(
        intrinsics=INTRINSICS,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
    )

    grid_extractor = ObjectGridExtractor(
        depth_converter=depth_to_3d,
        rows=GRID_ROWS,
        cols=GRID_COLS,
        border_ratio=GRID_BORDER,
        depth_radius=DEPTH_RADIUS,
    )

    feature_extractor = FeatureExtractor()
    pair_extractor = PairFeatureExtractor()

    print("3D components initialized.")

    # --------------------------------------------------------
    # Contract self-test
    # --------------------------------------------------------

    print("\n[5/5] Verifying feature pipeline...")

    test_depth = np.full(
        (CAMERA_HEIGHT, CAMERA_WIDTH),
        2.0,
        dtype=np.float32,
    )

    test_bbox = BoundingBox(
        100,
        100,
        300,
        400,
    )

    test_grid = grid_extractor.extract(
        depth_map=test_depth,
        bbox=test_bbox,
        object_id=1,
        object_class="person",
    )

    if test_grid is None:
        raise RuntimeError(
            "ObjectGridExtractor returned None during self-test."
        )

    if not isinstance(test_grid, Object3DGrid):
        raise TypeError(
            "ObjectGridExtractor must return Object3DGrid, "
            f"got {type(test_grid).__name__}"
        )

    test_features = feature_extractor.extract(
        test_grid
    )

    if not isinstance(test_features, dict):
        raise TypeError(
            "FeatureExtractor must return a feature dictionary."
        )

    # PairFeatureExtractor expects GRIDS, not feature dictionaries.
    test_pair_features = pair_extractor.extract(
        test_grid,
        test_grid,
    )

    if not isinstance(test_pair_features, dict):
        raise TypeError(
            "PairFeatureExtractor must return a feature dictionary."
        )

    test_vector = pair_extractor.to_vector(
        test_pair_features
    )

    if len(test_vector) != 34:
        raise RuntimeError(
            f"PairFeatureExtractor produced "
            f"{len(test_vector)} features, expected 34."
        )

    print(
        f"Grid self-test: {len(test_grid.points)} points"
    )

    print(
        f"Pair feature self-test: "
        f"{len(test_vector)} features"
    )

    print("Feature pipeline verified.")

    print("\nALL MODELS LOADED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# DETECTION
# ============================================================

def detect_objects(frame):

    if yolo_model is None:
        raise RuntimeError("YOLO model is not loaded.")

    # Explicitly request ONLY person and chair.
    # This removes unnecessary COCO classes and makes the
    # person/chair path deterministic.
    results = yolo_model.predict(
        source=frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        classes=[PERSON_CLASS, CHAIR_CLASS],
        device="cpu",
        verbose=False,
    )

    persons = []
    chairs = []

    if not results:
        return persons, chairs

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return persons, chairs

    boxes = result.boxes

    for i in range(len(boxes)):

        cls = int(
            boxes.cls[i].detach().cpu().item()
        )

        confidence = float(
            boxes.conf[i].detach().cpu().item()
        )

        coords = (
            boxes.xyxy[i]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        x1, y1, x2, y2 = coords.tolist()

        # Clip to actual frame dimensions.
        h, w = frame.shape[:2]

        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detection = {
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "confidence": confidence,
        }

        if cls == PERSON_CLASS:
            persons.append(detection)

        elif cls == CHAIR_CLASS:
            chairs.append(detection)

    # CRITICAL:
    # Independent trackers prevent person/chair cross-matching.
    persons = person_tracker.update(persons)
    chairs = chair_tracker.update(chairs)

    return persons, chairs


# ============================================================
# DEPTH
# ============================================================

def estimate_depth(frame):

    if depth_model is None:
        raise RuntimeError("Depth model is not loaded.")

    results = depth_model.predict(
        source=frame,
        imgsz=DEPTH_IMGSZ,
        device="cpu",
        verbose=False,
    )

    if not results:
        return None

    result = results[0]

    if not hasattr(result, "depth"):
        return None

    if result.depth is None:
        return None

    raw = result.depth.data

    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()

    depth = np.asarray(
        raw,
        dtype=np.float32,
    )

    depth = np.squeeze(depth)

    if depth.ndim != 2:
        raise RuntimeError(
            f"Unexpected depth shape: {depth.shape}"
        )

    if depth.shape != (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
    ):
        depth = cv2.resize(
            depth,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    # Invalid depth becomes NaN.
    invalid = (
        ~np.isfinite(depth)
        | (depth < MIN_DEPTH)
        | (depth > MAX_DEPTH)
    )

    depth[invalid] = np.nan

    return depth.astype(np.float32)


# ============================================================
# BBOX CONVERSION
# ============================================================

def make_bbox(detection):

    b = detection["bbox"]

    return BoundingBox(
        x1=float(b["x1"]),
        y1=float(b["y1"]),
        x2=float(b["x2"]),
        y2=float(b["y2"]),
    )


# ============================================================
# GRID EXTRACTION
# ============================================================

def extract_object_grid(
    detection,
    object_class,
    depth_map,
):
    if depth_map is None:
        return None

    depth_map = np.asarray(
        depth_map,
        dtype=np.float32,
    )

    if depth_map.ndim != 2:
        raise ValueError(
            f"depth_map must be 2D, "
            f"got {depth_map.shape}"
        )

    bbox = make_bbox(detection)

    grid = grid_extractor.extract(
        depth_map=depth_map,
        bbox=bbox,
        object_id=int(
            detection["object_id"]
        ),
        object_class=str(object_class),
    )

    if grid is not None and not isinstance(
        grid,
        Object3DGrid,
    ):
        raise TypeError(
            "Grid extractor returned "
            f"{type(grid).__name__}, expected Object3DGrid."
        )

    return grid


# ============================================================
# MAIN FRAME PROCESSING
# ============================================================

def process_frame(
    frame,
    frame_id,
    depth_map,
):

    persons, chairs = detect_objects(frame)

    result = {
        "frame_id": int(frame_id),
        "persons": persons,
        "chairs": chairs,
        "pairs": [],
        "detection": {
            "person_count": len(persons),
            "chair_count": len(chairs),
            "both_detected": bool(persons and chairs),
        },
    }

    # --------------------------------------------------------
    # IMPORTANT:
    # Detection result is returned even when depth is unavailable.
    # --------------------------------------------------------

    if depth_map is None:
        result["pipeline_status"] = (
            "DETECTION_OK_DEPTH_UNAVAILABLE"
        )
        return result

    # --------------------------------------------------------
    # Build person grids.
    # --------------------------------------------------------

    person_grids = {}

    for person in persons:

        try:
            grid = extract_object_grid(
                person,
                "person",
                depth_map,
            )

            if grid is not None and grid.point_count > 0:
                person_grids[
                    person["object_id"]
                ] = grid

        except Exception as exc:
            print(
                "[PERSON GRID ERROR]",
                exc,
            )

    # --------------------------------------------------------
    # Build chair grids.
    # --------------------------------------------------------

    chair_grids = {}

    for chair in chairs:

        try:
            grid = extract_object_grid(
                chair,
                "chair",
                depth_map,
            )

            if grid is not None and grid.point_count > 0:
                chair_grids[
                    chair["object_id"]
                ] = grid

        except Exception as exc:
            print(
                "[CHAIR GRID ERROR]",
                exc,
            )

    # --------------------------------------------------------
    # Person-chair pair prediction.
    # --------------------------------------------------------

    for person in persons:

        person_id = person["object_id"]

        person_grid = person_grids.get(
            person_id
        )

        if person_grid is None:
            continue

        for chair in chairs:

            chair_id = chair["object_id"]

            chair_grid = chair_grids.get(
                chair_id
            )

            if chair_grid is None:
                continue

            try:
                # ====================================================
                # CRITICAL FIX:
                #
                # PairFeatureExtractor.extract() expects:
                #
                #     Object3DGrid, Object3DGrid
                #
                # NOT:
                #
                #     feature_dict, feature_dict
                #
                # The previous live file passed:
                #     person["features"]
                #     chair["features"]
                #
                # which caused the feature pipeline to break.
                # ====================================================

                pair_features = pair_extractor.extract(
                    person_grid,
                    chair_grid,
                )

                # Deterministic 34-feature order.
                model_input = np.asarray(
                    pair_extractor.to_vector(
                        pair_features
                    ),
                    dtype=np.float32,
                ).reshape(
                    1,
                    -1,
                )

                if model_input.shape[1] != 34:
                    raise RuntimeError(
                        "Pair feature count is "
                        f"{model_input.shape[1]}, expected 34."
                    )

                # Verify the trained model's schema.
                if feature_columns:
                    if len(feature_columns) != 34:
                        raise RuntimeError(
                            "Stored model schema is not 34 features."
                        )

                predicted_cm = float(
                    distance_model.predict(
                        model_input
                    )[0]
                )

                if not np.isfinite(
                    predicted_cm
                ):
                    predicted_cm = None

                if predicted_cm is not None:
                    predicted_cm = float(
                        np.clip(
                            predicted_cm,
                            0.0,
                            MAX_DEPTH * 100.0,
                        )
                    )

                # ------------------------------------------------
                # Raw geometric distances.
                # ------------------------------------------------

                raw_3d = pair_features.get(
                    "raw_3d_distance",
                    np.nan,
                )

                raw_ground = pair_features.get(
                    "raw_ground_distance",
                    np.nan,
                )

                # Compatibility with any older feature naming.
                if not np.isfinite(raw_3d):
                    raw_3d = pair_features.get(
                        "pair_3d_distance",
                        np.nan,
                    )

                if not np.isfinite(raw_ground):
                    raw_ground = pair_features.get(
                        "pair_ground_distance",
                        np.nan,
                    )

                # ------------------------------------------------
                # Temporal smoothing.
                # ------------------------------------------------

                smooth_cm = None

                if predicted_cm is not None:
                    smooth_cm = distance_smoother.update(
                        (
                            int(person_id),
                            int(chair_id),
                        ),
                        predicted_cm,
                    )

                pairs_entry = {
                    "person_id": int(person_id),
                    "chair_id": int(chair_id),

                    "predicted_distance_cm": (
                        smooth_cm
                        if smooth_cm is not None
                        else predicted_cm
                    ),

                    "raw_model_distance_cm": predicted_cm,

                    "raw_3d_distance_m": (
                        float(raw_3d)
                        if np.isfinite(raw_3d)
                        else None
                    ),

                    "raw_ground_distance_m": (
                        float(raw_ground)
                        if np.isfinite(raw_ground)
                        else None
                    ),

                    "person_grid_points": int(
                        person_grid.point_count
                    ),

                    "chair_grid_points": int(
                        chair_grid.point_count
                    ),

                    "person_grid_valid_ratio": float(
                        person_grid.valid_ratio
                    ),

                    "chair_grid_valid_ratio": float(
                        chair_grid.valid_ratio
                    ),
                }

                pairs_entry["actual_distance_cm"] = None
                pairs_entry["error_cm"] = None

                pairs_entry["pair_score"] = (
                    float(
                        pair_features.get(
                            "raw_3d_distance",
                            np.inf,
                        )
                    )
                    if np.isfinite(
                        pair_features.get(
                            "raw_3d_distance",
                            np.nan,
                        )
                    )
                    else float("inf")
                )

                pairs_entry["status"] = "OK"

                result["pairs"].append(
                    pairs_entry
                )

            except Exception as exc:
                print(
                    "[PAIR/ML ERROR]",
                    f"P{person_id}-C{chair_id}:",
                    type(exc).__name__,
                    exc,
                )

    if persons and chairs:
        if result["pairs"]:
            result["pipeline_status"] = (
                "PERSON_AND_CHAIR_DETECTED_DISTANCE_OK"
            )
        else:
            result["pipeline_status"] = (
                "PERSON_AND_CHAIR_DETECTED_GRID_OR_ML_PENDING"
            )
    elif persons:
        result["pipeline_status"] = (
            "PERSON_DETECTED_NO_CHAIR"
        )
    elif chairs:
        result["pipeline_status"] = (
            "CHAIR_DETECTED_NO_PERSON"
        )
    else:
        result["pipeline_status"] = (
            "NO_PERSON_OR_CHAIR"
        )

    return result


# ============================================================
# DRAWING
# ============================================================

def draw_result(
    frame,
    result,
):

    output = frame.copy()

    persons = result.get(
        "persons",
        [],
    )

    chairs = result.get(
        "chairs",
        [],
    )

    pairs = result.get(
        "pairs",
        [],
    )

    # --------------------------------------------------------
    # PERSONS
    # --------------------------------------------------------

    for person in persons:

        b = person["bbox"]

        x1 = int(b["x1"])
        y1 = int(b["y1"])
        x2 = int(b["x2"])
        y2 = int(b["y2"])

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        label = (
            f"PERSON #{person['object_id']} "
            f"{person['confidence']:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # CHAIRS
    # --------------------------------------------------------

    for chair in chairs:

        b = chair["bbox"]

        x1 = int(b["x1"])
        y1 = int(b["y1"])
        x2 = int(b["x2"])
        y2 = int(b["y2"])

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"CHAIR #{chair['object_id']} "
            f"{chair['confidence']:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Detection status
    # --------------------------------------------------------

    status = result.get(
        "pipeline_status",
        "",
    )

    if persons and chairs:
        status_text = (
            f"PERSONS: {len(persons)} | "
            f"CHAIRS: {len(chairs)}"
        )
    elif persons:
        status_text = (
            f"PERSONS: {len(persons)} | "
            f"CHAIRS: 0"
        )
    elif chairs:
        status_text = (
            f"PERSONS: 0 | "
            f"CHAIRS: {len(chairs)}"
        )
    else:
        status_text = "PERSONS: 0 | CHAIRS: 0"

    cv2.putText(
        output,
        status_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Distances
    # --------------------------------------------------------

    y = 58

    for pair in pairs:

        distance = pair.get(
            "predicted_distance_cm"
        )

        if distance is None:
            continue

        text = (
            f"P{pair['person_id']} <-> "
            f"C{pair['chair_id']} : "
            f"{distance:.1f} cm"
        )

        cv2.putText(
            output,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 28

    # --------------------------------------------------------
    # Pipeline state
    # --------------------------------------------------------

    cv2.putText(
        output,
        status,
        (10, CAMERA_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global camera
    global latest_frame
    global latest_result
    global running

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2,
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS,
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print("\nCamera started.")
    print(
        f"Resolution: "
        f"{actual_width}x{actual_height}"
    )
    print(
        f"Target FPS: {CAMERA_FPS}"
    )

    frame_id = 0
    last_depth = None
    last_depth_frame = -1

    running = True

    while running:

        success, frame = camera.read()

        if not success or frame is None:
            print(
                "[CAMERA] Frame read failed."
            )
            time.sleep(0.05)
            continue

        frame_id += 1

        # ----------------------------------------------------
        # Depth update.
        # ----------------------------------------------------

        if (
            last_depth is None
            or frame_id - last_depth_frame >= DEPTH_INTERVAL
        ):
            try:
                last_depth = estimate_depth(
                    frame
                )

                last_depth_frame = frame_id

            except Exception as exc:
                print(
                    "[DEPTH ERROR]",
                    type(exc).__name__,
                    exc,
                )

        # ----------------------------------------------------
        # Process.
        # ----------------------------------------------------

        try:

            result = process_frame(
                frame,
                frame_id,
                last_depth,
            )

            display_frame = draw_result(
                frame,
                result,
            )

            with frame_lock:
                latest_frame = display_frame
                latest_result = result

        except Exception as exc:

            print(
                "[PROCESS ERROR]",
                type(exc).__name__,
                exc,
            )

            # Even if processing fails, run detection alone
            # so the user can still see person/chair boxes.
            try:
                persons, chairs = detect_objects(
                    frame
                )

                fallback = {
                    "frame_id": frame_id,
                    "persons": persons,
                    "chairs": chairs,
                    "pairs": [],
                    "pipeline_status": (
                        "DETECTION_ONLY_AFTER_PROCESS_ERROR"
                    ),
                }

                display_frame = draw_result(
                    frame,
                    fallback,
                )

                with frame_lock:
                    latest_frame = display_frame
                    latest_result = fallback

            except Exception as detection_exc:
                print(
                    "[DETECTION ERROR]",
                    type(detection_exc).__name__,
                    detection_exc,
                )

        time.sleep(
            max(
                0.0,
                (1.0 / CAMERA_FPS) * 0.05,
            )
        )


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():

    while True:

        with frame_lock:

            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                82,
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

        time.sleep(0.02)


# ============================================================
# WEB PAGE
# ============================================================

@app.route("/")
def index():

    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>BAS-HMR ML Distance</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    margin: 0;
    padding: 20px;
}

.container {
    width: 95%;
    max-width: 1000px;
    margin: auto;
}

h1 {
    margin-bottom: 10px;
}

img {
    width: 100%;
    max-width: 640px;
    border: 2px solid #444;
}

.info {
    margin-top: 15px;
    padding: 15px;
    background: #222;
    border-radius: 8px;
}

pre {
    white-space: pre-wrap;
    word-break: break-word;
}

</style>
</head>

<body>

<div class="container">

<h1>BAS-HMR Realtime ML Distance</h1>

<img src="/video_feed">

<div class="info">
    <p>
        <b>Detector:</b> YOLO11n
    </p>

    <p>
        <b>Objects:</b> Person + Chair
    </p>

    <p>
        <b>Depth:</b> YOLO26n-Depth
    </p>

    <p>
        <b>Distance model:</b> Ridge
    </p>

    <p>
        <b>Features:</b> 34
    </p>

    <p>
        <b>Output:</b> centimetres
    </p>

    <pre id="status">Loading...</pre>
</div>

</div>

<script>

async function updateStatus() {

    try {

        const response =
            await fetch("/api/status");

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
            "Status error: " + error;

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
# VIDEO
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
# STATUS
# ============================================================

@app.route("/api/status")
def status():

    with frame_lock:
        result = latest_result

    if result is None:
        return jsonify(
            {
                "status": "starting",
                "detector": "YOLO11n",
                "person_class": PERSON_CLASS,
                "chair_class": CHAIR_CLASS,
            }
        )

    return jsonify(
        {
            "status": "running",
            "frame_id": result.get(
                "frame_id"
            ),
            "pipeline_status": result.get(
                "pipeline_status"
            ),
            "person_count": len(
                result.get(
                    "persons",
                    [],
                )
            ),
            "chair_count": len(
                result.get(
                    "chairs",
                    [],
                )
            ),
            "persons": result.get(
                "persons",
                [],
            ),
            "chairs": result.get(
                "chairs",
                [],
            ),
            "pairs": result.get(
                "pairs",
                [],
            ),
        }
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health")
def health():

    return jsonify(
        {
            "status": "ok",
            "detector": "YOLO11n",
            "person_class": PERSON_CLASS,
            "chair_class": CHAIR_CLASS,
            "depth_model": "YOLO26n-Depth",
            "distance_model": (
                artifact.get(
                    "model_name"
                )
                if artifact
                else None
            ),
            "features": len(
                feature_columns
            ),
            "camera_open": (
                camera is not None
                and camera.isOpened()
            ),
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    load_models()

    worker = threading.Thread(
        target=camera_loop,
        daemon=True,
        name="bas-hmr-camera",
    )

    worker.start()

    print(
        "\n" + "=" * 70
    )

    print(
        "SERVER STARTED"
    )

    print(
        f"Open in Windows Chrome:"
    )

    print(
        f"http://localhost:{FLASK_PORT}"
    )

    print(
        f"API:"
    )

    print(
        f"http://localhost:{FLASK_PORT}/api/status"
    )

    print(
        "=" * 70
    )

    try:

        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    except KeyboardInterrupt:

        print(
            "\nStopping..."
        )

    finally:

        running = False

        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass

        print(
            "Camera released."
        )


if __name__ == "__main__":
    main()
