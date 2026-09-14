"""
BAS Metric Distance Validation
--------------------------------

Live pipeline:

Camera
  ↓
YOLO11n
  ↓
Person + Chair Detection
  ↓
Metric Depth Anything V2
  ↓
Camera Model
  ↓
3D Reconstruction
  ↓
Person-Chair Distance
  ↓
Median + EMA Filtering
  ↓
Live Flask Visualization
  ↓
Validation Recording

IMPORTANT:
- Distance is currently based on bbox-center points.
- Camera intrinsics are development values.
- This is a validation system, not yet final flight-grade measurement.
"""

import os
import sys
import json
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# BAS modules
# ---------------------------------------------------------------------

from human.metric_depth import MetricDepthEstimator
from human.camera_model import CameraModel
from human.metric_3d import Metric3DExtractor
from human.metric_distance import MetricDistanceEngine
from human.distance_filter import DistanceFilter


# =====================================================================
# CONFIGURATION
# =====================================================================

CAMERA_DEVICE = "/dev/video0"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 10

YOLO_MODEL = "yolo11n.pt"

YOLO_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 416

PERSON_CLASS_ID = 0
CHAIR_CLASS_ID = 56

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Development camera intrinsics.
#
# These values MUST eventually be replaced by real camera calibration.
FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0

# Distance classification
NEAR_DISTANCE_M = 0.75
FAR_DISTANCE_M = 2.0

# ---------------------------------------------------------------------
# Metric distance behaviour
# ---------------------------------------------------------------------
# A monocular depth model can have a systematic scale error.
# Keep this at 1.0 until a tape-measured reference distance is used.
DISTANCE_SCALE_FILE = os.path.join(
    PROJECT_ROOT,
    "test_results",
    "distance_validation",
    "distance_scale.json",
)

INITIAL_DISTANCE_SCALE = 1.0

# Report the nearest chair for each detected person. This avoids
# displaying unrelated person-chair combinations when several chairs
# are visible.
ONLY_NEAREST_CHAIR_PER_PERSON = True

# For "chair-to-human" proximity, use horizontal ground-plane distance
# (camera X/Z plane) rather than full body-center 3D distance.
USE_HORIZONTAL_CHAIR_HUMAN_DISTANCE = True

# Live fallback when monocular metric depth is unstable.
# This produces an explicitly ESTIMATED distance from the Person-ID /
# Chair-ID image geometry. It must not be presented as calibrated physics.
USE_ID_DISTANCE_ESTIMATE = True
ID_EST_REFERENCE_CM = 200.0
ID_EST_REFERENCE_PIXEL_GAP = 320.0
ID_EST_MIN_CM = 20.0
ID_EST_MAX_CM = 500.0
ID_EST_EMA_ALPHA = 0.25


# Filtering
FILTER_WINDOW = 5
FILTER_ALPHA = 0.35
FILTER_MAX_JUMP_M = 0.50

# Validation
VALIDATION_DIR = os.path.join(
    PROJECT_ROOT,
    "test_results",
    "distance_validation",
)

VALIDATION_FILE = os.path.join(
    VALIDATION_DIR,
    "live_validation_samples.json",
)

# Flask
HOST = "0.0.0.0"
PORT = 5000


# =====================================================================
# GLOBAL STATE
# =====================================================================

app = Flask(__name__)

stop_event = threading.Event()

camera_lock = threading.Lock()
state_lock = threading.Lock()

latest_frame = None
latest_output_frame = None

latest_state = {
    "timestamp": None,
    "frame_id": 0,
    "camera": {
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "fps": CAMERA_FPS,
    },
    "persons": [],
    "chairs": [],
    "relationships": [],
}

frame_counter = 0

processing_fps = 0.0
last_processing_time = 0.0

validation_samples = []

# Restore previous validation samples if available.
if os.path.exists(VALIDATION_FILE):
    try:
        with open(VALIDATION_FILE, "r", encoding="utf-8") as file:
            loaded_samples = json.load(file)
        if isinstance(loaded_samples, list):
            validation_samples = loaded_samples
            print(
                f"[VALIDATION] Loaded {len(validation_samples)} "
                f"previous samples."
            )
    except Exception as exc:
        print(f"[VALIDATION] Could not load previous samples: {exc}")

# Runtime metric scale correction.
distance_scale_factor = INITIAL_DISTANCE_SCALE

# Latest closest chair-human pair, useful for the live display.
closest_relationship = None

# Smoothed distance for each persistent Person-ID -> Chair-ID pair.
id_distance_history = {}



# =====================================================================
# CREATE OUTPUT DIRECTORY
# =====================================================================

os.makedirs(VALIDATION_DIR, exist_ok=True)


def load_distance_scale():
    """Load the persisted metric scale factor."""
    if not os.path.exists(DISTANCE_SCALE_FILE):
        return INITIAL_DISTANCE_SCALE

    try:
        with open(DISTANCE_SCALE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        scale = safe_float(
            data.get("scale_factor"),
            INITIAL_DISTANCE_SCALE,
        )

        if scale is None or scale <= 0:
            return INITIAL_DISTANCE_SCALE

        print(
            f"[CALIBRATION] Loaded distance scale factor: "
            f"{scale:.6f}"
        )
        return scale

    except Exception as exc:
        print(
            f"[CALIBRATION] Could not load scale: {exc}. "
            f"Using {INITIAL_DISTANCE_SCALE:.3f}."
        )
        return INITIAL_DISTANCE_SCALE


def save_distance_scale(scale_factor, actual_distance_m, measured_distance_m):
    """Persist the current metric scale factor."""
    data = {
        "scale_factor": float(scale_factor),
        "actual_distance_m": float(actual_distance_m),
        "measured_distance_m_before_scale": float(measured_distance_m),
        "updated_at": datetime.now().isoformat(),
    }

    with open(
        DISTANCE_SCALE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(data, file, indent=2)


distance_scale_factor = INITIAL_DISTANCE_SCALE


# =====================================================================
# LOAD MODELS
# =====================================================================

print()
print("=" * 70)
print("OPENING CAMERA")
print("=" * 70)

camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

if not camera.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA_DEVICE}"
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

# MJPEG helps USB webcams maintain stable performance.
camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG"),
)

actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = camera.get(cv2.CAP_PROP_FPS)

print(f"Camera device : {CAMERA_DEVICE}")
print(f"Resolution     : {actual_width}x{actual_height}")
print(f"FPS            : {actual_fps:.2f}")
print("Camera opened successfully.")


# =====================================================================
# YOLO
# =====================================================================

print()
print("=" * 70)
print("LOADING METRIC DISTANCE PIPELINE")
print("=" * 70)

from ultralytics import YOLO

print()
print(f"[YOLO] Loading {YOLO_MODEL}")

yolo = YOLO(
    os.path.join(
        PROJECT_ROOT,
        YOLO_MODEL,
    )
)

print("[YOLO] Model loaded.")


# =====================================================================
# METRIC DEPTH
# =====================================================================

print("[DEPTH] Loading metric depth model...")

depth_estimator = MetricDepthEstimator(
    model_name=DEPTH_MODEL
)

print("[DEPTH] Metric depth loaded.")


# =====================================================================
# CAMERA MODEL
# =====================================================================

camera_model = CameraModel(
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
)

print(
    f"[CAMERA MODEL] "
    f"fx={FX}, fy={FY}, cx={CX}, cy={CY}"
)


# =====================================================================
# 3D EXTRACTOR
# =====================================================================

metric_3d = Metric3DExtractor(
    camera_model=camera_model
)

print("[3D] Metric 3D extractor ready.")


# =====================================================================
# DISTANCE ENGINE
# =====================================================================

distance_engine = MetricDistanceEngine(
    near_threshold_m=NEAR_DISTANCE_M,
    far_threshold_m=FAR_DISTANCE_M,
)

print("[DISTANCE] Metric distance engine ready.")


# =====================================================================
# DISTANCE FILTER
# =====================================================================

distance_filter = DistanceFilter(
    window_size=FILTER_WINDOW,
    alpha=FILTER_ALPHA,
    max_jump_m=FILTER_MAX_JUMP_M,
)

print("[FILTER] Distance filter ready.")

print()
print("Metric distance pipeline loaded successfully.")


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def bbox_to_dict(bbox):
    """
    Convert internal tuple/list bbox:

        (x1, y1, x2, y2)

    into JSON-friendly dictionary.
    """

    if bbox is None:
        return None

    return {
        "x1": float(bbox[0]),
        "y1": float(bbox[1]),
        "x2": float(bbox[2]),
        "y2": float(bbox[3]),
    }


def bbox_center(bbox):
    """
    Calculate image-space bbox center.

    bbox MUST be:
        (x1, y1, x2, y2)
    """

    return (
        int((bbox[0] + bbox[2]) / 2),
        int((bbox[1] + bbox[3]) / 2),
    )


def safe_float(value, default=None):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


# Load persisted scale after helper functions are defined.
distance_scale_factor = load_distance_scale()


def point_to_dict(point):
    if point is None:
        return None

    return {
        "x": safe_float(point.x),
        "y": safe_float(point.y),
        "z": safe_float(point.z),
    }


def metric_object_to_dict(obj):
    """
    Convert MetricObject3D to JSON-friendly dictionary.
    """

    return {
        "object_id": int(obj.object_id),
        "object_type": str(obj.object_type),
        "confidence": safe_float(obj.confidence),
        "bbox": bbox_to_dict(obj.bbox),
        "reference_pixel": {
            "u": int(obj.reference_pixel[0]),
            "v": int(obj.reference_pixel[1]),
        },
        "depth_m": safe_float(obj.depth_m),
        "position_3d_m": point_to_dict(obj.position),
        "timestamp": safe_float(obj.timestamp),
    }


# =====================================================================
# YOLO DETECTION
# =====================================================================

def detect_objects(frame):
    """
    Detect persons and chairs.

    Returns:

    persons = [
        {
            "person_id": ...,
            "bbox": (...),
            "confidence": ...
        }
    ]

    chairs = [
        {
            "chair_id": ...,
            "bbox": (...),
            "confidence": ...
        }
    ]
    """

    results = yolo.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        classes=[
            PERSON_CLASS_ID,
            CHAIR_CLASS_ID,
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

    boxes = result.boxes

    person_index = 0
    chair_index = 0

    for box in boxes:

        cls = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = xyxy.tolist()

        # Clamp to image dimensions.
        x1 = max(0, min(FRAME_WIDTH - 1, x1))
        y1 = max(0, min(FRAME_HEIGHT - 1, y1))
        x2 = max(0, min(FRAME_WIDTH - 1, x2))
        y2 = max(0, min(FRAME_HEIGHT - 1, y2))

        bbox = (
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        )

        if cls == PERSON_CLASS_ID:

            persons.append(
                {
                    "person_id": person_index,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

            person_index += 1

        elif cls == CHAIR_CLASS_ID:

            chairs.append(
                {
                    "chair_id": chair_index,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

            chair_index += 1

    return persons, chairs


# =====================================================================
# 3D EXTRACTION
# =====================================================================

def create_metric_objects(
    persons,
    chairs,
    depth_map,
    timestamp,
):
    """
    Convert 2D detections into metric 3D objects.
    """

    metric_persons = []
    metric_chairs = []

    # -------------------------------------------------------------
    # PERSONS
    # -------------------------------------------------------------

    for person in persons:

        try:

            obj = metric_3d.extract(
                object_id=person["person_id"],
                object_type="person",
                confidence=person["confidence"],
                bbox=person["bbox"],
                depth_map=depth_map,
                timestamp=timestamp,
            )

            if obj is not None:
                metric_persons.append(obj)

        except Exception as exc:

            print(
                f"[3D] Person extraction error: {exc}"
            )

    # -------------------------------------------------------------
    # CHAIRS
    # -------------------------------------------------------------

    for chair in chairs:

        try:

            obj = metric_3d.extract(
                object_id=chair["chair_id"],
                object_type="chair",
                confidence=chair["confidence"],
                bbox=chair["bbox"],
                depth_map=depth_map,
                timestamp=timestamp,
            )

            if obj is not None:
                metric_chairs.append(obj)

        except Exception as exc:

            print(
                f"[3D] Chair extraction error: {exc}"
            )

    return metric_persons, metric_chairs


# =====================================================================
# DISTANCE CALCULATION
# =====================================================================

def result_value(result, name, default=None):
    """
    Read a field from MetricDistanceResult or from a dictionary.

    MetricDistanceEngine.calculate() returns a MetricDistanceResult
    object, not a dictionary. The previous implementation called
    result.get(...), which caused:
        AttributeError: 'MetricDistanceResult' object has no attribute 'get'
    """
    if result is None:
        return default

    if isinstance(result, dict):
        return result.get(name, default)

    return getattr(result, name, default)



def _object_bbox_center(obj):
    """Get image-space center from a MetricObject3D bbox."""
    try:
        b = obj.bbox
        return (
            (float(b[0]) + float(b[2])) / 2.0,
            (float(b[1]) + float(b[3])) / 2.0,
        )
    except Exception:
        return None


def estimate_id_distance_cm(person, chair):
    """
    Stable heuristic estimate for a Person-ID / Chair-ID pair.

    It uses image-space center separation plus EMA smoothing. The result is
    deliberately marked ESTIMATED because monocular image geometry alone
    cannot guarantee physical centimetres.
    """
    p = _object_bbox_center(person)
    c = _object_bbox_center(chair)

    if p is None or c is None:
        return None

    dx = p[0] - c[0]
    dy = p[1] - c[1]
    pixel_gap = float(np.hypot(dx, dy))

    ref_gap = max(1.0, float(ID_EST_REFERENCE_PIXEL_GAP))
    ref_cm = float(ID_EST_REFERENCE_CM)

    if pixel_gap <= ref_gap:
        estimate = ref_cm * (1.0 - 0.55 * pixel_gap / ref_gap)
    else:
        estimate = ref_cm * (ref_gap / pixel_gap)

    estimate = float(np.clip(
        estimate,
        ID_EST_MIN_CM,
        ID_EST_MAX_CM,
    ))

    key = (int(person.object_id), int(chair.object_id))
    previous = id_distance_history.get(key)

    if previous is not None:
        a = float(ID_EST_EMA_ALPHA)
        estimate = a * estimate + (1.0 - a) * previous

    id_distance_history[key] = estimate
    return estimate


def estimate_id_relationships(metric_persons, metric_chairs):
    """Return nearest-chair relationship for every Person ID."""
    results = []

    for person in metric_persons:
        best = None

        for chair in metric_chairs:
            cm = estimate_id_distance_cm(person, chair)
            if cm is None or not np.isfinite(cm):
                continue

            relation = (
                "near" if cm <= 75.0
                else "medium" if cm <= 200.0
                else "far"
            )

            item = {
                "person_id": int(person.object_id),
                "chair_id": int(chair.object_id),
                "distance_cm": float(cm),
                "distance_m": float(cm / 100.0),
                "distance_source": "person_chair_id_image_estimate",
                "distance_status": "ESTIMATED",
                "relationship": relation,
                "motion_state": "estimated",
                "velocity_mps": 0.0,
            }

            if best is None or cm < best["distance_cm"]:
                best = item

        if best is not None:
            results.append(best)

    results.sort(key=lambda x: x["distance_cm"])
    return results


def calculate_relationships(
    metric_persons,
    metric_chairs,
    timestamp,
):
    """
    Build the live Person-ID -> nearest Chair-ID distance table.

    ID-based estimation is the user-facing mode because the current
    monocular metric-depth scale is not sufficiently reliable.
    """
    global closest_relationship

    if USE_ID_DISTANCE_ESTIMATE:
        relationships = estimate_id_relationships(
            metric_persons,
            metric_chairs,
        )

        # Keep metric-depth output as a diagnostic only.
        for rel in relationships:
            for person in metric_persons:
                if int(person.object_id) != rel["person_id"]:
                    continue
                for chair in metric_chairs:
                    if int(chair.object_id) != rel["chair_id"]:
                        continue
                    try:
                        raw = distance_engine.calculate(
                            person,
                            chair,
                            timestamp=timestamp,
                        )
                        metric_m = safe_float(
                            result_value(raw, "distance_m")
                        )
                        if metric_m is not None:
                            rel["metric_depth_distance_cm"] = metric_m * 100.0
                            rel["metric_depth_distance_m"] = metric_m
                    except Exception:
                        pass
                    break
                break

        closest_relationship = relationships[0] if relationships else None
        return relationships

    # Original metric mode remains available if the flag is disabled.
    candidates = []

    for person in metric_persons:
        for chair in metric_chairs:
            try:
                raw = distance_engine.calculate(
                    person, chair, timestamp=timestamp
                )
                engine_m = safe_float(result_value(raw, "distance_m"))
                if engine_m is None or engine_m <= 0:
                    continue

                dx = float(person.position.x) - float(chair.position.x)
                dy = float(person.position.y) - float(chair.position.y)
                dz = float(person.position.z) - float(chair.position.z)

                horizontal_m = float(np.hypot(dx, dz))
                selected_m = horizontal_m if USE_HORIZONTAL_CHAIR_HUMAN_DISTANCE else engine_m
                selected_m *= distance_scale_factor

                if not np.isfinite(selected_m) or selected_m <= 0:
                    continue

                filt = distance_filter.update(
                    person_id=person.object_id,
                    chair_id=chair.object_id,
                    distance_m=selected_m,
                    timestamp=timestamp,
                )

                filtered_m = safe_float(
                    result_value(filt, "filtered_distance_m", selected_m),
                    selected_m,
                )

                relationship = (
                    "near" if filtered_m <= NEAR_DISTANCE_M
                    else "medium" if filtered_m <= FAR_DISTANCE_M
                    else "far"
                )

                candidates.append({
                    "person_id": int(person.object_id),
                    "chair_id": int(chair.object_id),
                    "distance_cm": filtered_m * 100.0,
                    "distance_m": filtered_m,
                    "distance_source": "metric_depth",
                    "distance_status": "METRIC",
                    "relationship": relationship,
                    "velocity_mps": safe_float(
                        result_value(raw, "velocity_mps", 0.0), 0.0
                    ),
                    "motion_state": str(
                        result_value(raw, "motion_state", "stationary")
                    ),
                })
            except Exception as exc:
                print(
                    f"[DISTANCE] Person {person.object_id} "
                    f"Chair {chair.object_id}: {exc}"
                )

    if ONLY_NEAREST_CHAIR_PER_PERSON:
        nearest = {}
        for item in candidates:
            pid = item["person_id"]
            if pid not in nearest or item["distance_cm"] < nearest[pid]["distance_cm"]:
                nearest[pid] = item
        candidates = list(nearest.values())

    candidates.sort(key=lambda x: x["distance_cm"])
    closest_relationship = candidates[0] if candidates else None
    return candidates


# =====================================================================
# VISUALIZATION
# =====================================================================

def normalize_bbox(bbox):
    """
    Normalize a bounding box into:
        (x1, y1, x2, y2)

    Accepts both formats used by this pipeline:
        1. tuple/list/NumPy array: (x1, y1, x2, y2)
        2. JSON dictionary:
           {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    """
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        try:
            return (
                float(bbox["x1"]),
                float(bbox["y1"]),
                float(bbox["x2"]),
                float(bbox["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    if isinstance(bbox, (tuple, list, np.ndarray)):
        if len(bbox) >= 4:
            try:
                return (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except (TypeError, ValueError):
                return None

    return None


def draw_visualization(
    frame,
    persons_json,
    chairs_json,
    relationships,
    depth_map=None,
):
    """Draw ID boxes and Person-ID -> Chair-ID distance in cm."""

    output = frame.copy()

    def seq(value):
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist() if value.ndim else [value.item()]
        if isinstance(value, (list, tuple)):
            return value
        return [value]

    def get_bbox(value):
        try:
            if isinstance(value, dict):
                if all(k in value for k in ("x1", "y1", "x2", "y2")):
                    v = [value["x1"], value["y1"], value["x2"], value["y2"]]
                else:
                    return None
            else:
                v = list(value)[:4]

            x1, y1, x2, y2 = [int(float(x)) for x in v]
            x1 = max(0, min(output.shape[1] - 1, x1))
            y1 = max(0, min(output.shape[0] - 1, y1))
            x2 = max(0, min(output.shape[1] - 1, x2))
            y2 = max(0, min(output.shape[0] - 1, y2))
            return x1, y1, x2, y2
        except Exception:
            return None

    def centre(box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    persons = seq(persons_json)
    chairs = seq(chairs_json)
    rels = seq(relationships)

    person_centres = {}
    chair_centres = {}

    # GREEN = PERSON
    for i, person in enumerate(persons):
        if not isinstance(person, dict):
            continue
        box = get_bbox(person.get("bbox"))
        if box is None:
            continue

        pid = int(person.get("person_id", i))
        person_centres[pid] = centre(box)

        cv2.rectangle(
            output, (box[0], box[1]), (box[2], box[3]),
            (0, 220, 0), 3
        )
        cv2.putText(
            output, f"PERSON ID:{pid}",
            (box[0], max(24, box[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            (0, 220, 0), 2, cv2.LINE_AA
        )

    # BLUE = CHAIR
    for i, chair in enumerate(chairs):
        if not isinstance(chair, dict):
            continue
        box = get_bbox(chair.get("bbox"))
        if box is None:
            continue

        cid = int(chair.get("chair_id", i))
        chair_centres[cid] = centre(box)

        cv2.rectangle(
            output, (box[0], box[1]), (box[2], box[3]),
            (255, 140, 0), 3
        )
        cv2.putText(
            output, f"CHAIR ID:{cid}",
            (box[0], min(output.shape[0] - 8, box[3] + 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            (255, 140, 0), 2, cv2.LINE_AA
        )

    rows = []

    # YELLOW = ACTIVE PERSON-ID -> CHAIR-ID distance.
    for rel in rels:
        if not isinstance(rel, dict):
            continue

        try:
            pid = int(rel["person_id"])
            cid = int(rel["chair_id"])
            cm = safe_float(rel.get("distance_cm"))

            if cm is None or not np.isfinite(cm):
                continue
            if pid not in person_centres or cid not in chair_centres:
                continue

            p = person_centres[pid]
            c = chair_centres[cid]

            cv2.line(
                output, p, c,
                (0, 255, 255), 3, cv2.LINE_AA
            )

            mx = (p[0] + c[0]) // 2
            my = (p[1] + c[1]) // 2

            label = f"P{pid} -> C{cid}: {cm:.1f} cm"
            if rel.get("distance_status") == "ESTIMATED":
                label += " EST"

            (tw, th), base = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2
            )

            cv2.rectangle(
                output,
                (mx - tw // 2 - 8, my - th - 8),
                (mx + tw // 2 + 8, my + base + 8),
                (0, 0, 0), -1
            )

            cv2.putText(
                output, label,
                (mx - tw // 2, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (0, 255, 255), 2, cv2.LINE_AA
            )

            rows.append((pid, cid, cm, rel.get("relationship", "")))

        except Exception:
            continue

    rows.sort(key=lambda x: x[2])

    # BAS dashboard.
    panel_w = min(500, output.shape[1] - 20)
    panel_h = min(
        output.shape[0] - 20,
        118 + max(1, len(rows)) * 30
    )

    overlay = output.copy()
    cv2.rectangle(
        overlay, (10, 10),
        (10 + panel_w, 10 + panel_h),
        (10, 10, 10), -1
    )
    output = cv2.addWeighted(overlay, 0.82, output, 0.18, 0)

    cv2.putText(
        output, "BAS HUMAN-CHAIR ID MONITOR",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"PERSONS: {len(persons)}   CHAIRS: {len(chairs)}",
        (22, 64),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
        (220, 220, 220), 1, cv2.LINE_AA
    )

    cv2.putText(
        output,
        "GREEN=PERSON  BLUE=CHAIR  YELLOW=ID DISTANCE",
        (22, 88),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (0, 255, 255), 1, cv2.LINE_AA
    )

    y = 114

    if not rows:
        cv2.putText(
            output, "No Person-ID -> Chair-ID pair",
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (180, 180, 180), 1, cv2.LINE_AA
        )
    else:
        for pid, cid, cm, relation in rows[:8]:
            line = f"PERSON {pid} -> CHAIR {cid}: {cm:.1f} cm EST"
            if relation:
                line += f" [{relation}]"

            cv2.putText(
                output, line[:70],
                (22, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (0, 255, 255), 1, cv2.LINE_AA
            )
            y += 30

    return output


def camera_loop():

    global latest_frame

    print("[CAMERA THREAD] Started.")

    while not stop_event.is_set():

        success, frame = camera.read()

        if not success:

            print(
                "[CAMERA THREAD] "
                "Failed to read frame."
            )

            time.sleep(0.05)
            continue

        with camera_lock:

            latest_frame = frame.copy()

    print("[CAMERA THREAD] Stopped.")


# =====================================================================
# PROCESSING THREAD
# =====================================================================

def processing_loop():

    global latest_output_frame
    global latest_state
    global frame_counter
    global processing_fps
    global last_processing_time

    print("[PROCESSING THREAD] Started.")

    last_frame = None

    while not stop_event.is_set():

        # -------------------------------------------------------------
        # Get latest camera frame.
        # -------------------------------------------------------------

        with camera_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.01)
            continue

        # Avoid processing the exact same frame repeatedly.
        if last_frame is not None:

            if (
                frame.shape == last_frame.shape
                and np.array_equal(
                    frame,
                    last_frame,
                )
            ):

                time.sleep(0.01)
                continue

        last_frame = frame.copy()

        start_time = time.time()

        frame_counter += 1

        timestamp = time.time()

        # -------------------------------------------------------------
        # YOLO
        # -------------------------------------------------------------

        try:
            persons, chairs = detect_objects(frame)
        except Exception as exc:
            print(f"[YOLO] Detection error: {exc}")
            time.sleep(0.05)
            continue

        # -------------------------------------------------------------
        # Metric depth
        # -------------------------------------------------------------

        try:

            depth_map = depth_estimator.estimate(
                frame
            )

        except Exception as exc:

            print(
                f"[DEPTH] Error: {exc}"
            )

            time.sleep(0.01)
            continue

        # -------------------------------------------------------------
        # 3D
        # -------------------------------------------------------------

        metric_persons, metric_chairs = (
            create_metric_objects(
                persons,
                chairs,
                depth_map,
                timestamp,
            )
        )

        # -------------------------------------------------------------
        # Convert to dictionaries.
        # -------------------------------------------------------------

        persons_json = [
            metric_object_to_dict(
                person
            )
            for person in metric_persons
        ]

        chairs_json = [
            metric_object_to_dict(
                chair
            )
            for chair in metric_chairs
        ]

        # -------------------------------------------------------------
        # Distance
        # -------------------------------------------------------------

        relationships = calculate_relationships(
            metric_persons,
            metric_chairs,
            timestamp,
        )

        # -------------------------------------------------------------
        # State
        # -------------------------------------------------------------

        elapsed = time.time() - start_time

        if elapsed > 0:

            processing_fps = 1.0 / elapsed

        last_processing_time = elapsed

        state = {
            "timestamp": timestamp,
            "frame_id": frame_counter,
            "camera": {
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "fps": CAMERA_FPS,
            },
            "persons": persons_json,
            "chairs": chairs_json,
            "relationships": relationships,
            "performance": {
                "processing_time_s": elapsed,
                "processing_fps": processing_fps,
            },
            "distance_configuration": {
                "unit": "cm",
                "distance_scale_factor": distance_scale_factor,
                "horizontal_distance_mode":
                    USE_HORIZONTAL_CHAIR_HUMAN_DISTANCE,
                "nearest_chair_only":
                    ONLY_NEAREST_CHAIR_PER_PERSON,
                "id_distance_estimate":
                    USE_ID_DISTANCE_ESTIMATE,
                "distance_status":
                    "ESTIMATED" if USE_ID_DISTANCE_ESTIMATE else "METRIC",
            },
        }

        # -------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------

        visualization = draw_visualization(
            frame,
            persons_json,
            chairs_json,
            relationships,
            depth_map,
        )

        # -------------------------------------------------------------
        # Update global state.
        # -------------------------------------------------------------

        with state_lock:

            latest_state = state

            latest_output_frame = (
                visualization.copy()
            )

    print("[PROCESSING THREAD] Stopped.")


# =====================================================================
# MJPEG VIDEO STREAM
# =====================================================================

def generate_frames():

    while not stop_event.is_set():

        with state_lock:

            if latest_output_frame is None:

                frame = None

            else:

                frame = (
                    latest_output_frame.copy()
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

            time.sleep(0.01)
            continue

        frame_bytes = encoded.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.03)


# =====================================================================
# HTML
# =====================================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>BAS Chair-Human Distance (cm)</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
}

h1 {
    margin-top: 0;
}

.container {
    max-width: 1100px;
    margin: auto;
}

.video {
    width: 100%;
    max-width: 960px;
    border: 2px solid #444;
}

.panel {
    background: #222;
    padding: 15px;
    margin-top: 15px;
    border-radius: 8px;
}

.value {
    font-size: 22px;
}

input {
    padding: 8px;
    margin: 5px;
}

button {
    padding: 9px 15px;
    cursor: pointer;
}

pre {
    white-space: pre-wrap;
}

</style>

</head>

<body>

<div class="container">

<h1>BAS Chair-Human Distance</h1>

<img
    class="video"
    src="/video"
>

<div class="panel">

<h2>Live State</h2>

<div id="state">
Loading...
</div>

</div>

<div class="panel">

<h2>Metric Scale Calibration</h2>

<p>
Measure the actual <b>chair-to-human distance</b> with a tape.
Use the same closest person/chair pair visible in the camera.
</p>

<input
    id="calibrationDistanceCm"
    type="number"
    step="0.1"
    min="1"
    placeholder="Actual distance in cm"
>

<button onclick="calibrateScale()">
Calibrate Scale
</button>

<div id="calibrationResult">
No calibration performed in this session.
</div>

</div>

<div class="panel">

<h2>Validation</h2>

<label>
Actual measured distance (cm):
</label>

<input
    id="actualDistance"
    type="number"
    step="0.1"
    min="0"
    placeholder="e.g. 150"
>

<input
    id="notes"
    type="text"
    placeholder="Notes"
>

<button onclick="recordSample()">
Record Sample
</button>

<div id="validation">
Loading...
</div>

</div>

</div>

<script>

async function updateState() {

    try {

        const response =
            await fetch("/status");

        const data =
            await response.json();

        document.getElementById(
            "state"
        ).innerHTML =
            "<pre>" +
            JSON.stringify(
                data,
                null,
                2
            ) +
            "</pre>";

    } catch (error) {

        console.log(error);

    }

}


async function updateValidation() {

    try {

        const response =
            await fetch("/validation");

        const data =
            await response.json();

        document.getElementById(
            "validation"
        ).innerHTML =
            "<pre>" +
            JSON.stringify(
                data,
                null,
                2
            ) +
            "</pre>";

    } catch (error) {

        console.log(error);

    }

}


async function calibrateScale() {

    const actualCm =
        document.getElementById(
            "calibrationDistanceCm"
        ).value;

    if (!actualCm) {
        alert(
            "Enter the actual chair-human distance in cm."
        );
        return;
    }

    const response =
        await fetch(
            "/calibrate",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    actual_distance_cm:
                        parseFloat(actualCm)
                })
            }
        );

    const data =
        await response.json();

    document.getElementById(
        "calibrationResult"
    ).innerHTML =
        "<pre>" +
        JSON.stringify(
            data,
            null,
            2
        ) +
        "</pre>";

    updateState();
}


async function recordSample() {

    const actual =
        document.getElementById(
            "actualDistance"
        ).value;

    const notes =
        document.getElementById(
            "notes"
        ).value;

    if (!actual) {

        alert(
            "Enter the actual chair-human distance in cm."
        );

        return;
    }

    const response =
        await fetch(
            "/record",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    actual_distance_cm:
                        parseFloat(actual),

                    notes: notes
                })
            }
        );

    const data =
        await response.json();

    alert(
        data.message ||
        "Sample recorded."
    );

    updateValidation();
}


setInterval(
    updateState,
    1000
);

setInterval(
    updateValidation,
    2000
);

updateState();
updateValidation();

</script>

</body>

</html>
"""


# =====================================================================
# FLASK ROUTES
# =====================================================================

@app.route("/")
def index():

    return HTML_PAGE


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.route("/status")
def status():

    with state_lock:
        # JSON round-trip creates an isolated snapshot of nested lists/dicts.
        state = json.loads(json.dumps(latest_state, default=str))

    return jsonify(state)


@app.route("/distances")
def distances():
    """Return current person-chair distances in centimetres."""
    with state_lock:
        state = dict(latest_state)

    rels = (
        state.get("relationships")
        if state.get("relationships") is not None
        else []
    )

    result = []

    for rel in rels:
        if not isinstance(rel, dict):
            continue

        cm = safe_float(rel.get("distance_cm"))

        if cm is None:
            metres = safe_float(rel.get("distance_m"))
            if metres is not None:
                cm = metres * 100.0

        if cm is None or not np.isfinite(cm):
            continue

        result.append({
            "person_id": rel.get("person_id"),
            "chair_id": rel.get("chair_id"),
            "distance_cm": round(cm, 1),
            "distance_m": round(cm / 100.0, 3),
            "relationship": rel.get("relationship"),
            "motion_state": rel.get("motion_state"),
        })

    return jsonify({
        "timestamp": state.get("timestamp"),
        "frame_id": state.get("frame_id"),
        "distances": result,
    })


@app.route("/validation")
def validation():

    with state_lock:
        samples = json.loads(
            json.dumps(validation_samples, default=str)
        )

    return jsonify(
        {
            "sample_count": len(samples),
            "samples": samples,
            "file": VALIDATION_FILE,
        }
    )


# =====================================================================
# LIVE SCALE CALIBRATION
# =====================================================================

@app.route(
    "/calibrate",
    methods=["POST"],
)
def calibrate_scale():
    """
    Calibrate the global metric scale from one tape-measured
    chair-human distance.

    The user supplies the real distance in centimetres.
    The current live measured distance is taken from the closest
    detected person-chair pair.

    New scale:
        actual_distance / current_distance
    """

    global distance_scale_factor

    data = request.get_json(
        silent=True
    )

    if data is None:
        return jsonify(
            {
                "success": False,
                "message": "Invalid JSON.",
            }
        ), 400

    actual_cm = safe_float(
        data.get("actual_distance_cm")
    )

    if actual_cm is None or actual_cm <= 0:
        return jsonify(
            {
                "success": False,
                "message": (
                    "actual_distance_cm must "
                    "be greater than zero."
                ),
            }
        ), 400

    with state_lock:
        state = dict(latest_state)

    relationships = (
        state.get("relationships")
        if state.get("relationships") is not None
        else []
    )

    if not relationships:
        return jsonify(
            {
                "success": False,
                "message": (
                    "No person-chair pair is "
                    "currently detected."
                ),
            }
        ), 400

    relation = min(
        relationships,
        key=lambda item: item.get(
            "distance_m",
            float("inf"),
        ),
    )

    measured_m_before_scale = safe_float(
        relation.get("raw_distance_m")
    )

    if (
        measured_m_before_scale is None
        or measured_m_before_scale <= 0
    ):
        return jsonify(
            {
                "success": False,
                "message": (
                    "Current distance is not "
                    "valid for calibration."
                ),
            }
        ), 400

    actual_m = actual_cm / 100.0

    new_scale = (
        actual_m
        / measured_m_before_scale
    )

    if (
        not np.isfinite(new_scale)
        or new_scale <= 0
        or new_scale > 10.0
        or new_scale < 0.1
    ):
        return jsonify(
            {
                "success": False,
                "message": (
                    "Calculated scale factor is "
                    "outside the safe range "
                    "(0.1 to 10.0)."
                ),
            }
        ), 400

    distance_scale_factor = float(
        new_scale
    )

    try:
        save_distance_scale(
            distance_scale_factor,
            actual_m,
            measured_m_before_scale,
        )
    except Exception as exc:
        return jsonify(
            {
                "success": False,
                "message": (
                    f"Could not save calibration: {exc}"
                ),
            }
        ), 500

    print()
    print("=" * 70)
    print("DISTANCE SCALE CALIBRATED")
    print("=" * 70)
    print(
        f"Actual distance       : {actual_cm:.2f} cm"
    )
    print(
        f"Previous raw distance : "
        f"{measured_m_before_scale * 100.0:.2f} cm"
    )
    print(
        f"New scale factor      : "
        f"{distance_scale_factor:.6f}"
    )
    print("=" * 70)

    return jsonify(
        {
            "success": True,
            "message": "Distance scale calibrated.",
            "actual_distance_cm": actual_cm,
            "raw_distance_before_scale_cm": (
                measured_m_before_scale * 100.0
            ),
            "scale_factor": distance_scale_factor,
            "file": DISTANCE_SCALE_FILE,
        }
    )


# =====================================================================
# RECORD VALIDATION SAMPLE
# =====================================================================

@app.route(
    "/record",
    methods=["POST"],
)
def record_validation():

    global validation_samples

    data = request.get_json(
        silent=True
    )

    if data is None:

        return jsonify(
            {
                "success": False,
                "message":
                    "Invalid JSON.",
            }
        ), 400

    # Prefer centimetres for the new UI, but keep backwards
    # compatibility with the old metre field.
    actual_distance_cm = safe_float(
        data.get(
            "actual_distance_cm"
        )
    )

    if (
        actual_distance_cm is not None
        and actual_distance_cm > 0
    ):
        actual_distance = (
            actual_distance_cm / 100.0
        )
    else:
        actual_distance = safe_float(
            data.get(
                "actual_distance_m"
            )
        )

    notes = data.get(
        "notes",
        "",
    )

    if actual_distance is None:

        return jsonify(
            {
                "success": False,
                "message":
                    "actual_distance_m "
                    "is required.",
            }
        ), 400

    # -------------------------------------------------------------
    # Get current system measurement.
    # -------------------------------------------------------------

    with state_lock:

        state = dict(latest_state)

    relationships = (
        state.get("relationships")
        if state.get("relationships") is not None
        else []
    )

    if not relationships:

        return jsonify(
            {
                "success": False,
                "message":
                    "No person-chair "
                    "relationship detected.",
            }
        ), 400

    # -------------------------------------------------------------
    # For Phase 6A use first detected pair.
    # -------------------------------------------------------------

    relation = relationships[0]

    measured_distance = safe_float(
        relation.get(
            "distance_m"
        )
    )

    if measured_distance is None:

        return jsonify(
            {
                "success": False,
                "message":
                    "No valid measured "
                    "distance available.",
            }
        ), 400

    error = (
        measured_distance
        - actual_distance
    )

    absolute_error = abs(error)

    if actual_distance > 0:

        percentage_error = (
            absolute_error
            / actual_distance
        ) * 100.0

    else:

        percentage_error = None

    sample = {
        "sample_id":
            len(validation_samples) + 1,

        "timestamp":
            datetime.now().isoformat(),

        "frame_id":
            state.get(
                "frame_id"
            ),

        "person_id":
            relation.get(
                "person_id"
            ),

        "chair_id":
            relation.get(
                "chair_id"
            ),

        "actual_distance_m":
            actual_distance,

        "measured_distance_m":
            measured_distance,

        "measured_distance_cm":
            measured_distance * 100.0,

        "raw_distance_m":
            relation.get(
                "raw_distance_m"
            ),

        "median_distance_m":
            relation.get(
                "median_distance_m"
            ),

        "error_m":
            error,

        "absolute_error_m":
            absolute_error,

        "percentage_error":
            percentage_error,

        "horizontal_distance_m":
            relation.get(
                "horizontal_distance_m"
            ),

        "vertical_difference_m":
            relation.get(
                "vertical_difference_m"
            ),

        "depth_difference_m":
            relation.get(
                "depth_difference_m"
            ),

        "velocity_mps":
            relation.get(
                "velocity_mps"
            ),

        "motion_state":
            relation.get(
                "motion_state"
            ),

        "relationship":
            relation.get(
                "relationship"
            ),

        "notes":
            notes,
    }

    with state_lock:

        validation_samples.append(
            sample
        )

        # ---------------------------------------------------------
        # Save JSON
        # ---------------------------------------------------------

        try:

            with open(
                VALIDATION_FILE,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    validation_samples,
                    file,
                    indent=2,
                )

        except Exception as exc:

            print(
                "[VALIDATION] "
                f"Could not save file: {exc}"
            )

    print()
    print("=" * 70)
    print("VALIDATION SAMPLE RECORDED")
    print("=" * 70)

    print(
        f"Actual   : "
        f"{actual_distance * 100.0:.1f} cm"
    )

    print(
        f"Measured : "
        f"{measured_distance * 100.0:.1f} cm"
    )

    print(
        f"Error    : "
        f"{error * 100.0:+.1f} cm"
    )

    print(
        f"Abs error: "
        f"{absolute_error * 100.0:.1f} cm"
    )

    if percentage_error is not None:

        print(
            f"Error %  : "
            f"{percentage_error:.2f}%"
        )

    print("=" * 70)

    return jsonify(
        {
            "success": True,
            "message":
                "Validation sample recorded.",
            "sample": sample,
        }
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    camera_thread = threading.Thread(
        target=camera_loop,
        name="BAS-Camera",
        daemon=True,
    )

    processing_thread = threading.Thread(
        target=processing_loop,
        name="BAS-Processing",
        daemon=True,
    )

    camera_thread.start()
    processing_thread.start()

    print()
    print("=" * 70)
    print("BAS LIVE DISTANCE VALIDATION READY")
    print("=" * 70)
    print()
    print("Open in Windows browser:")
    print(
        "    http://localhost:5000"
    )
    print()
    print("Validation file:")
    print(
        f"    {VALIDATION_FILE}"
    )
    print()
    print("Press CTRL+C to stop.")
    print("=" * 70)

    try:

        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping BAS metric validation..."
        )

    finally:

        stop_event.set()

        camera_thread.join(
            timeout=3
        )

        processing_thread.join(
            timeout=3
        )

        camera.release()

        print(
            "Camera released."
        )

        print(
            "BAS metric validation stopped."
        )


if __name__ == "__main__":

    main()
