import csv
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DEPTH_MODEL = "yolo26n-depth.pt"
DETECTION_MODEL = "yolo11n.pt"

CAMERA_DEVICE = "/dev/video0"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 10

YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.35

HOST = "0.0.0.0"
PORT = 5005

# Development camera intrinsics.
# Replace with calibrated values later.
FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0

DEPTH_SAMPLE_RADIUS = 5

# Save validation measurements here.
CSV_FILE = (
    "data/validation/yolo26_distance/"
    "yolo26_distance_validation.csv"
)


# COCO classes
PERSON_CLASS = 0
CHAIR_CLASS = 56


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Detection:

    object_id: int

    class_id: int

    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    depth_m: float = 0.0

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0

    missed: int = 0

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom_center_x(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def bottom_center_y(self):
        return self.y2


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.Lock()

latest_frame = None
latest_depth_map = None

latest_persons = []
latest_chairs = []

latest_distance_m = None
latest_ground_distance_m = None
latest_depth_difference_m = None

actual_distance_cm = None

latest_error_cm = None
latest_percentage_error = None

frame_id = 0

latest_inference_time = 0.0
latest_fps = 0.0

running = False


# ============================================================
# SIMPLE OBJECT TRACKER
# ============================================================

class SimpleTracker:

    def __init__(
        self,
        max_distance=100.0,
        max_missed=10
    ):

        self.max_distance = max_distance
        self.max_missed = max_missed

        self.next_id = 0

        self.objects = []

    @staticmethod
    def center(det):

        return np.array(
            [
                det.center_x,
                det.center_y
            ],
            dtype=np.float32
        )

    def update(self, detections):

        if not self.objects:

            for det in detections:

                det.object_id = self.next_id

                self.next_id += 1

                self.objects.append(det)

            return self.objects.copy()

        used = set()

        updated = []

        for old in self.objects:

            best_index = None
            best_distance = float("inf")

            old_center = self.center(old)

            for i, new in enumerate(detections):

                if i in used:
                    continue

                new_center = self.center(new)

                distance = np.linalg.norm(
                    old_center - new_center
                )

                if (
                    distance < best_distance
                    and distance <= self.max_distance
                ):

                    best_distance = distance
                    best_index = i

            if best_index is not None:

                new = detections[best_index]

                new.object_id = old.object_id

                new.missed = 0

                updated.append(new)

                used.add(best_index)

            else:

                old.missed += 1

                if old.missed <= self.max_missed:

                    updated.append(old)

        # New detections

        for i, det in enumerate(detections):

            if i in used:
                continue

            det.object_id = self.next_id

            self.next_id += 1

            updated.append(det)

        self.objects = updated

        return self.objects.copy()


# ============================================================
# TRACKERS
# ============================================================

person_tracker = SimpleTracker(
    max_distance=120.0
)

chair_tracker = SimpleTracker(
    max_distance=120.0
)


# ============================================================
# DEPTH EXTRACTION
# ============================================================

def extract_depth_map(
    result,
    frame_shape
):

    if not hasattr(result, "depth"):
        return None

    if result.depth is None:
        return None

    depth = (
        result.depth.data
        .detach()
        .cpu()
        .numpy()
    )

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    depth = np.squeeze(depth)

    if depth.ndim != 2:
        return None

    h = frame_shape[0]
    w = frame_shape[1]

    if depth.shape != (h, w):

        depth = cv2.resize(
            depth,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    return depth


# ============================================================
# ROBUST DEPTH
# ============================================================

def depth_at_point(
    depth_map,
    x,
    y,
    radius=DEPTH_SAMPLE_RADIUS
):

    if depth_map is None:
        return None

    h, w = depth_map.shape

    x = int(
        np.clip(
            x,
            0,
            w - 1
        )
    )

    y = int(
        np.clip(
            y,
            0,
            h - 1
        )
    )

    x1 = max(
        0,
        x - radius
    )

    x2 = min(
        w,
        x + radius + 1
    )

    y1 = max(
        0,
        y - radius
    )

    y2 = min(
        h,
        y + radius + 1
    )

    region = depth_map[
        y1:y2,
        x1:x2
    ]

    valid = region[
        np.isfinite(region)
        &
        (region > 0)
    ]

    if len(valid) == 0:
        return None

    return float(
        np.median(valid)
    )


def depth_inside_bbox(
    depth_map,
    detection
):

    if depth_map is None:
        return None

    h, w = depth_map.shape

    x1 = int(
        np.clip(
            detection.x1,
            0,
            w - 1
        )
    )

    y1 = int(
        np.clip(
            detection.y1,
            0,
            h - 1
        )
    )

    x2 = int(
        np.clip(
            detection.x2,
            0,
            w - 1
        )
    )

    y2 = int(
        np.clip(
            detection.y2,
            0,
            h - 1
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    # Sample several points rather than relying
    # on one center pixel.

    width = x2 - x1
    height = y2 - y1

    points = [

        (
            x1 + width * 0.50,
            y1 + height * 0.35
        ),

        (
            x1 + width * 0.35,
            y1 + height * 0.50
        ),

        (
            x1 + width * 0.65,
            y1 + height * 0.50
        ),

        (
            x1 + width * 0.50,
            y1 + height * 0.65
        ),

        (
            x1 + width * 0.50,
            y1 + height * 0.85
        )
    ]

    values = []

    for x, y in points:

        value = depth_at_point(
            depth_map,
            x,
            y
        )

        if value is not None:

            if np.isfinite(value) and value > 0:

                values.append(value)

    if not values:
        return None

    # Remove extreme outliers.

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) >= 3:

        low = np.percentile(
            values,
            20
        )

        high = np.percentile(
            values,
            80
        )

        filtered = values[
            (values >= low)
            &
            (values <= high)
        ]

        if len(filtered) > 0:

            values = filtered

    return float(
        np.median(values)
    )


# ============================================================
# PIXEL → 3D
# ============================================================

def pixel_to_3d(
    u,
    v,
    depth_m
):

    if depth_m is None:
        return None

    if depth_m <= 0:
        return None

    x = (
        (u - CX)
        * depth_m
        / FX
    )

    y = (
        (v - CY)
        * depth_m
        / FY
    )

    z = depth_m

    return (
        float(x),
        float(y),
        float(z)
    )


# ============================================================
# OBJECT 3D POSITION
# ============================================================

def calculate_object_3d(
    detection,
    depth_map
):

    depth_m = depth_inside_bbox(
        depth_map,
        detection
    )

    if depth_m is None:
        return False

    # Use bottom-center as reference point.
    u = detection.bottom_center_x
    v = detection.bottom_center_y

    point = pixel_to_3d(
        u,
        v,
        depth_m
    )

    if point is None:
        return False

    detection.depth_m = depth_m

    detection.x_m = point[0]
    detection.y_m = point[1]
    detection.z_m = point[2]

    return True


# ============================================================
# DISTANCE
# ============================================================

def calculate_distance(
    person,
    chair
):

    dx = (
        chair.x_m -
        person.x_m
    )

    dy = (
        chair.y_m -
        person.y_m
    )

    dz = (
        chair.z_m -
        person.z_m
    )

    distance = math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )

    # Ground-plane distance
    ground_distance = math.sqrt(
        dx * dx +
        dz * dz
    )

    depth_difference = abs(dz)

    return (
        distance,
        ground_distance,
        depth_difference
    )


# ============================================================
# FIND PERSON + CHAIR PAIR
# ============================================================

def find_nearest_pair(
    persons,
    chairs
):

    if not persons or not chairs:
        return None

    best_pair = None
    best_distance = float("inf")

    for person in persons:

        for chair in chairs:

            if (
                person.depth_m <= 0
                or chair.depth_m <= 0
            ):
                continue

            distance = math.sqrt(
                (
                    person.x_m -
                    chair.x_m
                ) ** 2
                +
                (
                    person.y_m -
                    chair.y_m
                ) ** 2
                +
                (
                    person.z_m -
                    chair.z_m
                ) ** 2
            )

            if distance < best_distance:

                best_distance = distance

                best_pair = (
                    person,
                    chair
                )

    return best_pair


# ============================================================
# DETECTION
# ============================================================

def detect_objects(
    detector,
    frame
):

    results = detector(
        frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        device="cpu",
        verbose=False
    )

    if not results:
        return [], []

    result = results[0]

    persons = []
    chairs = []

    if result.boxes is None:
        return persons, chairs

    boxes = result.boxes

    for i in range(
        len(boxes)
    ):

        cls = int(
            boxes.cls[i].item()
        )

        confidence = float(
            boxes.conf[i].item()
        )

        coords = (
            boxes.xyxy[i]
            .detach()
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = (
            coords.tolist()
        )

        detection = Detection(
            object_id=-1,
            class_id=cls,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )

        if cls == PERSON_CLASS:

            persons.append(
                detection
            )

        elif cls == CHAIR_CLASS:

            chairs.append(
                detection
            )

    return persons, chairs


# ============================================================
# SAVE VALIDATION RESULT
# ============================================================

def save_validation(
    actual_cm,
    calculated_m,
    ground_m,
    person,
    chair
):

    os.makedirs(
        os.path.dirname(CSV_FILE),
        exist_ok=True
    )

    file_exists = os.path.exists(
        CSV_FILE
    )

    error_cm = None
    percentage_error = None

    if actual_cm is not None:

        error_cm = (
            calculated_m * 100
            - actual_cm
        )

        if actual_cm != 0:

            percentage_error = (
                abs(error_cm)
                / actual_cm
            ) * 100

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow(
                [
                    "timestamp",
                    "frame_id",
                    "actual_distance_cm",
                    "calculated_distance_cm",
                    "ground_distance_cm",
                    "error_cm",
                    "percentage_error",
                    "person_id",
                    "person_depth_cm",
                    "person_x_cm",
                    "person_y_cm",
                    "person_z_cm",
                    "chair_id",
                    "chair_depth_cm",
                    "chair_x_cm",
                    "chair_y_cm",
                    "chair_z_cm"
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                frame_id,
                actual_cm,
                calculated_m * 100,
                ground_m * 100,
                error_cm,
                percentage_error,

                person.object_id,
                person.depth_m * 100,
                person.x_m * 100,
                person.y_m * 100,
                person.z_m * 100,

                chair.object_id,
                chair.depth_m * 100,
                chair.x_m * 100,
                chair.y_m * 100,
                chair.z_m * 100
            ]
        )


# ============================================================
# CAMERA / MODEL WORKER
# ============================================================

def processing_worker():

    global latest_frame
    global latest_depth_map

    global latest_persons
    global latest_chairs

    global latest_distance_m
    global latest_ground_distance_m
    global latest_depth_difference_m

    global latest_error_cm
    global latest_percentage_error

    global frame_id

    global latest_inference_time
    global latest_fps

    global running

    print("=" * 70)
    print("YOLO26 PERSON-CHAIR DISTANCE SYSTEM")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        f"[MODEL] Loading detection model: "
        f"{DETECTION_MODEL}"
    )

    detector = YOLO(
        DETECTION_MODEL
    )

    print(
        "[MODEL] Detection model loaded."
    )

    print(
        f"[MODEL] Loading depth model: "
        f"{DEPTH_MODEL}"
    )

    depth_model = YOLO(
        DEPTH_MODEL
    )

    print(
        "[MODEL] Depth model loaded."
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():

        print(
            "[ERROR] Could not open camera."
        )

        running = False

        return

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

    print(
        "[CAMERA] Camera opened."
    )

    running = True

    previous_time = time.perf_counter()

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    while running:

        success, frame = cap.read()

        if not success:

            time.sleep(0.05)

            continue

        frame_id += 1

        start = time.perf_counter()

        try:

            # ================================================
            # YOLO11 DETECTION
            # ================================================

            persons, chairs = detect_objects(
                detector,
                frame
            )

            # ================================================
            # YOLO26 DEPTH
            # ================================================

            depth_results = depth_model(
                frame,
                imgsz=YOLO_IMAGE_SIZE,
                device="cpu",
                verbose=False
            )

            depth_map = None

            if depth_results:

                depth_map = extract_depth_map(
                    depth_results[0],
                    frame.shape
                )

            # ================================================
            # TRACKING
            # ================================================

            persons = person_tracker.update(
                persons
            )

            chairs = chair_tracker.update(
                chairs
            )

            # ================================================
            # 3D POSITION
            # ================================================

            valid_persons = []

            for person in persons:

                if calculate_object_3d(
                    person,
                    depth_map
                ):

                    valid_persons.append(
                        person
                    )

            valid_chairs = []

            for chair in chairs:

                if calculate_object_3d(
                    chair,
                    depth_map
                ):

                    valid_chairs.append(
                        chair
                    )

            # ================================================
            # PERSON-CHAIR DISTANCE
            # ================================================

            pair = find_nearest_pair(
                valid_persons,
                valid_chairs
            )

            distance_m = None
            ground_m = None
            depth_difference_m = None

            if pair is not None:

                person, chair = pair

                (
                    distance_m,
                    ground_m,
                    depth_difference_m
                ) = calculate_distance(
                    person,
                    chair
                )

            else:

                person = None
                chair = None

            # ================================================
            # ERROR AGAINST ACTUAL DISTANCE
            # ================================================

            error_cm = None
            percentage_error = None

            with state_lock:

                current_actual = (
                    actual_distance_cm
                )

            if (
                current_actual is not None
                and distance_m is not None
            ):

                error_cm = (
                    distance_m * 100
                    - current_actual
                )

                if current_actual != 0:

                    percentage_error = (
                        abs(error_cm)
                        / current_actual
                    ) * 100

            # ================================================
            # PERFORMANCE
            # ================================================

            inference_time = (
                time.perf_counter()
                - start
            )

            now = time.perf_counter()

            delta = (
                now -
                previous_time
            )

            if delta > 0:

                current_fps = (
                    1.0 / delta
                )

            else:

                current_fps = 0.0

            previous_time = now

            # ================================================
            # UPDATE STATE
            # ================================================

            with state_lock:

                latest_frame = frame.copy()

                latest_depth_map = (
                    depth_map.copy()
                    if depth_map is not None
                    else None
                )

                latest_persons = (
                    valid_persons
                )

                latest_chairs = (
                    valid_chairs
                )

                latest_distance_m = (
                    distance_m
                )

                latest_ground_distance_m = (
                    ground_m
                )

                latest_depth_difference_m = (
                    depth_difference_m
                )

                latest_error_cm = (
                    error_cm
                )

                latest_percentage_error = (
                    percentage_error
                )

                latest_inference_time = (
                    inference_time
                )

                latest_fps = (
                    current_fps
                )

            # ================================================
            # TERMINAL LOG
            # ================================================

            if distance_m is not None:

                message = (
                    f"[FRAME {frame_id}] "
                    f"Distance="
                    f"{distance_m * 100:.1f} cm"
                )

                if current_actual is not None:

                    message += (
                        f" | Actual="
                        f"{current_actual:.1f} cm"
                    )

                if error_cm is not None:

                    message += (
                        f" | Error="
                        f"{error_cm:+.1f} cm"
                    )

                print(message)

            else:

                print(
                    f"[FRAME {frame_id}] "
                    f"No valid person-chair pair"
                )

        except Exception as e:

            print(
                f"[ERROR] Frame {frame_id}: {e}"
            )

    cap.release()

    print(
        "[CAMERA] Camera released."
    )


# ============================================================
# DRAW FRAME
# ============================================================

def create_frame():

    with state_lock:

        if latest_frame is None:
            return None

        frame = latest_frame.copy()

        persons = list(
            latest_persons
        )

        chairs = list(
            latest_chairs
        )

        distance_m = (
            latest_distance_m
        )

        ground_m = (
            latest_ground_distance_m
        )

        actual_cm = (
            actual_distance_cm
        )

        error_cm = (
            latest_error_cm
        )

        percentage_error = (
            latest_percentage_error
        )

        fps = latest_fps

    # --------------------------------------------------------
    # PERSON BOXES
    # --------------------------------------------------------

    for person in persons:

        cv2.rectangle(
            frame,
            (
                int(person.x1),
                int(person.y1)
            ),
            (
                int(person.x2),
                int(person.y2)
            ),
            (0, 255, 0),
            2
        )

        text = (
            f"Person {person.object_id} "
            f"{person.depth_m * 100:.1f} cm"
        )

        cv2.putText(
            frame,
            text,
            (
                int(person.x1),
                max(
                    20,
                    int(person.y1) - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # CHAIR BOXES
    # --------------------------------------------------------

    for chair in chairs:

        cv2.rectangle(
            frame,
            (
                int(chair.x1),
                int(chair.y1)
            ),
            (
                int(chair.x2),
                int(chair.y2)
            ),
            (255, 255, 0),
            2
        )

        text = (
            f"Chair {chair.object_id} "
            f"{chair.depth_m * 100:.1f} cm"
        )

        cv2.putText(
            frame,
            text,
            (
                int(chair.x1),
                max(
                    20,
                    int(chair.y1) - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    y = 30

    if distance_m is not None:

        cv2.putText(
            frame,
            f"YOLO26 Distance: "
            f"{distance_m * 100:.1f} cm",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        y += 30

        if ground_m is not None:

            cv2.putText(
                frame,
                f"Ground Distance: "
                f"{ground_m * 100:.1f} cm",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            y += 30

    # --------------------------------------------------------
    # ACTUAL DISTANCE
    # --------------------------------------------------------

    if actual_cm is not None:

        cv2.putText(
            frame,
            f"Actual Distance: "
            f"{actual_cm:.1f} cm",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        y += 30

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if error_cm is not None:

        cv2.putText(
            frame,
            f"Error: "
            f"{error_cm:+.1f} cm",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        y += 30

    if percentage_error is not None:

        cv2.putText(
            frame,
            f"Error: "
            f"{percentage_error:.2f}%",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        y += 30

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "YOLO26 experimental / uncalibrated",
        (
            20,
            frame.shape[0] - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        2
    )

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not success:
        return None

    return encoded.tobytes()


# ============================================================
# STREAM
# ============================================================

def video_stream():

    while True:

        frame = create_frame()

        if frame is None:

            time.sleep(0.05)

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>BAS-HMR YOLO26 Distance</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    margin: 0;
    padding: 20px;
}

h1 {
    margin-bottom: 5px;
}

img {
    width: 640px;
    max-width: 100%;
    border-radius: 8px;
}

.panel {
    max-width: 680px;
    background: #222;
    padding: 20px;
    border-radius: 12px;
}

input {
    padding: 10px;
    font-size: 18px;
    width: 160px;
}

button {
    padding: 10px 20px;
    font-size: 18px;
    cursor: pointer;
}

.info {
    margin-top: 20px;
    font-size: 18px;
    line-height: 1.7;
}

.value {
    font-weight: bold;
}

</style>

</head>

<body>

<h1>BAS-HMR — YOLO26 Distance</h1>

<p>
Experimental Person ↔ Chair Metric Distance
</p>

<div class="panel">

<img src="/video_feed">

<div class="info">

<p>
YOLO26 Distance:
<span id="distance" class="value">
--
</span>
</p>

<p>
Ground Distance:
<span id="ground" class="value">
--
</span>
</p>

<p>
Actual Distance:
<span id="actual" class="value">
--
</span>
</p>

<p>
Error:
<span id="error" class="value">
--
</span>
</p>

<p>
Percentage Error:
<span id="percentage" class="value">
--
</span>
</p>

<hr>

<h3>Enter Tape-Measured Actual Distance</h3>

<input
    id="actualInput"
    type="number"
    step="0.1"
    placeholder="cm"
/>

<button onclick="setActual()">
Set Actual Distance
</button>

<br><br>

<button onclick="saveMeasurement()">
Save Measurement
</button>

<p id="message"></p>

</div>

</div>


<script>

async function updateStatus() {

    try {

        const response =
            await fetch("/api/status");

        const data =
            await response.json();

        if (data.distance_cm !== null) {

            document.getElementById(
                "distance"
            ).innerText =
                data.distance_cm.toFixed(1)
                + " cm";

        }

        if (data.ground_distance_cm !== null) {

            document.getElementById(
                "ground"
            ).innerText =
                data.ground_distance_cm.toFixed(1)
                + " cm";

        }

        if (data.actual_distance_cm !== null) {

            document.getElementById(
                "actual"
            ).innerText =
                data.actual_distance_cm.toFixed(1)
                + " cm";

        }

        if (data.error_cm !== null) {

            document.getElementById(
                "error"
            ).innerText =
                data.error_cm.toFixed(1)
                + " cm";

        }

        if (
            data.percentage_error !== null
        ) {

            document.getElementById(
                "percentage"
            ).innerText =
                data.percentage_error.toFixed(2)
                + " %";

        }

    }

    catch (error) {

        console.log(error);

    }

}


async function setActual() {

    const value =
        document.getElementById(
            "actualInput"
        ).value;

    if (!value) {

        return;

    }

    await fetch(
        "/api/set_actual",
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                actual_distance_cm:
                    parseFloat(value)
            })
        }
    );

    document.getElementById(
        "message"
    ).innerText =
        "Actual distance set to "
        + value
        + " cm";

}


async function saveMeasurement() {

    const response =
        await fetch(
            "/api/save",
            {
                method: "POST"
            }
        );

    const data =
        await response.json();

    document.getElementById(
        "message"
    ).innerText =
        data.message;

}


setInterval(
    updateStatus,
    500
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

    return HTML


@app.route("/video_feed")
def video_feed():

    return Response(
        video_stream(),
        mimetype=(
            "multipart/x-mixed-replace;"
            "boundary=frame"
        )
    )


@app.route("/api/status")
def status():

    with state_lock:

        return jsonify(
            {
                "running": running,

                "frame_id": frame_id,

                "persons": len(
                    latest_persons
                ),

                "chairs": len(
                    latest_chairs
                ),

                "distance_cm": (
                    latest_distance_m * 100
                    if latest_distance_m
                    is not None
                    else None
                ),

                "ground_distance_cm": (
                    latest_ground_distance_m
                    * 100
                    if latest_ground_distance_m
                    is not None
                    else None
                ),

                "depth_difference_cm": (
                    latest_depth_difference_m
                    * 100
                    if latest_depth_difference_m
                    is not None
                    else None
                ),

                "actual_distance_cm": (
                    actual_distance_cm
                ),

                "error_cm": (
                    latest_error_cm
                ),

                "percentage_error": (
                    latest_percentage_error
                ),

                "fps": latest_fps,

                "inference_time_s":
                    latest_inference_time,

                "models": {
                    "detection":
                        DETECTION_MODEL,
                    "depth":
                        DEPTH_MODEL
                },

                "calibrated": False
            }
        )


@app.route(
    "/api/set_actual",
    methods=["POST"]
)
def set_actual():

    global actual_distance_cm

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify(
            {
                "success": False,
                "message":
                    "No JSON data received."
            }
        ), 400

    value = data.get(
        "actual_distance_cm"
    )

    if value is None:

        return jsonify(
            {
                "success": False,
                "message":
                    "actual_distance_cm missing."
            }
        ), 400

    try:

        value = float(value)

    except ValueError:

        return jsonify(
            {
                "success": False,
                "message":
                    "Invalid distance."
            }
        ), 400

    if value <= 0:

        return jsonify(
            {
                "success": False,
                "message":
                    "Distance must be > 0."
            }
        ), 400

    with state_lock:

        actual_distance_cm = value

    return jsonify(
        {
            "success": True,
            "actual_distance_cm": value
        }
    )


@app.route(
    "/api/save",
    methods=["POST"]
)
def save():

    with state_lock:

        if (
            actual_distance_cm
            is None
        ):

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Set actual distance first."
                }
            ), 400

        if latest_distance_m is None:

            return jsonify(
                {
                    "success": False,
                    "message":
                        "No valid person-chair "
                        "distance available."
                }
            ), 400

        pair = find_nearest_pair(
            latest_persons,
            latest_chairs
        )

        if pair is None:

            return jsonify(
                {
                    "success": False,
                    "message":
                        "No valid person-chair pair."
                }
            ), 400

        person, chair = pair

        distance_m = (
            latest_distance_m
        )

        ground_m = (
            latest_ground_distance_m
        )

        actual_cm = (
            actual_distance_cm
        )

    save_validation(
        actual_cm,
        distance_m,
        ground_m,
        person,
        chair
    )

    return jsonify(
        {
            "success": True,
            "message":
                "Measurement saved to CSV.",
            "file":
                CSV_FILE
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    worker = threading.Thread(
        target=processing_worker,
        daemon=True
    )

    worker.start()

    print()
    print("=" * 70)
    print("BAS-HMR YOLO26 DISTANCE SERVER")
    print("=" * 70)

    print()
    print(
        f"Browser:"
    )

    print(
        f"http://localhost:{PORT}"
    )

    print()

    print(
        f"API:"
    )

    print(
        f"http://localhost:{PORT}/api/status"
    )

    print()

    print(
        "Person + Chair → YOLO26 Depth → 3D Distance"
    )

    print()

    print(
        "Press CTRL+C to stop."
    )

    print("=" * 70)

    try:

        app.run(
            host=HOST,
            port=PORT,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    except KeyboardInterrupt:

        print(
            "\n[SERVER] Stopping..."
        )

    finally:

        global running

        running = False


if __name__ == "__main__":

    main()
