from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO


# ============================================================
# BAS SCENE DISTANCE CALIBRATION - FIXED V2
# ============================================================
#
# Purpose:
#   Build a REAL-WORLD calibration dataset for Person <-> Chair
#   distance estimation without the checkerboard workflow.
#
# Important:
#   - YOLO detection indices are NOT used as persistent IDs.
#   - A lightweight centroid tracker assigns stable IDs.
#   - IDs identify objects; they do not define physical distance.
#   - Ground-truth distances MUST be measured with a tape measure.
#
# Output:
#   calibration/scene_calibration_samples.csv
#   calibration/scene_distance_calibration.json
#
# Run:
#   python human/distance/scene_calibration.py
#
# Browser:
#   http://localhost:5002
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

CAMERA_DEVICE = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10

YOLO_MODEL = "yolo11n.pt"

PERSON_CLASS = 0
CHAIR_CLASS = 56

CONFIDENCE = 0.35
IMAGE_SIZE = 416

OUTPUT_DIR = Path("calibration")
CSV_FILE = OUTPUT_DIR / "scene_calibration_samples.csv"
MODEL_FILE = OUTPUT_DIR / "scene_distance_calibration.json"

MIN_DISTANCE_M = 0.20
MAX_DISTANCE_M = 5.00

MAX_TRACK_DISTANCE_PX = 100.0
MAX_MISSED_FRAMES = 10

# Calibration should have enough data for several different distances.
MIN_FIT_SAMPLES = 12


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

camera = None
model = None

latest_frame = None
latest_detections = []

running = True

state_lock = threading.Lock()

samples = []


# ============================================================
# TRACKER
# ============================================================

class Track:
    def __init__(self, track_id, detection):
        self.track_id = int(track_id)
        self.center = tuple(detection["center"])
        self.bbox = tuple(detection["bbox"])
        self.width = float(detection["width"])
        self.height = float(detection["height"])
        self.confidence = float(detection["confidence"])
        self.missed = 0
        self.age = 1


class CentroidTracker:
    """
    Small CPU-friendly tracker.

    It is intentionally simple because this module is for
    calibration data collection, not final production tracking.
    """

    def __init__(
        self,
        max_distance_px=100.0,
        max_missed_frames=10
    ):
        self.max_distance_px = float(max_distance_px)
        self.max_missed_frames = int(max_missed_frames)

        self.next_id = 0
        self.tracks = {}

    @staticmethod
    def distance(a, b):
        return math.hypot(
            float(a[0]) - float(b[0]),
            float(a[1]) - float(b[1])
        )

    def update(self, detections):
        detections = list(detections)

        if not self.tracks:
            for detection in detections:
                track = Track(self.next_id, detection)
                self.tracks[self.next_id] = track
                self.next_id += 1

            return self._export()

        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))

        pairs = []

        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                d = self.distance(
                    track.center,
                    detection["center"]
                )
                pairs.append(
                    (d, track_id, index)
                )

        pairs.sort(key=lambda item: item[0])

        for d, track_id, index in pairs:
            if d > self.max_distance_px:
                continue

            if track_id not in unmatched_tracks:
                continue

            if index not in unmatched_detections:
                continue

            detection = detections[index]
            track = self.tracks[track_id]

            track.center = tuple(detection["center"])
            track.bbox = tuple(detection["bbox"])
            track.width = float(detection["width"])
            track.height = float(detection["height"])
            track.confidence = float(detection["confidence"])
            track.missed = 0
            track.age += 1

            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missed += 1

            if track.missed > self.max_missed_frames:
                del self.tracks[track_id]

        for index in unmatched_detections:
            track = Track(
                self.next_id,
                detections[index]
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

        return self._export()

    def _export(self):
        output = []

        for track in self.tracks.values():
            output.append({
                "id": track.track_id,
                "center": track.center,
                "bbox": track.bbox,
                "width": track.width,
                "height": track.height,
                "confidence": track.confidence,
                "age": track.age,
                "missed": track.missed
            })

        return output


person_tracker = CentroidTracker(
    MAX_TRACK_DISTANCE_PX,
    MAX_MISSED_FRAMES
)

chair_tracker = CentroidTracker(
    MAX_TRACK_DISTANCE_PX,
    MAX_MISSED_FRAMES
)


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>BAS Scene Distance Calibration</title>

<style>

body {
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 0;
    padding: 20px;
}

.container {
    width: 1000px;
    max-width: 96%;
    margin: auto;
}

img {
    width: 640px;
    max-width: 100%;
    border: 2px solid #555;
}

.panel {
    background: #202020;
    border-radius: 8px;
    padding: 16px;
    margin: 15px 0;
}

input {
    padding: 12px;
    font-size: 18px;
    width: 180px;
    text-align: center;
}

button {
    padding: 12px 22px;
    margin: 8px;
    font-size: 16px;
    cursor: pointer;
}

pre {
    text-align: left;
    white-space: pre-wrap;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    border: 1px solid #555;
    padding: 7px;
}

.good {
    color: #66ff66;
}

.warn {
    color: #ffcc66;
}

</style>

</head>


<body>

<div class="container">

<h1>BAS Scene Distance Calibration V2</h1>

<p>
Real-world Person ↔ Chair calibration
</p>

<img src="/video">


<div class="panel">

<h2>Live Objects</h2>

<pre id="detections">Loading...</pre>

</div>


<div class="panel">

<h2>Capture Ground Truth</h2>

<p>
Use a tape measure to measure the real ground distance
between the person's reference point and chair's reference point.
</p>

<input
    id="distance"
    type="number"
    step="0.01"
    min="0.20"
    max="5.00"
    placeholder="1.50"
>

<br>

<button onclick="captureSample()">
CAPTURE SAMPLE
</button>

</div>


<div class="panel">

<h2>Fit Calibration Model</h2>

<button onclick="fitModel()">
FIT MODEL
</button>

<pre id="fit"></pre>

</div>


<div class="panel">

<h2>Collected Samples</h2>

<div id="samples">
Loading...
</div>

</div>

</div>


<script>


async function updateStatus() {

    try {

        const response =
            await fetch("/status");

        const data =
            await response.json();

        document.getElementById(
            "detections"
        ).innerText =
            JSON.stringify(
                data,
                null,
                2
            );

    } catch (error) {

        console.log(error);

    }
}


async function captureSample() {

    const input =
        document.getElementById(
            "distance"
        );

    const distance =
        parseFloat(
            input.value
        );

    if (!Number.isFinite(distance)) {

        alert(
            "Enter the tape-measured distance."
        );

        return;
    }


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
                    distance_m:
                        distance
                })
            }
        );


    const data =
        await response.json();


    alert(
        data.message
    );


    updateSamples();
}


async function fitModel() {

    const response =
        await fetch(
            "/fit",
            {
                method: "POST"
            }
        );


    const data =
        await response.json();


    document.getElementById(
        "fit"
    ).innerText =
        JSON.stringify(
            data,
            null,
            2
        );

}


async function updateSamples() {

    try {

        const response =
            await fetch(
                "/samples"
            );

        const data =
            await response.json();


        let html =
            "<table>" +
            "<tr>" +
            "<th>#</th>" +
            "<th>Person ID</th>" +
            "<th>Chair ID</th>" +
            "<th>Pixel Gap</th>" +
            "<th>Person H</th>" +
            "<th>Person W</th>" +
            "<th>Chair W</th>" +
            "<th>Chair H</th>" +
            "<th>Ground Truth</th>" +
            "</tr>";


        for (
            const s of data.samples
        ) {

            html +=
                "<tr>" +
                "<td>" +
                s.sample_index +
                "</td>" +

                "<td>" +
                s.person_id +
                "</td>" +

                "<td>" +
                s.chair_id +
                "</td>" +

                "<td>" +
                s.pixel_gap.toFixed(1) +
                "</td>" +

                "<td>" +
                s.person_height.toFixed(1) +
                "</td>" +

                "<td>" +
                s.person_width.toFixed(1) +
                "</td>" +

                "<td>" +
                s.chair_width.toFixed(1) +
                "</td>" +

                "<td>" +
                s.chair_height.toFixed(1) +
                "</td>" +

                "<td>" +
                s.ground_truth_m.toFixed(2) +
                " m</td>" +

                "</tr>";
        }


        html +=
            "</table>";


        document.getElementById(
            "samples"
        ).innerHTML =
            html;

    } catch (error) {

        console.log(error);

    }
}


setInterval(
    updateStatus,
    1000
);

setInterval(
    updateSamples,
    2000
);


updateStatus();
updateSamples();

</script>

</body>

</html>
"""


# ============================================================
# DETECTION
# ============================================================

def detect_objects(frame):
    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        classes=[
            PERSON_CLASS,
            CHAIR_CLASS
        ],
        device="cpu",
        verbose=False
    )

    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = [
            float(value)
            for value in box.xyxy[0].tolist()
        ]

        width = max(
            1.0,
            x2 - x1
        )

        height = max(
            1.0,
            y2 - y1
        )

        center = (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        )

        detections.append({
            "class_id": class_id,
            "confidence": confidence,
            "bbox": (
                x1,
                y1,
                x2,
                y2
            ),
            "center": center,
            "width": width,
            "height": height
        })

    return detections


# ============================================================
# REFERENCE POINTS
# ============================================================

def person_reference_point(track):
    """
    Person ground/contact reference.

    We use lower-center of the bounding box.
    This is intentionally kept simple until pose-based
    reference points are integrated.
    """

    x1, y1, x2, y2 = track["bbox"]

    return (
        (x1 + x2) / 2.0,
        y1 + (y2 - y1) * 0.97
    )


def chair_reference_point(track):
    """
    Chair floor/contact reference.
    """

    x1, y1, x2, y2 = track["bbox"]

    return (
        (x1 + x2) / 2.0,
        y1 + (y2 - y1) * 0.98
    )


# ============================================================
# PERSON-CHAIR ASSOCIATION
# ============================================================

def select_nearest_pair(persons, chairs):
    if not persons or not chairs:
        return None

    best = None
    best_distance = float("inf")

    for person in persons:

        px, py = person["center"]

        for chair in chairs:

            cx, cy = chair["center"]

            distance = math.hypot(
                px - cx,
                py - cy
            )

            if distance < best_distance:

                best_distance = distance

                best = (
                    person,
                    chair,
                    best_distance
                )

    return best


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_features(person, chair):
    person_ref = person_reference_point(person)
    chair_ref = chair_reference_point(chair)

    px, py = person_ref
    cx, cy = chair_ref

    dx = px - cx
    dy = py - cy

    pixel_gap = math.hypot(
        dx,
        dy
    )

    return {
        "person_id": int(person["id"]),
        "chair_id": int(chair["id"]),

        "pixel_gap": float(pixel_gap),

        "horizontal_gap": float(abs(dx)),
        "vertical_gap": float(abs(dy)),

        "person_width": float(
            person["width"]
        ),

        "person_height": float(
            person["height"]
        ),

        "chair_width": float(
            chair["width"]
        ),

        "chair_height": float(
            chair["height"]
        ),

        "person_ref_x": float(px),
        "person_ref_y": float(py),

        "chair_ref_x": float(cx),
        "chair_ref_y": float(cy),

        "person_confidence": float(
            person["confidence"]
        ),

        "chair_confidence": float(
            chair["confidence"]
        )
    }


# ============================================================
# DRAW
# ============================================================

def draw_detection(
    frame,
    track,
    class_name,
    color
):

    x1, y1, x2, y2 = [
        int(value)
        for value in track["bbox"]
    ]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    label = (
        f"{class_name} "
        f"{track['id']}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2
    )


    rx, ry = (
        person_reference_point(track)
        if class_name == "PERSON"
        else chair_reference_point(track)
    )

    cv2.circle(
        frame,
        (int(rx), int(ry)),
        5,
        color,
        -1
    )


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global latest_frame
    global latest_detections

    while running:

        if camera is None:
            time.sleep(0.1)
            continue

        ok, frame = camera.read()

        if not ok:
            time.sleep(0.05)
            continue

        detections = detect_objects(frame)

        person_detections = [
            d for d in detections
            if d["class_id"] == PERSON_CLASS
        ]

        chair_detections = [
            d for d in detections
            if d["class_id"] == CHAIR_CLASS
        ]

        persons = person_tracker.update(
            person_detections
        )

        chairs = chair_tracker.update(
            chair_detections
        )

        display = frame.copy()

        for person in persons:

            draw_detection(
                display,
                person,
                "PERSON",
                (0, 255, 0)
            )

        for chair in chairs:

            draw_detection(
                display,
                chair,
                "CHAIR",
                (255, 0, 0)
            )

        pair = select_nearest_pair(
            persons,
            chairs
        )

        pair_info = None

        if pair is not None:

            person, chair, gap = pair

            px, py = [
                int(v)
                for v in person_reference_point(
                    person
                )
            ]

            cx, cy = [
                int(v)
                for v in chair_reference_point(
                    chair
                )
            ]

            cv2.line(
                display,
                (px, py),
                (cx, cy),
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"P{person['id']} -> C{chair['id']}",
                (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"PIXEL GAP: {gap:.1f}",
                (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            pair_info = {
                "person_id": int(
                    person["id"]
                ),
                "chair_id": int(
                    chair["id"]
                ),
                "pixel_gap": float(gap)
            }


        with state_lock:

            latest_frame = display.copy()

            latest_detections = {
                "persons": persons,
                "chairs": chairs,
                "pair": pair_info
            }


        time.sleep(
            1.0 / FPS
        )


# ============================================================
# VIDEO STREAM
# ============================================================

def video_generator():

    while running:

        with state_lock:

            if latest_frame is None:
                frame = np.zeros(
                    (HEIGHT, WIDTH, 3),
                    dtype=np.uint8
                )

                cv2.putText(
                    frame,
                    "Waiting for camera...",
                    (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            else:
                frame = latest_frame.copy()


        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )

        time.sleep(0.05)


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
        video_generator(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


@app.route("/status")
def status():

    with state_lock:

        data = {
            "camera": (
                camera is not None
                and camera.isOpened()
            ),
            "samples": len(samples),
            "detections": latest_detections
        }

    return jsonify(data)


@app.route(
    "/samples"
)
def get_samples():

    return jsonify({
        "samples": samples
    })


# ============================================================
# CAPTURE
# ============================================================

@app.route(
    "/capture",
    methods=["POST"]
)
def capture():

    data = request.get_json(
        silent=True
    ) or {}

    try:
        ground_truth = float(
            data["distance_m"]
        )
    except (
        KeyError,
        TypeError,
        ValueError
    ):

        return jsonify({
            "success": False,
            "message":
                "Enter a valid distance in metres."
        })


    if not (
        MIN_DISTANCE_M
        <= ground_truth
        <= MAX_DISTANCE_M
    ):

        return jsonify({
            "success": False,
            "message":
                "Distance must be between "
                f"{MIN_DISTANCE_M:.2f} and "
                f"{MAX_DISTANCE_M:.2f} metres."
        })


    with state_lock:

        detection_state = latest_detections

        persons = list(
            detection_state.get(
                "persons",
                []
            )
        )

        chairs = list(
            detection_state.get(
                "chairs",
                []
            )
        )


    if not persons:

        return jsonify({
            "success": False,
            "message":
                "No person detected."
        })


    if not chairs:

        return jsonify({
            "success": False,
            "message":
                "No chair detected."
        })


    pair = select_nearest_pair(
        persons,
        chairs
    )


    if pair is None:

        return jsonify({
            "success": False,
            "message":
                "Could not create Person-Chair pair."
        })


    person, chair, _ = pair

    features = build_features(
        person,
        chair
    )

    features["sample_index"] = (
        len(samples) + 1
    )

    features["ground_truth_m"] = (
        ground_truth
    )

    features["timestamp"] = (
        time.time()
    )

    samples.append(
        features
    )

    write_samples_csv()

    return jsonify({
        "success": True,
        "message":
            f"Sample {len(samples)} captured.",
        "sample":
            features
    })


# ============================================================
# CSV
# ============================================================

def write_samples_csv():

    if not samples:
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        samples[0].keys()
    )

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            samples
        )


# ============================================================
# MODEL FIT
# ============================================================

def fit_model():

    if len(samples) < MIN_FIT_SAMPLES:

        return {
            "success": False,
            "message":
                "Need at least "
                f"{MIN_FIT_SAMPLES} samples. "
                f"Currently {len(samples)}."
        }


    feature_names = [
        "pixel_gap",
        "horizontal_gap",
        "vertical_gap",
        "person_width",
        "person_height",
        "chair_width",
        "chair_height"
    ]


    X = np.asarray(
        [
            [
                1.0,
                *[
                    float(
                        sample[name]
                    )
                    for name in feature_names
                ]
            ]
            for sample in samples
        ],
        dtype=np.float64
    )


    y = np.asarray(
        [
            float(
                sample["ground_truth_m"]
            )
            for sample in samples
        ],
        dtype=np.float64
    )


    coefficients, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )


    prediction = X @ coefficients

    error = prediction - y

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )


    result = {

        "model":
            "multicue_linear_v2",

        "feature_names":
            feature_names,

        "coefficients":
            [
                float(value)
                for value in coefficients
            ],

        "samples":
            len(samples),

        "training_mae_m":
            mae,

        "training_mae_cm":
            mae * 100.0,

        "training_rmse_m":
            rmse,

        "training_rmse_cm":
            rmse * 100.0,

        "warning":
            "Training error is not independent validation accuracy."
    }


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_FILE.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )


    return {
        "success": True,
        **result
    }


@app.route(
    "/fit",
    methods=["POST"]
)
def fit_route():

    return jsonify(
        fit_model()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global camera
    global model
    global running

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 70)
    print("BAS SCENE DISTANCE CALIBRATION V2")
    print("=" * 70)

    print()
    print("Loading YOLO model...")

    model = YOLO(
        YOLO_MODEL
    )

    print("YOLO loaded.")


    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        FPS
    )


    if not camera.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"{CAMERA_DEVICE}"
        )


    print("Camera opened successfully.")


    worker = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    worker.start()


    print()
    print("Open in Windows Chrome:")
    print()
    print("http://localhost:5002")
    print()
    print("IMPORTANT:")
    print("1. Put one person and one chair in view.")
    print("2. Measure their real ground distance with a tape.")
    print("3. Enter that measured value in metres.")
    print("4. Capture many different distances.")
    print("5. Do NOT use guessed values.")
    print()
    print("Recommended:")
    print("0.50, 0.75, 1.00, 1.25, 1.50,")
    print("1.75, 2.00, 2.50 and 3.00 metres.")
    print()


    try:

        app.run(
            host="0.0.0.0",
            port=5002,
            threaded=True
        )

    finally:

        running = False

        if camera is not None:
            camera.release()

        print("Camera released.")


if __name__ == "__main__":
    main()
