import sys
import time
import threading
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BAS-HMR — LIVE DISTANCE TEST V2")
print("=" * 70)


# ============================================================
# IMPORT PIPELINE
# ============================================================

print("\n[1] Loading pipeline modules...")

from ultralytics import YOLO

from human.hmr.hmr_processor import HMRProcessor

from human.distance.live_metric_distance import (
    MetricDepthEstimator,
)

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

print("[OK] Pipeline modules imported.")


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 10

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

PERSON_CLASS = 0
CHAIR_CLASS = 56

# HMR + depth are CPU intensive.
PROCESS_INTERVAL = 10

# Same development intrinsics used during V2 collection.
FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "distance"
    / "distance_model_v1.joblib"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n[2] Loading trained distance model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Distance model not found:\n{MODEL_PATH}"
    )

model_package = joblib.load(
    MODEL_PATH
)

distance_model = model_package["model"]

feature_names = model_package[
    "feature_names"
]

print("[OK] Distance model loaded.")
print(
    "[OK] Model:",
    model_package["model_name"]
)
print(
    "[OK] Version:",
    model_package["version"]
)
print(
    "[OK] Features:",
    len(feature_names)
)

if len(feature_names) != 34:
    raise RuntimeError(
        "Model does not contain exactly 34 features."
    )


# ============================================================
# LOAD YOLO
# ============================================================

print("\n[3] Loading YOLO11n...")

yolo = YOLO("yolo11n.pt")

print("[OK] YOLO11n loaded.")
print("[OK] Device: CPU")


# ============================================================
# LOAD HMR2
# ============================================================

print("\n[4] Loading HMR2...")

hmr = HMRProcessor()

print("[OK] HMR2 loaded.")


# ============================================================
# LOAD METRIC DEPTH
# ============================================================

print("\n[5] Loading metric depth...")

depth_estimator = MetricDepthEstimator()

print("[OK] Metric depth loaded.")


# ============================================================
# LOAD 3D FEATURE PIPELINE
# ============================================================

print("\n[6] Loading 3D feature modules...")

intrinsics = CameraIntrinsics(
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY,
)

depth_converter = DepthTo3D(
    intrinsics=intrinsics,
    min_depth=0.05,
    max_depth=20.0,
)

grid_extractor = ObjectGridExtractor(
    depth_converter=depth_converter,
    rows=5,
    cols=5,
    border_ratio=0.1,
    depth_radius=2,
)

feature_extractor = FeatureExtractor()

pair_feature_extractor = (
    PairFeatureExtractor()
)

print("[OK] 3D feature modules loaded.")


# ============================================================
# CAMERA
# ============================================================

print("\n[7] Opening camera...")

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2,
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
    FPS,
)

camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG"),
)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open /dev/video0"
    )

print("[OK] Camera opened.")


# ============================================================
# SHARED STATE
# ============================================================

lock = threading.Lock()

latest_frame = None

latest_result = {
    "status": "starting",
    "frame": 0,

    "person_detected": False,
    "chair_detected": False,

    "features_valid": False,

    "distance_cm": None,

    "hmr_success": False,
    "hmr_joint_count": 0,

    "processing_time_sec": 0.0,
}

running = True


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.6,
    thickness=2,
):
    """
    Draw text with a dark outline so that it
    remains visible against the camera image.
    """

    x, y = position

    # Dark outline
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )

    # Main text
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def center_of_bbox(bbox):
    """
    Return center point of a bounding box.
    """

    x1, y1, x2, y2 = bbox

    cx = int(
        (x1 + x2) / 2
    )

    cy = int(
        (y1 + y2) / 2
    )

    return cx, cy


def draw_distance_connection(
    frame,
    person_bbox,
    chair_bbox,
    distance_cm,
):
    """
    Draw a visual connection from the person
    to the chair and show predicted distance.
    """

    person_center = center_of_bbox(
        person_bbox
    )

    chair_center = center_of_bbox(
        chair_bbox
    )

    px, py = person_center
    cx, cy = chair_center

    # --------------------------------------------------------
    # Draw center points
    # --------------------------------------------------------

    cv2.circle(
        frame,
        person_center,
        7,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        chair_center,
        7,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Draw connection line
    # --------------------------------------------------------

    cv2.line(
        frame,
        person_center,
        chair_center,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Calculate midpoint
    # --------------------------------------------------------

    mid_x = int(
        (px + cx) / 2
    )

    mid_y = int(
        (py + cy) / 2
    )

    # --------------------------------------------------------
    # Distance label
    # --------------------------------------------------------

    if distance_cm is not None:

        label = (
            f"{distance_cm:.1f} cm"
        )

    else:

        label = "-- cm"


    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65
    thickness = 2

    (text_w, text_h), baseline = (
        cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )
    )

    padding = 8

    box_x1 = (
        mid_x
        - text_w // 2
        - padding
    )

    box_y1 = (
        mid_y
        - text_h
        - padding
    )

    box_x2 = (
        mid_x
        + text_w // 2
        + padding
    )

    box_y2 = (
        mid_y
        + baseline
        + padding
    )

    # Keep label inside frame
    box_x1 = max(
        5,
        box_x1
    )

    box_y1 = max(
        5,
        box_y1
    )

    box_x2 = min(
        FRAME_WIDTH - 5,
        box_x2
    )

    box_y2 = min(
        FRAME_HEIGHT - 5,
        box_y2
    )

    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (0, 0, 0),
        -1,
    )

    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (255, 255, 255),
        1,
    )

    # --------------------------------------------------------
    # Draw label
    # --------------------------------------------------------

    text_x = (
        mid_x
        - text_w // 2
    )

    text_y = (
        mid_y
        + text_h // 2
    )

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# CAMERA PROCESSING LOOP
# ============================================================

def camera_loop():

    global latest_frame
    global latest_result
    global running

    frame_number = 0

    last_distance = None

    last_person_bbox = None
    last_chair_bbox = None

    last_hmr_success = False
    last_hmr_joint_count = 0

    last_features_valid = False

    last_processing_time = 0.0

    while running:

        ok, frame = camera.read()

        if not ok:

            print(
                "[WARN] Camera frame read failed."
            )

            time.sleep(0.1)

            continue

        frame_number += 1

        display = frame.copy()


        # ====================================================
        # YOLO DETECTION
        # ====================================================

        results = yolo.predict(
            frame,
            classes=[
                PERSON_CLASS,
                CHAIR_CLASS,
            ],
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            device="cpu",
            verbose=False,
        )


        persons = []
        chairs = []


        if results:

            result = results[0]

            if result.boxes is not None:

                for box in result.boxes:

                    cls = int(
                        box.cls[0].item()
                    )

                    conf = float(
                        box.conf[0].item()
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    detection = {
                        "bbox": (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ),
                        "confidence": conf,
                    }

                    if cls == PERSON_CLASS:

                        persons.append(
                            detection
                        )

                    elif cls == CHAIR_CLASS:

                        chairs.append(
                            detection
                        )


        # ====================================================
        # DRAW DETECTION BOXES
        # ====================================================

        for person in persons:

            x1, y1, x2, y2 = (
                person["bbox"]
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            draw_text(
                display,
                f"PERSON "
                f"{person['confidence']:.2f}",
                (
                    x1,
                    max(
                        20,
                        y1 - 8,
                    ),
                ),
                0.55,
                2,
            )


        for chair in chairs:

            x1, y1, x2, y2 = (
                chair["bbox"]
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            draw_text(
                display,
                f"CHAIR "
                f"{chair['confidence']:.2f}",
                (
                    x1,
                    min(
                        FRAME_HEIGHT - 10,
                        y2 + 20,
                    ),
                ),
                0.55,
                2,
            )


        # ====================================================
        # PROCESS PERSON + CHAIR
        # ====================================================

        if (
            frame_number % PROCESS_INTERVAL == 0
            and len(persons) > 0
            and len(chairs) > 0
        ):

            start_time = time.time()

            # ------------------------------------------------
            # Current person and chair
            # ------------------------------------------------

            person = persons[0]
            chair = chairs[0]

            person_bbox = person["bbox"]
            chair_bbox = chair["bbox"]

            last_person_bbox = (
                person_bbox
            )

            last_chair_bbox = (
                chair_bbox
            )


            # ------------------------------------------------
            # HMR
            # ------------------------------------------------

            hmr_success = False
            hmr_joint_count = 0

            try:

                hmr_result = hmr.process_frame(
                    frame,
                    frame_number,
                    time.time(),
                    [person_bbox],
                )

                if hmr_result:

                    hmr_success = True

                    if "persons" in hmr_result:

                        if len(
                            hmr_result[
                                "persons"
                            ]
                        ) > 0:

                            hmr_joint_count = (
                                hmr_result[
                                    "persons"
                                ][0]
                                .get(
                                    "pose",
                                    {}
                                )
                                .get(
                                    "joint_count",
                                    0,
                                )
                            )

            except Exception as exc:

                print(
                    "[WARN] HMR failed:",
                    exc,
                )


            # ------------------------------------------------
            # METRIC DEPTH
            # ------------------------------------------------

            try:

                depth_map = (
                    depth_estimator.estimate(
                        frame
                    )
                )

            except Exception as exc:

                print(
                    "[WARN] Depth estimation failed:",
                    exc,
                )

                depth_map = None


            # ------------------------------------------------
            # FEATURE EXTRACTION
            # ------------------------------------------------

            features_valid = False
            predicted_distance = None

            if depth_map is not None:

                try:

                    person_grid = (
                        grid_extractor.extract(
                            depth_map,
                            BoundingBox(
                                x1=person_bbox[0],
                                y1=person_bbox[1],
                                x2=person_bbox[2],
                                y2=person_bbox[3],
                            ),
                            object_id=0,
                            object_class="person",
                        )
                    )


                    chair_grid = (
                        grid_extractor.extract(
                            depth_map,
                            BoundingBox(
                                x1=chair_bbox[0],
                                y1=chair_bbox[1],
                                x2=chair_bbox[2],
                                y2=chair_bbox[3],
                            ),
                            object_id=1,
                            object_class="chair",
                        )
                    )


                    if (
                        person_grid is not None
                        and chair_grid is not None
                    ):

                        # ------------------------------------------------
                        # Person features
                        # ------------------------------------------------

                        person_features = (
                            feature_extractor.extract(
                                person_grid
                            )
                        )


                        # ------------------------------------------------
                        # Chair features
                        # ------------------------------------------------

                        chair_features = (
                            feature_extractor.extract(
                                chair_grid
                            )
                        )


                        # ------------------------------------------------
                        # Pair features
                        # ------------------------------------------------

                        pair_features = (
                            pair_feature_extractor.extract(
                                person_grid,
                                chair_grid,
                            )
                        )


                        # ------------------------------------------------
                        # Convert pair features to model vector
                        # ------------------------------------------------

                        vector = (
                            pair_feature_extractor.to_vector(
                                pair_features
                            )
                        )


                        vector = np.asarray(
                            vector,
                            dtype=np.float32,
                        ).reshape(
                            1,
                            -1,
                        )


                        # ------------------------------------------------
                        # Strict schema validation
                        # ------------------------------------------------

                        if vector.shape[1] != 34:

                            raise ValueError(
                                "Feature vector has "
                                f"{vector.shape[1]} features. "
                                "Expected 34."
                            )


                        if not np.isfinite(
                            vector
                        ).all():

                            raise ValueError(
                                "Feature vector contains "
                                "NaN or Inf."
                            )


                        # ------------------------------------------------
                        # MODEL PREDICTION
                        # ------------------------------------------------

                        prediction = (
                            distance_model.predict(
                                vector
                            )
                        )

                        predicted_distance = float(
                            prediction[0]
                        )

                        features_valid = True

                        last_distance = (
                            predicted_distance
                        )

                        last_features_valid = True

                        last_hmr_success = (
                            hmr_success
                        )

                        last_hmr_joint_count = (
                            hmr_joint_count
                        )


                        print(
                            "\n"
                            + "-" * 60
                        )

                        print(
                            "[LIVE DISTANCE PREDICTION]"
                        )

                        print(
                            f"Frame       : "
                            f"{frame_number}"
                        )

                        print(
                            f"Distance    : "
                            f"{predicted_distance:.2f} cm"
                        )

                        print(
                            f"Features    : "
                            f"{vector.shape[1]}"
                        )

                        print(
                            f"HMR         : "
                            f"{hmr_success}"
                        )

                        print(
                            f"HMR joints  : "
                            f"{hmr_joint_count}"
                        )

                        print(
                            "-" * 60
                        )


                except Exception as exc:

                    last_features_valid = False

                    print(
                        "[WARN] "
                        "Feature/model processing failed:"
                    )

                    print(
                        f"       {exc}"
                    )


            last_processing_time = (
                time.time()
                - start_time
            )


            # ------------------------------------------------
            # UPDATE STATE
            # ------------------------------------------------

            with lock:

                latest_result = {

                    "status": "processed",

                    "frame": frame_number,

                    "person_detected": True,

                    "chair_detected": True,

                    "features_valid": (
                        features_valid
                    ),

                    "distance_cm": (
                        predicted_distance
                    ),

                    "hmr_success": (
                        hmr_success
                    ),

                    "hmr_joint_count": (
                        hmr_joint_count
                    ),

                    "processing_time_sec": (
                        last_processing_time
                    ),
                }


        # ====================================================
        # DRAW CONNECTION LINE
        # ====================================================

        if (
            last_person_bbox is not None
            and last_chair_bbox is not None
        ):

            draw_distance_connection(
                display,
                last_person_bbox,
                last_chair_bbox,
                last_distance,
            )


        # ====================================================
        # TOP INFORMATION PANEL
        # ====================================================

        overlay = display.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (FRAME_WIDTH, 145),
            (0, 0, 0),
            -1,
        )

        display = cv2.addWeighted(
            overlay,
            0.55,
            display,
            0.45,
            0,
        )


        draw_text(
            display,
            "BAS-HMR | LIVE DISTANCE",
            (15, 27),
            0.65,
            2,
        )


        draw_text(
            display,
            f"Person detected: "
            f"{len(persons)}",
            (15, 55),
            0.5,
            2,
        )


        draw_text(
            display,
            f"Chair detected: "
            f"{len(chairs)}",
            (15, 80),
            0.5,
            2,
        )


        if last_distance is not None:

            draw_text(
                display,
                f"DISTANCE: "
                f"{last_distance:.1f} cm",
                (15, 112),
                0.7,
                2,
            )

        else:

            draw_text(
                display,
                "DISTANCE: --",
                (15, 112),
                0.7,
                2,
            )


        # ====================================================
        # BOTTOM STATUS
        # ====================================================

        draw_text(
            display,
            f"Frame: {frame_number}",
            (
                15,
                FRAME_HEIGHT - 42,
            ),
            0.5,
            2,
        )


        if last_features_valid:

            draw_text(
                display,
                "34 FEATURES: OK",
                (
                    15,
                    FRAME_HEIGHT - 18,
                ),
                0.5,
                2,
            )

        else:

            draw_text(
                display,
                "34 FEATURES: WAITING",
                (
                    15,
                    FRAME_HEIGHT - 18,
                ),
                0.5,
                2,
            )


        # ====================================================
        # SAVE FRAME
        # ====================================================

        with lock:

            latest_frame = (
                display.copy()
            )


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


HTML = """
<!DOCTYPE html>

<html>

<head>

<title>BAS-HMR Live Distance</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 0;
    padding: 20px;
}

h1 {
    margin-bottom: 20px;
}

.video-container {
    width: 100%;
    max-width: 900px;
    margin: auto;
}

img {
    width: 100%;
    height: auto;
    border: 2px solid white;
}

.panel {
    width: 100%;
    max-width: 900px;
    margin: 20px auto;
    padding: 18px;
    box-sizing: border-box;
    background: #222;
    text-align: left;
}

.distance {
    font-size: 34px;
    font-weight: bold;
    margin-bottom: 12px;
}

.status {
    font-family: monospace;
    white-space: pre-wrap;
}

</style>

</head>

<body>

<h1>BAS-HMR Live Distance Test V2</h1>

<div class="video-container">

<img src="/video">

</div>

<div class="panel">

<div class="distance">

Predicted Distance:
<span id="distance">--</span>

</div>

<div class="status" id="status">
Loading...
</div>

</div>


<script>

async function updateStatus() {

    try {

        const response =
            await fetch("/status");

        const data =
            await response.json();


        if (
            data.distance_cm !== null
            && data.distance_cm !== undefined
        ) {

            document.getElementById(
                "distance"
            ).innerText =
                data.distance_cm.toFixed(1)
                + " cm";

        } else {

            document.getElementById(
                "distance"
            ).innerText =
                "--";

        }


        document.getElementById(
            "status"
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

    return render_template_string(
        HTML
    )


@app.route("/status")
def status():

    with lock:

        return jsonify(
            latest_result
        )


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


# ============================================================
# VIDEO GENERATOR
# ============================================================

def generate_frames():

    while running:

        with lock:

            if latest_frame is None:

                continue

            frame = (
                latest_frame.copy()
            )


        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                80,
            ],
        )


        if not ok:

            continue


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    thread = threading.Thread(
        target=camera_loop,
        daemon=True,
    )

    thread.start()


    print("\n")
    print("=" * 70)
    print("BAS-HMR LIVE DISTANCE SERVER V2")
    print("=" * 70)

    print(
        "\nOpen browser:"
    )

    print(
        "http://localhost:5012"
    )

    print(
        "\nVisualization:"
    )

    print(
        "PERSON ●────────────● CHAIR"
    )

    print(
        "             XX.X cm"
    )

    print(
        "\nPress CTRL+C to stop."
    )

    print("=" * 70)


    try:

        app.run(
            host="0.0.0.0",
            port=5012,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    except KeyboardInterrupt:

        print(
            "\n[STOP] CTRL+C received."
        )

    finally:

        running = False

        camera.release()

        print(
            "[OK] Camera released."
        )

        print(
            "[OK] Live distance test stopped."
        )
