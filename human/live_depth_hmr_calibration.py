"""
Live Depth ↔ HMR2 Calibration Collector

Pipeline:
    Webcam
       ↓
    YOLO Person Detection
       ↓
    Depth Anything V2
       ↓
    HMR2
       ↓
    Relative Depth + HMR Z
       ↓
    Calibration Samples

Run:
    python -m human.live_depth_hmr_calibration

Open:
    http://localhost:5000
"""

import os
import json
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from ultralytics import YOLO
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from hmr.hmr_processor import HMRProcessor


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_DEVICE = "/dev/video0"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 416

DEPTH_MODEL = "depth-anything/depth-anything-v2-small-hf"

MAX_CALIBRATION_SAMPLES = 10

OUTPUT_DIR = "test_results/depth_hmr_calibration"
SAMPLES_FILE = os.path.join(
    OUTPUT_DIR,
    "calibration_samples.json"
)

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None
latest_jpeg = None

frame_lock = threading.Lock()

running = True

frame_id = 0

calibration_samples = []

calibration_status = "Waiting for person..."

worker_error = None


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 60)
print("Loading YOLO...")
print("=" * 60)

yolo_model = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# LOAD DEPTH ANYTHING V2
# ============================================================

print("=" * 60)
print("Loading Depth Anything V2...")
print("=" * 60)

depth_processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL
)

depth_model.eval()

print("Depth Anything V2 loaded.")


# ============================================================
# LOAD HMR2
# ============================================================

print("=" * 60)
print("Loading HMR2...")
print("=" * 60)

hmr_processor = HMRProcessor()

print("HMR2 loaded.")


# ============================================================
# CAMERA
# ============================================================

print("=" * 60)
print("Opening camera...")
print("=" * 60)

camera = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

if not camera.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA_DEVICE}"
    )

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

# MJPEG is generally more stable for the webcam in WSL.
camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        *"MJPG"
    )
)

print("Camera opened successfully.")


# ============================================================
# DEPTH FUNCTION
# ============================================================

def estimate_depth(frame):
    """
    Estimate relative depth using Depth Anything V2.

    Returns:
        depth_map
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    inputs = depth_processor(
        images=pil_image,
        return_tensors="pt"
    )

    with __import__("torch").no_grad():

        outputs = depth_model(
            **inputs
        )

    predicted_depth = outputs.predicted_depth

    predicted_depth = (
        predicted_depth
        .squeeze()
        .cpu()
        .numpy()
    )

    predicted_depth = cv2.resize(
        predicted_depth,
        (
            frame.shape[1],
            frame.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )

    return predicted_depth


# ============================================================
# GET CENTRAL DEPTH
# ============================================================

def get_person_depth(
    depth_map,
    bbox
):
    """
    Get robust depth from the central region
    of the detected person's bounding box.
    """

    x1 = max(
        0,
        int(bbox["x1"])
    )

    y1 = max(
        0,
        int(bbox["y1"])
    )

    x2 = min(
        depth_map.shape[1] - 1,
        int(bbox["x2"])
    )

    y2 = min(
        depth_map.shape[0] - 1,
        int(bbox["y2"])
    )

    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1

    # Use central 60% region.
    cx1 = int(
        x1 + width * 0.20
    )

    cx2 = int(
        x2 - width * 0.20
    )

    cy1 = int(
        y1 + height * 0.20
    )

    cy2 = int(
        y2 - height * 0.20
    )

    region = depth_map[
        cy1:cy2,
        cx1:cx2
    ]

    if region.size == 0:
        return None

    valid = region[
        np.isfinite(region)
    ]

    if valid.size == 0:
        return None

    return float(
        np.median(valid)
    )


# ============================================================
# NORMALIZE DEPTH
# ============================================================

def normalize_depth(
    depth_map,
    value
):
    """
    Convert raw relative depth into 0-1 range.

    IMPORTANT:
        This is NOT metres.
    """

    finite_values = depth_map[
        np.isfinite(depth_map)
    ]

    if finite_values.size == 0:
        return None

    minimum = float(
        np.min(finite_values)
    )

    maximum = float(
        np.max(finite_values)
    )

    if maximum - minimum < 1e-8:
        return 0.0

    normalized = (
        value - minimum
    ) / (
        maximum - minimum
    )

    return float(
        np.clip(
            normalized,
            0.0,
            1.0
        )
    )


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_samples():

    data = {
        "created_at": time.time(),

        "camera": {
            "device": CAMERA_DEVICE,
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fps": CAMERA_FPS,
        },

        "depth_model": DEPTH_MODEL,

        "hmr_model": "HMR2",

        "sample_count": len(
            calibration_samples
        ),

        "samples": calibration_samples,
    }

    with open(
        SAMPLES_FILE,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

    print()
    print("=" * 60)
    print("Calibration samples saved.")
    print("=" * 60)
    print(
        f"File: {SAMPLES_FILE}"
    )
    print(
        f"Samples: {len(calibration_samples)}"
    )


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.55,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    frame,
    current_frame_id
):

    global calibration_status
    global worker_error

    try:

        # ----------------------------------------------------
        # YOLO PERSON DETECTION
        # ----------------------------------------------------

        results = yolo_model.predict(
            source=frame,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            classes=[0],
            device="cpu",
            verbose=False
        )

        persons = []

        if results:

            result = results[0]

            if result.boxes is not None:

                for box in result.boxes:

                    xyxy = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    confidence = float(
                        box.conf[0]
                        .cpu()
                        .item()
                    )

                    x1, y1, x2, y2 = map(
                        float,
                        xyxy
                    )

                    persons.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": confidence,
                        }
                    )

        # ----------------------------------------------------
        # NO PERSON
        # ----------------------------------------------------

        if not persons:

            calibration_status = (
                "No person detected"
            )

            draw_text(
                frame,
                calibration_status,
                (20, 30)
            )

            return frame

        # ----------------------------------------------------
        # USE FIRST PERSON
        # ----------------------------------------------------

        bbox = persons[0]

        x1 = int(
            bbox["x1"]
        )

        y1 = int(
            bbox["y1"]
        )

        x2 = int(
            bbox["x2"]
        )

        y2 = int(
            bbox["y2"]
        )

        # ----------------------------------------------------
        # DRAW PERSON BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # PERSON CENTER
        # ----------------------------------------------------

        center_x = (
            x1 + x2
        ) // 2

        center_y = (
            y1 + y2
        ) // 2

        cv2.circle(
            frame,
            (
                center_x,
                center_y
            ),
            5,
            (0, 255, 255),
            -1
        )

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------

        calibration_status = (
            "Estimating depth..."
        )

        depth_map = estimate_depth(
            frame
        )

        raw_depth = get_person_depth(
            depth_map,
            bbox
        )

        if raw_depth is None:

            calibration_status = (
                "Depth unavailable"
            )

            draw_text(
                frame,
                calibration_status,
                (20, 30)
            )

            return frame

        relative_depth = normalize_depth(
            depth_map,
            raw_depth
        )

        if relative_depth is None:

            calibration_status = (
                "Depth normalization failed"
            )

            draw_text(
                frame,
                calibration_status,
                (20, 30)
            )

            return frame

        # ----------------------------------------------------
        # IMPORTANT HMR2 FIX
        # ----------------------------------------------------
        #
        # HMRProcessor.process_frame()
        # expects:
        #
        #     (x1, y1, x2, y2)
        #
        # It does NOT expect:
        #
        #     {
        #         "x1": ...,
        #         "y1": ...,
        #         ...
        #     }
        #
        # Therefore convert the YOLO dictionary here.
        # ----------------------------------------------------

        hmr_bbox = (
            float(bbox["x1"]),
            float(bbox["y1"]),
            float(bbox["x2"]),
            float(bbox["y2"]),
        )

        calibration_status = (
            "Running HMR2..."
        )

        # ----------------------------------------------------
        # HMR2
        # ----------------------------------------------------

        hmr_result = (
            hmr_processor.process_frame(
                frame=frame,
                frame_id=current_frame_id,
                timestamp=time.time(),
                boxes=[hmr_bbox],
            )
        )

        # ----------------------------------------------------
        # CHECK HMR RESULT
        # ----------------------------------------------------

        if not hmr_result:

            calibration_status = (
                "HMR2 returned no result"
            )

            draw_text(
                frame,
                calibration_status,
                (20, 30)
            )

            return frame

        hmr_persons = hmr_result.get(
            "persons",
            []
        )

        if not hmr_persons:

            calibration_status = (
                "HMR2 detected no person"
            )

            draw_text(
                frame,
                calibration_status,
                (20, 30)
            )

            return frame

        # ----------------------------------------------------
        # GET HMR PERSON
        # ----------------------------------------------------

        hmr_person = (
            hmr_persons[0]
        )

        camera_translation = (
            hmr_person.get(
                "camera_translation"
            )
        )

        if not camera_translation:

            calibration_status = (
                "HMR2 camera translation unavailable"
            )

            draw_text(
                frame,
                calibration_status,
                (20, 30)
            )

            return frame

        # ----------------------------------------------------
        # HMR Z
        # ----------------------------------------------------

        hmr_z = float(
            camera_translation["z"]
        )

        hmr_x = float(
            camera_translation["x"]
        )

        hmr_y = float(
            camera_translation["y"]
        )

        # ----------------------------------------------------
        # STORE SAMPLE
        # ----------------------------------------------------

        if (
            len(calibration_samples)
            < MAX_CALIBRATION_SAMPLES
        ):

            sample = {

                "sample_id": (
                    len(calibration_samples)
                    + 1
                ),

                "frame_id": (
                    current_frame_id
                ),

                "timestamp": time.time(),

                "bbox": {
                    "x1": bbox["x1"],
                    "y1": bbox["y1"],
                    "x2": bbox["x2"],
                    "y2": bbox["y2"],
                },

                "relative_depth": (
                    relative_depth
                ),

                "raw_depth": (
                    raw_depth
                ),

                "hmr_camera_translation": {

                    "x": hmr_x,

                    "y": hmr_y,

                    "z": hmr_z,
                },
            }

            calibration_samples.append(
                sample
            )

            save_samples()

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if (
            len(calibration_samples)
            >= MAX_CALIBRATION_SAMPLES
        ):

            calibration_status = (
                "Calibration collection complete"
            )

        else:

            calibration_status = (
                "Calibration sample collected"
            )

        # ----------------------------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------------------------

        draw_text(
            frame,
            f"Person: {bbox['confidence']:.2f}",
            (20, 30)
        )

        draw_text(
            frame,
            f"Relative Depth: {relative_depth:.3f}",
            (20, 55)
        )

        draw_text(
            frame,
            f"HMR X: {hmr_x:.3f}",
            (20, 80)
        )

        draw_text(
            frame,
            f"HMR Y: {hmr_y:.3f}",
            (20, 105)
        )

        draw_text(
            frame,
            f"HMR Z: {hmr_z:.3f}",
            (20, 130)
        )

        draw_text(
            frame,
            (
                f"Samples: "
                f"{len(calibration_samples)}/"
                f"{MAX_CALIBRATION_SAMPLES}"
            ),
            (20, 155)
        )

        draw_text(
            frame,
            calibration_status,
            (20, 185)
        )

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        draw_text(
            frame,
            "Depth is relative - NOT metres",
            (20, 450),
            scale=0.55,
            thickness=2
        )

        return frame

    except Exception as e:

        worker_error = repr(e)

        calibration_status = (
            "Worker error"
        )

        print(
            "Calibration worker error:",
            repr(e)
        )

        # Show error on frame
        draw_text(
            frame,
            f"ERROR: {str(e)[:70]}",
            (20, 30)
        )

        return frame


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker():

    global latest_frame
    global latest_jpeg
    global frame_id
    global running

    while running:

        success, frame = camera.read()

        if not success:

            print(
                "Camera frame read failed."
            )

            time.sleep(0.1)

            continue

        frame_id += 1

        processed = process_frame(
            frame,
            frame_id
        )

        # ----------------------------------------------------
        # ENCODE JPEG
        # ----------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            processed,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                80
            ]
        )

        if not success:
            continue

        jpeg = encoded.tobytes()

        with frame_lock:

            latest_frame = processed

            latest_jpeg = jpeg


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():

    while running:

        with frame_lock:

            jpeg = latest_jpeg

        if jpeg is None:

            time.sleep(0.05)

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
            + b"\r\n"
        )

        time.sleep(0.03)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return """
<!DOCTYPE html>

<html>

<head>

<title>BAS HMR - Depth HMR Calibration</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

h1 {
    margin-top: 20px;
}

img {
    width: 80%;
    max-width: 1000px;
    border: 2px solid white;
}

.info {
    margin-top: 15px;
    font-size: 18px;
}

.warning {
    margin-top: 15px;
    color: #ffcc00;
}

</style>

</head>

<body>

<h1>
BAS HMR - Depth ↔ HMR2 Calibration
</h1>

<img src="/video_feed">

<div class="info">

<p>
Move closer and farther from the camera
to collect different depth samples.
</p>

<p>
10 samples will be collected automatically.
</p>

</div>

<div class="warning">

Depth Anything values are relative,
not physical metres.

</div>

</body>

</html>
"""


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# STATUS API
# ============================================================

@app.route("/status")
def status():

    return jsonify(
        {
            "running": running,

            "frame_id": frame_id,

            "sample_count": len(
                calibration_samples
            ),

            "max_samples": (
                MAX_CALIBRATION_SAMPLES
            ),

            "status": calibration_status,

            "worker_error": worker_error,

            "output_file": SAMPLES_FILE,
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    worker = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    worker.start()

    print()
    print("=" * 60)
    print("BAS HMR Depth ↔ HMR2 Calibration")
    print("=" * 60)

    print(
        f"Camera: {CAMERA_DEVICE}"
    )

    print(
        f"Web interface:"
    )

    print(
        f"http://localhost:{WEB_PORT}"
    )

    print()

    print(
        "Move the person closer/farther "
        "from the camera."
    )

    print(
        f"Collecting "
        f"{MAX_CALIBRATION_SAMPLES} samples."
    )

    print()

    try:

        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping server..."
        )

    finally:

        running = False

        time.sleep(0.5)

        camera.release()

        save_samples()

        print(
            "Camera released."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()