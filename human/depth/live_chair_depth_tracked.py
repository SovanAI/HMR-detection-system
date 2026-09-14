from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from flask import Flask, Response, render_template_string
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

from human.chair_tracker import ChairTracker, ChairDetection


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

YOLO_MODEL_PATH = ROOT / "yolo11n.pt"

CAMERA_DEVICE = "/dev/video0"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

IMG_SIZE = 416
CONFIDENCE = 0.35


# ============================================================
# YOLO CLASSES
# ============================================================

PERSON_CLASS = 0
CHAIR_CLASS = 56


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL CAMERA STATE
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

latest_result = None
result_lock = threading.Lock()

running = True


# ============================================================
# MODELS
# ============================================================

print("=" * 70)
print("BAS-HMR — LIVE CHAIR DEPTH + TRACKING")
print("=" * 70)

print("\nLoading YOLO11n...")

yolo_model = YOLO(str(YOLO_MODEL_PATH))

print("YOLO11n loaded.")

print("\nLoading Depth Anything V2 Small...")

depth_pipeline = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=-1,
)

print("Depth Anything V2 loaded.")


# ============================================================
# CHAIR TRACKER
# ============================================================

chair_tracker = ChairTracker(
    max_center_distance=120.0,
    min_iou=0.05,
    max_missed_frames=10,
)


# ============================================================
# CAMERA READER
# ============================================================

def camera_reader():
    global latest_frame
    global running

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        running = False
        return

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
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    print("\nCamera opened successfully.")

    while running:

        success, frame = camera.read()

        if not success:
            time.sleep(0.05)
            continue

        with frame_lock:
            latest_frame = frame.copy()

    camera.release()

    print("Camera released.")


# ============================================================
# GET LATEST FRAME
# ============================================================

def get_latest_frame():

    with frame_lock:

        if latest_frame is None:
            return None

        return latest_frame.copy()


# ============================================================
# DEPTH ESTIMATION
# ============================================================

def estimate_depth(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(rgb)

    result = depth_pipeline(pil_image)

    depth_image = result["depth"]

    depth_map = np.array(
        depth_image,
        dtype=np.float32,
    )

    depth_map = cv2.resize(
        depth_map,
        (
            frame.shape[1],
            frame.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return depth_map


# ============================================================
# CHAIR DEPTH
# ============================================================

def calculate_chair_depth(
    depth_map,
    bbox,
):

    x1, y1, x2, y2 = bbox

    height, width = depth_map.shape

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))

    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_width = x2 - x1
    box_height = y2 - y1

    # Ignore outer 20% of the chair bounding box.
    # This reduces background/edge contamination.
    margin_x = int(box_width * 0.20)
    margin_y = int(box_height * 0.20)

    sx1 = x1 + margin_x
    sx2 = x2 - margin_x

    sy1 = y1 + margin_y
    sy2 = y2 - margin_y

    if sx2 <= sx1 or sy2 <= sy1:
        return float(np.median(depth_map[y1:y2, x1:x2]))

    region = depth_map[
        sy1:sy2,
        sx1:sx2,
    ]

    if region.size == 0:
        return 0.0

    return float(np.median(region))


# ============================================================
# NORMALIZE DEPTH
# ============================================================

def normalize_depth(
    depth_value,
    depth_map,
):

    min_depth = float(np.min(depth_map))
    max_depth = float(np.max(depth_map))

    denominator = max_depth - min_depth

    if denominator < 1e-6:
        return 0.0

    normalized = (
        depth_value - min_depth
    ) / denominator

    return float(
        np.clip(normalized, 0.0, 1.0)
    )


# ============================================================
# DEPTH VISUALIZATION
# ============================================================

def create_depth_visualization(depth_map):

    depth_normalized = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    depth_normalized = depth_normalized.astype(
        np.uint8
    )

    depth_color = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_INFERNO,
    )

    return depth_color


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color,
    scale=0.55,
    thickness=2,
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# INFERENCE WORKER
# ============================================================

def inference_worker():

    global latest_result
    global running

    frame_counter = 0

    print("\nInference worker started.")

    while running:

        frame = get_latest_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        frame_counter += 1

        display_frame = frame.copy()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = yolo_model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        person_boxes = []
        chair_detections = []

        if result.boxes is not None:

            for box in result.boxes:

                cls = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                x1, y1, x2, y2 = (
                    map(
                        int,
                        box.xyxy[0].tolist(),
                    )
                )

                bbox = (
                    x1,
                    y1,
                    x2,
                    y2,
                )

                # ------------------------------------------------
                # PERSON
                # ------------------------------------------------

                if cls == PERSON_CLASS:

                    person_boxes.append(
                        (
                            bbox,
                            confidence,
                        )
                    )

                # ------------------------------------------------
                # CHAIR
                # ------------------------------------------------

                elif cls == CHAIR_CLASS:

                    chair_detections.append(
                        ChairDetection(
                            bbox=bbox,
                            confidence=confidence,
                        )
                    )

        # ----------------------------------------------------
        # CHAIR TRACKING
        # ----------------------------------------------------

        tracked_chairs = chair_tracker.update(
            chair_detections
        )

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------

        depth_map = estimate_depth(
            frame
        )

        depth_visual = create_depth_visualization(
            depth_map
        )

        # ----------------------------------------------------
        # DEPTH RANGE
        # ----------------------------------------------------

        min_depth = float(
            np.min(depth_map)
        )

        max_depth = float(
            np.max(depth_map)
        )

        # ----------------------------------------------------
        # DRAW PERSONS
        # ----------------------------------------------------

        for bbox, confidence in person_boxes:

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            draw_text(
                display_frame,
                f"PERSON {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                (0, 255, 0),
            )

        # ----------------------------------------------------
        # DRAW TRACKED CHAIRS
        # ----------------------------------------------------

        chair_information = []

        for chair in tracked_chairs:

            depth_value = calculate_chair_depth(
                depth_map,
                chair.bbox,
            )

            relative_depth = normalize_depth(
                depth_value,
                depth_map,
            )

            x1, y1, x2, y2 = chair.bbox

            cx, cy = chair.center

            # -----------------------------------------------
            # CHAIR BOX
            # -----------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3,
            )

            # -----------------------------------------------
            # CENTER
            # -----------------------------------------------

            cv2.circle(
                display_frame,
                (cx, cy),
                6,
                (255, 0, 0),
                -1,
            )

            # -----------------------------------------------
            # LABEL
            # -----------------------------------------------

            label = (
                f"CHAIR #{chair.chair_id} "
                f"{chair.confidence:.2f}"
            )

            draw_text(
                display_frame,
                label,
                (x1, max(20, y1 - 10)),
                (255, 0, 0),
                scale=0.55,
                thickness=2,
            )

            draw_text(
                display_frame,
                f"Depth: {relative_depth:.3f}",
                (x1, min(
                    display_frame.shape[0] - 10,
                    y2 + 20
                )),
                (255, 0, 0),
                scale=0.50,
                thickness=2,
            )

            chair_information.append(
                {
                    "chair_id": chair.chair_id,
                    "bbox": chair.bbox,
                    "center": chair.center,
                    "confidence": chair.confidence,
                    "depth": relative_depth,
                }
            )

            # ------------------------------------------------
            # DEPTH MAP CHAIR CENTER
            # ------------------------------------------------

            cv2.circle(
                depth_visual,
                (cx, cy),
                7,
                (255, 255, 255),
                -1,
            )

            draw_text(
                depth_visual,
                f"#{chair.chair_id} {relative_depth:.2f}",
                (
                    max(0, x1),
                    max(20, y1),
                ),
                (255, 255, 255),
                scale=0.55,
                thickness=2,
            )

        # ----------------------------------------------------
        # INFORMATION PANEL
        # ----------------------------------------------------

        cv2.rectangle(
            display_frame,
            (0, 0),
            (300, 105),
            (0, 0, 0),
            -1,
        )

        draw_text(
            display_frame,
            "BAS CHAIR DEPTH + TRACKING",
            (10, 25),
            (255, 255, 255),
            scale=0.55,
            thickness=2,
        )

        draw_text(
            display_frame,
            f"Persons: {len(person_boxes)}",
            (10, 50),
            (0, 255, 0),
        )

        draw_text(
            display_frame,
            f"Chairs: {len(tracked_chairs)}",
            (10, 75),
            (255, 0, 0),
        )

        draw_text(
            display_frame,
            f"Frame: {frame_counter}",
            (10, 98),
            (255, 255, 255),
            scale=0.45,
            thickness=1,
        )

        # ----------------------------------------------------
        # DEPTH INFORMATION
        # ----------------------------------------------------

        cv2.rectangle(
            depth_visual,
            (0, 0),
            (270, 50),
            (0, 0, 0),
            -1,
        )

        draw_text(
            depth_visual,
            "DEPTH MAP",
            (10, 25),
            (255, 255, 255),
            scale=0.60,
            thickness=2,
        )

        draw_text(
            depth_visual,
            f"Range: {min_depth:.2f} - {max_depth:.2f}",
            (10, 45),
            (255, 255, 255),
            scale=0.35,
            thickness=1,
        )

        # ----------------------------------------------------
        # SIDE-BY-SIDE DISPLAY
        # ----------------------------------------------------

        combined = np.hstack(
            (
                display_frame,
                depth_visual,
            )
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        with result_lock:

            latest_result = {
                "frame": combined,
                "chairs": chair_information,
                "persons": len(person_boxes),
                "frame_id": frame_counter,
            }

        # ----------------------------------------------------
        # Small CPU-friendly pause
        # ----------------------------------------------------

        time.sleep(0.01)

    print("Inference worker stopped.")


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():

    while running:

        with result_lock:

            if latest_result is None:
                frame = None
            else:
                frame = latest_result["frame"].copy()

        if frame is None:
            time.sleep(0.05)
            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
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
# HTML
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>BAS-HMR Chair Tracking</title>

<style>

body {
    margin: 0;
    background: #101010;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

h1 {
    margin-top: 25px;
    margin-bottom: 5px;
}

.subtitle {
    color: #aaa;
    margin-bottom: 20px;
}

.stream {
    width: min(1280px, 95%);
    border: 2px solid #444;
}

.warning {
    color: #ffd000;
    margin-top: 12px;
}

.legend {
    margin: 15px;
    font-size: 18px;
}

.person {
    color: #00ff44;
}

.chair {
    color: #3388ff;
}

</style>

</head>

<body>

<h1>BAS-HMR — Chair Tracking + Depth</h1>

<div class="subtitle">
    YOLO11 + Chair Tracker + Depth Anything V2
</div>

<div class="legend">
    <span class="person">■ PERSON</span>
    &nbsp;&nbsp;&nbsp;
    <span class="chair">■ CHAIR</span>
</div>

<img
    class="stream"
    src="/video"
    alt="BAS-HMR live chair tracking stream"
/>

<div class="warning">
    Depth shown here is RELATIVE depth, not metres.
</div>

</body>

</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML_PAGE
    )


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# START SYSTEM
# ============================================================

if __name__ == "__main__":

    camera_thread = threading.Thread(
        target=camera_reader,
        daemon=True,
    )

    inference_thread = threading.Thread(
        target=inference_worker,
        daemon=True,
    )

    camera_thread.start()

    time.sleep(1)

    inference_thread.start()

    print("\n")
    print("=" * 70)
    print("LIVE SERVER READY")
    print("=" * 70)
    print()
    print("Open Windows Chrome:")
    print()
    print("http://localhost:5000")
    print()
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
        )

    except KeyboardInterrupt:

        print("\nStopping BAS-HMR...")

    finally:

        running = False

        time.sleep(1)

        print("Shutdown complete.")
