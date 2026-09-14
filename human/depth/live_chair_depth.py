from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch

from PIL import Image
from flask import Flask, Response

from ultralytics import YOLO
from transformers import pipeline


# ============================================================
# BAS-HMR
# LIVE CHAIR + DEPTH PERCEPTION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CAMERA = "/dev/video0"

YOLO_MODEL = ROOT / "yolo11n.pt"

IMG_SIZE = 416
YOLO_CONFIDENCE = 0.35

PERSON_CLASS = 0
CHAIR_CLASS = 56

PORT = 5000


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# MODEL LOADING
# ============================================================

print("=" * 70)
print("BAS-HMR — LIVE CHAIR + DEPTH PERCEPTION")
print("=" * 70)

print()
print("Loading YOLO11...")
print()

yolo = YOLO(str(YOLO_MODEL))

print("YOLO11 loaded.")
print()

print("Loading Depth Anything V2 Small...")
print()

depth_model = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=-1,
)

print("Depth model loaded.")
print()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_FPS,
    10
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        *"MJPG"
    )
)


# ============================================================
# SHARED STATE
# ============================================================

latest_frame = None

latest_output = None

lock = threading.Lock()

running = True


# ============================================================
# CAMERA READER
# ============================================================

def camera_reader():

    global latest_frame
    global running

    while running:

        success, frame = cap.read()

        if not success:
            time.sleep(0.01)
            continue

        with lock:

            latest_frame = frame.copy()


# ============================================================
# DEPTH SAMPLING
# ============================================================

def get_chair_depth(
    depth_array,
    bbox,
):
    """
    Extract robust relative depth from the
    interior region of the chair bounding box.

    Returns:
        median_depth
        normalized_depth
    """

    x1, y1, x2, y2 = bbox

    height, width = depth_array.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))

    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None, None

    box_width = x2 - x1
    box_height = y2 - y1

    # Ignore edges of bounding box.
    margin_x = int(box_width * 0.20)
    margin_y = int(box_height * 0.20)

    sx1 = x1 + margin_x
    sx2 = x2 - margin_x

    sy1 = y1 + margin_y
    sy2 = y2 - margin_y

    if sx2 <= sx1 or sy2 <= sy1:
        sx1, sx2 = x1, x2
        sy1, sy2 = y1, y2

    region = depth_array[
        sy1:sy2,
        sx1:sx2
    ]

    if region.size == 0:
        return None, None

    values = region.astype(
        np.float32
    ).flatten()

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return None, None

    median_depth = float(
        np.median(values)
    )

    min_depth = float(
        np.min(depth_array)
    )

    max_depth = float(
        np.max(depth_array)
    )

    if max_depth - min_depth < 1e-6:

        normalized = 0.0

    else:

        normalized = (
            median_depth - min_depth
        ) / (
            max_depth - min_depth
        )

    return median_depth, float(normalized)


# ============================================================
# INFERENCE WORKER
# ============================================================

def inference_worker():

    global latest_output
    global running

    frame_number = 0

    while running:

        # ----------------------------------------------------
        # Get latest frame
        # ----------------------------------------------------

        with lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.01)
            continue

        frame_number += 1

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = yolo.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=YOLO_CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        persons = []
        chairs = []

        if result.boxes is not None:

            for box in result.boxes:

                cls = int(
                    box.cls[0]
                )

                conf = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                detection = {
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
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

        # ----------------------------------------------------
        # DEPTH ESTIMATION
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb
        )

        depth_result = depth_model(
            pil_image
        )

        depth_image = depth_result[
            "depth"
        ]

        depth_array = np.array(
            depth_image
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Resize depth to camera frame
        # ----------------------------------------------------

        depth_array = cv2.resize(
            depth_array,
            (
                frame.shape[1],
                frame.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        # ----------------------------------------------------
        # DEPTH VISUALIZATION
        # ----------------------------------------------------

        depth_visual = cv2.normalize(
            depth_array,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        depth_visual = (
            depth_visual
            .astype(np.uint8)
        )

        depth_color = cv2.applyColorMap(
            depth_visual,
            cv2.COLORMAP_INFERNO
        )

        # ----------------------------------------------------
        # CHAIR DEPTH
        # ----------------------------------------------------

        chair_data = []

        for chair_id, chair in enumerate(chairs):

            bbox = chair["bbox"]

            depth_value, normalized_depth = (
                get_chair_depth(
                    depth_array,
                    bbox
                )
            )

            x1, y1, x2, y2 = bbox

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )

            chair_info = {
                "chair_id": chair_id,
                "bbox": bbox,
                "confidence": chair[
                    "confidence"
                ],
                "center": [
                    cx,
                    cy
                ],
                "depth": depth_value,
                "normalized_depth": normalized_depth,
            }

            chair_data.append(
                chair_info
            )

            # ------------------------------------------------
            # Draw chair
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3,
            )

            cv2.circle(
                frame,
                (cx, cy),
                7,
                (255, 0, 0),
                -1,
            )

            label = (
                f"CHAIR {chair_id} "
                f"{chair['confidence']:.2f}"
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    max(0, y1 - 30)
                ),
                (
                    x1 + 180,
                    y1
                ),
                (255, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    max(20, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # ------------------------------------------------
            # Depth text
            # ------------------------------------------------

            if normalized_depth is not None:

                depth_text = (
                    f"Depth: "
                    f"{normalized_depth:.3f}"
                )

                cv2.putText(
                    frame,
                    depth_text,
                    (
                        x1,
                        min(
                            frame.shape[0] - 10,
                            y2 + 20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2,
                )

        # ----------------------------------------------------
        # Draw persons
        # ----------------------------------------------------

        for person_id, person in enumerate(
            persons
        ):

            x1, y1, x2, y2 = (
                person["bbox"]
            )

            conf = person[
                "confidence"
            ]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            label = (
                f"PERSON {person_id} "
                f"{conf:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(20, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # ----------------------------------------------------
        # Information panel
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (300, 110),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            "BAS CHAIR DEPTH",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Persons: {len(persons)}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Chairs: {len(chairs)}",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (10, 103),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        # ----------------------------------------------------
        # Combine camera + depth
        # ----------------------------------------------------

        combined = np.hstack(
            [
                frame,
                depth_color
            ]
        )

        # ----------------------------------------------------
        # Add depth-map title
        # ----------------------------------------------------

        cv2.rectangle(
            combined,
            (
                frame.shape[1],
                0
            ),
            (
                frame.shape[1] + 250,
                40
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            combined,
            "DEPTH MAP",
            (
                frame.shape[1] + 10,
                27
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Encode JPEG
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            combined,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not success:
            continue

        output = buffer.tobytes()

        with lock:

            latest_output = output


# ============================================================
# FLASK PAGE
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>BAS-HMR Chair Depth</title>

        <style>

            body {
                margin: 0;
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }

            h1 {
                margin: 20px 0 5px 0;
            }

            p {
                color: #aaa;
            }

            img {
                width: 1280px;
                max-width: 96%;
                border: 2px solid #444;
            }

            .legend {
                margin: 12px;
                font-size: 18px;
            }

            .person {
                color: #00ff00;
            }

            .chair {
                color: #0088ff;
            }

            .warning {
                margin-top: 10px;
                color: #ffcc00;
            }

        </style>

    </head>

    <body>

        <h1>BAS-HMR — Chair Depth Perception</h1>

        <p>
            Live YOLO11 + Depth Anything V2
        </p>

        <div class="legend">

            <span class="person">
                🟩 PERSON
            </span>

            &nbsp;&nbsp;&nbsp;

            <span class="chair">
                🟦 CHAIR
            </span>

        </div>

        <img src="/video">

        <p class="warning">
            Depth shown here is RELATIVE depth, not metres.
        </p>

    </body>

    </html>
    """


# ============================================================
# VIDEO STREAM
# ============================================================

@app.route("/video")
def video():

    def generate():

        while True:

            with lock:

                frame = latest_output

            if frame is None:

                time.sleep(0.05)

                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

            time.sleep(0.01)

    return Response(
        generate(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    camera_thread = threading.Thread(
        target=camera_reader,
        daemon=True
    )

    inference_thread = threading.Thread(
        target=inference_worker,
        daemon=True
    )

    camera_thread.start()

    inference_thread.start()

    print()
    print("=" * 70)
    print("LIVE SERVER READY")
    print("=" * 70)
    print()
    print("Open Windows Chrome:")
    print()
    print("http://localhost:5000")
    print()
    print("Press Ctrl+C to stop.")
    print()

    try:

        app.run(
            host="0.0.0.0",
            port=PORT,
            threaded=True,
            debug=False,
        )

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    finally:

        running = False

        cap.release()
