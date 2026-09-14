from __future__ import annotations

import time
from pathlib import Path

import cv2
from flask import Flask, Response
from ultralytics import YOLO


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# MODEL
# ============================================================

MODEL = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "cardboard_yolo11n_v2"
    / "weights"
    / "best.pt"
)


# ============================================================
# CAMERA
# ============================================================

CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10


# ============================================================
# YOLO
# ============================================================

CONFIDENCE = 0.50
IMAGE_SIZE = 416


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("BAS LIVE CARDBOARD BOX DETECTION - V2")
print("=" * 70)

print()
print("Model:")
print(MODEL)

if not MODEL.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL}"
    )

model = YOLO(str(MODEL))

print()
print("Model loaded:")
print(model.names)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2,
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open camera "
        f"{CAMERA}"
    )


cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG"),
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

time.sleep(1.0)


print()
print("Camera:")
print(
    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
)

print(
    f"FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}"
)

print()


# ============================================================
# FRAME GENERATOR
# ============================================================

def generate_frames():

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "WARNING: Camera frame failed."
            )

            time.sleep(0.1)

            continue

        frame_id += 1

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        boxes = result.boxes

        box_count = (
            len(boxes)
            if boxes is not None
            else 0
        )

        # ----------------------------------------------------
        # Annotated frame
        # ----------------------------------------------------

        annotated = frame.copy()

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        if boxes is not None:

            for index, box in enumerate(boxes):

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy,
                )

                confidence = float(
                    box.conf[0]
                    .cpu()
                    .item()
                )

                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                )

                # ------------------------------------------------
                # Center
                # ------------------------------------------------

                cv2.circle(
                    annotated,
                    (
                        center_x,
                        center_y,
                    ),
                    6,
                    (0, 0, 255),
                    -1,
                )

                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                label = (
                    f"CARDBOARD BOX "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 10,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # Center coordinates
                # ------------------------------------------------

                center_text = (
                    f"Center: "
                    f"{center_x},"
                    f"{center_y}"
                )

                cv2.putText(
                    annotated,
                    center_text,
                    (
                        x1,
                        min(
                            HEIGHT - 10,
                            y2 + 25,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status = (
            f"CARDBOARD BOXES: "
            f"{box_count}"
        )

        cv2.putText(
            annotated,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Frame number
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            f"Frame: {frame_id}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Encode JPEG
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80,
            ],
        )

        if not success:

            continue

        frame_bytes = buffer.tobytes()

        # ----------------------------------------------------
        # MJPEG stream
        # ----------------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# WEB PAGE
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            BAS Live Cardboard Box Detection
        </title>

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

            img {
                width: 640px;
                max-width: 95vw;
                border: 3px solid white;
            }

            p {
                font-size: 18px;
            }

        </style>

    </head>

    <body>

        <h1>
            BAS Live Cardboard Box Detection V2
        </h1>

        <img src="/video">

        <p>
            Live YOLO11n cardboard-box detection
        </p>

    </body>

    </html>
    """


# ============================================================
# VIDEO
# ============================================================

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
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LIVE SERVER STARTED")
    print("=" * 70)

    print()
    print(
        "Open this address in Windows Chrome:"
    )

    print()
    print(
        "http://localhost:5000"
    )

    print()

    print(
        "Press CTRL+C in WSL to stop."
    )

    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
    )
