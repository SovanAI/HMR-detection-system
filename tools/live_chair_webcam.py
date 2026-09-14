from pathlib import Path
import cv2
from flask import Flask, Response
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "yolo11n.pt"
CAMERA = "/dev/video0"

IMG_SIZE = 416
CONFIDENCE = 0.35

PERSON_CLASS = 0
CHAIR_CLASS = 56

app = Flask(__name__)

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )

# Webcam configuration
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)


def generate_frames():

    frame_id = 0

    while True:

        success, frame = cap.read()

        if not success:
            continue

        frame_id += 1

        # YOLO inference
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        persons = []
        chairs = []

        if result.boxes is not None:

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

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
                    persons.append(detection)

                elif cls == CHAIR_CLASS:
                    chairs.append(detection)

        # --------------------------------------------------
        # DRAW PERSONS
        # --------------------------------------------------

        for i, person in enumerate(persons):

            x1, y1, x2, y2 = person["bbox"]
            conf = person["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            label = f"PERSON {i} | {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 30)),
                (x1 + 190, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

            # Person center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                frame,
                (cx, cy),
                6,
                (0, 255, 0),
                -1,
            )

        # --------------------------------------------------
        # DRAW CHAIRS
        # --------------------------------------------------

        for i, chair in enumerate(chairs):

            x1, y1, x2, y2 = chair["bbox"]
            conf = chair["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3,
            )

            label = f"CHAIR {i} | {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 30)),
                (x1 + 180, y1),
                (255, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Chair center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                frame,
                (cx, cy),
                6,
                (255, 0, 0),
                -1,
            )

        # --------------------------------------------------
        # INFORMATION PANEL
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (270, 95),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            "BAS CHAIR DETECTION",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Persons : {len(persons)}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Chairs  : {len(chairs)}",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )

        # Frame number
        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (500, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # JPEG encode
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>

    <head>
        <title>BAS Chair Detection</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
            }

            h1 {
                margin-top: 20px;
            }

            img {
                width: 640px;
                max-width: 95%;
                border: 3px solid white;
            }

            .legend {
                margin: 15px;
                font-size: 18px;
            }

            .person {
                color: #00ff00;
            }

            .chair {
                color: #0088ff;
            }

        </style>

    </head>

    <body>

        <h1>BAS — Person + Chair Detection</h1>

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

    </body>

    </html>
    """


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":

    print("=" * 60)
    print("BAS LIVE CHAIR DETECTION")
    print("=" * 60)

    print()
    print("Open this in Windows Chrome:")
    print()
    print("http://localhost:5000")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
    )
