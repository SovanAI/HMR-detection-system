import cv2
import time
import threading
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"
CAMERA = "/dev/video0"

CONFIDENCE = 0.25
IMAGE_SIZE = 640

PERSON_CLASS = 0
CHAIR_CLASS = 56

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5006

# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

latest_frame = None
latest_jpeg = None

frame_lock = threading.Lock()

status = {
    "camera": False,
    "persons": 0,
    "chairs": 0,
    "fps": 0.0,
    "frame_count": 0,
    "model": MODEL_PATH,
}


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("Loading YOLO11...")
print("=" * 70)

model = YOLO(MODEL_PATH)

print("YOLO11 loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)

if not cap.isOpened():
    raise RuntimeError("Could not open /dev/video0")

status["camera"] = True

print("Camera opened successfully.")


# ============================================================
# DETECTION LOOP
# ============================================================

def detection_loop():

    global latest_frame
    global latest_jpeg

    frame_count = 0
    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("\nCamera frame failed")
            time.sleep(0.1)
            continue

        frame_count += 1

        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            classes=[PERSON_CLASS, CHAIR_CLASS],
            verbose=False,
        )

        result = results[0]

        persons = []
        chairs = []

        if result.boxes is not None:

            for box in result.boxes:

                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detection = {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                }

                if cls == PERSON_CLASS:
                    persons.append(detection)

                elif cls == CHAIR_CLASS:
                    chairs.append(detection)

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        output = frame.copy()

        # PERSONS
        for i, person in enumerate(persons):

            x1, y1, x2, y2 = person["bbox"]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"PERSON {i + 1} "
                f"{person['confidence']:.2f}"
            )

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # CHAIRS
        for i, chair in enumerate(chairs):

            x1, y1, x2, y2 = chair["bbox"]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )

            label = (
                f"CHAIR {i + 1} "
                f"{chair['confidence']:.2f}"
            )

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = time.time() - start_time

        fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )

        # ----------------------------------------------------
        # STATUS PANEL
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (0, 0),
            (640, 65),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            f"PERSONS : {len(persons)}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"CHAIRS  : {len(chairs)}",
            (200, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"FPS : {fps:.2f}",
            (390, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            "YOLO11 PERSON + CHAIR",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # JPEG
        # ----------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            output,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80,
            ],
        )

        if success:

            jpeg_bytes = encoded.tobytes()

            with frame_lock:

                latest_frame = output.copy()
                latest_jpeg = jpeg_bytes

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status["persons"] = len(persons)
        status["chairs"] = len(chairs)
        status["fps"] = round(fps, 2)
        status["frame_count"] = frame_count

        print(
            f"\rFrame: {frame_count:05d} | "
            f"Persons: {len(persons)} | "
            f"Chairs: {len(chairs)} | "
            f"FPS: {fps:.2f}",
            end="",
            flush=True,
        )


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_stream():

    while True:

        with frame_lock:

            frame = latest_jpeg

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

<title>BAS-HMR | Person Chair Detection</title>

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

.container {
    display: inline-block;
}

img {
    width: 960px;
    max-width: 95vw;
    border: 3px solid #444;
}

.panel {
    margin-top: 15px;
    font-size: 20px;
}

.status {
    margin: 8px;
}

</style>

</head>

<body>

<h1>BAS-HMR</h1>

<div class="container">

<img src="/video_feed">

<div class="panel">

<div class="status">
Camera: <span id="camera">...</span>
</div>

<div class="status">
Persons: <span id="persons">...</span>
</div>

<div class="status">
Chairs: <span id="chairs">...</span>
</div>

<div class="status">
FPS: <span id="fps">...</span>
</div>

</div>

</div>

<script>

async function updateStatus() {

    try {

        const response =
            await fetch("/api/status");

        const data =
            await response.json();

        document.getElementById("camera")
            .textContent = data.camera;

        document.getElementById("persons")
            .textContent = data.persons;

        document.getElementById("chairs")
            .textContent = data.chairs;

        document.getElementById("fps")
            .textContent = data.fps;

    }

    catch (error) {

        console.log(error);

    }

}

setInterval(updateStatus, 1000);

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

    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():

    return jsonify(status)


@app.route("/api/health")
def health():

    return jsonify(
        {
            "status": "running",
            "camera": status["camera"],
            "model": MODEL_PATH,
        }
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    detection_thread = threading.Thread(
        target=detection_loop,
        daemon=True,
    )

    detection_thread.start()

    print("\n")
    print("=" * 70)
    print("FLASK SERVER STARTING")
    print("=" * 70)
    print("Open in Windows browser:")
    print()
    print("http://localhost:5006")
    print()
    print("Video stream:")
    print("http://localhost:5006/video_feed")
    print()
    print("API:")
    print("http://localhost:5006/api/status")
    print("=" * 70)

    try:

        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    finally:

        cap.release()