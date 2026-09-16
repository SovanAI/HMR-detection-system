import cv2
import csv
import json
import os
import time
from datetime import datetime

from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO


# ============================================================
# BAS-HMR DATASET COLLECTOR V1.1
# YOLO11 + Manual Ground Truth + Dataset Capture
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "data")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")

CSV_FILE = os.path.join(
    DATA_DIR,
    "distance_dataset_v1.csv"
)

CAMERA = "/dev/video0"

MODEL_PATH = "../yolo11n.pt"

WIDTH = 640
HEIGHT = 480
FPS = 10

CONFIDENCE = 0.25

PORT = 5010


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)


# ============================================================
# CSV COLUMNS
# ============================================================

CSV_COLUMNS = [
    "sample_id",
    "timestamp",

    "distance_cm",

    "person_id",
    "chair_id",

    "pose",
    "position",
    "chair_type",
    "lighting",

    "person_confidence",
    "chair_confidence",

    "person_x1",
    "person_y1",
    "person_x2",
    "person_y2",

    "chair_x1",
    "chair_y1",
    "chair_x2",
    "chair_y2",

    "image_path",
]


# ============================================================
# CREATE CSV
# ============================================================

if not os.path.exists(CSV_FILE):

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# YOLO
# ============================================================

print("=" * 70)
print("BAS-HMR DATASET COLLECTOR V1.1")
print("=" * 70)

print("\n[1] Loading YOLO11n...")

model = YOLO(MODEL_PATH)

print("[OK] YOLO11n loaded.")
print("[OK] Device: CPU")


# ============================================================
# CAMERA
# ============================================================

print("\n[2] Opening camera...")

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )


cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    FPS
)

print("[OK] Camera opened.")


# ============================================================
# SESSION
# ============================================================

session_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

session_file = os.path.join(
    SESSION_DIR,
    f"session_{session_id}.json"
)


session_data = {
    "session_id": session_id,
    "created": datetime.now().isoformat(),
    "samples": []
}


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None

latest_persons = []

latest_chairs = []

sample_counter = 0

state = {

    "distance_cm": "",

    "pose": "standing",

    "position": "front",

    "chair_type": "unknown",

    "lighting": "normal",
}


# ============================================================
# OPTIONS
# ============================================================

POSES = [
    "standing",
    "sitting",
    "walking",
    "leaning",
    "front_facing",
    "back_facing",
    "side_profile",
]

POSITIONS = [
    "front",
    "left",
    "right",
    "diagonal",
]

CHAIR_TYPES = [
    "unknown",
    "office",
    "plastic",
    "wooden",
    "other",
]

LIGHTING_TYPES = [
    "normal",
    "low",
    "bright",
]


# ============================================================
# DETECTION
# ============================================================

def detect(frame):

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=640,
        classes=[0, 56],
        device="cpu",
        verbose=False,
    )

    result = results[0]

    persons = []

    chairs = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detection = {

                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),

                "confidence": confidence,
            }

            if class_id == 0:

                persons.append(
                    detection
                )

            elif class_id == 56:

                chairs.append(
                    detection
                )

    return persons, chairs


# ============================================================
# DRAW
# ============================================================

def draw_detections(
    frame,
    persons,
    chairs
):

    display = frame.copy()


    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    for i, person in enumerate(persons):

        x1 = int(person["x1"])
        y1 = int(person["y1"])
        x2 = int(person["x2"])
        y2 = int(person["y2"])

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"PERSON {i+1} "
            f"{person['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


    # --------------------------------------------------------
    # CHAIR
    # --------------------------------------------------------

    for i, chair in enumerate(chairs):

        x1 = int(chair["x1"])
        y1 = int(chair["y1"])
        x2 = int(chair["x2"])
        y2 = int(chair["y2"])

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            display,
            f"CHAIR {i+1} "
            f"{chair['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )


    # --------------------------------------------------------
    # PANEL
    # --------------------------------------------------------

    cv2.rectangle(
        display,
        (0, 0),
        (640, 135),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        display,
        f"Persons: {len(persons)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        display,
        f"Chairs: {len(chairs)}",
        (180, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    distance = state["distance_cm"]

    if not distance:

        distance = "NOT SET"


    cv2.putText(
        display,
        f"Distance: {distance} cm",
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        display,
        f"Pose: {state['pose']}",
        (10, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        display,
        f"Position: {state['position']}",
        (10, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        display,
        f"Samples: {sample_counter}",
        (10, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    return display


# ============================================================
# SAVE SESSION
# ============================================================

def save_session():

    with open(
        session_file,
        "w"
    ) as f:

        json.dump(
            session_data,
            f,
            indent=2
        )


# ============================================================
# CAPTURE SAMPLE
# ============================================================

def capture_sample():

    global sample_counter

    if latest_frame is None:

        return False, "Camera frame unavailable."


    if len(latest_persons) == 0:

        return False, "No person detected."


    if len(latest_chairs) == 0:

        return False, "No chair detected."


    if state["distance_cm"] == "":

        return False, "Distance has not been entered."


    try:

        distance = float(
            state["distance_cm"]
        )

    except ValueError:

        return False, "Invalid distance."


    if distance <= 0:

        return False, "Distance must be greater than zero."


    # --------------------------------------------------------
    # USE FIRST PERSON + FIRST CHAIR
    # --------------------------------------------------------

    person = latest_persons[0]

    chair = latest_chairs[0]


    sample_counter += 1


    sample_id = (
        f"sample_{sample_counter:06d}"
    )


    timestamp = datetime.now().isoformat()


    image_filename = (
        f"{sample_id}.jpg"
    )


    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    cv2.imwrite(
        image_path,
        latest_frame
    )


    # --------------------------------------------------------
    # CSV ROW
    # --------------------------------------------------------

    row = {

        "sample_id":
            sample_id,

        "timestamp":
            timestamp,

        "distance_cm":
            distance,

        "person_id":
            1,

        "chair_id":
            1,

        "pose":
            state["pose"],

        "position":
            state["position"],

        "chair_type":
            state["chair_type"],

        "lighting":
            state["lighting"],

        "person_confidence":
            person["confidence"],

        "chair_confidence":
            chair["confidence"],

        "person_x1":
            person["x1"],

        "person_y1":
            person["y1"],

        "person_x2":
            person["x2"],

        "person_y2":
            person["y2"],

        "chair_x1":
            chair["x1"],

        "chair_y1":
            chair["y1"],

        "chair_x2":
            chair["x2"],

        "chair_y2":
            chair["y2"],

        "image_path":
            f"images/{image_filename}",
    }


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS
        )

        writer.writerow(row)


    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    session_data["samples"].append(
        row
    )

    save_session()


    print()
    print("=" * 70)
    print("[SAVED] DATASET SAMPLE")
    print("=" * 70)

    print(
        f"Sample    : {sample_id}"
    )

    print(
        f"Distance  : {distance} cm"
    )

    print(
        f"Pose      : {state['pose']}"
    )

    print(
        f"Position  : {state['position']}"
    )

    print(
        f"Chair     : {state['chair_type']}"
    )

    print(
        f"Lighting  : {state['lighting']}"
    )

    print("=" * 70)


    return True, sample_id


# ============================================================
# CAMERA GENERATOR
# ============================================================

def generate_frames():

    global latest_frame
    global latest_persons
    global latest_chairs


    while True:

        success, frame = cap.read()


        if not success:

            time.sleep(0.1)

            continue


        persons, chairs = detect(
            frame
        )


        latest_frame = frame.copy()

        latest_persons = persons

        latest_chairs = chairs


        display = draw_detections(
            frame,
            persons,
            chairs
        )


        success, buffer = cv2.imencode(
            ".jpg",
            display
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


# ============================================================
# WEB UI
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>BAS-HMR Dataset Collector</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

body {
    margin: 0;
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 1100px;
    margin: auto;
    padding: 20px;
}

h1 {
    text-align: center;
}

.layout {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    justify-content: center;
}

.camera {
    flex: 1;
    min-width: 600px;
}

.camera img {
    width: 100%;
    border: 2px solid #444;
}

.controls {
    width: 300px;
    background: #1d1d1d;
    padding: 20px;
    border-radius: 10px;
}

label {
    display: block;
    margin-top: 12px;
    margin-bottom: 5px;
}

input, select, button {
    width: 100%;
    box-sizing: border-box;
    padding: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
    border: none;
}

button {
    margin-top: 15px;
    cursor: pointer;
    font-size: 16px;
}

.capture {
    font-weight: bold;
}

.status {
    margin-top: 15px;
    padding: 10px;
    background: #222;
    border-radius: 5px;
}

.samples {
    margin-top: 10px;
    font-size: 18px;
}

</style>

</head>


<body>

<div class="container">

<h1>BAS-HMR Dataset Collector V1.1</h1>


<div class="layout">


<div class="camera">

<img src="/video">

</div>


<div class="controls">

<h2>Sample Labels</h2>


<label>
Distance (cm)
</label>

<input
id="distance"
type="number"
min="1"
step="1"
placeholder="Example: 100"
>


<label>
Pose
</label>

<select id="pose">

<option>standing</option>
<option>sitting</option>
<option>walking</option>
<option>leaning</option>
<option>front_facing</option>
<option>back_facing</option>
<option>side_profile</option>

</select>


<label>
Person–Chair Position
</label>

<select id="position">

<option>front</option>
<option>left</option>
<option>right</option>
<option>diagonal</option>

</select>


<label>
Chair Type
</label>

<select id="chair_type">

<option>unknown</option>
<option>office</option>
<option>plastic</option>
<option>wooden</option>
<option>other</option>

</select>


<label>
Lighting
</label>

<select id="lighting">

<option>normal</option>
<option>low</option>
<option>bright</option>

</select>


<button
class="capture"
onclick="captureSample()"
>
CAPTURE SAMPLE
</button>


<div class="status" id="status">
Ready.
</div>


<div class="samples">
Samples: <span id="count">0</span>
</div>


</div>

</div>

</div>


<script>

async function captureSample() {

    const distance =
        document.getElementById(
            "distance"
        ).value;


    const pose =
        document.getElementById(
            "pose"
        ).value;


    const position =
        document.getElementById(
            "position"
        ).value;


    const chair_type =
        document.getElementById(
            "chair_type"
        ).value;


    const lighting =
        document.getElementById(
            "lighting"
        ).value;


    if (!distance) {

        document.getElementById(
            "status"
        ).innerText =
            "Enter distance first.";

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

                    distance_cm:
                        distance,

                    pose:
                        pose,

                    position:
                        position,

                    chair_type:
                        chair_type,

                    lighting:
                        lighting
                })
            }
        );


    const data =
        await response.json();


    document.getElementById(
        "status"
    ).innerText =
        data.message;


    if (data.success) {

        document.getElementById(
            "count"
        ).innerText =
            data.sample_count;
    }

}

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


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


@app.route(
    "/capture",
    methods=["POST"]
)
def capture():

    data = request.get_json()


    state["distance_cm"] = str(
        data.get(
            "distance_cm",
            ""
        )
    )

    state["pose"] = data.get(
        "pose",
        "standing"
    )

    state["position"] = data.get(
        "position",
        "front"
    )

    state["chair_type"] = data.get(
        "chair_type",
        "unknown"
    )

    state["lighting"] = data.get(
        "lighting",
        "normal"
    )


    success, message = (
        capture_sample()
    )


    return jsonify({

        "success": success,

        "message":
            message
            if success
            else f"NOT SAVED: {message}",

        "sample_count":
            sample_counter,
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)

    print(
        f"Dataset Collector running on port {PORT}"
    )

    print(
        f"Open: http://localhost:{PORT}"
    )

    print("=" * 70)


    try:

        app.run(
            host="0.0.0.0",
            port=PORT,
            threaded=True,
            debug=False
        )

    finally:

        cap.release()

        save_session()

        print(
            "[OK] Camera released."
        )

        print(
            f"[OK] Samples: {sample_counter}"
        )