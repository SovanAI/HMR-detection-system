import cv2
import time
import threading
import numpy as np

from flask import Flask, Response, jsonify, render_template_string

from ultralytics import YOLO
from human.hmr.hmr_processor import HMRProcessor


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "/dev/video0"

YOLO_MODEL = "yolo11n.pt"

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

PERSON_CLASS = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5008

# HMR processing interval.
#
# HMR is very slow on CPU, therefore we do NOT run HMR
# on every camera frame.
#
HMR_INTERVAL = 30


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.Lock()

latest_jpeg = None

latest_raw_frame = None

latest_person_boxes = []

latest_hmr_result = None

latest_hmr_frame_id = -1

frame_id = 0

running = True


status = {
    "camera": False,
    "yolo": False,
    "hmr": False,

    "frame_id": 0,

    "persons": 0,

    "hmr_persons": 0,

    "hmr_frame_id": -1,

    "fps": 0.0,

    "hmr_status": "WAITING",
}


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("Loading YOLO11...")
print("=" * 70)

yolo_model = YOLO(YOLO_MODEL)

status["yolo"] = True

print("YOLO11 loaded successfully.")


# ============================================================
# LOAD HMR
# ============================================================

print("=" * 70)
print("Loading HMR2...")
print("=" * 70)

hmr_processor = HMRProcessor()

status["hmr"] = True

print("HMR2 loaded successfully.")


# ============================================================
# CAMERA
# ============================================================

print("=" * 70)
print("Opening camera...")
print("=" * 70)

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open /dev/video0"
    )


status["camera"] = True

print("Camera opened successfully.")


# ============================================================
# HMR WORKER
# ============================================================

def hmr_worker():

    global latest_hmr_result
    global latest_hmr_frame_id

    last_processed_frame = -1

    while running:

        # ----------------------------------------------------
        # Get newest frame and boxes
        # ----------------------------------------------------

        with state_lock:

            frame = (
                latest_raw_frame.copy()
                if latest_raw_frame is not None
                else None
            )

            boxes = list(
                latest_person_boxes
            )

            current_frame_id = frame_id

        if frame is None:

            time.sleep(0.05)

            continue


        # ----------------------------------------------------
        # Avoid repeatedly processing same frame
        # ----------------------------------------------------

        if current_frame_id == last_processed_frame:

            time.sleep(0.05)

            continue


        # ----------------------------------------------------
        # Only run HMR when enough frames have passed
        # ----------------------------------------------------

        if (
            current_frame_id % HMR_INTERVAL != 0
        ):

            time.sleep(0.05)

            continue


        last_processed_frame = current_frame_id


        # ----------------------------------------------------
        # No person
        # ----------------------------------------------------

        if len(boxes) == 0:

            with state_lock:

                latest_hmr_result = {
                    "frame_id": current_frame_id,
                    "persons": [],
                    "model": "HMR2",
                }

                latest_hmr_frame_id = (
                    current_frame_id
                )

                status[
                    "hmr_status"
                ] = "NO PERSON"

            continue


        # ----------------------------------------------------
        # Convert boxes to NumPy
        # ----------------------------------------------------

        boxes_array = np.asarray(
            boxes,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # HMR INFERENCE
        # ----------------------------------------------------

        try:

            print(
                f"\n[HMR WORKER] "
                f"Processing frame "
                f"{current_frame_id} | "
                f"persons={len(boxes_array)}"
            )


            result = (
                hmr_processor.process_frame(
                    frame,
                    current_frame_id,
                    time.time(),
                    boxes_array
                )
            )


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            with state_lock:

                latest_hmr_result = result

                latest_hmr_frame_id = (
                    current_frame_id
                )

                status[
                    "hmr_frame_id"
                ] = current_frame_id

                status[
                    "hmr_persons"
                ] = len(
                    result.get(
                        "persons",
                        []
                    )
                )

                status[
                    "hmr_status"
                ] = "ACTIVE"


            print(
                f"[HMR WORKER] "
                f"Completed frame "
                f"{current_frame_id}"
            )


        except Exception as e:

            print(
                f"\n[HMR ERROR] {e}"
            )

            with state_lock:

                status[
                    "hmr_status"
                ] = f"ERROR: {type(e).__name__}"


# ============================================================
# PROJECT 3D JOINTS TO IMAGE
# ============================================================

def project_hmr_joints(
    frame,
    hmr_person,
):

    output = frame


    # --------------------------------------------------------
    # Get joints
    # --------------------------------------------------------

    pose = hmr_person.get(
        "pose",
        {}
    )

    joints = pose.get(
        "joints_3d",
        []
    )


    if len(joints) == 0:

        return output


    # --------------------------------------------------------
    # Get camera translation
    # --------------------------------------------------------

    camera_translation = (
        hmr_person.get(
            "camera_translation",
            {}
        )
    )


    # --------------------------------------------------------
    # HMR 3D coordinates are model coordinates.
    #
    # For this visualization we use a simple projection
    # relative to the HMR camera translation.
    #
    # This is visualization only.
    #
    # It is NOT yet our metric camera calibration.
    # --------------------------------------------------------

    tx = float(
        camera_translation.get(
            "x",
            0.0
        )
    )

    ty = float(
        camera_translation.get(
            "y",
            0.0
        )
    )

    tz = float(
        camera_translation.get(
            "z",
            1.0
        )
    )


    # --------------------------------------------------------
    # Get bbox
    # --------------------------------------------------------

    bbox = hmr_person.get(
        "bbox",
        {}
    )

    bx1 = float(
        bbox.get(
            "x1",
            0
        )
    )

    by1 = float(
        bbox.get(
            "y1",
            0
        )
    )

    bx2 = float(
        bbox.get(
            "x2",
            frame.shape[1]
        )
    )

    by2 = float(
        bbox.get(
            "y2",
            frame.shape[0]
        )
    )


    bw = max(
        bx2 - bx1,
        1
    )

    bh = max(
        by2 - by1,
        1
    )


    # --------------------------------------------------------
    # Approximate visualization scale
    # --------------------------------------------------------

    scale_x = bw * 0.45

    scale_y = bh * 0.45


    # --------------------------------------------------------
    # Project joints
    # --------------------------------------------------------

    projected = []


    for joint in joints:

        x = float(
            joint["x"]
        )

        y = float(
            joint["y"]
        )

        z = float(
            joint["z"]
        )


        depth = (
            z - tz
        )


        if abs(depth) < 0.01:

            depth = 0.01


        px = (
            (bx1 + bx2) / 2
            + x * scale_x
        )

        py = (
            (by1 + by2) / 2
            - y * scale_y
        )


        px = int(
            np.clip(
                px,
                0,
                frame.shape[1] - 1
            )
        )

        py = int(
            np.clip(
                py,
                0,
                frame.shape[0] - 1
            )
        )


        projected.append(
            (
                px,
                py
            )
        )


    # --------------------------------------------------------
    # Skeleton connections
    #
    # HMR output contains 44 joints.
    #
    # The exact mapping can vary by model configuration.
    # We therefore use a conservative connection structure
    # for visualization.
    # --------------------------------------------------------

    skeleton = [

        (0, 1),
        (1, 2),
        (2, 3),

        (0, 4),
        (4, 5),
        (5, 6),

        (0, 7),
        (7, 8),
        (8, 9),

        (9, 10),

        (8, 11),
        (11, 12),
        (12, 13),

        (8, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),

        (0, 20),
        (20, 21),
        (21, 22),

    ]


    # --------------------------------------------------------
    # Draw skeleton
    # --------------------------------------------------------

    for a, b in skeleton:

        if (
            a >= len(projected)
            or b >= len(projected)
        ):

            continue


        cv2.line(
            output,
            projected[a],
            projected[b],
            (0, 200, 255),
            2,
        )


    # --------------------------------------------------------
    # Draw joints
    # --------------------------------------------------------

    for point in projected:

        cv2.circle(
            output,
            point,
            4,
            (0, 0, 255),
            -1,
        )


    return output


# ============================================================
# DRAW HMR INFORMATION
# ============================================================

def draw_hmr_information(
    frame,
    hmr_result,
):

    output = frame


    if hmr_result is None:

        return output


    persons = hmr_result.get(
        "persons",
        []
    )


    for person in persons:

        person_id = person.get(
            "person_id",
            0
        )


        camera = person.get(
            "camera_translation",
            {}
        )


        cx = camera.get(
            "x",
            0.0
        )

        cy = camera.get(
            "y",
            0.0
        )

        cz = camera.get(
            "z",
            0.0
        )


        pose = person.get(
            "pose",
            {}
        )


        joint_count = pose.get(
            "joint_count",
            0
        )


        bbox = person.get(
            "bbox",
            {}
        )


        x1 = int(
            bbox.get(
                "x1",
                0
            )
        )

        y1 = int(
            bbox.get(
                "y1",
                0
            )
        )


        # ----------------------------------------------------
        # Information panel
        # ----------------------------------------------------

        panel_x = max(
            5,
            x1
        )

        panel_y = max(
            100,
            y1
        )


        lines = [

            f"HMR PERSON {person_id + 1}",

            "HMR: ACTIVE",

            f"Joints: {joint_count}",

            f"Cam X: {cx:.3f}",

            f"Cam Y: {cy:.3f}",

            f"Cam Z: {cz:.3f}",

        ]


        for index, text in enumerate(
            lines
        ):

            y = (
                panel_y
                + index * 20
            )


            cv2.putText(
                output,
                text,
                (
                    panel_x,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


    return output


# ============================================================
# MAIN CAMERA + YOLO LOOP
# ============================================================

def camera_loop():

    global latest_raw_frame
    global latest_person_boxes
    global latest_jpeg
    global frame_id


    start_time = time.time()

    local_frame_count = 0


    while running:

        ret, frame = cap.read()


        if not ret or frame is None:

            print(
                "\n[CAMERA] "
                "Frame capture failed"
            )

            time.sleep(0.1)

            continue


        frame_id += 1

        local_frame_count += 1


        # ----------------------------------------------------
        # YOLO PERSON DETECTION
        # ----------------------------------------------------

        try:

            results = yolo_model.predict(

                source=frame,

                conf=YOLO_CONF,

                imgsz=YOLO_IMGSZ,

                device="cpu",

                classes=[
                    PERSON_CLASS
                ],

                verbose=False,
            )


            result = results[0]


        except Exception as e:

            print(
                f"\n[YOLO ERROR] {e}"
            )

            continue


        persons = []


        if result.boxes is not None:

            for box in result.boxes:

                cls = int(
                    box.cls[0]
                )

                if cls != PERSON_CLASS:

                    continue


                confidence = float(
                    box.conf[0]
                )


                x1, y1, x2, y2 = map(

                    float,

                    box.xyxy[0].tolist()

                )


                persons.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )


        # ----------------------------------------------------
        # Store newest data for HMR worker
        # ----------------------------------------------------

        with state_lock:

            latest_raw_frame = (
                frame.copy()
            )

            latest_person_boxes = (
                persons.copy()
            )


        # ----------------------------------------------------
        # Draw YOLO
        # ----------------------------------------------------

        output = frame.copy()


        for i, box in enumerate(
            persons
        ):

            x1, y1, x2, y2 = box


            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            cv2.rectangle(

                output,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2,

            )


            cv2.putText(

                output,

                f"PERSON {i + 1}",

                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 0),

                2,

            )


        # ----------------------------------------------------
        # Get newest HMR result
        # ----------------------------------------------------

        with state_lock:

            hmr_result = (
                latest_hmr_result
            )

            hmr_frame = (
                latest_hmr_frame_id
            )


        # ----------------------------------------------------
        # Draw HMR
        # ----------------------------------------------------

        if hmr_result is not None:

            for person in hmr_result.get(
                "persons",
                []
            ):

                output = (
                    project_hmr_joints(
                        output,
                        person
                    )
                )


            output = (
                draw_hmr_information(
                    output,
                    hmr_result
                )
            )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )


        fps = (

            local_frame_count
            / elapsed

            if elapsed > 0

            else 0.0

        )


        # ----------------------------------------------------
        # STATUS PANEL
        # ----------------------------------------------------

        cv2.rectangle(

            output,

            (0, 0),

            (640, 75),

            (0, 0, 0),

            -1,

        )


        cv2.putText(

            output,

            f"PERSONS: {len(persons)}",

            (10, 22),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 0),

            2,

        )


        cv2.putText(

            output,

            f"FPS: {fps:.2f}",

            (200, 22),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2,

        )


        hmr_status_text = (
            status["hmr_status"]
        )


        cv2.putText(

            output,

            f"HMR: {hmr_status_text}",

            (350, 22),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 200, 255),

            2,

        )


        cv2.putText(

            output,

            f"HMR FRAME: {hmr_frame}",

            (10, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            1,

        )


        cv2.putText(

            output,

            "YOLO11 + HMR2",

            (390, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            1,

        )


        # ----------------------------------------------------
        # Encode JPEG
        # ----------------------------------------------------

        success, encoded = (
            cv2.imencode(
                ".jpg",
                output,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80
                ]
            )
        )


        if success:

            with state_lock:

                latest_jpeg = (
                    encoded.tobytes()
                )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        with state_lock:

            status["frame_id"] = (
                frame_id
            )

            status["persons"] = (
                len(persons)
            )

            status["fps"] = round(
                fps,
                2
            )


        print(

            f"\rFrame: {frame_id:05d} | "

            f"Persons: {len(persons)} | "

            f"HMR frame: {hmr_frame} | "

            f"FPS: {fps:.2f}",

            end="",

            flush=True,

        )


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_stream():

    while running:

        with state_lock:

            frame = latest_jpeg


        if frame is None:

            time.sleep(
                0.05
            )

            continue


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame

            + b"\r\n"

        )


# ============================================================
# HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>BAS-HMR | HMR2 Human 3D</title>

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

    margin-bottom: 5px;

}

.subtitle {

    color: #aaa;

    margin-bottom: 20px;

}

.video {

    width: 960px;

    max-width: 95vw;

    border: 3px solid #444;

}

.panel {

    margin: 20px auto;

    padding: 15px;

    max-width: 900px;

    background: #1c1c1c;

    border-radius: 8px;

}

.stat {

    display: inline-block;

    margin: 10px 25px;

    font-size: 18px;

}

.value {

    font-weight: bold;

}

</style>

</head>

<body>

<h1>BAS-HMR</h1>

<div class="subtitle">

YOLO11 + HMR2 Human 3D Visualization

</div>

<img

    class="video"

    src="/video_feed"

>

<div class="panel">

<div class="stat">

Camera:

<span

    class="value"

    id="camera">

...

</span>

</div>

<div class="stat">

Persons:

<span

    class="value"

    id="persons">

...

</span>

</div>

<div class="stat">

HMR Persons:

<span

    class="value"

    id="hmr_persons">

...

</span>

</div>

<div class="stat">

HMR:

<span

    class="value"

    id="hmr">

...

</span>

</div>

<div class="stat">

FPS:

<span

    class="value"

    id="fps">

...

</span>

</div>

</div>


<script>

async function updateStatus() {

    try {

        const response =
            await fetch("/api/status");

        const data =
            await response.json();

        document.getElementById(
            "camera"
        ).textContent =
            data.camera;

        document.getElementById(
            "persons"
        ).textContent =
            data.persons;

        document.getElementById(
            "hmr_persons"
        ).textContent =
            data.hmr_persons;

        document.getElementById(
            "hmr"
        ).textContent =
            data.hmr_status;

        document.getElementById(
            "fps"
        ).textContent =
            data.fps;

    }

    catch (error) {

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
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


@app.route("/video_feed")
def video_feed():

    return Response(

        generate_stream(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

    )


@app.route("/api/status")
def api_status():

    with state_lock:

        return jsonify(
            status.copy()
        )


@app.route("/api/health")
def health():

    return jsonify({

        "status": "running",

        "camera": status["camera"],

        "yolo": status["yolo"],

        "hmr": status["hmr"],

        "model": "YOLO11 + HMR2",

        "device": "cpu",

    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("BAS-HMR HMR2 LIVE VISUALIZATION")
    print("=" * 70)
    print()
    print("Browser:")
    print()
    print("http://localhost:5008")
    print()
    print("API:")
    print()
    print("http://localhost:5008/api/status")
    print()
    print("=" * 70)


    # --------------------------------------------------------
    # Start HMR worker
    # --------------------------------------------------------

    hmr_thread = threading.Thread(

        target=hmr_worker,

        daemon=True,

    )

    hmr_thread.start()


    # --------------------------------------------------------
    # Start camera thread
    # --------------------------------------------------------

    camera_thread = threading.Thread(

        target=camera_loop,

        daemon=True,

    )

    camera_thread.start()


    # --------------------------------------------------------
    # Flask
    # --------------------------------------------------------

    try:

        app.run(

            host=FLASK_HOST,

            port=FLASK_PORT,

            threaded=True,

            debug=False,

            use_reloader=False,

        )

    finally:

        running = False

        cap.release()

        print(
            "\nCamera released."
        )
