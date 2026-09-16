import cv2
import json
import time
import threading
import numpy as np

from flask import Flask, Response, jsonify

from ultralytics import YOLO
from human.hmr.hmr_processor import HMRProcessor


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# HMR is CPU-heavy.
# Run HMR once every N frames and display cached result.
HMR_INTERVAL = 30

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5011


# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

camera = None
running = True

latest_frame = None
latest_hmr_result = None

frame_lock = threading.Lock()
hmr_lock = threading.Lock()

frame_counter = 0

last_hmr_frame = -1
last_hmr_time = 0.0


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("=" * 70)
print("BAS-HMR 3D HUMAN COORDINATE DIAGNOSTIC")
print("=" * 70)

print("[YOLO] Loading YOLO11n...")

yolo = YOLO("yolo11n.pt")

print("[YOLO] Loaded successfully.")
print("[YOLO] Device: CPU")


# ============================================================
# LOAD HMR2
# ============================================================

print()
print("[HMR] Loading HMR2...")

hmr_processor = HMRProcessor()

print("[HMR] HMR2 loaded successfully.")
print("[HMR] Device: CPU")


# ============================================================
# CAMERA
# ============================================================

print()
print("[CAMERA] Opening camera...")

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

# Prefer MJPG for USB webcam
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not camera.isOpened():
    raise RuntimeError("Could not open /dev/video0")

print("[CAMERA] Camera opened successfully.")
print(f"[CAMERA] Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print(f"[CAMERA] FPS target: {CAMERA_FPS}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_numpy(value):
    """
    Convert different possible HMR output formats to numpy.
    """
    if value is None:
        return None

    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()

        return np.asarray(value, dtype=np.float32)

    except Exception as e:
        print(f"[WARN] Could not convert value to numpy: {e}")
        return None


def extract_person_coordinates(person):
    """
    Extract and summarize HMR 3D information.

    Expected HMR processor output:

        person["joints_3d"]
        person["camera_translation"]

    The joints are treated as HMR's 3D joint coordinates.
    The translated coordinates are marked as estimated
    camera-frame coordinates.
    """

    joints = safe_numpy(person.get("joints_3d"))
    camera_translation = safe_numpy(
        person.get("camera_translation")
    )

    result = {
        "person_id": person.get("person_id"),
        "bbox": person.get("bbox"),
        "joint_count": 0,
        "joints_shape": None,
        "joints_3d": None,
        "camera_translation": None,
        "camera_frame_joints_3d": None,
        "centroid": None,
        "camera_frame_centroid": None,
        "min_xyz": None,
        "max_xyz": None,
    }

    if joints is None:
        return result

    # --------------------------------------------------------
    # Normalize joint array shape
    # --------------------------------------------------------

    # Typical shape:
    # (N, 3)
    #
    # Sometimes it may arrive as:
    # (1, N, 3)

    if joints.ndim == 3 and joints.shape[0] == 1:
        joints = joints[0]

    if joints.ndim != 2 or joints.shape[1] != 3:
        print(
            "[WARN] Unexpected joints_3d shape:",
            joints.shape
        )
        return result

    result["joint_count"] = int(joints.shape[0])
    result["joints_shape"] = list(joints.shape)

    result["joints_3d"] = joints.tolist()

    # --------------------------------------------------------
    # Joint statistics
    # --------------------------------------------------------

    centroid = np.mean(joints, axis=0)
    min_xyz = np.min(joints, axis=0)
    max_xyz = np.max(joints, axis=0)

    result["centroid"] = centroid.tolist()
    result["min_xyz"] = min_xyz.tolist()
    result["max_xyz"] = max_xyz.tolist()

    # --------------------------------------------------------
    # Camera translation
    # --------------------------------------------------------

    if camera_translation is not None:

        camera_translation = camera_translation.reshape(-1)

        if camera_translation.size >= 3:

            camera_translation = camera_translation[:3]

            result["camera_translation"] = (
                camera_translation.tolist()
            )

            # HMR joints are generally root-relative.
            #
            # For diagnosis we form:
            #
            # camera-frame estimate =
            #       joint_3d + camera_translation
            #
            # This is explicitly an estimated camera-frame
            # representation and must later be validated.
            # ------------------------------------------------

            camera_frame_joints = (
                joints + camera_translation.reshape(1, 3)
            )

            camera_frame_centroid = np.mean(
                camera_frame_joints,
                axis=0
            )

            result["camera_frame_joints_3d"] = (
                camera_frame_joints.tolist()
            )

            result["camera_frame_centroid"] = (
                camera_frame_centroid.tolist()
            )

    return result


def print_hmr_debug(hmr_output, frame_id):
    """
    Print actual HMR numerical coordinates to terminal.
    """

    print()
    print("=" * 70)
    print(f"[HMR 3D DEBUG] FRAME {frame_id}")
    print("=" * 70)

    if not hmr_output:
        print("[HMR 3D DEBUG] No HMR result.")
        return

    persons = hmr_output.get("persons", [])

    print(f"[HMR 3D DEBUG] Persons detected: {len(persons)}")

    for i, person in enumerate(persons):

        info = extract_person_coordinates(person)

        print()
        print(f"--------------- PERSON {i + 1} ---------------")

        print("Person ID:")
        print(" ", info["person_id"])

        print("Bounding Box:")
        print(" ", info["bbox"])

        print("Joint count:")
        print(" ", info["joint_count"])

        print("Joint array shape:")
        print(" ", info["joints_shape"])

        print()
        print("HMR camera translation [X Y Z]:")
        print(" ", info["camera_translation"])

        print()
        print("HMR joints centroid [X Y Z]:")
        print(" ", info["centroid"])

        print()
        print("HMR joints minimum [X Y Z]:")
        print(" ", info["min_xyz"])

        print()
        print("HMR joints maximum [X Y Z]:")
        print(" ", info["max_xyz"])

        print()
        print("Estimated camera-frame centroid [X Y Z]:")
        print(" ", info["camera_frame_centroid"])

        # ----------------------------------------------------
        # Print first 5 joints
        # ----------------------------------------------------

        joints = info["joints_3d"]

        if joints:

            print()
            print("First 5 HMR 3D joints:")

            for joint_index, xyz in enumerate(joints[:5]):

                print(
                    f"  Joint {joint_index:02d}: "
                    f"X={xyz[0]: .5f}, "
                    f"Y={xyz[1]: .5f}, "
                    f"Z={xyz[2]: .5f}"
                )

        # ----------------------------------------------------
        # Print first 5 translated joints
        # ----------------------------------------------------

        translated = info["camera_frame_joints_3d"]

        if translated:

            print()
            print("First 5 estimated camera-frame joints:")

            for joint_index, xyz in enumerate(
                translated[:5]
            ):

                print(
                    f"  Joint {joint_index:02d}: "
                    f"X={xyz[0]: .5f}, "
                    f"Y={xyz[1]: .5f}, "
                    f"Z={xyz[2]: .5f}"
                )

    print()
    print("=" * 70)


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_persons(frame):
    """
    Detect only people using YOLO.
    """

    results = yolo.predict(
        source=frame,
        classes=[0],       # COCO person
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        device="cpu",
        verbose=False
    )

    boxes = []

    if not results:
        return boxes

    result = results[0]

    if result.boxes is None:
        return boxes

    for box in result.boxes:

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = xyxy.astype(int)

        confidence = float(
            box.conf[0].cpu().numpy()
        )

        boxes.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": confidence,
            }
        )

    return boxes


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_hmr_information(
    frame,
    person_boxes,
    hmr_result
):

    output = frame.copy()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 75),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        output,
        "BAS-HMR | HUMAN 3D COORDINATE DIAGNOSTIC",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        "HMR2: CPU | YOLO11n: PERSON",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # YOLO boxes
    # --------------------------------------------------------

    for idx, box in enumerate(person_boxes):

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        conf = box["confidence"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Person {idx + 1} {conf:.2f}",
            (x1, max(95, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # HMR information
    # --------------------------------------------------------

    if hmr_result is None:
        return output

    persons = hmr_result.get("persons", [])

    y_offset = 95

    for i, person in enumerate(persons):

        info = extract_person_coordinates(person)

        # ----------------------------------------------------
        # Match HMR bbox
        # ----------------------------------------------------

        bbox = info["bbox"]

        if bbox is not None:

            try:
                x1 = int(bbox["x1"])
                y1 = int(bbox["y1"])
            except Exception:
                x1 = 10
                y1 = 95

        else:
            x1 = 10
            y1 = 95

        # ----------------------------------------------------
        # Main status
        # ----------------------------------------------------

        lines = [
            f"HMR Person {i + 1}",
            f"Joints: {info['joint_count']}",
        ]

        if info["camera_translation"] is not None:

            t = info["camera_translation"]

            lines.append(
                f"Cam T: "
                f"{t[0]:.2f}, "
                f"{t[1]:.2f}, "
                f"{t[2]:.2f}"
            )

        if info["camera_frame_centroid"] is not None:

            c = info["camera_frame_centroid"]

            lines.append(
                f"3D C: "
                f"{c[0]:.2f}, "
                f"{c[1]:.2f}, "
                f"{c[2]:.2f}"
            )

        # ----------------------------------------------------
        # Draw text near person
        # ----------------------------------------------------

        panel_y = max(85, y1)

        for line in lines:

            cv2.putText(
                output,
                line,
                (x1, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            panel_y += 20

        # ----------------------------------------------------
        # 3D status in lower-left
        # ----------------------------------------------------

        if i == 0:

            bottom_y = 105

            text = (
                f"HMR3D: "
                f"{info['joint_count']} joints"
            )

            cv2.putText(
                output,
                text,
                (10, bottom_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return output


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():

    global latest_frame

    while running:

        with frame_lock:

            if latest_frame is None:
                time.sleep(0.01)
                continue

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


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BAS-HMR 3D Diagnostic</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial;
                margin: 0;
                padding: 20px;
                text-align: center;
            }

            h1 {
                margin-bottom: 10px;
            }

            img {
                width: 640px;
                max-width: 95vw;
                border: 2px solid white;
            }

            .panel {
                margin-top: 20px;
                padding: 15px;
                background: #222;
                text-align: left;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
            }

            pre {
                white-space: pre-wrap;
            }

        </style>
    </head>

    <body>

        <h1>BAS-HMR Human 3D Coordinate Diagnostic</h1>

        <img src="/video">

        <div class="panel">

            <h2>HMR Output</h2>

            <pre id="output">
Waiting for HMR...
            </pre>

        </div>

        <script>

        async function updateData() {

            try {

                const response =
                    await fetch("/api/hmr3d");

                const data =
                    await response.json();

                document.getElementById("output")
                    .textContent =
                    JSON.stringify(data, null, 2);

            }
            catch(error) {

                console.log(error);

            }

        }

        setInterval(updateData, 1000);

        updateData();

        </script>

    </body>
    </html>
    """


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/hmr3d")
def api_hmr3d():

    with hmr_lock:

        if latest_hmr_result is None:

            return jsonify(
                {
                    "status": "waiting",
                    "message": "HMR has not produced a result yet."
                }
            )

        output = latest_hmr_result

        diagnostic_persons = []

        for person in output.get("persons", []):

            diagnostic_persons.append(
                extract_person_coordinates(person)
            )

        return jsonify(
            {
                "status": "ok",
                "frame_id": output.get("frame_id"),
                "timestamp": output.get("timestamp"),
                "model": output.get("model"),
                "device": output.get("device"),
                "persons": diagnostic_persons,
            }
        )


# ============================================================
# PROCESSING LOOP
# ============================================================

def processing_loop():

    global latest_frame
    global latest_hmr_result
    global frame_counter
    global last_hmr_frame
    global last_hmr_time

    print()
    print("=" * 70)
    print("STARTING PROCESSING LOOP")
    print("=" * 70)
    print()
    print("Browser:")
    print(f"  http://localhost:{FLASK_PORT}")
    print()
    print("API:")
    print(f"  http://localhost:{FLASK_PORT}/api/hmr3d")
    print()

    while running:

        ret, frame = camera.read()

        if not ret:

            print("[CAMERA] Failed to read frame.")
            time.sleep(0.1)
            continue

        frame_counter += 1

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        person_boxes = detect_persons(frame)

        # ----------------------------------------------------
        # HMR
        # ----------------------------------------------------

        if (
            last_hmr_frame < 0
            or frame_counter - last_hmr_frame >= HMR_INTERVAL
        ):

            print()
            print(
                f"[HMR] Processing frame "
                f"{frame_counter} | "
                f"YOLO persons={len(person_boxes)}"
            )

            hmr_start = time.time()

            try:

                hmr_result = hmr_processor.process_frame(
                    frame,
                    frame_counter,
                    time.time(),
                    person_boxes
                )

                hmr_elapsed = time.time() - hmr_start

                with hmr_lock:
                    latest_hmr_result = hmr_result

                last_hmr_frame = frame_counter
                last_hmr_time = time.time()

                print(
                    f"[HMR] Completed in "
                    f"{hmr_elapsed:.2f} sec"
                )

                # ------------------------------------------------
                # RAW KEYS
                # ------------------------------------------------

                persons = hmr_result.get(
                    "persons",
                    []
                )

                print(
                    f"[HMR] Persons returned: "
                    f"{len(persons)}"
                )

                if persons:

                    print(
                        "[HMR] RAW PERSON KEYS:"
                    )

                    print(
                        sorted(
                            persons[0].keys()
                        )
                    )

                # ------------------------------------------------
                # PRINT NUMERICAL 3D DATA
                # ------------------------------------------------

                print_hmr_debug(
                    hmr_result,
                    frame_counter
                )

            except Exception as e:

                print()
                print(
                    "[HMR ERROR]",
                    repr(e)
                )

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        with hmr_lock:
            hmr_cached = latest_hmr_result

        display = draw_hmr_information(
            frame,
            person_boxes,
            hmr_cached
        )

        # ----------------------------------------------------
        # Status bar
        # ----------------------------------------------------

        status = (
            f"Frame: {frame_counter} | "
            f"YOLO: {len(person_boxes)} | "
            f"HMR: "
            f"{0 if hmr_cached is None else len(hmr_cached.get('persons', []))}"
        )

        cv2.putText(
            display,
            status,
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        with frame_lock:
            latest_frame = display


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    worker = threading.Thread(
        target=processing_loop,
        daemon=True
    )

    worker.start()

    try:

        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True
        )

    finally:

        running = False

        if camera is not None:
            camera.release()
