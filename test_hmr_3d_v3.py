from __future__ import annotations

import json
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from ultralytics import YOLO

from human.hmr.hmr_processor import HMRProcessor


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_DEVICE = "/dev/video0"

YOLO_MODEL = "yolo11n.pt"

HOST = "0.0.0.0"
PORT = 5013

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# HMR is CPU-heavy.
# Run HMR once every N frames.
HMR_INTERVAL = 30


# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

running = True

latest_frame = None
latest_hmr_result = {
    "frame_id": 0,
    "timestamp": 0.0,
    "model": "HMR2",
    "device": "cpu",
    "persons": [],
}

state_lock = threading.Lock()


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("=" * 80)
print("BAS-HMR HUMAN 3D DIAGNOSTIC V3")
print("=" * 80)

print()
print("[1/3] Loading YOLO11n...")

yolo_model = YOLO(YOLO_MODEL)

print("[OK] YOLO11n loaded.")
print("[OK] Device: CPU")


# ============================================================
# LOAD HMR2
# ============================================================

print()
print("[2/3] Loading HMR2...")

hmr_processor = HMRProcessor()

print("[OK] HMR2 loaded.")
print("[OK] Device: CPU")


# ============================================================
# CAMERA
# ============================================================

print()
print("[3/3] Opening camera...")

camera = cv2.VideoCapture(CAMERA_DEVICE)

if not camera.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA_DEVICE}"
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

camera.set(cv2.CAP_PROP_FPS, 10)

print("[OK] Camera opened.")
print("[OK] Resolution: 640x480")
print("[OK] FPS: 10")


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def make_json_safe(value):
    """
    Convert numpy / nested values into JSON-safe values.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(v)
            for v in value
        ]

    return value


# ============================================================
# NUMERICAL ARRAY EXTRACTION
# ============================================================

def safe_xyz_array(value):
    """
    Convert a possible joints_3d value into an
    Nx3 numpy array.

    Returns None if conversion is impossible.
    """

    if value is None:
        return None

    try:

        arr = np.asarray(
            value,
            dtype=np.float32
        )

        if arr.size == 0:
            return None

        # Flatten accidental batch dimension.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim != 2:
            return None

        if arr.shape[1] < 3:
            return None

        arr = arr[:, :3]

        return arr

    except Exception:
        return None


# ============================================================
# HMR RESULT DIAGNOSTICS
# ============================================================

def inspect_hmr_result(result):
    """
    Print the actual HMR2 output structure.

    This is the most important diagnostic function.
    """

    print()
    print("-" * 80)
    print("HMR 3D RESULT INSPECTION")
    print("-" * 80)

    print(
        "[RAW] result type:",
        type(result)
    )

    if not isinstance(result, dict):

        print(
            "[ERROR] HMR result is not a dictionary."
        )

        return

    print(
        "[RAW] top-level keys:",
        list(result.keys())
    )

    persons = result.get(
        "persons",
        []
    )

    print(
        "[RAW] persons:",
        len(persons)
    )

    if not persons:

        print(
            "[INFO] No HMR persons returned."
        )

        return

    for person_index, person in enumerate(persons):

        print()
        print(
            f"[PERSON {person_index}]"
        )

        print(
            "keys:",
            list(person.keys())
        )

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        bbox = person.get("bbox")

        print(
            "bbox:",
            bbox
        )

        # ----------------------------------------------------
        # CAMERA TRANSLATION
        # ----------------------------------------------------

        camera_translation = person.get(
            "camera_translation"
        )

        print(
            "camera_translation:",
            camera_translation
        )

        if isinstance(
            camera_translation,
            dict
        ):

            try:

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
                        0.0
                    )
                )

                print(
                    f"CAMERA T = "
                    f"X={tx:.5f}, "
                    f"Y={ty:.5f}, "
                    f"Z={tz:.5f}"
                )

            except Exception as exc:

                print(
                    "[WARNING] Could not parse "
                    "camera translation:",
                    repr(exc)
                )

        # ----------------------------------------------------
        # JOINTS 3D
        # ----------------------------------------------------

        joints_raw = person.get(
            "joints_3d"
        )

        print(
            "joints_3d type:",
            type(joints_raw)
        )

        joints = safe_xyz_array(
            joints_raw
        )

        if joints is None:

            print(
                "[ERROR] joints_3d could not "
                "be converted to Nx3."
            )

            continue

        print(
            "joints_3d shape:",
            joints.shape
        )

        print(
            "joint count:",
            len(joints)
        )

        print()
        print(
            "FIRST 5 3D JOINTS:"
        )

        for joint_id, xyz in enumerate(
            joints[:5]
        ):

            print(
                f"  joint {joint_id:02d}: "
                f"X={xyz[0]: .5f} "
                f"Y={xyz[1]: .5f} "
                f"Z={xyz[2]: .5f}"
            )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        minimum = np.min(
            joints,
            axis=0
        )

        maximum = np.max(
            joints,
            axis=0
        )

        centroid = np.mean(
            joints,
            axis=0
        )

        print()
        print(
            "3D JOINT STATISTICS:"
        )

        print(
            f"MIN: "
            f"X={minimum[0]: .5f} "
            f"Y={minimum[1]: .5f} "
            f"Z={minimum[2]: .5f}"
        )

        print(
            f"MAX: "
            f"X={maximum[0]: .5f} "
            f"Y={maximum[1]: .5f} "
            f"Z={maximum[2]: .5f}"
        )

        print(
            f"CENTROID: "
            f"X={centroid[0]: .5f} "
            f"Y={centroid[1]: .5f} "
            f"Z={centroid[2]: .5f}"
        )

        # ----------------------------------------------------
        # CAMERA FRAME ESTIMATE
        # ----------------------------------------------------

        if isinstance(
            camera_translation,
            dict
        ):

            try:

                translation = np.array(
                    [
                        float(
                            camera_translation["x"]
                        ),
                        float(
                            camera_translation["y"]
                        ),
                        float(
                            camera_translation["z"]
                        ),
                    ],
                    dtype=np.float32
                )

                camera_frame_joints = (
                    joints
                    + translation
                )

                cf_centroid = np.mean(
                    camera_frame_joints,
                    axis=0
                )

                print()
                print(
                    "ESTIMATED CAMERA-FRAME "
                    "3D JOINTS:"
                )

                print(
                    f"CENTROID: "
                    f"X={cf_centroid[0]: .5f} "
                    f"Y={cf_centroid[1]: .5f} "
                    f"Z={cf_centroid[2]: .5f}"
                )

                print(
                    "NOTE: This is a diagnostic "
                    "translation of the HMR "
                    "root-relative joints."
                )

            except Exception as exc:

                print(
                    "[WARNING] Camera-frame "
                    "calculation failed:",
                    repr(exc)
                )

    print()
    print(
        "-" * 80
    )


# ============================================================
# DRAW HMR INFORMATION
# ============================================================

def draw_hmr_overlay(
    frame,
    hmr_result
):

    output = frame.copy()

    persons = hmr_result.get(
        "persons",
        []
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    cv2.rectangle(
        output,
        (0, 0),
        (640, 80),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        "BAS-HMR HUMAN 3D DIAGNOSTIC V3",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"HMR persons: {len(persons)}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # PERSON DATA
    # --------------------------------------------------------

    y_text = 105

    for person_index, person in enumerate(
        persons
    ):

        bbox = person.get(
            "bbox"
        )

        if isinstance(
            bbox,
            dict
        ):

            try:

                x1 = int(
                    float(
                        bbox["x1"]
                    )
                )

                y1 = int(
                    float(
                        bbox["y1"]
                    )
                )

                x2 = int(
                    float(
                        bbox["x2"]
                    )
                )

                y2 = int(
                    float(
                        bbox["y2"]
                    )
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

            except Exception:
                pass

        joints = safe_xyz_array(
            person.get(
                "joints_3d"
            )
        )

        camera_translation = person.get(
            "camera_translation"
        )

        cv2.putText(
            output,
            f"PERSON {person_index}",
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y_text += 25

        if joints is not None:

            cv2.putText(
                output,
                f"3D joints: {len(joints)}",
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            y_text += 22

            centroid = np.mean(
                joints,
                axis=0
            )

            cv2.putText(
                output,
                (
                    f"J centroid: "
                    f"{centroid[0]:.2f}, "
                    f"{centroid[1]:.2f}, "
                    f"{centroid[2]:.2f}"
                ),
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            y_text += 22

        if isinstance(
            camera_translation,
            dict
        ):

            try:

                tx = float(
                    camera_translation["x"]
                )

                ty = float(
                    camera_translation["y"]
                )

                tz = float(
                    camera_translation["z"]
                )

                cv2.putText(
                    output,
                    (
                        f"Cam T: "
                        f"{tx:.2f}, "
                        f"{ty:.2f}, "
                        f"{tz:.2f}"
                    ),
                    (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

                y_text += 22

            except Exception:
                pass

        y_text += 10

    return output


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_person_boxes(frame):

    results = yolo_model.predict(
        source=frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        classes=[0],
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

        try:

            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
            )

            if len(xyxy) < 4:
                continue

            x1 = float(
                xyxy[0]
            )

            y1 = float(
                xyxy[1]
            )

            x2 = float(
                xyxy[2]
            )

            y2 = float(
                xyxy[3]
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # HMRProcessor requires:
            #
            # [
            #     [x1, y1, x2, y2],
            #     ...
            # ]
            #
            # NOT dictionaries.
            # ------------------------------------------------

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )

            print(
                "[YOLO] Person box:",
                (
                    f"{x1:.1f}, "
                    f"{y1:.1f}, "
                    f"{x2:.1f}, "
                    f"{y2:.1f}"
                ),
                f"conf={confidence:.3f}"
            )

        except Exception as exc:

            print(
                "[YOLO ERROR]",
                repr(exc)
            )

    return boxes


# ============================================================
# PROCESSING LOOP
# ============================================================

def processing_loop():

    global latest_frame
    global latest_hmr_result
    global running

    frame_id = 0

    last_hmr_frame = -1

    while running:

        success, frame = camera.read()

        if not success:

            print(
                "[CAMERA ERROR] "
                "Could not read frame."
            )

            time.sleep(0.1)

            continue

        frame_id += 1

        timestamp = time.time()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        boxes = detect_person_boxes(
            frame
        )

        print()
        print(
            f"[FRAME {frame_id}] "
            f"YOLO persons={len(boxes)}"
        )

        # ----------------------------------------------------
        # HMR
        # ----------------------------------------------------

        should_run_hmr = (
            frame_id == 1
            or
            frame_id - last_hmr_frame
            >= HMR_INTERVAL
        )

        if should_run_hmr:

            last_hmr_frame = frame_id

            hmr_start = time.time()

            try:

                print()
                print(
                    "=" * 80
                )

                print(
                    f"[HMR] Processing "
                    f"frame {frame_id}"
                )

                print(
                    f"[HMR] Input boxes:"
                )

                print(
                    boxes
                )

                # ------------------------------------------------
                # THIS IS THE CRITICAL FIX
                # ------------------------------------------------

                hmr_result = (
                    hmr_processor.process_frame(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        boxes=boxes
                    )
                )

                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                with state_lock:

                    latest_hmr_result = (
                        make_json_safe(
                            hmr_result
                        )
                    )

                # ------------------------------------------------
                # INSPECT RESULT
                # ------------------------------------------------

                inspect_hmr_result(
                    hmr_result
                )

                elapsed = (
                    time.time()
                    - hmr_start
                )

                print(
                    f"[HMR] Finished in "
                    f"{elapsed:.2f} sec"
                )

                print(
                    "=" * 80
                )

            except Exception as exc:

                print()
                print(
                    "=" * 80
                )

                print(
                    "[HMR ERROR]"
                )

                print(
                    "TYPE:",
                    type(exc)
                )

                print(
                    "MESSAGE:",
                    repr(exc)
                )

                print(
                    "=" * 80
                )

        # ----------------------------------------------------
        # DISPLAY FRAME
        # ----------------------------------------------------

        with state_lock:

            current_hmr = (
                latest_hmr_result.copy()
            )

        display_frame = (
            draw_hmr_overlay(
                frame,
                current_hmr
            )
        )

        with state_lock:

            latest_frame = (
                display_frame
            )


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_video():

    while running:

        with state_lock:

            frame = (
                None
                if latest_frame is None
                else latest_frame.copy()
            )

        if frame is None:

            time.sleep(0.03)

            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                80
            ]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            encoded.tobytes()
            +
            b"\r\n"
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

<title>BAS HMR Human 3D Diagnostic V3</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    margin: 20px;
}

h1 {
    font-size: 24px;
}

img {
    width: 640px;
    height: 480px;
    border: 2px solid white;
}

pre {
    background: #222;
    padding: 15px;
    width: 610px;
    overflow: auto;
}

</style>

</head>

<body>

<h1>BAS-HMR Human 3D Diagnostic V3</h1>

<img src="/video">

<h2>Raw HMR2 JSON</h2>

<pre id="json">Loading...</pre>

<script>

async function updateJSON() {

    try {

        const response =
            await fetch("/api/hmr3d");

        const data =
            await response.json();

        document.getElementById(
            "json"
        ).textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    } catch(error) {

        document.getElementById(
            "json"
        ).textContent =
            error.toString();

    }

}

setInterval(
    updateJSON,
    1000
);

updateJSON();

</script>

</body>

</html>
"""


# ============================================================
# VIDEO ROUTE
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_video(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# HMR API
# ============================================================

@app.route("/api/hmr3d")
def api_hmr3d():

    with state_lock:

        return jsonify(
            latest_hmr_result
        )


# ============================================================
# HEALTH API
# ============================================================

@app.route("/api/health")
def health():

    with state_lock:

        return jsonify(
            {
                "status": "running",
                "camera": CAMERA_DEVICE,
                "model": "HMR2",
                "device": "cpu",
                "hmr_interval": HMR_INTERVAL,
                "persons": len(
                    latest_hmr_result.get(
                        "persons",
                        []
                    )
                )
            }
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("PROCESSING LOOP STARTED")
    print("=" * 80)

    print()
    print(
        "Open browser:"
    )

    print(
        f"http://localhost:{PORT}"
    )

    print()
    print(
        f"Raw HMR API:"
    )

    print(
        f"http://localhost:{PORT}/api/hmr3d"
    )

    print()
    print(
        "Press CTRL+C to stop."
    )

    worker = threading.Thread(
        target=processing_loop,
        daemon=True
    )

    worker.start()

    try:

        app.run(
            host=HOST,
            port=PORT,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping..."
        )

    finally:

        running = False

        time.sleep(1)

        camera.release()

        print(
            "Camera released."
        )

        print(
            "BAS-HMR stopped."
        )
