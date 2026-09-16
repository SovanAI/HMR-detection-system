import os
import sys
import time
import json
import threading
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_DEVICE = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# Run expensive HMR only every N frames.
HMR_INTERVAL = 30

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5013


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None
latest_hmr_result = {
    "status": "waiting",
    "frame_id": 0,
    "persons": []
}

frame_lock = threading.Lock()
hmr_lock = threading.Lock()

running = True


# ============================================================
# IMPORTS
# ============================================================

print("=" * 80)
print("BAS-HMR HUMAN 3D DIAGNOSTIC V4")
print("=" * 80)


print("\n[1/3] Loading YOLO11n...")

try:
    from ultralytics import YOLO

    yolo = YOLO("yolo11n.pt")

    print("[OK] YOLO11n loaded.")
    print("[OK] Device: CPU")

except Exception as e:
    print("[ERROR] Failed to load YOLO11n")
    print(e)
    traceback.print_exc()
    sys.exit(1)


print("\n[2/3] Loading HMR2...")

try:
    from human.hmr.hmr_processor import HMRProcessor

    hmr_processor = HMRProcessor()

    print("[OK] HMR2 loaded.")
    print("[OK] Device: CPU")

except Exception as e:
    print("[ERROR] Failed to load HMR2")
    print(e)
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# CAMERA FUNCTIONS
# ============================================================

def open_camera():
    """
    Open webcam using V4L2 explicitly.

    The previous version successfully opened the camera but
    cap.read() repeatedly failed. This version configures the
    Linux V4L2 backend explicitly and forces MJPEG.
    """

    print("\n[CAMERA] Opening:", CAMERA_DEVICE)

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[CAMERA ERROR] Could not open camera.")
        return None

    # Request MJPEG.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Reduce buffering/latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("[CAMERA] Requested:", WIDTH, "x", HEIGHT, "@", FPS, "FPS")
    print(
        "[CAMERA] Actual:",
        actual_width,
        "x",
        actual_height,
        "@",
        actual_fps,
        "FPS"
    )

    # --------------------------------------------------------
    # Camera warm-up
    # --------------------------------------------------------

    print("[CAMERA] Warming up...")

    for i in range(10):
        ok, frame = cap.read()

        if ok and frame is not None:
            print(
                "[CAMERA] Warm-up successful on attempt",
                i + 1
            )

            print(
                "[CAMERA] Frame shape:",
                frame.shape
            )

            return cap

        time.sleep(0.15)

    print("[CAMERA ERROR] Camera opened but produced no frames.")

    cap.release()

    return None


def reopen_camera():
    """
    Release and reopen the webcam.
    """

    print("\n[CAMERA] Attempting camera recovery...")

    time.sleep(1.0)

    cap = open_camera()

    if cap is not None:
        print("[CAMERA] Recovery successful.")
    else:
        print("[CAMERA] Recovery failed.")

    return cap


# ============================================================
# PERSON DETECTION
# ============================================================

def detect_person_boxes(frame):
    """
    YOLO11 person detection.

    IMPORTANT:
    HMRProcessor requires numeric bounding boxes:

        [x1, y1, x2, y2]

    NOT dictionaries.
    """

    results = yolo.predict(
        source=frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        classes=[0],       # person only
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

        x1, y1, x2, y2 = map(float, xyxy)

        # Clamp to image.
        x1 = max(0.0, min(x1, frame.shape[1] - 1))
        y1 = max(0.0, min(y1, frame.shape[0] - 1))
        x2 = max(0.0, min(x2, frame.shape[1] - 1))
        y2 = max(0.0, min(y2, frame.shape[0] - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # CRITICAL:
        # This MUST be a numeric list.
        boxes.append([
            x1,
            y1,
            x2,
            y2
        ])

    return boxes


# ============================================================
# HMR PROCESSING
# ============================================================

def process_hmr(frame, frame_id):

    global latest_hmr_result

    try:

        boxes = detect_person_boxes(frame)

        print(
            f"\n[FRAME {frame_id}] "
            f"YOLO persons = {len(boxes)}"
        )

        if len(boxes) > 0:
            print("[HMR INPUT] Numeric boxes:")
            print(boxes)

        else:
            result = {
                "status": "no_person",
                "frame_id": frame_id,
                "model": "HMR2",
                "device": "cpu",
                "persons": []
            }

            with hmr_lock:
                latest_hmr_result = result

            return result

        # ----------------------------------------------------
        # HMR2
        # ----------------------------------------------------

        print("[HMR] Running HMR2...")

        start = time.time()

        result = hmr_processor.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=time.time(),
            boxes=boxes
        )

        elapsed = time.time() - start

        print(
            f"[HMR] Completed in {elapsed:.2f} seconds"
        )

        inspect_hmr_result(result)

        with hmr_lock:
            latest_hmr_result = result

        return result

    except Exception as e:

        print("\n[HMR ERROR]")
        print(type(e).__name__, ":", str(e))

        traceback.print_exc()

        result = {
            "status": "error",
            "frame_id": frame_id,
            "error_type": type(e).__name__,
            "error": str(e),
            "persons": []
        }

        with hmr_lock:
            latest_hmr_result = result

        return result


# ============================================================
# HMR RESULT INSPECTION
# ============================================================

def inspect_hmr_result(result):

    print("\n" + "-" * 70)
    print("HMR RESULT INSPECTION")
    print("-" * 70)

    print(
        "Result type:",
        type(result).__name__
    )

    if not isinstance(result, dict):
        print("Unexpected HMR result format.")
        print(result)
        return

    print(
        "Top-level keys:",
        list(result.keys())
    )

    persons = result.get("persons", [])

    print(
        "Person count:",
        len(persons)
    )

    if len(persons) == 0:
        print("No HMR persons returned.")
        print("-" * 70)
        return

    for index, person in enumerate(persons):

        print(
            f"\nPERSON {index + 1}"
        )

        print(
            "Keys:",
            list(person.keys())
        )

        print(
            "Person ID:",
            person.get("person_id")
        )

        print(
            "BBox:",
            person.get("bbox")
        )

        camera_translation = person.get(
            "camera_translation"
        )

        print(
            "Camera translation:",
            camera_translation
        )

        joints = person.get("joints_3d")

        if joints is None:

            print(
                "joints_3d: NOT PRESENT"
            )

            continue

        try:

            joints_np = np.asarray(
                joints,
                dtype=np.float32
            )

            print(
                "joints_3d type:",
                type(joints).__name__
            )

            print(
                "joints_3d shape:",
                joints_np.shape
            )

            if joints_np.ndim == 2 and joints_np.shape[1] == 3:

                print(
                    "Joint count:",
                    joints_np.shape[0]
                )

                print(
                    "First 5 joints:"
                )

                print(
                    joints_np[:5]
                )

                print(
                    "XYZ minimum:",
                    joints_np.min(axis=0)
                )

                print(
                    "XYZ maximum:",
                    joints_np.max(axis=0)
                )

                print(
                    "Root-relative centroid:",
                    joints_np.mean(axis=0)
                )

                # --------------------------------------------
                # Diagnostic camera-frame estimate
                # --------------------------------------------

                if (
                    camera_translation is not None
                    and len(camera_translation) == 3
                ):

                    translation_np = np.asarray(
                        camera_translation,
                        dtype=np.float32
                    )

                    camera_frame_joints = (
                        joints_np + translation_np
                    )

                    print(
                        "Estimated camera-frame centroid:"
                    )

                    print(
                        camera_frame_joints.mean(axis=0)
                    )

                    print(
                        "Estimated camera-frame XYZ min:"
                    )

                    print(
                        camera_frame_joints.min(axis=0)
                    )

                    print(
                        "Estimated camera-frame XYZ max:"
                    )

                    print(
                        camera_frame_joints.max(axis=0)
                    )

        except Exception as e:

            print(
                "Could not inspect joints:",
                e
            )

    print("-" * 70)


# ============================================================
# DRAW DEBUG INFORMATION
# ============================================================

def draw_debug(frame, boxes):

    output = frame.copy()

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Person {i + 1}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        output,
        "BAS-HMR V4",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "CPU | YOLO11 + HMR2",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return output


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BAS-HMR Human 3D V4</title>

        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
            }

            img {
                width: 80%;
                max-width: 900px;
                border: 2px solid #444;
            }

            a {
                color: #6cf;
            }

            pre {
                text-align: left;
                width: 80%;
                max-width: 900px;
                margin: auto;
                background: #222;
                padding: 15px;
                overflow-x: auto;
            }
        </style>
    </head>

    <body>

        <h1>BAS-HMR Human 3D Diagnostic V4</h1>

        <img src="/video">

        <h3>
            <a href="/api/hmr3d" target="_blank">
                Open Raw HMR JSON
            </a>
        </h3>

        <h3>
            <a href="/api/health" target="_blank">
                Health
            </a>
        </h3>

    </body>
    </html>
    """


@app.route("/api/hmr3d")
def api_hmr3d():

    with hmr_lock:
        result = latest_hmr_result

    return jsonify(result)


@app.route("/api/health")
def api_health():

    with frame_lock:
        frame_available = (
            latest_frame is not None
        )

    return jsonify({
        "running": running,
        "camera": CAMERA_DEVICE,
        "frame_available": frame_available,
        "hmr_interval": HMR_INTERVAL
    })


@app.route("/video")
def video():

    def generate():

        global latest_frame

        while running:

            with frame_lock:

                if latest_frame is None:
                    frame = None
                else:
                    frame = latest_frame.copy()

            if frame is not None:

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        80
                    ]
                )

                if ok:

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + encoded.tobytes()
                        + b"\r\n"
                    )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# FLASK THREAD
# ============================================================

def run_flask():

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    global running
    global latest_frame

    print("\n" + "=" * 80)
    print("PROCESSING LOOP STARTING")
    print("=" * 80)

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = open_camera()

    while cap is None:

        print(
            "\n[CAMERA] Waiting before retry..."
        )

        time.sleep(2)

        cap = open_camera()

    print("\n[OK] Camera ready.")

    # --------------------------------------------------------
    # Start Flask AFTER camera is confirmed working.
    # --------------------------------------------------------

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    print("\nOpen browser:")
    print(
        f"http://localhost:{FLASK_PORT}"
    )

    print("\nRaw HMR API:")
    print(
        f"http://localhost:{FLASK_PORT}/api/hmr3d"
    )

    print("\nHealth:")
    print(
        f"http://localhost:{FLASK_PORT}/api/health"
    )

    print("\nPress CTRL+C to stop.\n")

    frame_id = 0
    camera_failures = 0

    # --------------------------------------------------------
    # Main processing loop
    # --------------------------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            # -----------------------------------------------
            # Camera failure
            # -----------------------------------------------

            if not ok or frame is None:

                camera_failures += 1

                print(
                    f"[CAMERA ERROR] "
                    f"Could not read frame "
                    f"({camera_failures}/5)"
                )

                if camera_failures >= 5:

                    print(
                        "\n[CAMERA] Too many read failures."
                    )

                    try:
                        cap.release()
                    except Exception:
                        pass

                    cap = reopen_camera()

                    if cap is None:

                        print(
                            "[CAMERA] Recovery failed."
                        )

                        time.sleep(2)

                    camera_failures = 0

                continue

            # -----------------------------------------------
            # Successful frame
            # -----------------------------------------------

            camera_failures = 0

            frame_id += 1

            # Keep camera display alive.
            with frame_lock:
                latest_frame = frame.copy()

            # -----------------------------------------------
            # YOLO detection every frame
            # -----------------------------------------------

            try:

                boxes = detect_person_boxes(frame)

            except Exception as e:

                print(
                    "[YOLO ERROR]",
                    e
                )

                boxes = []

            # -----------------------------------------------
            # Draw boxes
            # -----------------------------------------------

            display_frame = draw_debug(
                frame,
                boxes
            )

            with frame_lock:
                latest_frame = display_frame

            # -----------------------------------------------
            # HMR every N frames
            # -----------------------------------------------

            if frame_id == 1 or (
                frame_id % HMR_INTERVAL == 0
            ):

                # Run HMR in this thread.
                process_hmr(
                    frame,
                    frame_id
                )

            # -----------------------------------------------
            # Small delay
            # -----------------------------------------------

            time.sleep(0.001)

    except KeyboardInterrupt:

        print("\n[STOP] CTRL+C received.")

    finally:

        running = False

        print(
            "[CAMERA] Releasing camera..."
        )

        try:
            cap.release()
        except Exception:
            pass

        print(
            "[OK] Camera released."
        )

        print(
            "BAS-HMR stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
