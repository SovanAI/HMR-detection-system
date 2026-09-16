import sys
import time
import json
import threading
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify

# ============================================================
# CONFIG
# ============================================================

CAMERA_DEVICE = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# HMR is expensive on CPU.
HMR_INTERVAL = 30

PORT = 5014


# ============================================================
# GLOBAL STATE
# ============================================================

running = True

latest_frame = None
latest_result = {
    "status": "waiting",
    "persons": []
}

frame_lock = threading.Lock()
result_lock = threading.Lock()


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("BAS-HMR HUMAN 3D JOINT DIAGNOSTIC V5")
print("=" * 80)


# ============================================================
# LOAD YOLO
# ============================================================

print("\n[1/3] Loading YOLO11n...")

try:

    from ultralytics import YOLO

    yolo = YOLO("yolo11n.pt")

    print("[OK] YOLO11n loaded.")
    print("[OK] Device: CPU")

except Exception as e:

    print("[ERROR] YOLO11n loading failed.")
    print(e)

    traceback.print_exc()

    sys.exit(1)


# ============================================================
# LOAD HMR
# ============================================================

print("\n[2/3] Loading HMR2...")

try:

    from human.hmr.hmr_processor import HMRProcessor

    hmr_processor = HMRProcessor()

    print("[OK] HMR2 loaded.")
    print("[OK] Device: CPU")

except Exception as e:

    print("[ERROR] HMR2 loading failed.")
    print(e)

    traceback.print_exc()

    sys.exit(1)


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    print("\n[CAMERA] Opening:", CAMERA_DEVICE)

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():

        print("[CAMERA ERROR] Could not open camera.")

        return None

    # Force MJPEG.
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

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print(
        "[CAMERA] Actual:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "@",
        cap.get(cv2.CAP_PROP_FPS),
        "FPS"
    )

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

    print(
        "[CAMERA ERROR] Camera opened but no frames received."
    )

    cap.release()

    return None


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_person_boxes(frame):

    results = yolo.predict(
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

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = map(
            float,
            xyxy
        )

        x1 = max(
            0.0,
            min(x1, frame.shape[1] - 1)
        )

        y1 = max(
            0.0,
            min(y1, frame.shape[0] - 1)
        )

        x2 = max(
            0.0,
            min(x2, frame.shape[1] - 1)
        )

        y2 = max(
            0.0,
            min(y2, frame.shape[0] - 1)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # IMPORTANT:
        # HMR requires numeric boxes.
        boxes.append([
            x1,
            y1,
            x2,
            y2
        ])

    return boxes


# ============================================================
# SAFE NUMPY CONVERSION
# ============================================================

def to_numpy(value):

    if value is None:
        return None

    try:

        if hasattr(value, "detach"):

            value = (
                value
                .detach()
                .cpu()
                .numpy()
            )

        elif hasattr(value, "cpu"):

            value = (
                value
                .cpu()
                .numpy()
            )

        return np.asarray(
            value,
            dtype=np.float32
        )

    except Exception:

        return None


# ============================================================
# FIND 3D KEYPOINTS
# ============================================================

def extract_3d_from_raw(raw_output):

    """
    Try to locate pred_keypoints_3d in the raw HMR output.

    This function is deliberately diagnostic.
    It does not assume a single HMR output structure.
    """

    if raw_output is None:
        return None

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(raw_output, dict):

        possible_keys = [
            "pred_keypoints_3d",
            "pred_keypoints3d",
            "keypoints_3d",
            "joints_3d"
        ]

        for key in possible_keys:

            if key in raw_output:

                arr = to_numpy(
                    raw_output[key]
                )

                if arr is not None:

                    return arr

        # Recursive search.
        for value in raw_output.values():

            found = extract_3d_from_raw(value)

            if found is not None:
                return found

    # --------------------------------------------------------
    # List / tuple
    # --------------------------------------------------------

    elif isinstance(
        raw_output,
        (list, tuple)
    ):

        for value in raw_output:

            found = extract_3d_from_raw(value)

            if found is not None:
                return found

    return None


# ============================================================
# NORMALIZE JOINT ARRAY
# ============================================================

def normalize_joints(joints):

    if joints is None:
        return None

    arr = np.asarray(
        joints,
        dtype=np.float32
    )

    # Common forms:
    #
    # (N, 3)
    # (1, N, 3)

    if arr.ndim == 3 and arr.shape[0] == 1:

        arr = arr[0]

    if (
        arr.ndim != 2
        or arr.shape[1] != 3
    ):

        return None

    return arr


# ============================================================
# PRINT 3D JOINT INFORMATION
# ============================================================

def print_joint_diagnostics(
    joints,
    camera_translation,
    person_id
):

    print("\n")
    print("=" * 70)
    print(
        f"3D HUMAN JOINT DIAGNOSTIC "
        f"| PERSON {person_id}"
    )
    print("=" * 70)

    if joints is None:

        print("[ERROR] No 3D joints available.")

        return

    joints = normalize_joints(joints)

    if joints is None:

        print(
            "[ERROR] Invalid joint array."
        )

        return

    print(
        "Joint array shape:",
        joints.shape
    )

    print(
        "Joint count:",
        joints.shape[0]
    )

    print("\nFirst 5 joints:")

    print(joints[:5])

    print("\nXYZ minimum:")

    print(
        np.min(
            joints,
            axis=0
        )
    )

    print("\nXYZ maximum:")

    print(
        np.max(
            joints,
            axis=0
        )
    )

    print("\nXYZ mean:")

    print(
        np.mean(
            joints,
            axis=0
        )
    )

    print("\nXYZ standard deviation:")

    print(
        np.std(
            joints,
            axis=0
        )
    )

    # --------------------------------------------------------
    # Root joint
    # --------------------------------------------------------

    root = joints[0]

    print("\nRoot-relative joint 0:")

    print(root)

    # --------------------------------------------------------
    # Camera translation
    # --------------------------------------------------------

    if camera_translation is not None:

        try:

            t = np.asarray(
                camera_translation,
                dtype=np.float32
            )

            if t.shape == (3,):

                print(
                    "\nHMR camera translation:"
                )

                print(t)

                camera_joints = (
                    joints + t
                )

                print(
                    "\nEstimated camera-frame "
                    "joint centroid:"
                )

                print(
                    np.mean(
                        camera_joints,
                        axis=0
                    )
                )

                print(
                    "\nEstimated camera-frame "
                    "XYZ minimum:"
                )

                print(
                    np.min(
                        camera_joints,
                        axis=0
                    )
                )

                print(
                    "\nEstimated camera-frame "
                    "XYZ maximum:"
                )

                print(
                    np.max(
                        camera_joints,
                        axis=0
                    )
                )

        except Exception as e:

            print(
                "[WARNING] Could not combine "
                "joints with camera translation:",
                e
            )

    print("=" * 70)


# ============================================================
# HMR PROCESSING
# ============================================================

def run_hmr(
    frame,
    frame_id,
    boxes
):

    global latest_result

    if len(boxes) == 0:

        result = {
            "status": "no_person",
            "frame_id": frame_id,
            "model": "HMR2",
            "device": "cpu",
            "persons": []
        }

        with result_lock:
            latest_result = result

        return result

    print(
        f"\n[FRAME {frame_id}] "
        f"HMR input persons = {len(boxes)}"
    )

    print(
        "[HMR INPUT] Numeric boxes:"
    )

    print(boxes)

    try:

        start = time.time()

        # ----------------------------------------------------
        # Existing HMR processor
        # ----------------------------------------------------

        result = hmr_processor.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=time.time(),
            boxes=boxes
        )

        elapsed = time.time() - start

        print(
            f"[HMR] Completed in "
            f"{elapsed:.2f} seconds"
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # The existing processor returns a JSON-friendly
        # dictionary. We inspect that result first.
        # ----------------------------------------------------

        print(
            "[HMR] Returned keys:",
            list(result.keys())
            if isinstance(result, dict)
            else type(result)
        )

        persons = (
            result.get("persons", [])
            if isinstance(result, dict)
            else []
        )

        print(
            "[HMR] Returned persons:",
            len(persons)
        )

        # ----------------------------------------------------
        # Inspect every person
        # ----------------------------------------------------

        for person in persons:

            print("\n" + "-" * 60)

            person_id = person.get(
                "person_id",
                0
            )

            print(
                "Person ID:",
                person_id
            )

            print(
                "Person keys:",
                list(person.keys())
            )

            print(
                "BBox:",
                person.get("bbox")
            )

            print(
                "Camera translation:",
                person.get(
                    "camera_translation"
                )
            )

            # ------------------------------------------------
            # Look inside pose
            # ------------------------------------------------

            pose = person.get(
                "pose"
            )

            if pose is not None:

                print(
                    "Pose keys:",
                    list(pose.keys())
                    if isinstance(
                        pose,
                        dict
                    )
                    else type(pose)
                )

                if isinstance(
                    pose,
                    dict
                ):

                    joints = pose.get(
                        "joints_3d"
                    )

                    if joints is not None:

                        joints = normalize_joints(
                            joints
                        )

                        print(
                            "[SUCCESS] "
                            "3D joints found "
                            "inside pose."
                        )

                        print_joint_diagnostics(
                            joints,
                            person.get(
                                "camera_translation"
                            ),
                            person_id
                        )

                    else:

                        print(
                            "[INFO] pose exists, "
                            "but pose.joints_3d "
                            "is not present."
                        )

            # ------------------------------------------------
            # Direct joints
            # ------------------------------------------------

            direct_joints = person.get(
                "joints_3d"
            )

            if direct_joints is not None:

                direct_joints = normalize_joints(
                    direct_joints
                )

                print(
                    "[SUCCESS] Direct "
                    "joints_3d found."
                )

                print_joint_diagnostics(
                    direct_joints,
                    person.get(
                        "camera_translation"
                    ),
                    person_id
                )

        with result_lock:

            latest_result = result

        return result

    except Exception as e:

        print(
            "\n[HMR ERROR]",
            type(e).__name__,
            ":",
            str(e)
        )

        traceback.print_exc()

        result = {
            "status": "error",
            "frame_id": frame_id,
            "error_type": type(e).__name__,
            "error": str(e),
            "persons": []
        }

        with result_lock:
            latest_result = result

        return result


# ============================================================
# DRAW
# ============================================================

def draw_debug(
    frame,
    boxes
):

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
            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        output,
        "BAS-HMR V5",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "3D JOINT DIAGNOSTIC",
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

        <title>BAS-HMR V5</title>

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

        </style>

    </head>

    <body>

        <h1>BAS-HMR V5</h1>

        <h2>Human 3D Joint Diagnostic</h2>

        <img src="/video">

        <p>
            <a
                href="/api/hmr3d"
                target="_blank"
            >
                Raw HMR JSON
            </a>
        </p>

        <p>
            <a
                href="/api/health"
                target="_blank"
            >
                Health
            </a>
        </p>

    </body>

    </html>
    """


@app.route("/api/hmr3d")
def api_hmr3d():

    with result_lock:
        result = latest_result

    return jsonify(result)


@app.route("/api/health")
def api_health():

    with frame_lock:

        available = (
            latest_frame is not None
        )

    return jsonify({
        "running": running,
        "camera": CAMERA_DEVICE,
        "frame_available": available,
        "hmr_interval": HMR_INTERVAL
    })


@app.route("/video")
def video():

    def generate():

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
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# FLASK THREAD
# ============================================================

def run_flask():

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global running
    global latest_frame

    print("\n" + "=" * 80)
    print("PROCESSING LOOP STARTING")
    print("=" * 80)

    cap = open_camera()

    while cap is None:

        print(
            "[CAMERA] Retrying in 2 seconds..."
        )

        time.sleep(2)

        cap = open_camera()

    print(
        "\n[OK] Camera ready."
    )

    # --------------------------------------------------------
    # Flask
    # --------------------------------------------------------

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    print("\nOpen browser:")
    print(
        f"http://localhost:{PORT}"
    )

    print("\nRaw HMR API:")
    print(
        f"http://localhost:{PORT}/api/hmr3d"
    )

    print("\nHealth:")
    print(
        f"http://localhost:{PORT}/api/health"
    )

    print("\nPress CTRL+C to stop.\n")

    frame_id = 0
    camera_failures = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok or frame is None:

                camera_failures += 1

                print(
                    "[CAMERA ERROR]",
                    camera_failures
                )

                if camera_failures >= 5:

                    print(
                        "[CAMERA] Reopening..."
                    )

                    cap.release()

                    cap = open_camera()

                    if cap is None:

                        time.sleep(2)

                    camera_failures = 0

                continue

            camera_failures = 0

            frame_id += 1

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            try:

                boxes = detect_person_boxes(
                    frame
                )

            except Exception as e:

                print(
                    "[YOLO ERROR]",
                    e
                )

                boxes = []

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            display = draw_debug(
                frame,
                boxes
            )

            with frame_lock:

                latest_frame = display

            # ------------------------------------------------
            # HMR
            # ------------------------------------------------

            if (
                frame_id == 1
                or frame_id % HMR_INTERVAL == 0
            ):

                run_hmr(
                    frame,
                    frame_id,
                    boxes
                )

            time.sleep(0.001)

    except KeyboardInterrupt:

        print(
            "\n[STOP] CTRL+C received."
        )

    finally:

        running = False

        try:
            cap.release()
        except Exception:
            pass

        print(
            "[OK] Camera released."
        )

        print(
            "BAS-HMR V5 stopped."
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
