import sys
import time
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

HMR_INTERVAL = 30

PORT = 5015


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
print("BAS-HMR HUMAN 3D JOINT STRUCTURE DIAGNOSTIC V5.1")
print("=" * 80)


# ============================================================
# YOLO
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
# HMR
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

        print(
            "[CAMERA ERROR] Could not open camera."
        )

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
        int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        ),
        "x",
        int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        ),
        "@",
        cap.get(
            cv2.CAP_PROP_FPS
        ),
        "FPS"
    )

    print(
        "[CAMERA] Warming up..."
    )

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
        "[CAMERA ERROR] "
        "Camera opened but no frames received."
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
            min(
                x1,
                frame.shape[1] - 1
            )
        )

        y1 = max(
            0.0,
            min(
                y1,
                frame.shape[0] - 1
            )
        )

        x2 = max(
            0.0,
            min(
                x2,
                frame.shape[1] - 1
            )
        )

        y2 = max(
            0.0,
            min(
                y2,
                frame.shape[0] - 1
            )
        )

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )

    return boxes


# ============================================================
# JOINT STRUCTURE INSPECTOR
# ============================================================

def inspect_joint_structure(joints):

    print("\n")
    print("=" * 80)
    print("RAW joints_3d STRUCTURE")
    print("=" * 80)

    print(
        "Python type:",
        type(joints).__name__
    )

    if joints is None:

        print(
            "[ERROR] joints_3d is None"
        )

        return

    if isinstance(joints, dict):

        print(
            "Dictionary keys:",
            list(joints.keys())
        )

        print(
            "Dictionary content:"
        )

        print(joints)

        return

    if isinstance(
        joints,
        (list, tuple)
    ):

        print(
            "Number of elements:",
            len(joints)
        )

        if len(joints) == 0:

            print(
                "[WARNING] Empty joints_3d."
            )

            return

        print(
            "\nFirst element type:",
            type(joints[0]).__name__
        )

        print(
            "\nFirst element:"
        )

        print(joints[0])

        if len(joints) > 1:

            print(
                "\nSecond element:"
            )

            print(joints[1])

        if len(joints) > 2:

            print(
                "\nThird element:"
            )

            print(joints[2])

        print(
            "\nFull structure summary:"
        )

        for i, joint in enumerate(
            joints[:10]
        ):

            print(
                f"Joint {i}: "
                f"type={type(joint).__name__} "
                f"value={joint}"
            )

        return

    # NumPy / tensor / other.
    print(
        "Representation:"
    )

    print(joints)

    print("=" * 80)


# ============================================================
# CONVERT ONE JOINT
# ============================================================

def convert_joint_to_xyz(joint):

    """
    Convert one joint representation into:

        [x, y, z]

    Supports several common dictionary/list formats.
    """

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(joint, dict):

        # Exact lowercase form.
        if all(
            key in joint
            for key in ["x", "y", "z"]
        ):

            return [
                float(joint["x"]),
                float(joint["y"]),
                float(joint["z"])
            ]

        # Uppercase.
        if all(
            key in joint
            for key in ["X", "Y", "Z"]
        ):

            return [
                float(joint["X"]),
                float(joint["Y"]),
                float(joint["Z"])
            ]

        # Alternative common naming.
        if all(
            key in joint
            for key in [
                "x_coord",
                "y_coord",
                "z_coord"
            ]
        ):

            return [
                float(joint["x_coord"]),
                float(joint["y_coord"]),
                float(joint["z_coord"])
            ]

        # Nested coordinate dictionary.
        for key in [
            "position",
            "coordinates",
            "coord",
            "xyz",
            "point"
        ]:

            if key in joint:

                nested = joint[key]

                converted = (
                    convert_joint_to_xyz(
                        nested
                    )
                )

                if converted is not None:

                    return converted

        return None

    # --------------------------------------------------------
    # List / tuple / ndarray
    # --------------------------------------------------------

    if isinstance(
        joint,
        (list, tuple, np.ndarray)
    ):

        try:

            arr = np.asarray(
                joint,
                dtype=np.float32
            ).reshape(-1)

            if arr.size >= 3:

                return [
                    float(arr[0]),
                    float(arr[1]),
                    float(arr[2])
                ]

        except Exception:

            return None

    return None


# ============================================================
# CONVERT ALL JOINTS
# ============================================================

def convert_joints_to_array(joints):

    if joints is None:
        return None

    # --------------------------------------------------------
    # Already a numerical array.
    # --------------------------------------------------------

    try:

        arr = np.asarray(
            joints,
            dtype=np.float32
        )

        if (
            arr.ndim == 2
            and arr.shape[1] == 3
        ):

            return arr

    except Exception:

        pass

    # --------------------------------------------------------
    # List of joint objects.
    # --------------------------------------------------------

    if isinstance(
        joints,
        (list, tuple)
    ):

        xyz_list = []

        for joint in joints:

            xyz = convert_joint_to_xyz(
                joint
            )

            if xyz is not None:

                xyz_list.append(xyz)

        if len(xyz_list) > 0:

            return np.asarray(
                xyz_list,
                dtype=np.float32
            )

    return None


# ============================================================
# PRINT NUMERICAL 3D JOINTS
# ============================================================

def print_3d_joints(
    joints,
    camera_translation,
    person_id
):

    print("\n")
    print("=" * 80)
    print(
        f"3D HUMAN JOINT RESULT "
        f"| PERSON {person_id}"
    )
    print("=" * 80)

    array = convert_joints_to_array(
        joints
    )

    if array is None:

        print(
            "[ERROR] Could not convert "
            "joints_3d to numerical XYZ."
        )

        print(
            "\nRaw structure will be inspected."
        )

        inspect_joint_structure(
            joints
        )

        return False

    print(
        "[SUCCESS] joints converted "
        "to numerical XYZ."
    )

    print(
        "Shape:",
        array.shape
    )

    print(
        "Joint count:",
        array.shape[0]
    )

    print(
        "\nFirst 5 joints:"
    )

    print(
        array[:5]
    )

    print(
        "\nXYZ minimum:"
    )

    print(
        np.min(
            array,
            axis=0
        )
    )

    print(
        "\nXYZ maximum:"
    )

    print(
        np.max(
            array,
            axis=0
        )
    )

    print(
        "\nXYZ mean:"
    )

    print(
        np.mean(
            array,
            axis=0
        )
    )

    print(
        "\nXYZ standard deviation:"
    )

    print(
        np.std(
            array,
            axis=0
        )
    )

    # --------------------------------------------------------
    # Camera translation
    # --------------------------------------------------------

    if camera_translation is not None:

        try:

            if isinstance(
                camera_translation,
                dict
            ):

                translation = np.asarray(
                    [
                        camera_translation["x"],
                        camera_translation["y"],
                        camera_translation["z"]
                    ],
                    dtype=np.float32
                )

            else:

                translation = np.asarray(
                    camera_translation,
                    dtype=np.float32
                )

            if translation.shape == (3,):

                print(
                    "\nHMR camera translation:"
                )

                print(
                    translation
                )

                # Diagnostic only.
                camera_joints = (
                    array + translation
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
                "[WARNING] "
                "Could not calculate camera-frame joints:",
                e
            )

    print("=" * 80)

    return True


# ============================================================
# HMR
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

        persons = result.get(
            "persons",
            []
        )

        print(
            "[HMR] Returned persons:",
            len(persons)
        )

        for person in persons:

            person_id = person.get(
                "person_id",
                0
            )

            print("\n" + "-" * 70)

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

            camera_translation = (
                person.get(
                    "camera_translation"
                )
            )

            print(
                "Camera translation:",
                camera_translation
            )

            # ------------------------------------------------
            # Pose
            # ------------------------------------------------

            pose = person.get(
                "pose"
            )

            if not isinstance(
                pose,
                dict
            ):

                print(
                    "[ERROR] pose is not a dictionary."
                )

                continue

            print(
                "Pose keys:",
                list(pose.keys())
            )

            joints = pose.get(
                "joints_3d"
            )

            print(
                "joints_3d Python type:",
                type(joints).__name__
            )

            # ------------------------------------------------
            # FIRST:
            # Print exact raw structure.
            # ------------------------------------------------

            inspect_joint_structure(
                joints
            )

            # ------------------------------------------------
            # SECOND:
            # Try conversion.
            # ------------------------------------------------

            success = print_3d_joints(
                joints,
                camera_translation,
                person_id
            )

            if success:

                print(
                    "\n[CHECKPOINT V5.1]"
                )

                print(
                    "[PASS] HMR 3D joints "
                    "are numerically accessible."
                )

            else:

                print(
                    "\n[CHECKPOINT V5.1]"
                )

                print(
                    "[INFO] HMR 3D structure "
                    "needs one more parser adjustment."
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
        "BAS-HMR V5.1",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "3D JOINT STRUCTURE",
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

        <title>BAS-HMR V5.1</title>

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

        <h1>BAS-HMR V5.1</h1>

        <h2>HMR 3D Joint Structure Diagnostic</h2>

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

            # ------------------------------------------------
            # IMPORTANT:
            # Never call cap.read() if cap is None.
            # ------------------------------------------------

            if cap is None:

                print(
                    "[CAMERA] Camera unavailable. "
                    "Trying to reopen..."
                )

                time.sleep(2)

                cap = open_camera()

                continue

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

                    try:
                        cap.release()
                    except Exception:
                        pass

                    cap = None

                    time.sleep(1)

                    cap = open_camera()

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

        if cap is not None:

            try:
                cap.release()
            except Exception:
                pass

        print(
            "[OK] Camera released."
        )

        print(
            "BAS-HMR V5.1 stopped."
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
