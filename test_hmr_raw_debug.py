import cv2
import json
import time
import threading
import numpy as np

from flask import Flask, Response, jsonify
from ultralytics import YOLO

from human.hmr.hmr_processor import HMRProcessor


# ============================================================
# CONFIG
# ============================================================

CAMERA_INDEX = 0

WIDTH = 640
HEIGHT = 480
FPS = 10

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# HMR is CPU intensive.
HMR_INTERVAL = 30

PORT = 5012


# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

camera = None
running = True

latest_frame = None
latest_raw_hmr = None

frame_counter = 0
last_hmr_frame = -1

frame_lock = threading.Lock()
hmr_lock = threading.Lock()


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("BAS-HMR RAW HMR2 OUTPUT DIAGNOSTIC")
print("=" * 80)


# ============================================================
# YOLO
# ============================================================

print()
print("[1/3] Loading YOLO11n...")

yolo = YOLO("yolo11n.pt")

print("[OK] YOLO11n loaded.")
print("[OK] Device: CPU")


# ============================================================
# HMR2
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

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    FPS
)

camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)


if not camera.isOpened():
    raise RuntimeError(
        "Could not open /dev/video0"
    )


print("[OK] Camera opened.")
print(f"[OK] Resolution: {WIDTH}x{HEIGHT}")
print(f"[OK] FPS: {FPS}")


# ============================================================
# SAFE TYPE DESCRIPTION
# ============================================================

def describe_value(value, name="value", depth=0):
    """
    Recursively describe an HMR object without assuming
    its structure.

    IMPORTANT:
    This function does NOT convert dictionaries to float.
    It only reports their structure.
    """

    indent = "  " * depth

    if depth > 5:
        print(
            f"{indent}{name}: <maximum inspection depth>"
        )
        return

    # --------------------------------------------------------
    # None
    # --------------------------------------------------------

    if value is None:

        print(
            f"{indent}{name}: None"
        )

        return

    # --------------------------------------------------------
    # Python dictionary
    # --------------------------------------------------------

    if isinstance(value, dict):

        print(
            f"{indent}{name}: DICT "
            f"({len(value)} keys)"
        )

        for key, item in value.items():

            print(
                f"{indent}  KEY: {repr(key)}"
            )

            describe_value(
                item,
                f"[{key}]",
                depth + 1
            )

        return

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(value, list):

        print(
            f"{indent}{name}: LIST "
            f"(length={len(value)})"
        )

        if len(value) == 0:
            return

        # Only inspect first few entries
        for i, item in enumerate(value[:3]):

            describe_value(
                item,
                f"[{i}]",
                depth + 1
            )

        if len(value) > 3:

            print(
                f"{indent}  ... "
                f"{len(value) - 3} more items"
            )

        return

    # --------------------------------------------------------
    # Tuple
    # --------------------------------------------------------

    if isinstance(value, tuple):

        print(
            f"{indent}{name}: TUPLE "
            f"(length={len(value)})"
        )

        for i, item in enumerate(value[:3]):

            describe_value(
                item,
                f"[{i}]",
                depth + 1
            )

        return

    # --------------------------------------------------------
    # Numpy
    # --------------------------------------------------------

    if isinstance(value, np.ndarray):

        print(
            f"{indent}{name}: NUMPY ARRAY"
        )

        print(
            f"{indent}  dtype={value.dtype}"
        )

        print(
            f"{indent}  shape={value.shape}"
        )

        print(
            f"{indent}  ndim={value.ndim}"
        )

        if value.size > 0:

            flat = value.reshape(-1)

            print(
                f"{indent}  first values="
                f"{flat[:10]}"
            )

        return

    # --------------------------------------------------------
    # Torch tensor
    # --------------------------------------------------------

    if hasattr(value, "detach") and hasattr(
        value,
        "cpu"
    ):

        print(
            f"{indent}{name}: TORCH TENSOR"
        )

        try:

            print(
                f"{indent}  shape={tuple(value.shape)}"
            )

        except Exception:
            pass

        try:

            print(
                f"{indent}  dtype={value.dtype}"
            )

        except Exception:
            pass

        try:

            print(
                f"{indent}  device={value.device}"
            )

        except Exception:
            pass

        try:

            arr = (
                value.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            print(
                f"{indent}  first values="
                f"{arr[:10]}"
            )

        except Exception as e:

            print(
                f"{indent}  tensor preview failed: "
                f"{e}"
            )

        return

    # --------------------------------------------------------
    # Number
    # --------------------------------------------------------

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):

        print(
            f"{indent}{name}: NUMBER "
            f"value={value}"
        )

        return

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(value, str):

        print(
            f"{indent}{name}: STRING "
            f"value={repr(value)}"
        )

        return

    # --------------------------------------------------------
    # Other object
    # --------------------------------------------------------

    print(
        f"{indent}{name}: "
        f"{type(value).__name__}"
    )

    print(
        f"{indent}  repr="
        f"{repr(value)[:500]}"
    )


# ============================================================
# RAW JSON-SAFE CONVERSION
# ============================================================

def make_json_safe(value, depth=0):
    """
    Convert arbitrary HMR output into something Flask
    can serialize.

    This is intentionally conservative.
    """

    if depth > 8:
        return "<max-depth>"

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool)
    ):
        return value

    if isinstance(
        value,
        (np.integer, np.floating)
    ):
        return value.item()

    if isinstance(value, np.ndarray):

        return {
            "__type__": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "data_preview": value.reshape(-1)[:20].tolist()
        }

    if hasattr(value, "detach") and hasattr(
        value,
        "cpu"
    ):

        try:

            tensor = (
                value.detach()
                .cpu()
            )

            return {
                "__type__": "torch.Tensor",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(value.device),
                "data_preview": (
                    tensor
                    .reshape(-1)[:20]
                    .numpy()
                    .tolist()
                )
            }

        except Exception:

            return {
                "__type__": "torch.Tensor"
            }

    if isinstance(value, dict):

        result = {}

        for key, item in value.items():

            result[str(key)] = make_json_safe(
                item,
                depth + 1
            )

        return result

    if isinstance(value, (list, tuple)):

        return [
            make_json_safe(
                item,
                depth + 1
            )
            for item in value[:20]
        ]

    return {
        "__type__": type(value).__name__,
        "repr": repr(value)[:500]
    }


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_persons(frame):

    results = yolo.predict(
        source=frame,
        classes=[0],
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

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = (
            xyxy.astype(int)
        )

        confidence = float(
            box.conf[0]
            .cpu()
            .numpy()
        )

        boxes.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": confidence
            }
        )

    return boxes


# ============================================================
# RAW HMR INSPECTION
# ============================================================

def inspect_hmr_result(
    result,
    frame_id
):

    print()
    print()
    print("#" * 80)
    print(
        f"RAW HMR INSPECTION - FRAME {frame_id}"
    )
    print("#" * 80)

    if result is None:

        print(
            "[RAW] HMR returned None."
        )

        return

    print()
    print(
        "[RAW] TOP LEVEL TYPE:"
    )

    print(
        " ",
        type(result)
    )

    print()
    print(
        "[RAW] COMPLETE TOP-LEVEL STRUCTURE:"
    )

    describe_value(
        result,
        "HMR_RESULT"
    )

    print()
    print(
        "#" * 80
    )
    print(
        "IMPORTANT: No coordinate conversion has "
        "been performed."
    )
    print(
        "#" * 80
    )
    print()


# ============================================================
# DRAW
# ============================================================

def draw_frame(
    frame,
    person_boxes,
    hmr_result
):

    output = frame.copy()

    # Header
    cv2.rectangle(
        output,
        (0, 0),
        (WIDTH, 70),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output,
        "BAS-HMR RAW 3D DIAGNOSTIC",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "YOLO11n + HMR2 | RAW OUTPUT MODE",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # YOLO boxes
    # --------------------------------------------------------

    for i, box in enumerate(person_boxes):

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"YOLO Person {i + 1}",
            (x1, max(85, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # --------------------------------------------------------
    # HMR status
    # --------------------------------------------------------

    hmr_count = 0

    if hmr_result is not None:

        if isinstance(
            hmr_result,
            dict
        ):

            persons = hmr_result.get(
                "persons",
                []
            )

            if isinstance(persons, list):
                hmr_count = len(persons)

    cv2.putText(
        output,
        f"HMR persons: {hmr_count}",
        (10, HEIGHT - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Frame: {frame_counter}",
        (10, HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    return output


# ============================================================
# MJPEG
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
# WEB PAGE
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>BAS-HMR Raw HMR Diagnostic</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial;
                margin: 0;
                padding: 20px;
                text-align: center;
            }

            img {
                width: 640px;
                max-width: 95vw;
                border: 2px solid white;
            }

            .panel {
                max-width: 1000px;
                margin: 20px auto;
                padding: 20px;
                background: #222;
                text-align: left;
            }

            pre {
                white-space: pre-wrap;
                word-wrap: break-word;
            }

        </style>

    </head>

    <body>

        <h1>BAS-HMR Raw HMR2 Diagnostic</h1>

        <img src="/video">

        <div class="panel">

            <h2>Raw HMR Structure</h2>

            <pre id="data">
Waiting for HMR result...
            </pre>

        </div>

        <script>

        async function updateData() {

            try {

                const response =
                    await fetch("/api/raw-hmr");

                const data =
                    await response.json();

                document.getElementById(
                    "data"
                ).textContent =
                    JSON.stringify(
                        data,
                        null,
                        2
                    );

            }
            catch(error) {

                document.getElementById(
                    "data"
                ).textContent =
                    error.toString();

            }

        }

        setInterval(
            updateData,
            1000
        );

        updateData();

        </script>

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
        )
    )


# ============================================================
# API
# ============================================================

@app.route("/api/raw-hmr")
def raw_hmr_api():

    with hmr_lock:

        if latest_raw_hmr is None:

            return jsonify(
                {
                    "status": "waiting",
                    "message":
                        "Waiting for HMR..."
                }
            )

        return jsonify(
            {
                "status": "ok",
                "raw_hmr":
                    make_json_safe(
                        latest_raw_hmr
                    )
            }
        )


# ============================================================
# PROCESSING LOOP
# ============================================================

def processing_loop():

    global frame_counter
    global latest_frame
    global latest_raw_hmr
    global last_hmr_frame

    print()
    print("=" * 80)
    print("PROCESSING LOOP STARTED")
    print("=" * 80)

    print()
    print(
        f"Open browser:"
    )

    print(
        f"http://localhost:{PORT}"
    )

    print()

    print(
        f"Raw API:"
    )

    print(
        f"http://localhost:{PORT}/api/raw-hmr"
    )

    print()

    while running:

        ret, frame = camera.read()

        if not ret:

            print(
                "[CAMERA ERROR] "
                "Could not read frame."
            )

            time.sleep(0.1)

            continue

        frame_counter += 1

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        person_boxes = detect_persons(
            frame
        )

        # ----------------------------------------------------
        # HMR
        # ----------------------------------------------------

        if (
            last_hmr_frame < 0
            or
            frame_counter - last_hmr_frame
            >= HMR_INTERVAL
        ):

            print()
            print(
                f"[HMR] Processing frame "
                f"{frame_counter}"
            )

            print(
                f"[HMR] YOLO persons: "
                f"{len(person_boxes)}"
            )

            hmr_start = time.time()

            try:

                result = (
                    hmr_processor.process_frame(
                        frame,
                        frame_counter,
                        time.time(),
                        person_boxes
                    )
                )

                elapsed = (
                    time.time()
                    - hmr_start
                )

                print(
                    f"[HMR] Finished in "
                    f"{elapsed:.2f} seconds"
                )

                # IMPORTANT:
                # Save RAW result BEFORE
                # doing any interpretation.
                with hmr_lock:

                    latest_raw_hmr = result

                last_hmr_frame = (
                    frame_counter
                )

                # ------------------------------------------------
                # Inspect raw structure
                # ------------------------------------------------

                inspect_hmr_result(
                    result,
                    frame_counter
                )

            except Exception as e:

                print()
                print(
                    "=" * 80
                )

                print(
                    "[HMR ERROR]"
                )

                print(
                    "TYPE:",
                    type(e)
                )

                print(
                    "MESSAGE:",
                    repr(e)
                )

                print(
                    "=" * 80
                )

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        with hmr_lock:

            cached_hmr = (
                latest_raw_hmr
            )

        display = draw_frame(
            frame,
            person_boxes,
            cached_hmr
        )

        with frame_lock:

            latest_frame = display


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    worker = threading.Thread(
        target=processing_loop,
        daemon=True
    )

    worker.start()

    try:

        app.run(
            host="0.0.0.0",
            port=PORT,
            threaded=True
        )

    finally:

        running = False

        if camera is not None:

            camera.release()
