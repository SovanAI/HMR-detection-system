import cv2
import numpy as np
import os
import time
import threading
from flask import Flask, Response, render_template_string, jsonify


# ============================================================
# SETTINGS
# ============================================================

CAMERA_DEVICE = "/dev/video0"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

CHECKERBOARD = (9, 6)

# 25 mm checkerboard squares
SQUARE_SIZE = 0.025

CALIBRATION_DIR = "calibration/images"
OUTPUT_FILE = "calibration/camera_intrinsics.npz"

TARGET_SAMPLES = 20

os.makedirs(CALIBRATION_DIR, exist_ok=True)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

camera = None

latest_frame = None
latest_display = None

object_points = []
image_points = []

sample_count = 0
last_message = "Waiting for checkerboard..."

running = True

lock = threading.Lock()


# ============================================================
# CHECKERBOARD OBJECT POINTS
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_worker():

    global camera
    global latest_frame
    global latest_display
    global last_message
    global running

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not camera.isOpened():

        last_message = "ERROR: Camera could not be opened."

        print(last_message)

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        IMAGE_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        IMAGE_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        10
    )

    print("Camera opened successfully.")

    while running:

        ret, frame = camera.read()

        if not ret:

            time.sleep(0.05)

            continue

        with lock:

            latest_frame = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame.copy()

        if found:

            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )
            )

            cv2.drawChessboardCorners(
                display,
                CHECKERBOARD,
                corners_refined,
                found
            )

            cv2.putText(
                display,
                "CHECKERBOARD DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            corners_refined = None

            cv2.putText(
                display,
                "CHECKERBOARD NOT DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            display,
            f"Samples: {sample_count}/{TARGET_SAMPLES}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Open browser and press CAPTURE",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        with lock:

            latest_display = display.copy()

        time.sleep(0.01)


# ============================================================
# JPEG STREAM
# ============================================================

def generate_frames():

    while running:

        with lock:

            if latest_display is None:

                frame = np.zeros(
                    (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                    dtype=np.uint8
                )

                cv2.putText(
                    frame,
                    "Waiting for camera...",
                    (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            else:

                frame = latest_display.copy()

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

        time.sleep(0.05)


# ============================================================
# CAPTURE SAMPLE
# ============================================================

@app.route("/capture", methods=["POST"])
def capture():

    global sample_count
    global last_message

    with lock:

        if latest_frame is None:

            return jsonify({
                "success": False,
                "message": "No camera frame available."
            })

        frame = latest_frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:

        last_message = (
            "Checkerboard not detected. "
            "Move or rotate the board."
        )

        return jsonify({
            "success": False,
            "message": last_message,
            "samples": sample_count
        })

    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
    )

    object_points.append(
        objp.copy()
    )

    image_points.append(
        corners_refined.copy()
    )

    filename = os.path.join(
        CALIBRATION_DIR,
        f"calibration_{sample_count:02d}.jpg"
    )

    cv2.imwrite(
        filename,
        frame
    )

    sample_count += 1

    last_message = (
        f"Sample {sample_count} captured successfully."
    )

    print(
        f"[+] Sample {sample_count} captured"
    )

    return jsonify({
        "success": True,
        "message": last_message,
        "samples": sample_count
    })


# ============================================================
# CALIBRATE
# ============================================================

@app.route("/calibrate", methods=["POST"])
def calibrate():

    global last_message

    if sample_count < 10:

        return jsonify({
            "success": False,
            "message": (
                f"Need at least 10 samples. "
                f"Currently have {sample_count}."
            )
        })

    print()
    print("=" * 70)
    print("CALIBRATING CAMERA")
    print("=" * 70)

    image_size = (
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    )

    ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    # --------------------------------------------------------
    # Reprojection error
    # --------------------------------------------------------

    total_error = 0

    for i in range(len(object_points)):

        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            distortion
        )

        error = cv2.norm(
            image_points[i],
            projected_points,
            cv2.NORM_L2
        ) / len(projected_points)

        total_error += error

    mean_error = (
        total_error / len(object_points)
    )

    # --------------------------------------------------------
    # Intrinsics
    # --------------------------------------------------------

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(
        OUTPUT_FILE,

        camera_matrix=camera_matrix,

        distortion_coefficients=distortion,

        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,

        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,

        checkerboard_width=CHECKERBOARD[0],
        checkerboard_height=CHECKERBOARD[1],

        square_size=SQUARE_SIZE,

        reprojection_error=mean_error
    )

    print()
    print("=" * 70)
    print("CAMERA CALIBRATION COMPLETE")
    print("=" * 70)

    print()
    print("Camera Matrix:")
    print(camera_matrix)

    print()
    print("Distortion:")
    print(distortion.ravel())

    print()
    print("Intrinsic Parameters:")
    print(f"fx = {fx:.6f}")
    print(f"fy = {fy:.6f}")
    print(f"cx = {cx:.6f}")
    print(f"cy = {cy:.6f}")

    print()
    print(f"Samples: {sample_count}")
    print(
        f"Mean reprojection error: "
        f"{mean_error:.6f}"
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")

    print("=" * 70)

    last_message = (
        "Calibration complete! "
        f"Reprojection error = {mean_error:.6f}"
    )

    return jsonify({

        "success": True,

        "message": last_message,

        "samples": sample_count,

        "fx": float(fx),
        "fy": float(fy),

        "cx": float(cx),
        "cy": float(cy),

        "reprojection_error": float(mean_error),

        "output": OUTPUT_FILE
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    return jsonify({

        "camera": camera is not None
        and camera.isOpened(),

        "samples": sample_count,

        "target_samples": TARGET_SAMPLES,

        "message": last_message
    })


# ============================================================
# VIDEO
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>BAS Camera Calibration</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

.container {
    width: 850px;
    margin: auto;
}

img {
    width: 640px;
    height: 480px;
    border: 2px solid #555;
}

button {
    padding: 14px 25px;
    margin: 10px;
    font-size: 16px;
    cursor: pointer;
}

#status {
    font-size: 18px;
    margin: 15px;
}

</style>

</head>

<body>

<div class="container">

<h1>BAS Webcam Camera Calibration</h1>

<p>
Checkerboard:
<strong>9 × 6 inner corners</strong>
</p>

<img src="/video">

<div>

<button onclick="captureSample()">
CAPTURE SAMPLE
</button>

<button onclick="calibrateCamera()">
CALIBRATE CAMERA
</button>

</div>

<div id="status">
Loading...
</div>

</div>


<script>

async function captureSample() {

    const response = await fetch(
        "/capture",
        {method: "POST"}
    );

    const data = await response.json();

    document.getElementById("status").innerText =
        data.message +
        " | Samples: " +
        data.samples;
}


async function calibrateCamera() {

    document.getElementById("status").innerText =
        "Calibrating... please wait.";

    const response = await fetch(
        "/calibrate",
        {method: "POST"}
    );

    const data = await response.json();

    if (data.success) {

        document.getElementById("status").innerText =
            data.message +
            " | fx=" + data.fx.toFixed(3) +
            " fy=" + data.fy.toFixed(3) +
            " cx=" + data.cx.toFixed(3) +
            " cy=" + data.cy.toFixed(3);

    } else {

        document.getElementById("status").innerText =
            data.message;
    }
}


async function updateStatus() {

    const response = await fetch(
        "/status"
    );

    const data = await response.json();

    document.getElementById("status").innerText =
        data.message +
        " | Samples: " +
        data.samples +
        "/" +
        data.target_samples;
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
# INDEX
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    thread.start()

    print()
    print("=" * 70)
    print("BAS CAMERA CALIBRATION SERVER")
    print("=" * 70)

    print()
    print("Open this in Windows Chrome:")
    print()
    print("http://localhost:5000")
    print()

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True
        )

    finally:

        running = False

        if camera is not None:

            camera.release()

        print("Camera released.")
