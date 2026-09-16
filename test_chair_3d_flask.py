import cv2
import time
import threading
import numpy as np

from flask import Flask, Response, jsonify, render_template_string

from ultralytics import YOLO

from human.distance.grid.depth_to_3d import (
    CameraIntrinsics,
    DepthTo3D,
)

from human.distance.grid.object_grid import (
    BoundingBox,
    ObjectGridExtractor,
)

from human.distance.grid.feature_extractor import (
    FeatureExtractor,
)


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "/dev/video0"

YOLO_MODEL = "yolo11n.pt"
DEPTH_MODEL = "yolo26n-depth.pt"

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

DEPTH_IMGSZ = 640

PERSON_CLASS = 0
CHAIR_CLASS = 56

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

DEPTH_INTERVAL = 3

GRID_ROWS = 5
GRID_COLS = 5

DEPTH_RADIUS = 2
BORDER_RATIO = 0.10

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5009


# ============================================================
# CAMERA INTRINSICS
# ============================================================

INTRINSICS = CameraIntrinsics(
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.Lock()

latest_jpeg = None

latest_depth_map = None

latest_chairs = []

latest_chair_grids = []

latest_chair_features = []

frame_id = 0

running = True


status = {
    "camera": False,
    "yolo": False,
    "depth": False,

    "frame_id": 0,

    "chairs": 0,

    "valid_3d_chairs": 0,

    "fps": 0.0,

    "depth_status": "WAITING",

    "last_depth_frame": -1,
}


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("Loading YOLO11...")
print("=" * 70)

yolo_model = YOLO(
    YOLO_MODEL
)

status["yolo"] = True

print("YOLO11 loaded successfully.")


# ============================================================
# LOAD YOLO26 DEPTH
# ============================================================

print("=" * 70)
print("Loading YOLO26n Depth...")
print("=" * 70)

depth_model = YOLO(
    DEPTH_MODEL
)

status["depth"] = True

print("YOLO26n Depth loaded successfully.")


# ============================================================
# DEPTH → 3D
# ============================================================

depth_converter = DepthTo3D(
    intrinsics=INTRINSICS,
    min_depth=0.05,
    max_depth=20.0,
)


# ============================================================
# OBJECT GRID
# ============================================================

grid_extractor = ObjectGridExtractor(
    depth_converter=depth_converter,
    rows=GRID_ROWS,
    cols=GRID_COLS,
    border_ratio=BORDER_RATIO,
    depth_radius=DEPTH_RADIUS,
)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_extractor = FeatureExtractor()


# ============================================================
# CAMERA
# ============================================================

print("=" * 70)
print("Opening webcam...")
print("=" * 70)

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2,
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG"),
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT,
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS,
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open /dev/video0"
    )


status["camera"] = True

print("Camera opened successfully.")


# ============================================================
# DEPTH EXTRACTION
# ============================================================

def calculate_depth(frame):

    """
    Run YOLO26 metric depth.

    Returns:
        depth_map in metres.
    """

    results = depth_model.predict(
        source=frame,
        imgsz=DEPTH_IMGSZ,
        device="cpu",
        verbose=False,
    )

    if not results:

        return None

    result = results[0]

    if not hasattr(result, "depth"):

        return None

    if result.depth is None:

        return None

    depth_tensor = result.depth.data

    depth = (
        depth_tensor
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Remove singleton dimensions
    # --------------------------------------------------------

    depth = np.squeeze(
        depth
    )

    if depth.ndim != 2:

        return None

    # --------------------------------------------------------
    # Resize to camera resolution
    # --------------------------------------------------------

    if (
        depth.shape[1] != CAMERA_WIDTH
        or depth.shape[0] != CAMERA_HEIGHT
    ):

        depth = cv2.resize(
            depth,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    # --------------------------------------------------------
    # Invalid depth → NaN
    # --------------------------------------------------------

    depth = depth.astype(
        np.float32
    )

    invalid = (
        ~np.isfinite(depth)
        | (depth <= 0)
        | (depth < 0.05)
        | (depth > 20.0)
    )

    depth[invalid] = np.nan

    return depth


# ============================================================
# CHAIR 3D GRID
# ============================================================

def build_chair_grid(
    depth_map,
    chair,
):

    bbox = BoundingBox(

        x1=float(
            chair["x1"]
        ),

        y1=float(
            chair["y1"]
        ),

        x2=float(
            chair["x2"]
        ),

        y2=float(
            chair["y2"]
        ),

    )

    grid = grid_extractor.extract(

        depth_map=depth_map,

        bbox=bbox,

        object_id=int(
            chair["id"]
        ),

        object_class="chair",

    )

    return grid


# ============================================================
# DRAW CHAIR 3D GRID
# ============================================================

def draw_chair_grid(
    frame,
    grid,
):

    if grid is None:

        return frame


    output = frame


    # --------------------------------------------------------
    # Draw all valid grid points
    # --------------------------------------------------------

    for point in grid.points:

        x = int(
            point.u
        )

        y = int(
            point.v
        )

        cv2.circle(

            output,

            (x, y),

            4,

            (255, 255, 0),

            -1,

        )


    # --------------------------------------------------------
    # Draw grid rows
    # --------------------------------------------------------

    points = grid.points

    for i in range(
        len(points) - 1
    ):

        p1 = points[i]

        p2 = points[i + 1]

        # Only connect points from same row
        if (
            p1.row == p2.row
            and p2.col == p1.col + 1
        ):

            cv2.line(

                output,

                (
                    int(p1.u),
                    int(p1.v)
                ),

                (
                    int(p2.u),
                    int(p2.v)
                ),

                (255, 180, 0),

                1,

            )


    # --------------------------------------------------------
    # Draw grid columns
    # --------------------------------------------------------

    for i in range(
        len(points)
    ):

        current = points[i]

        for j in range(
            i + 1,
            len(points)
        ):

            other = points[j]

            if (
                current.col == other.col
                and other.row
                == current.row + 1
            ):

                cv2.line(

                    output,

                    (
                        int(current.u),
                        int(current.v)
                    ),

                    (
                        int(other.u),
                        int(other.v)
                    ),

                    (255, 180, 0),

                    1,

                )

                break


    # --------------------------------------------------------
    # Draw 3D centroid projection
    # --------------------------------------------------------

    centroid = grid.xyz_array()

    if len(centroid) > 0:

        center_3d = np.mean(
            centroid,
            axis=0
        )

        z = center_3d[2]

        if (
            np.isfinite(z)
            and z > 0.05
        ):

            u = (
                INTRINSICS.fx
                * center_3d[0]
                / z
                + INTRINSICS.cx
            )

            v = (
                INTRINSICS.fy
                * center_3d[1]
                / z
                + INTRINSICS.cy
            )

            if (
                0 <= u < frame.shape[1]
                and 0 <= v < frame.shape[0]
            ):

                cv2.circle(

                    output,

                    (
                        int(u),
                        int(v)
                    ),

                    7,

                    (0, 0, 255),

                    -1,

                )


    return output


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global latest_jpeg
    global latest_depth_map
    global latest_chairs
    global latest_chair_grids
    global latest_chair_features
    global frame_id


    start_time = time.time()

    local_count = 0


    while running:

        ret, frame = cap.read()


        if not ret or frame is None:

            print(
                "\n[CAMERA] "
                "Frame capture failed"
            )

            time.sleep(
                0.1
            )

            continue


        frame_id += 1

        local_count += 1


        # ====================================================
        # YOLO11 DETECTION
        # ====================================================

        try:

            results = yolo_model.predict(

                source=frame,

                conf=YOLO_CONF,

                imgsz=YOLO_IMGSZ,

                device="cpu",

                classes=[
                    PERSON_CLASS,
                    CHAIR_CLASS,
                ],

                verbose=False,

            )

        except Exception as e:

            print(
                f"\n[YOLO ERROR] {e}"
            )

            continue


        result = results[0]


        chairs = []


        if result.boxes is not None:

            chair_index = 0


            for box in result.boxes:

                cls = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )


                if cls != CHAIR_CLASS:

                    continue


                x1, y1, x2, y2 = map(

                    float,

                    box.xyxy[0].tolist()

                )


                chairs.append({

                    "id": chair_index,

                    "x1": x1,

                    "y1": y1,

                    "x2": x2,

                    "y2": y2,

                    "confidence": confidence,

                })


                chair_index += 1


        # ====================================================
        # DEPTH
        # ====================================================

        depth_map = None


        if (
            frame_id % DEPTH_INTERVAL == 0
            or latest_depth_map is None
        ):

            try:

                start_depth = time.time()

                depth_map = calculate_depth(
                    frame
                )

                depth_time = (
                    time.time()
                    - start_depth
                )


                if depth_map is not None:

                    status[
                        "depth_status"
                    ] = (
                        f"ACTIVE "
                        f"{depth_time:.2f}s"
                    )

                    status[
                        "last_depth_frame"
                    ] = frame_id


            except Exception as e:

                print(
                    f"\n[DEPTH ERROR] {e}"
                )

                depth_map = None


        else:

            with state_lock:

                if latest_depth_map is not None:

                    depth_map = (
                        latest_depth_map.copy()
                    )


        # ====================================================
        # CHAIR 3D
        # ====================================================

        chair_grids = []

        chair_features = []


        if depth_map is not None:

            for chair in chairs:

                try:

                    grid = build_chair_grid(
                        depth_map,
                        chair
                    )


                    if grid is None:

                        continue


                    features = (
                        feature_extractor.extract(
                            grid
                        )
                    )


                    chair_grids.append(
                        grid
                    )

                    chair_features.append(
                        features
                    )


                except Exception as e:

                    print(
                        f"\n[GRID ERROR] "
                        f"Chair {chair['id']}: "
                        f"{e}"
                    )


        # ====================================================
        # DRAW
        # ====================================================

        output = frame.copy()


        # ----------------------------------------------------
        # Chair detections
        # ----------------------------------------------------

        for chair in chairs:

            x1 = int(
                chair["x1"]
            )

            y1 = int(
                chair["y1"]
            )

            x2 = int(
                chair["x2"]
            )

            y2 = int(
                chair["y2"]
            )


            cv2.rectangle(

                output,

                (x1, y1),

                (x2, y2),

                (255, 255, 0),

                2,

            )


            cv2.putText(

                output,

                (
                    f"CHAIR "
                    f"{chair['id'] + 1} "
                    f"{chair['confidence']:.2f}"
                ),

                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 0),

                2,

            )


        # ----------------------------------------------------
        # Chair grids
        # ----------------------------------------------------

        for grid in chair_grids:

            output = draw_chair_grid(
                output,
                grid
            )


        # ----------------------------------------------------
        # Chair feature information
        # ----------------------------------------------------

        for index, features in enumerate(
            chair_features
        ):

            if index >= len(
                chairs
            ):

                break


            chair = chairs[index]


            x = int(
                chair["x1"]
            )

            y = int(
                chair["y2"]
            ) + 20


            centroid_x = features.get(
                "centroid_x",
                float("nan")
            )

            centroid_y = features.get(
                "centroid_y",
                float("nan")
            )

            centroid_z = features.get(
                "centroid_z",
                float("nan")
            )

            width_3d = features.get(
                "width_3d",
                float("nan")
            )

            height_3d = features.get(
                "height_3d",
                float("nan")
            )

            depth_3d = features.get(
                "depth_3d",
                float("nan")
            )

            valid_ratio = features.get(
                "valid_ratio",
                0.0
            )


            lines = [

                (
                    f"3D "
                    f"X:{centroid_x:.2f}m "
                    f"Y:{centroid_y:.2f}m "
                    f"Z:{centroid_z:.2f}m"
                ),

                (
                    f"SIZE "
                    f"W:{width_3d:.2f}m "
                    f"H:{height_3d:.2f}m "
                    f"D:{depth_3d:.2f}m"
                ),

                (
                    f"VALID "
                    f"{valid_ratio * 100:.0f}%"
                ),

            ]


            for line_index, text in enumerate(
                lines
            ):

                cv2.putText(

                    output,

                    text,

                    (
                        x,
                        y + line_index * 18
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.43,

                    (255, 255, 255),

                    1,

                    cv2.LINE_AA,

                )


        # ====================================================
        # FPS
        # ====================================================

        elapsed = (
            time.time()
            - start_time
        )


        fps = (

            local_count
            / elapsed

            if elapsed > 0

            else 0.0

        )


        # ====================================================
        # STATUS PANEL
        # ====================================================

        cv2.rectangle(

            output,

            (0, 0),

            (640, 85),

            (0, 0, 0),

            -1,

        )


        cv2.putText(

            output,

            f"CHAIRS: {len(chairs)}",

            (10, 22),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 0),

            2,

        )


        cv2.putText(

            output,

            (
                f"3D CHAIRS: "
                f"{len(chair_grids)}"
            ),

            (150, 22),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 0),

            2,

        )


        cv2.putText(

            output,

            f"FPS: {fps:.2f}",

            (310, 22),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2,

        )


        cv2.putText(

            output,

            (
                "DEPTH: "
                f"{status['depth_status']}"
            ),

            (10, 48),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (255, 255, 255),

            1,

        )


        cv2.putText(

            output,

            "YOLO26 3D CHAIR",

            (390, 48),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (255, 255, 255),

            1,

        )


        cv2.putText(

            output,

            (
                f"FRAME: {frame_id}"
            ),

            (10, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (180, 180, 180),

            1,

        )


        # ====================================================
        # JPEG
        # ====================================================

        success, encoded = cv2.imencode(

            ".jpg",

            output,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                80,
            ],

        )


        if success:

            with state_lock:

                latest_jpeg = (
                    encoded.tobytes()
                )

                if depth_map is not None:

                    latest_depth_map = (
                        depth_map.copy()
                    )

                latest_chairs = (
                    chairs.copy()
                )

                latest_chair_grids = (
                    chair_grids.copy()
                )

                latest_chair_features = (
                    chair_features.copy()
                )


        # ====================================================
        # STATUS
        # ====================================================

        with state_lock:

            status["frame_id"] = (
                frame_id
            )

            status["chairs"] = (
                len(chairs)
            )

            status["valid_3d_chairs"] = (
                len(chair_grids)
            )

            status["fps"] = round(
                fps,
                2
            )


        print(

            f"\rFrame: {frame_id:05d} | "

            f"Chairs: {len(chairs)} | "

            f"3D: {len(chair_grids)} | "

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

<title>BAS-HMR | Chair 3D</title>

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

    margin: 10px 20px;

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

YOLO11 + YOLO26 Metric Depth + Chair 3D

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

YOLO:

<span

    class="value"

    id="yolo">

...

</span>

</div>

<div class="stat">

Depth:

<span

    class="value"

    id="depth">

...

</span>

</div>

<div class="stat">

Chairs:

<span

    class="value"

    id="chairs">

...

</span>

</div>

<div class="stat">

3D Chairs:

<span

    class="value"

    id="chairs3d">

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
            "yolo"
        ).textContent =
            data.yolo;

        document.getElementById(
            "depth"
        ).textContent =
            data.depth_status;

        document.getElementById(
            "chairs"
        ).textContent =
            data.chairs;

        document.getElementById(
            "chairs3d"
        ).textContent =
            data.valid_3d_chairs;

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

        "depth": status["depth"],

        "model": (
            "YOLO11 + YOLO26 "
            "Metric Depth"
        ),

        "device": "cpu",

    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("BAS-HMR CHAIR 3D VISUALIZATION")
    print("=" * 70)

    print()
    print("Browser:")
    print()
    print("http://localhost:5009")

    print()
    print("API:")
    print()
    print("http://localhost:5009/api/status")

    print()
    print("=" * 70)


    camera_thread = threading.Thread(

        target=camera_loop,

        daemon=True,

    )

    camera_thread.start()


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
