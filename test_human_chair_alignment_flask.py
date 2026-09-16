import cv2
import time
import threading
import numpy as np

from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
)

from ultralytics import YOLO

# ============================================================
# PROJECT HMR PROCESSOR
# ============================================================

from human.hmr.hmr_processor import HMRProcessor

# ============================================================
# 3D DEPTH / GRID
# ============================================================

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

CAMERA_DEVICE = "/dev/video0"

YOLO_MODEL_PATH = "yolo11n.pt"
DEPTH_MODEL_PATH = "yolo26n-depth.pt"

YOLO_CONF = 0.25
YOLO_IMGSZ = 640

DEPTH_IMGSZ = 640

PERSON_CLASS = 0
CHAIR_CLASS = 56

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

# CPU optimization
DEPTH_INTERVAL = 3
HMR_INTERVAL = 30

# Chair 3D grid
GRID_ROWS = 5
GRID_COLS = 5
DEPTH_RADIUS = 2
BORDER_RATIO = 0.10

# Flask
HOST = "0.0.0.0"
PORT = 5010


# ============================================================
# DEVELOPMENT CAMERA INTRINSICS
# ============================================================
#
# These are development values.
# They are NOT a final calibrated camera model.
#
# fx/fy = focal length
# cx/cy = principal point
#
# ============================================================

INTRINSICS = CameraIntrinsics(
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
)


# ============================================================
# HMR → DEPTH COORDINATE TRANSFORM
# ============================================================
#
# IMPORTANT
# ------------------------------------------------------------
# We are NOT claiming this is the final calibrated transform.
#
# Initially:
#
#       P_depth = R @ P_hmr + T
#
# where:
#
#       R = identity
#       T = zero
#
# This allows us to inspect both systems safely.
#
# Later, once we determine the actual coordinate-frame
# transformation, ONLY these values need to change.
#
# ============================================================

HMR_TO_DEPTH_ROTATION = np.eye(
    3,
    dtype=np.float32,
)

HMR_TO_DEPTH_TRANSLATION = np.zeros(
    3,
    dtype=np.float32,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# THREAD-SAFE GLOBAL STATE
# ============================================================

state_lock = threading.Lock()

running = True

latest_jpeg = None

latest_depth = None

latest_people = []

latest_chairs = []

latest_hmr_people = []

latest_chair_features = []

latest_alignment = []

frame_id = 0


# ============================================================
# STATUS
# ============================================================

status = {

    "camera": False,

    "yolo": False,

    "depth": False,

    "hmr": False,

    "frame": 0,

    "people": 0,

    "chairs": 0,

    "hmr_people": 0,

    "3d_chairs": 0,

    "depth_status": "WAITING",

    "hmr_status": "WAITING",

    "fps": 0.0,

    "error": "",

}


# ============================================================
# LOAD YOLO11
# ============================================================

print()
print("=" * 70)
print("LOADING YOLO11")
print("=" * 70)

try:

    yolo_model = YOLO(
        YOLO_MODEL_PATH
    )

    status["yolo"] = True

    print("[OK] YOLO11 loaded.")

except Exception as e:

    yolo_model = None

    status["error"] = (
        f"YOLO11 load error: {e}"
    )

    print(
        f"[ERROR] YOLO11: {e}"
    )


# ============================================================
# LOAD YOLO26 DEPTH
# ============================================================

print()
print("=" * 70)
print("LOADING YOLO26n DEPTH")
print("=" * 70)

try:

    depth_model = YOLO(
        DEPTH_MODEL_PATH
    )

    status["depth"] = True

    print("[OK] YOLO26n Depth loaded.")

except Exception as e:

    depth_model = None

    status["error"] = (
        f"Depth model load error: {e}"
    )

    print(
        f"[ERROR] YOLO26 Depth: {e}"
    )


# ============================================================
# LOAD HMR2 THROUGH PROJECT PROCESSOR
# ============================================================

print()
print("=" * 70)
print("LOADING HMR2")
print("=" * 70)

try:

    hmr_processor = HMRProcessor()

    status["hmr"] = True

    print("[OK] HMRProcessor initialized.")

except Exception as e:

    hmr_processor = None

    status["error"] = (
        f"HMR initialization error: {e}"
    )

    print(
        f"[ERROR] HMR2: {e}"
    )


# ============================================================
# DEPTH → 3D
# ============================================================

depth_converter = DepthTo3D(

    intrinsics=INTRINSICS,

    min_depth=0.05,

    max_depth=20.0,

)


# ============================================================
# CHAIR GRID EXTRACTOR
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
# CAMERA INITIALIZATION
# ============================================================

print()
print("=" * 70)
print("OPENING CAMERA")
print("=" * 70)

cap = cv2.VideoCapture(

    CAMERA_DEVICE,

    cv2.CAP_V4L2,

)


if cap.isOpened():

    cap.set(

        cv2.CAP_PROP_FOURCC,

        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),

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

    print(
        "[ERROR] Cannot open /dev/video0"
    )

else:

    status["camera"] = True

    print(
        "[OK] Camera opened."
    )


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return default


# ============================================================
# SAFE VECTOR
# ============================================================

def safe_vector(
    value,
    length=3,
):

    try:

        arr = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        if len(arr) >= length:

            arr = arr[:length]

            if np.isfinite(arr).all():

                return arr

    except Exception:

        pass

    return np.zeros(
        length,
        dtype=np.float32,
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_objects(frame):

    people = []

    chairs = []


    if yolo_model is None:

        return people, chairs


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

        return people, chairs


    if not results:

        return people, chairs


    result = results[0]


    if result.boxes is None:

        return people, chairs


    person_id = 0

    chair_id = 0


    for box in result.boxes:

        try:

            cls = int(
                box.cls[0]
            )

            confidence = safe_float(
                box.conf[0]
            )

            coords = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            if len(coords) < 4:

                continue


            x1, y1, x2, y2 = (
                map(
                    float,
                    coords[:4],
                )
            )


            # Clip coordinates
            x1 = max(
                0.0,
                min(
                    x1,
                    CAMERA_WIDTH - 1,
                ),
            )

            y1 = max(
                0.0,
                min(
                    y1,
                    CAMERA_HEIGHT - 1,
                ),
            )

            x2 = max(
                0.0,
                min(
                    x2,
                    CAMERA_WIDTH - 1,
                ),
            )

            y2 = max(
                0.0,
                min(
                    y2,
                    CAMERA_HEIGHT - 1,
                ),
            )


            if x2 <= x1 or y2 <= y1:

                continue


            if cls == PERSON_CLASS:

                people.append({

                    "id": person_id,

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "confidence": confidence,

                })

                person_id += 1


            elif cls == CHAIR_CLASS:

                chairs.append({

                    "id": chair_id,

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "confidence": confidence,

                })

                chair_id += 1


        except Exception as e:

            print(
                f"\n[DETECTION ITEM ERROR] {e}"
            )

            continue


    return people, chairs


# ============================================================
# DEPTH EXTRACTION
# ============================================================

def calculate_depth(frame):

    if depth_model is None:

        return None


    try:

        results = depth_model.predict(

            source=frame,

            imgsz=DEPTH_IMGSZ,

            device="cpu",

            verbose=False,

        )

    except Exception as e:

        print(
            f"\n[DEPTH ERROR] {e}"
        )

        return None


    if not results:

        return None


    result = results[0]


    if not hasattr(
        result,
        "depth",
    ):

        return None


    if result.depth is None:

        return None


    try:

        depth = (

            result.depth.data

            .detach()

            .float()

            .cpu()

            .numpy()

        )

    except Exception as e:

        print(
            f"\n[DEPTH TENSOR ERROR] {e}"
        )

        return None


    depth = np.squeeze(
        depth
    )


    if depth.ndim != 2:

        return None


    try:

        depth = cv2.resize(

            depth,

            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            ),

            interpolation=cv2.INTER_LINEAR,

        )

    except Exception as e:

        print(
            f"\n[DEPTH RESIZE ERROR] {e}"
        )

        return None


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
# HMR PROCESSING
# ============================================================

def process_hmr(
    frame,
    people,
    current_frame_id,
):

    if hmr_processor is None:

        return []


    if not people:

        return []


    boxes = np.asarray(

        [

            [

                p["x1"],
                p["y1"],
                p["x2"],
                p["y2"],

            ]

            for p in people

        ],

        dtype=np.float32,

    )


    try:

        result = (
            hmr_processor.process_frame(

                frame=frame,

                frame_id=current_frame_id,

                timestamp=time.time(),

                boxes=boxes,

            )
        )


    except Exception as e:

        print(
            f"\n[HMR ERROR] {e}"
        )

        return []


    if not isinstance(
        result,
        dict,
    ):

        return []


    persons = result.get(
        "persons",
        [],
    )


    if not isinstance(
        persons,
        list,
    ):

        return []


    return persons


# ============================================================
# HMR HUMAN 3D REFERENCE
# ============================================================

def extract_human_3d(
    hmr_person,
):

    if not isinstance(
        hmr_person,
        dict,
    ):

        return None


    joints_raw = hmr_person.get(
        "joints_3d",
        None,
    )


    if joints_raw is None:

        return None


    try:

        joints = np.asarray(

            joints_raw,

            dtype=np.float32,

        )

    except Exception:

        return None


    if joints.ndim != 2:

        return None


    if joints.shape[1] < 3:

        return None


    joints = joints[:, :3]


    valid = np.isfinite(
        joints
    ).all(
        axis=1
    )


    joints = joints[
        valid
    ]


    if len(joints) == 0:

        return None


    # --------------------------------------------------------
    # HMR keypoints are root-relative.
    #
    # camera_translation gives the estimated body/root
    # translation relative to the camera.
    #
    # Therefore:
    #
    #       camera_joint = joint + camera_translation
    #
    # --------------------------------------------------------

    camera_translation = safe_vector(

        hmr_person.get(
            "camera_translation",
            [0.0, 0.0, 0.0],
        ),

        length=3,

    )


    camera_joints = (
        joints
        + camera_translation
    )


    valid_camera = np.isfinite(
        camera_joints
    ).all(
        axis=1
    )


    camera_joints = camera_joints[
        valid_camera
    ]


    if len(camera_joints) == 0:

        return None


    centroid = np.mean(

        camera_joints,

        axis=0,

    )


    minimum = np.min(

        camera_joints,

        axis=0,

    )


    maximum = np.max(

        camera_joints,

        axis=0,

    )


    dimensions = (
        maximum
        - minimum
    )


    return {

        "person_id":
            hmr_person.get(
                "person_id",
                0,
            ),

        "joints_camera":
            camera_joints,

        "centroid":
            centroid,

        "dimensions":
            dimensions,

        "camera_translation":
            camera_translation,

        "joint_count":
            len(camera_joints),

    }


# ============================================================
# HMR → DEPTH TRANSFORM
# ============================================================

def transform_hmr_point(
    point,
):

    point = safe_vector(
        point,
        length=3,
    )


    transformed = (

        HMR_TO_DEPTH_ROTATION
        @ point

    ) + HMR_TO_DEPTH_TRANSLATION


    return transformed.astype(
        np.float32
    )


# ============================================================
# BUILD CHAIR GRID
# ============================================================

def build_chair_grid(
    depth,
    chair,
):

    if depth is None:

        return None


    try:

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


        grid = (
            grid_extractor.extract(

                depth_map=depth,

                bbox=bbox,

                object_id=int(
                    chair["id"]
                ),

                object_class="chair",

            )
        )


        return grid


    except Exception as e:

        print(
            f"\n[CHAIR GRID ERROR] {e}"
        )

        return None


# ============================================================
# CHAIR 3D REFERENCE
# ============================================================

def extract_chair_3d(
    grid,
):

    if grid is None:

        return None


    try:

        xyz = np.asarray(

            grid.xyz_array(),

            dtype=np.float32,

        )

    except Exception as e:

        print(
            f"\n[CHAIR XYZ ERROR] {e}"
        )

        return None


    if xyz.ndim != 2:

        return None


    if xyz.shape[1] < 3:

        return None


    xyz = xyz[:, :3]


    valid = np.isfinite(
        xyz
    ).all(
        axis=1
    )


    xyz = xyz[
        valid
    ]


    if len(xyz) == 0:

        return None


    centroid = np.mean(
        xyz,
        axis=0,
    )


    minimum = np.min(
        xyz,
        axis=0,
    )


    maximum = np.max(
        xyz,
        axis=0,
    )


    dimensions = (
        maximum
        - minimum
    )


    return {

        "centroid":
            centroid,

        "dimensions":
            dimensions,

        "points":
            xyz,

        "point_count":
            len(xyz),

        "valid_ratio":
            safe_float(
                getattr(
                    grid,
                    "valid_ratio",
                    0.0,
                )
            ),

    }


# ============================================================
# HUMAN ↔ CHAIR RELATIONSHIP
# ============================================================

def calculate_relationship(
    hmr_person,
    chair_grid,
    chair_id,
):

    human = extract_human_3d(
        hmr_person
    )


    chair = extract_chair_3d(
        chair_grid
    )


    if human is None:

        return None


    if chair is None:

        return None


    # --------------------------------------------------------
    # Transform HMR coordinates
    # --------------------------------------------------------

    human_centroid_hmr = (
        human["centroid"]
    )


    human_centroid_depth = (
        transform_hmr_point(
            human_centroid_hmr
        )
    )


    # --------------------------------------------------------
    # Chair is already in YOLO26 camera coordinates
    # --------------------------------------------------------

    chair_centroid = (
        chair["centroid"]
    )


    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    delta = (

        chair_centroid
        - human_centroid_depth

    )


    distance_3d = float(

        np.linalg.norm(
            delta
        )

    )


    ground_delta = np.array(

        [

            delta[0],

            0.0,

            delta[2],

        ],

        dtype=np.float32,

    )


    ground_distance = float(

        np.linalg.norm(
            ground_delta
        )

    )


    return {

        "person_id":
            int(
                hmr_person.get(
                    "person_id",
                    0,
                )
            ),

        "chair_id":
            int(chair_id),

        "human_hmr_x":
            float(
                human_centroid_hmr[0]
            ),

        "human_hmr_y":
            float(
                human_centroid_hmr[1]
            ),

        "human_hmr_z":
            float(
                human_centroid_hmr[2]
            ),

        "human_depth_x":
            float(
                human_centroid_depth[0]
            ),

        "human_depth_y":
            float(
                human_centroid_depth[1]
            ),

        "human_depth_z":
            float(
                human_centroid_depth[2]
            ),

        "chair_x":
            float(
                chair_centroid[0]
            ),

        "chair_y":
            float(
                chair_centroid[1]
            ),

        "chair_z":
            float(
                chair_centroid[2]
            ),

        "delta_x":
            float(delta[0]),

        "delta_y":
            float(delta[1]),

        "delta_z":
            float(delta[2]),

        "raw_3d_distance":
            distance_3d,

        "raw_ground_distance":
            ground_distance,

        "human_joint_count":
            int(
                human["joint_count"]
            ),

        "chair_point_count":
            int(
                chair["point_count"]
            ),

        "chair_valid_ratio":
            float(
                chair["valid_ratio"]
            ),

    }


# ============================================================
# DRAW CHAIR GRID
# ============================================================

def draw_chair_grid(
    frame,
    grid,
):

    if grid is None:

        return


    try:

        points = grid.points

    except Exception:

        return


    for point in points:

        try:

            u = int(
                point.u
            )

            v = int(
                point.v
            )

            cv2.circle(

                frame,

                (u, v),

                3,

                (255, 180, 0),

                -1,

            )

        except Exception:

            continue


# ============================================================
# DRAW HUMAN 3D STATUS
# ============================================================

def draw_human_status(
    frame,
    hmr_person,
):

    human = extract_human_3d(
        hmr_person
    )


    if human is None:

        return


    bbox = hmr_person.get(
        "bbox",
        None,
    )


    if isinstance(
        bbox,
        dict,
    ):

        x1 = int(
            safe_float(
                bbox.get(
                    "x1",
                    5,
                )
            )
        )

        y1 = int(
            safe_float(
                bbox.get(
                    "y1",
                    5,
                )
            )
        )

    else:

        x1 = 10
        y1 = 90


    centroid = human[
        "centroid"
    ]


    text = (

        "HMR C3D: "

        f"{centroid[0]:.2f}, "

        f"{centroid[1]:.2f}, "

        f"{centroid[2]:.2f}"

    )


    cv2.putText(

        frame,

        text,

        (
            max(5, x1),
            max(90, y1),
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (0, 255, 255),

        1,

        cv2.LINE_AA,

    )


# ============================================================
# DRAW RELATIONSHIP
# ============================================================

def draw_relationship(
    frame,
    relationship,
):

    if relationship is None:

        return


    lines = [

        (
            "HMR->DEPTH: "
            f"("
            f"{relationship['human_depth_x']:.2f}, "
            f"{relationship['human_depth_y']:.2f}, "
            f"{relationship['human_depth_z']:.2f}"
            f")"
        ),

        (
            "CHAIR: "
            f"("
            f"{relationship['chair_x']:.2f}, "
            f"{relationship['chair_y']:.2f}, "
            f"{relationship['chair_z']:.2f}"
            f")"
        ),

        (
            "DELTA: "
            f"("
            f"{relationship['delta_x']:.2f}, "
            f"{relationship['delta_y']:.2f}, "
            f"{relationship['delta_z']:.2f}"
            f")"
        ),

        (
            "RAW 3D: "
            f"{relationship['raw_3d_distance']:.2f} m"
        ),

        (
            "GROUND: "
            f"{relationship['raw_ground_distance']:.2f} m"
        ),

    ]


    y = 150


    for text in lines:

        cv2.putText(

            frame,

            text,

            (10, y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            (255, 255, 255),

            1,

            cv2.LINE_AA,

        )

        y += 18


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global latest_jpeg
    global latest_depth
    global latest_people
    global latest_chairs
    global latest_hmr_people
    global latest_chair_features
    global latest_alignment
    global frame_id


    start_time = time.time()

    frame_counter = 0


    cached_depth = None

    cached_hmr = []

    cached_grids = []

    cached_relationships = []


    while running:

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        try:

            ret, frame = cap.read()

        except Exception as e:

            print(
                f"\n[CAMERA ERROR] {e}"
            )

            time.sleep(
                0.1
            )

            continue


        if not ret or frame is None:

            print(
                "\n[CAMERA] Frame read failed"
            )

            time.sleep(
                0.1
            )

            continue


        frame_id += 1

        frame_counter += 1


        # ====================================================
        # YOLO
        # ====================================================

        people, chairs = (
            detect_objects(frame)
        )


        # ====================================================
        # DEPTH
        # ====================================================

        if (

            cached_depth is None

            or frame_id % DEPTH_INTERVAL == 0

        ):

            t0 = time.time()


            new_depth = calculate_depth(
                frame
            )


            if new_depth is not None:

                cached_depth = (
                    new_depth
                )

                depth_time = (
                    time.time()
                    - t0
                )

                status[
                    "depth_status"
                ] = (
                    f"ACTIVE "
                    f"{depth_time:.2f}s"
                )

            else:

                status[
                    "depth_status"
                ] = "NO DEPTH"


        # ====================================================
        # HMR
        # ====================================================

        if (

            frame_id % HMR_INTERVAL == 0

            or not cached_hmr

        ):

            status[
                "hmr_status"
            ] = "RUNNING"


            new_hmr = process_hmr(

                frame,

                people,

                frame_id,

            )


            if new_hmr:

                cached_hmr = new_hmr

                status[
                    "hmr_status"
                ] = "ACTIVE"

            else:

                status[
                    "hmr_status"
                ] = "NO HUMAN 3D"


        # ====================================================
        # CHAIR 3D
        # ====================================================

        new_grids = []


        if cached_depth is not None:

            for chair in chairs:

                grid = build_chair_grid(

                    cached_depth,

                    chair,

                )


                if grid is not None:

                    chair_3d = (
                        extract_chair_3d(
                            grid
                        )
                    )


                    if chair_3d is not None:

                        new_grids.append(

                            (
                                chair,
                                grid,
                                chair_3d,
                            )

                        )


        cached_grids = new_grids


        # ====================================================
        # HUMAN ↔ CHAIR RELATIONSHIPS
        # ====================================================

        relationships = []


        if cached_hmr and cached_grids:

            for person in cached_hmr:

                best = None

                best_distance = None


                for (

                    chair,

                    grid,

                    chair_3d,

                ) in cached_grids:


                    relationship = (
                        calculate_relationship(

                            person,

                            grid,

                            chair["id"],

                        )
                    )


                    if relationship is None:

                        continue


                    d = relationship[
                        "raw_3d_distance"
                    ]


                    if (

                        best_distance is None

                        or d < best_distance

                    ):

                        best_distance = d

                        best = (
                            relationship
                        )


                if best is not None:

                    relationships.append(
                        best
                    )


        cached_relationships = (
            relationships
        )


        # ====================================================
        # DRAW
        # ====================================================

        output = frame.copy()


        # ----------------------------------------------------
        # Person detections
        # ----------------------------------------------------

        for person in people:

            x1 = int(
                person["x1"]
            )

            y1 = int(
                person["y1"]
            )

            x2 = int(
                person["x2"]
            )

            y2 = int(
                person["y2"]
            )


            cv2.rectangle(

                output,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2,

            )


            cv2.putText(

                output,

                (
                    f"PERSON "
                    f"{person['id'] + 1} "
                    f"{person['confidence']:.2f}"
                ),

                (
                    x1,
                    max(
                        18,
                        y1 - 6,
                    ),
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (0, 255, 0),

                2,

                cv2.LINE_AA,

            )


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
                        18,
                        y1 - 6,
                    ),
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (255, 255, 0),

                2,

                cv2.LINE_AA,

            )


        # ----------------------------------------------------
        # Chair 3D grids
        # ----------------------------------------------------

        for (

            chair,

            grid,

            chair_3d,

        ) in cached_grids:

            draw_chair_grid(

                output,

                grid,

            )


            c = chair_3d[
                "centroid"
            ]


            x1 = int(
                chair["x1"]
            )

            y2 = int(
                chair["y2"]
            )


            text = (

                f"C3D "
                f"X:{c[0]:.2f} "
                f"Y:{c[1]:.2f} "
                f"Z:{c[2]:.2f}"

            )


            cv2.putText(

                output,

                text,

                (
                    x1,
                    min(
                        CAMERA_HEIGHT - 8,
                        y2 + 18,
                    ),
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.38,

                (255, 255, 255),

                1,

                cv2.LINE_AA,

            )


        # ----------------------------------------------------
        # HMR status
        # ----------------------------------------------------

        for hmr_person in cached_hmr:

            draw_human_status(

                output,

                hmr_person,

            )


        # ----------------------------------------------------
        # Relationship
        # ----------------------------------------------------

        if cached_relationships:

            draw_relationship(

                output,

                cached_relationships[0],

            )


        else:

            cv2.putText(

                output,

                "WAITING FOR HUMAN + CHAIR 3D",

                (10, 150),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (0, 255, 255),

                1,

                cv2.LINE_AA,

            )


        # ====================================================
        # TOP STATUS PANEL
        # ====================================================

        cv2.rectangle(

            output,

            (0, 0),

            (CAMERA_WIDTH, 78),

            (0, 0, 0),

            -1,

        )


        cv2.putText(

            output,

            (
                f"PERSON: "
                f"{len(people)}"
            ),

            (10, 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0, 255, 0),

            1,

        )


        cv2.putText(

            output,

            (
                f"CHAIR: "
                f"{len(chairs)}"
            ),

            (100, 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (255, 255, 0),

            1,

        )


        cv2.putText(

            output,

            (
                f"CHAIR3D: "
                f"{len(cached_grids)}"
            ),

            (185, 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (255, 255, 255),

            1,

        )


        cv2.putText(

            output,

            (
                f"HMR: "
                f"{len(cached_hmr)}"
            ),

            (300, 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0, 255, 255),

            1,

        )


        cv2.putText(

            output,

            (
                f"FPS: "
                f"{status['fps']:.2f}"
            ),

            (380, 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (255, 255, 255),

            1,

        )


        cv2.putText(

            output,

            (
                "HMR->DEPTH: "
                "IDENTITY/DEV"
            ),

            (10, 48),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.40,

            (0, 200, 255),

            1,

        )


        cv2.putText(

            output,

            (
                f"D:{status['depth_status']}"
            ),

            (260, 48),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.38,

            (255, 255, 255),

            1,

        )


        cv2.putText(

            output,

            (
                f"H:{status['hmr_status']}"
            ),

            (500, 48),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.38,

            (255, 255, 255),

            1,

        )


        cv2.putText(

            output,

            (
                f"FRAME {frame_id}"
            ),

            (500, 68),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.35,

            (160, 160, 160),

            1,

        )


        # ====================================================
        # FPS
        # ====================================================

        elapsed = (
            time.time()
            - start_time
        )


        fps = (

            frame_counter
            / elapsed

            if elapsed > 0

            else 0.0

        )


        status["fps"] = round(
            fps,
            2,
        )


        # ====================================================
        # JPEG
        # ====================================================

        try:

            ok, encoded = cv2.imencode(

                ".jpg",

                output,

                [

                    cv2.IMWRITE_JPEG_QUALITY,

                    80,

                ],

            )


            if ok:

                with state_lock:

                    latest_jpeg = (
                        encoded.tobytes()
                    )

                    latest_depth = (
                        None

                        if cached_depth is None

                        else cached_depth.copy()
                    )

                    latest_people = (
                        people.copy()
                    )

                    latest_chairs = (
                        chairs.copy()
                    )

                    latest_hmr_people = (
                        cached_hmr.copy()
                    )

                    latest_alignment = (
                        cached_relationships.copy()
                    )


        except Exception as e:

            print(
                f"\n[JPEG ERROR] {e}"
            )


        # ====================================================
        # STATUS
        # ====================================================

        with state_lock:

            status["frame"] = frame_id

            status["people"] = len(
                people
            )

            status["chairs"] = len(
                chairs
            )

            status["hmr_people"] = len(
                cached_hmr
            )

            status["3d_chairs"] = len(
                cached_grids
            )


        print(

            f"\rFrame {frame_id:05d} | "

            f"Person {len(people)} | "

            f"Chair {len(chairs)} | "

            f"HMR {len(cached_hmr)} | "

            f"Chair3D {len(cached_grids)} | "

            f"FPS {fps:.2f}",

            end="",

            flush=True,

        )


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_stream():

    while running:

        with state_lock:

            jpeg = latest_jpeg


        if jpeg is None:

            time.sleep(
                0.05
            )

            continue


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + jpeg

            + b"\r\n"

        )


# ============================================================
# HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>BAS-HMR Human Chair Alignment</title>

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

    max-width: 900px;

    background: #1c1c1c;

    padding: 18px;

    border-radius: 8px;

}

.item {

    display: inline-block;

    margin: 8px 18px;

}

.value {

    font-weight: bold;

}

.note {

    color: #aaa;

    margin-top: 15px;

    font-size: 13px;

}

</style>

</head>

<body>

<h1>BAS-HMR</h1>

<div class="subtitle">

Human 3D ↔ Chair 3D Integration

</div>

<img

    class="video"

    src="/video_feed"

/>

<div class="panel">

<div class="item">

Camera:

<span

    class="value"

    id="camera">

...

</span>

</div>

<div class="item">

YOLO:

<span

    class="value"

    id="yolo">

...

</span>

</div>

<div class="item">

Depth:

<span

    class="value"

    id="depth">

...

</span>

</div>

<div class="item">

HMR:

<span

    class="value"

    id="hmr">

...

</span>

</div>

<div class="item">

People:

<span

    class="value"

    id="people">

...

</span>

</div>

<div class="item">

Chairs:

<span

    class="value"

    id="chairs">

...

</span>

</div>

<div class="item">

Chair 3D:

<span

    class="value"

    id="chairs3d">

...

</span>

</div>

<div class="item">

FPS:

<span

    class="value"

    id="fps">

...

</span>

</div>

<div class="note">

Current HMR → YOLO26 transform:
IDENTITY / DEVELOPMENT MODE

</div>

</div>


<script>

async function updateStatus() {

    try {

        const response =
            await fetch(
                "/api/status"
            );

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
            "hmr"
        ).textContent =
            data.hmr_status;


        document.getElementById(
            "people"
        ).textContent =
            data.people;


        document.getElementById(
            "chairs"
        ).textContent =
            data.chairs;


        document.getElementById(
            "chairs3d"
        ).textContent =
            data["3d_chairs"];


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
# ROUTES
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

            "multipart/x-mixed-replace;"

            " boundary=frame"

        ),

    )


@app.route("/api/status")
def api_status():

    with state_lock:

        return jsonify(
            status.copy()
        )


@app.route("/api/alignment")
def api_alignment():

    with state_lock:

        return jsonify({

            "frame":

                frame_id,

            "transform": {

                "rotation":

                    HMR_TO_DEPTH_ROTATION.tolist(),

                "translation":

                    HMR_TO_DEPTH_TRANSLATION.tolist(),

                "mode":

                    "IDENTITY_DEVELOPMENT",

            },

            "alignment":

                latest_alignment,

        })


@app.route("/api/health")
def api_health():

    return jsonify({

        "status":
            "running",

        "camera":
            status["camera"],

        "yolo":
            status["yolo"],

        "depth":
            status["depth"],

        "hmr":
            status["hmr"],

        "device":
            "cpu",

        "hmr_to_depth":
            "identity_development",

    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "BAS-HMR HUMAN ↔ CHAIR 3D INTEGRATION"
    )
    print("=" * 70)

    print()

    print(
        "Camera:",
        CAMERA_DEVICE,
    )

    print(
        "YOLO:",
        YOLO_MODEL_PATH,
    )

    print(
        "Depth:",
        DEPTH_MODEL_PATH,
    )

    print(
        "HMR interval:",
        HMR_INTERVAL,
        "frames",
    )

    print(
        "Depth interval:",
        DEPTH_INTERVAL,
        "frames",
    )

    print()

    print(
        "Browser:"
    )

    print(
        "http://localhost:5010"
    )

    print()

    print(
        "Alignment API:"
    )

    print(
        "http://localhost:5010/api/alignment"
    )

    print()

    print(
        "Health API:"
    )

    print(
        "http://localhost:5010/api/health"
    )

    print()

    print("=" * 70)


    # --------------------------------------------------------
    # Start camera worker
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

            host=HOST,

            port=PORT,

            threaded=True,

            debug=False,

            use_reloader=False,

        )

    except KeyboardInterrupt:

        print(
            "\nStopping..."
        )

    finally:

        running = False

        try:

            cap.release()

        except Exception:

            pass

        print(
            "\nCamera released."
        )