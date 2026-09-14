import os
import sys
import time
import threading
import queue

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from pipeline.camera_worker import CameraWorker
from yolo.yolo_processor import YOLOProcessor
from hmr.hmr_processor import HMRProcessor


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "/dev/video0"

YOLO_MODEL = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt",
)

YOLO_CONFIDENCE = 0.30

# Run YOLO every N camera frames.
YOLO_INTERVAL = 3

# Number of seconds to wait between plot refreshes.
PLOT_INTERVAL = 0.01


# ============================================================
# 25 VERIFIED JOINT NAMES
# ============================================================

JOINT_NAMES = {
    0: "nose",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "pelvis",
    9: "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    15: "right_eye",
    16: "left_eye",
    17: "right_ear",
    18: "left_ear",
    19: "left_big_toe",
    20: "left_small_toe",
    21: "left_heel",
    22: "right_big_toe",
    23: "right_small_toe",
    24: "right_heel",
}


# ============================================================
# SKELETON CONNECTIONS
# ============================================================

SKELETON_EDGES = [

    # Head
    ("nose", "neck"),

    # Right arm
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    # Left arm
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    # Torso
    ("neck", "pelvis"),

    # Right leg
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    # Left leg
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    # Face
    ("nose", "right_eye"),
    ("nose", "left_eye"),
    ("right_eye", "right_ear"),
    ("left_eye", "left_ear"),

    # Feet
    ("left_ankle", "left_heel"),
    ("left_heel", "left_big_toe"),
    ("left_heel", "left_small_toe"),

    ("right_ankle", "right_heel"),
    ("right_heel", "right_big_toe"),
    ("right_heel", "right_small_toe"),
]


# ============================================================
# LATEST HMR FRAME BUFFER
# ============================================================

class LatestFrameBuffer:
    """
    Stores only the newest frame waiting for HMR.
    """

    def __init__(self):
        self.item = None
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            self.item = item

    def get(self):
        with self.lock:
            item = self.item
            self.item = None
            return item

    def clear(self):
        with self.lock:
            self.item = None


# ============================================================
# BACKGROUND HMR WORKER
# ============================================================

class HMRLiveWorker:

    def __init__(self):

        self.buffer = LatestFrameBuffer()

        self.result_queue = queue.Queue(
            maxsize=5
        )

        self.running = False
        self.thread = None

        self.processor = None

        self.processed_frames = 0

    def start(self):

        print("[HMR] Loading HMR2...")

        self.processor = HMRProcessor()

        print("[HMR] HMR2 loaded")
        print("[HMR] Device: CPU")

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

        print("[HMR] Background worker started")

    def submit(
        self,
        frame,
        frame_id,
        timestamp,
        boxes,
    ):

        self.buffer.put({
            "frame": frame,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "boxes": boxes,
        })

    def _run(self):

        while self.running:

            item = self.buffer.get()

            if item is None:

                time.sleep(0.005)

                continue

            try:

                frame = item["frame"]
                frame_id = item["frame_id"]
                timestamp = item["timestamp"]
                boxes = item["boxes"]

                print(
                    f"[HMR] Processing frame "
                    f"{frame_id} | "
                    f"persons={len(boxes)}"
                )

                start = time.time()

                result = self.processor.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    boxes=boxes,
                )

                elapsed = time.time() - start

                self.processed_frames += 1

                print(
                    f"[HMR] Completed frame "
                    f"{frame_id} | "
                    f"{elapsed:.2f}s"
                )

                result["inference_time"] = elapsed

                # If old results are waiting, discard one.
                if self.result_queue.full():

                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.result_queue.put_nowait(
                    result
                )

            except Exception as e:

                print(
                    f"[HMR] Error: "
                    f"{type(e).__name__}: {e}"
                )

    def get_result(self):

        try:

            return self.result_queue.get_nowait()

        except queue.Empty:

            return None

    def stop(self):

        print("[HMR] Stopping...")

        self.running = False

        self.buffer.clear()

        if self.thread is not None:

            self.thread.join(
                timeout=30
            )

        print(
            f"[HMR] Processed: "
            f"{self.processed_frames} frames"
        )

        print("[HMR] Stopped")


# ============================================================
# CONVERT RESULT TO NAMED JOINTS
# ============================================================

def convert_joints(person):

    named = {}

    for joint in person["pose"]["joints_3d"]:

        joint_id = int(
            joint["joint_id"]
        )

        if joint_id not in JOINT_NAMES:
            continue

        name = JOINT_NAMES[joint_id]

        named[name] = {
            "x": float(joint["x"]),
            "y": float(joint["y"]),
            "z": float(joint["z"]),
        }

    return named


# ============================================================
# UPDATE 3D VIEW
# ============================================================

def draw_skeleton(
    ax,
    person,
    frame_id,
    inference_time,
):

    ax.clear()

    joints = convert_joints(
        person
    )

    # --------------------------------------------------------
    # Plot joints
    # --------------------------------------------------------

    xs = []
    ys = []
    zs = []

    for joint in joints.values():

        xs.append(joint["x"])
        ys.append(joint["y"])
        zs.append(joint["z"])

    if xs:

        ax.scatter(
            xs,
            ys,
            zs,
            s=45,
        )

    # --------------------------------------------------------
    # Plot skeleton
    # --------------------------------------------------------

    for joint_a, joint_b in SKELETON_EDGES:

        if (
            joint_a not in joints
            or joint_b not in joints
        ):
            continue

        a = joints[joint_a]
        b = joints[joint_b]

        ax.plot(
            [a["x"], b["x"]],
            [a["y"], b["y"]],
            [a["z"], b["z"]],
            linewidth=2,
        )

    # --------------------------------------------------------
    # Camera/HMR information
    # --------------------------------------------------------

    camera_translation = person.get(
        "camera_translation",
        {}
    )

    camera_z = camera_translation.get(
        "z",
        0.0
    )

    ax.set_title(
        f"BAS-HMR LIVE 3D\n"
        f"Frame: {frame_id} | "
        f"HMR: {inference_time:.2f}s | "
        f"Depth: {camera_z:.2f}"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # --------------------------------------------------------
    # Keep human centered around pelvis
    # --------------------------------------------------------

    pelvis = joints.get(
        "pelvis"
    )

    if pelvis is not None:

        center_x = pelvis["x"]
        center_y = pelvis["y"]
        center_z = pelvis["z"]

    elif xs:

        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        center_z = sum(zs) / len(zs)

    else:

        center_x = 0.0
        center_y = 0.0
        center_z = 0.0

    radius = 1.2

    ax.set_xlim(
        center_x - radius,
        center_x + radius,
    )

    ax.set_ylim(
        center_y - radius,
        center_y + radius,
    )

    ax.set_zlim(
        center_z - radius,
        center_z + radius,
    )

    # Camera-friendly viewpoint
    ax.view_init(
        elev=15,
        azim=-70,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("             BAS-HMR LIVE 3D VISUALIZER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = CameraWorker(
        camera=CAMERA,
        width=640,
        height=480,
        fps=30,
    )

    print("[1] Starting camera...")

    camera.start()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print()
    print("[2] Loading YOLO...")

    yolo = YOLOProcessor(
        model_path=YOLO_MODEL,
        device="cpu",
    )

    print("[YOLO] Ready")

    # --------------------------------------------------------
    # HMR
    # --------------------------------------------------------

    print()
    print("[3] Starting background HMR...")

    hmr = HMRLiveWorker()

    hmr.start()

    # --------------------------------------------------------
    # Matplotlib
    # --------------------------------------------------------

    print()
    print("[4] Starting 3D viewer...")

    plt.ion()

    fig = plt.figure(
        figsize=(9, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    plt.show(
        block=False
    )

    print()
    print("=" * 70)
    print("LIVE 3D VISUALIZER STARTED")
    print("=" * 70)

    print()
    print("Camera   : /dev/video0")
    print("YOLO     : YOLO11n CPU")
    print("HMR      : HMR2 CPU")
    print()
    print("Only the newest waiting frame is sent to HMR.")
    print("Close the 3D window or press Ctrl+C to stop.")
    print()

    frame_counter = 0

    latest_result = None

    try:

        while True:

            # ------------------------------------------------
            # Read camera
            # ------------------------------------------------

            item = camera.read()

            if item is None:
                continue

            frame = item["frame"]
            frame_id = item["frame_id"]
            timestamp = item["timestamp"]

            frame_counter += 1

            # ------------------------------------------------
            # YOLO frequency control
            # ------------------------------------------------

            if frame_counter % YOLO_INTERVAL == 0:

                yolo_result = yolo.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )

                persons = yolo_result.get(
                    "persons",
                    []
                )

                # ------------------------------------------------
                # Send newest person frame to HMR
                # ------------------------------------------------

                if persons:

                    boxes = []

                    for person in persons:

                        bbox = person["bbox"]

                        boxes.append([
                            bbox["x1"],
                            bbox["y1"],
                            bbox["x2"],
                            bbox["y2"],
                        ])

                    hmr.submit(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        boxes=np.asarray(
                            boxes,
                            dtype=np.float32,
                        ),
                    )

            # ------------------------------------------------
            # Check HMR result
            # ------------------------------------------------

            result = hmr.get_result()

            if result is not None:

                latest_result = result

                persons = result.get(
                    "persons",
                    []
                )

                if persons:

                    person = persons[0]

                    draw_skeleton(
                        ax=ax,
                        person=person,
                        frame_id=result["frame_id"],
                        inference_time=result.get(
                            "inference_time",
                            0.0,
                        ),
                    )

                    fig.canvas.draw_idle()

                    fig.canvas.flush_events()

            # ------------------------------------------------
            # Keep GUI alive
            # ------------------------------------------------

            plt.pause(
                PLOT_INTERVAL
            )

    except KeyboardInterrupt:

        print()
        print("[SYSTEM] Ctrl+C received")

    finally:

        print()
        print("=" * 70)
        print("STOPPING LIVE 3D VISUALIZER")
        print("=" * 70)

        hmr.stop()

        camera.stop()

        plt.close(
            fig
        )

        print("[SYSTEM] Complete")


if __name__ == "__main__":
    main()
