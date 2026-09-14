from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from pipeline.camera_worker import CameraWorker
from yolo.yolo_processor import YOLOProcessor
from hmr.hmr_processor import HMRProcessor


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_DEVICE = "/dev/video0"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

YOLO_MODEL = "yolo11n.pt"
YOLO_DEVICE = "cpu"

YOLO_INTERVAL = 10

PERSON_CONFIDENCE_THRESHOLD = 0.30

OUTPUT_DIR = Path(
    "output/live_3d_json"
)

MAX_STRUCTURAL_JOINTS = 25


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

# ------------------------------------------------------------
# These affect ONLY visualization.
# JSON coordinates remain the real HMR coordinates.
# ------------------------------------------------------------

# Make skeleton visually larger.
SKELETON_SCALE = 2.2

# Joint point size.
JOINT_SIZE = 45

# Bone thickness.
BONE_WIDTH = 4

# Label size.
LABEL_SIZE = 14

# Minimum distance between person centers
# in the visualization.
MIN_PERSON_SEPARATION = 3.5

# Extra space around the displayed skeletons.
VIEW_PADDING = 1.0

# Minimum viewing volume.
MIN_VIEW_SIZE = 7.0


# ============================================================
# JOINT NAMES
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

SKELETON_CONNECTIONS = [

    # Head
    (0, 1),

    # Right arm
    (1, 2),
    (2, 3),
    (3, 4),

    # Left arm
    (1, 5),
    (5, 6),
    (6, 7),

    # Torso
    (1, 8),

    # Right leg
    (8, 9),
    (9, 10),
    (10, 11),

    # Left leg
    (8, 12),
    (12, 13),
    (13, 14),

    # Face
    (0, 15),
    (0, 16),
    (15, 17),
    (16, 18),

    # Left foot
    (14, 19),
    (14, 20),
    (14, 21),

    # Right foot
    (11, 22),
    (11, 23),
    (11, 24),
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def create_output_directory() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# NUMPY -> JSON CONVERSION
# ============================================================

def convert_numpy(
    value: Any,
) -> Any:

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, dict):
        return {
            key: convert_numpy(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            convert_numpy(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            convert_numpy(item)
            for item in value
        ]

    return value


# ============================================================
# SAVE JSON
# ============================================================

def save_hmr_json(
    result: dict[str, Any],
) -> None:

    frame_id = int(
        result.get(
            "frame_id",
            -1,
        )
    )

    output_path = (
        OUTPUT_DIR
        / f"frame_{frame_id:06d}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            convert_numpy(result),
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"[JSON] Saved: {output_path}"
    )


# ============================================================
# YOLO -> HMR BOXES
# ============================================================

def extract_hmr_boxes(
    yolo_result: dict[str, Any],
) -> list[list[float]]:

    boxes = []

    persons = yolo_result.get(
        "persons",
        [],
    )

    for person in persons:

        confidence = float(
            person.get(
                "confidence",
                0.0,
            )
        )

        if (
            confidence
            < PERSON_CONFIDENCE_THRESHOLD
        ):
            continue

        bbox = person.get(
            "bbox"
        )

        if bbox is None:
            continue

        try:

            x1 = float(
                bbox["x1"]
            )

            y1 = float(
                bbox["y1"]
            )

            x2 = float(
                bbox["x2"]
            )

            y2 = float(
                bbox["y2"]
            )

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

    return boxes


# ============================================================
# EXTRACT JOINTS
# ============================================================

def extract_joints(
    person: dict[str, Any],
) -> dict[
    int,
    tuple[float, float, float],
]:

    joints = (
        person
        .get(
            "pose",
            {},
        )
        .get(
            "joints_3d",
            [],
        )
    )

    points = {}

    for joint in joints:

        try:

            joint_id = int(
                joint["joint_id"]
            )

            if (
                joint_id
                >= MAX_STRUCTURAL_JOINTS
            ):
                continue

            x = float(
                joint["x"]
            )

            y = float(
                joint["y"]
            )

            z = float(
                joint["z"]
            )

            points[joint_id] = (
                x,
                y,
                z,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

    return points


# ============================================================
# NORMALIZE PERSON FOR VISUALIZATION
# ============================================================

def normalize_person(
    points: dict[
        int,
        tuple[float, float, float],
    ],
) -> dict[
    int,
    tuple[float, float, float],
]:

    if not points:
        return {}

    # --------------------------------------------------------
    # Use pelvis as the person's local origin.
    # --------------------------------------------------------

    if 8 in points:

        origin = np.asarray(
            points[8],
            dtype=float,
        )

    else:

        origin = np.asarray(
            list(points.values()),
            dtype=float,
        ).mean(
            axis=0
        )

    normalized = {}

    for joint_id, point in points.items():

        p = np.asarray(
            point,
            dtype=float,
        )

        local = (
            p - origin
        )

        # ----------------------------------------------------
        # Scale only for visualization.
        # ----------------------------------------------------

        local *= SKELETON_SCALE

        normalized[joint_id] = (
            float(local[0]),
            float(local[1]),
            float(local[2]),
        )

    return normalized


# ============================================================
# CALCULATE PERSON DISPLAY POSITIONS
# ============================================================

def calculate_person_positions(
    persons: list[
        dict[str, Any]
    ],
) -> list[
    tuple[float, float, float]
]:

    count = len(persons)

    if count == 0:
        return []

    positions = []

    # --------------------------------------------------------
    # One person -> center.
    # --------------------------------------------------------

    if count == 1:

        return [
            (
                0.0,
                0.0,
                0.0,
            )
        ]

    # --------------------------------------------------------
    # Multiple people.
    #
    # Place skeletons along X so their body structures
    # remain visually separated.
    # --------------------------------------------------------

    spacing = MIN_PERSON_SEPARATION

    total_width = (
        (count - 1)
        * spacing
    )

    start_x = (
        -total_width / 2.0
    )

    for index in range(count):

        x = (
            start_x
            + index * spacing
        )

        positions.append(
            (
                x,
                0.0,
                0.0,
            )
        )

    return positions


# ============================================================
# DRAW ONE PERSON
# ============================================================

def draw_person_skeleton(
    ax,
    person: dict[str, Any],
    person_number: int,
    display_position: tuple[
        float,
        float,
        float,
    ],
) -> None:

    points = extract_joints(
        person
    )

    if not points:
        return

    # Normalize body around pelvis.
    normalized = normalize_person(
        points
    )

    if not normalized:
        return

    display_offset = np.asarray(
        display_position,
        dtype=float,
    )

    # --------------------------------------------------------
    # Draw larger joint points.
    # --------------------------------------------------------

    for (
        joint_id,
        point,
    ) in normalized.items():

        point_array = (
            np.asarray(
                point,
                dtype=float,
            )
            + display_offset
        )

        x = point_array[0]
        y = point_array[1]
        z = point_array[2]

        # Display axes:
        #
        # X -> horizontal
        # Z -> depth
        # Y -> vertical

        ax.scatter(
            x,
            z,
            y,
            s=JOINT_SIZE,
            depthshade=True,
        )

    # --------------------------------------------------------
    # Draw connecting bones.
    # --------------------------------------------------------

    for (
        start_id,
        end_id,
    ) in SKELETON_CONNECTIONS:

        if start_id not in normalized:
            continue

        if end_id not in normalized:
            continue

        start = (
            np.asarray(
                normalized[start_id],
                dtype=float,
            )
            + display_offset
        )

        end = (
            np.asarray(
                normalized[end_id],
                dtype=float,
            )
            + display_offset
        )

        ax.plot(
            [
                start[0],
                end[0],
            ],
            [
                start[2],
                end[2],
            ],
            [
                start[1],
                end[1],
            ],
            linewidth=BONE_WIDTH,
        )

    # --------------------------------------------------------
    # Person label.
    # --------------------------------------------------------

    if 0 in normalized:

        nose = (
            np.asarray(
                normalized[0],
                dtype=float,
            )
            + display_offset
        )

        ax.text(
            nose[0],
            nose[2],
            nose[1],
            f"P{person_number}",
            fontsize=LABEL_SIZE,
        )


# ============================================================
# UPDATE VIEW
# ============================================================

def update_view(
    ax,
    persons: list[
        dict[str, Any]
    ],
    display_positions: list[
        tuple[float, float, float]
    ],
) -> None:

    if not persons:

        ax.set_xlim(
            -MIN_VIEW_SIZE,
            MIN_VIEW_SIZE,
        )

        ax.set_ylim(
            -MIN_VIEW_SIZE,
            MIN_VIEW_SIZE,
        )

        ax.set_zlim(
            -MIN_VIEW_SIZE,
            MIN_VIEW_SIZE,
        )

        return

    displayed_points = []

    for index, person in enumerate(
        persons
    ):

        points = extract_joints(
            person
        )

        normalized = normalize_person(
            points
        )

        if not normalized:
            continue

        offset = np.asarray(
            display_positions[
                index
            ],
            dtype=float,
        )

        for point in normalized.values():

            displayed_points.append(
                np.asarray(
                    point,
                    dtype=float,
                )
                + offset
            )

    if not displayed_points:
        return

    array = np.asarray(
        displayed_points,
        dtype=float,
    )

    minimum = array.min(
        axis=0
    )

    maximum = array.max(
        axis=0
    )

    center = (
        minimum
        + maximum
    ) / 2.0

    span = (
        maximum
        - minimum
    )

    largest_span = max(
        float(span.max()),
        4.0,
    )

    # --------------------------------------------------------
    # Tighter view around the enlarged skeletons.
    # --------------------------------------------------------

    half_size = max(
        largest_span / 2.0
        + VIEW_PADDING,
        MIN_VIEW_SIZE / 2.0,
    )

    ax.set_xlim(
        center[0] - half_size,
        center[0] + half_size,
    )

    ax.set_ylim(
        center[2] - half_size,
        center[2] + half_size,
    )

    ax.set_zlim(
        center[1] - half_size,
        center[1] + half_size,
    )

    try:

        ax.set_box_aspect(
            (
                1,
                1,
                1,
            )
        )

    except Exception:
        pass


# ============================================================
# HMR BACKGROUND WORKER
# ============================================================

class HMRWorker:

    def __init__(self):

        print(
            "[HMR] Loading HMR2 model..."
        )

        self.processor = (
            HMRProcessor()
        )

        self.lock = (
            threading.Lock()
        )

        self.pending_frame = None

        self.latest_result = None

        self.stop_requested = False

        self.processed_frames = 0

        self.thread = threading.Thread(
            target=self._worker_loop,
            name="BAS-HMR-Worker",
            daemon=False,
        )

    def start(self):

        print(
            "[HMR] Background worker started."
        )

        self.thread.start()

    def submit(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        boxes: list[list[float]],
    ) -> None:

        with self.lock:

            if self.stop_requested:
                return

            self.pending_frame = {

                "frame": frame.copy(),

                "frame_id": int(
                    frame_id
                ),

                "timestamp": float(
                    timestamp
                ),

                "boxes": boxes,
            }

    def get_latest_result(
        self,
    ) -> Optional[
        dict[str, Any]
    ]:

        with self.lock:

            return self.latest_result

    def _worker_loop(
        self,
    ) -> None:

        while True:

            with self.lock:

                if (
                    self.stop_requested
                    and self.pending_frame
                    is None
                ):
                    break

                current = (
                    self.pending_frame
                )

                self.pending_frame = None

            if current is None:

                time.sleep(
                    0.05
                )

                continue

            frame = current[
                "frame"
            ]

            frame_id = current[
                "frame_id"
            ]

            timestamp = current[
                "timestamp"
            ]

            boxes = current[
                "boxes"
            ]

            try:

                start_time = (
                    time.perf_counter()
                )

                result = (
                    self.processor.process_frame(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        boxes=boxes,
                    )
                )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                with self.lock:

                    self.latest_result = (
                        result
                    )

                    self.processed_frames += 1

                detected = len(
                    result.get(
                        "persons",
                        [],
                    )
                )

                print(
                    f"[HMR] Completed "
                    f"frame {frame_id} | "
                    f"persons={detected} | "
                    f"time={elapsed:.2f}s"
                )

                save_hmr_json(
                    result
                )

            except Exception as error:

                print(
                    f"[HMR] ERROR "
                    f"frame {frame_id}: "
                    f"{error}"
                )

        print(
            "[HMR] Worker stopped. "
            f"Processed="
            f"{self.processed_frames}"
        )

    def stop(self):

        print(
            "[HMR] Stopping worker..."
        )

        with self.lock:

            self.stop_requested = True

            self.pending_frame = None

        self.thread.join()

        print(
            "[HMR] Worker shutdown complete."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        " BAS-HMR LARGE MULTI-PERSON "
        "3D STRUCTURAL VIEWER"
    )
    print("=" * 70)
    print()

    create_output_directory()

    camera = None
    hmr_worker = None
    figure = None

    last_displayed_frame = -1

    try:

        # ====================================================
        # CAMERA
        # ====================================================

        print(
            "[CAMERA] Starting..."
        )

        camera = CameraWorker(
            camera=CAMERA_DEVICE,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        )

        camera.start()

        print(
            "[CAMERA] Ready."
        )

        # ====================================================
        # YOLO
        # ====================================================

        print(
            "[YOLO] Loading..."
        )

        yolo = YOLOProcessor(
            model_path=YOLO_MODEL,
            device=YOLO_DEVICE,
        )

        print(
            "[YOLO] Ready."
        )

        # ====================================================
        # HMR
        # ====================================================

        hmr_worker = HMRWorker()

        hmr_worker.start()

        # ====================================================
        # 3D FIGURE
        # ====================================================

        plt.ion()

        figure = plt.figure(
            "BAS-HMR Large Multi-Person 3D",
            figsize=(
                15,
                11,
            ),
        )

        ax = figure.add_subplot(
            111,
            projection="3d",
        )

        ax.grid(
            True
        )

        try:

            ax.set_box_aspect(
                (
                    1,
                    1,
                    1,
                )
            )

        except Exception:
            pass

        ax.view_init(
            elev=12,
            azim=-72,
        )

        ax.set_xlabel(
            "X"
        )

        ax.set_ylabel(
            "Z"
        )

        ax.set_zlabel(
            "Y"
        )

        ax.set_title(
            "BAS-HMR | Waiting..."
        )

        ax.set_xlim(
            -MIN_VIEW_SIZE,
            MIN_VIEW_SIZE,
        )

        ax.set_ylim(
            -MIN_VIEW_SIZE,
            MIN_VIEW_SIZE,
        )

        ax.set_zlim(
            -MIN_VIEW_SIZE,
            MIN_VIEW_SIZE,
        )

        plt.show(
            block=False
        )

        print()
        print("=" * 70)
        print(
            " LIVE STRUCTURAL VIEWER STARTED"
        )
        print("=" * 70)
        print()
        print(
            "Multi-person detection : ENABLED"
        )
        print(
            "Large skeleton display : ENABLED"
        )
        print(
            "Skeleton scaling       : "
            f"{SKELETON_SCALE}x"
        )
        print(
            "Person separation      : "
            f"{MIN_PERSON_SEPARATION}"
        )
        print(
            "JSON recording         : ENABLED"
        )
        print()
        print(
            "Press Ctrl+C to stop."
        )
        print()

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            packet = camera.read()

            if packet is None:

                time.sleep(
                    0.001
                )

                continue

            frame = packet[
                "frame"
            ]

            frame_id = int(
                packet["frame_id"]
            )

            timestamp = float(
                packet["timestamp"]
            )

            # =================================================
            # YOLO
            # =================================================

            if (
                frame_id
                % YOLO_INTERVAL
                == 0
            ):

                try:

                    yolo_result = (
                        yolo.process_frame(
                            frame=frame,
                            frame_id=frame_id,
                            timestamp=timestamp,
                        )
                    )

                    persons = (
                        yolo_result.get(
                            "persons",
                            [],
                        )
                    )

                    boxes = (
                        extract_hmr_boxes(
                            yolo_result
                        )
                    )

                    print(
                        f"[YOLO] Frame "
                        f"{frame_id} | "
                        f"people="
                        f"{len(persons)} | "
                        f"boxes="
                        f"{len(boxes)}"
                    )

                    hmr_worker.submit(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        boxes=boxes,
                    )

                except Exception as error:

                    print(
                        f"[YOLO] ERROR "
                        f"frame {frame_id}: "
                        f"{error}"
                    )

            # =================================================
            # HMR RESULT
            # =================================================

            result = (
                hmr_worker.get_latest_result()
            )

            if result is not None:

                result_frame = int(
                    result.get(
                        "frame_id",
                        -1,
                    )
                )

                if (
                    result_frame
                    != last_displayed_frame
                ):

                    last_displayed_frame = (
                        result_frame
                    )

                    result_persons = (
                        result.get(
                            "persons",
                            [],
                        )
                    )

                    # =========================================
                    # Calculate visualization positions.
                    # =========================================

                    display_positions = (
                        calculate_person_positions(
                            result_persons
                        )
                    )

                    # =========================================
                    # Clear old frame.
                    # =========================================

                    ax.clear()

                    # =========================================
                    # Draw every person.
                    # =========================================

                    for (
                        person_index,
                        person,
                    ) in enumerate(
                        result_persons
                    ):

                        draw_person_skeleton(
                            ax=ax,
                            person=person,
                            person_number=person_index,
                            display_position=(
                                display_positions[
                                    person_index
                                ]
                            ),
                        )

                    # =========================================
                    # Update view.
                    # =========================================

                    update_view(
                        ax=ax,
                        persons=result_persons,
                        display_positions=(
                            display_positions
                        ),
                    )

                    # =========================================
                    # Restore visualization settings.
                    # =========================================

                    ax.grid(
                        True
                    )

                    try:

                        ax.set_box_aspect(
                            (
                                1,
                                1,
                                1,
                            )
                        )

                    except Exception:
                        pass

                    ax.view_init(
                        elev=12,
                        azim=-72,
                    )

                    ax.set_xlabel(
                        "X"
                    )

                    ax.set_ylabel(
                        "Z"
                    )

                    ax.set_zlabel(
                        "Y"
                    )

                    ax.set_title(
                        "BAS-HMR | "
                        f"Frame={result_frame} | "
                        f"Persons="
                        f"{len(result_persons)}"
                    )

                    figure.canvas.draw_idle()

                    figure.canvas.flush_events()

            # =================================================
            # KEEP GUI RESPONSIVE
            # =================================================

            plt.pause(
                0.001
            )

    except KeyboardInterrupt:

        print()
        print(
            "[VIEWER] Ctrl+C received."
        )

    except Exception as error:

        print()
        print(
            f"[VIEWER] ERROR: {error}"
        )

    finally:

        print()
        print("=" * 70)
        print(
            " SHUTTING DOWN"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # HMR
        # ----------------------------------------------------

        if hmr_worker is not None:

            try:

                hmr_worker.stop()

            except Exception as error:

                print(
                    f"[HMR] Shutdown error: "
                    f"{error}"
                )

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        if camera is not None:

            try:

                stop_method = getattr(
                    camera,
                    "stop",
                    None,
                )

                if callable(
                    stop_method
                ):

                    stop_method()

            except Exception as error:

                print(
                    f"[CAMERA] Stop error: "
                    f"{error}"
                )

            try:

                release_method = getattr(
                    camera,
                    "release",
                    None,
                )

                if callable(
                    release_method
                ):

                    release_method()

            except Exception as error:

                print(
                    f"[CAMERA] Release error: "
                    f"{error}"
                )

            print(
                "[CAMERA] Camera released."
            )

        # ----------------------------------------------------
        # GUI
        # ----------------------------------------------------

        try:

            plt.ioff()

            plt.close(
                "all"
            )

        except Exception as error:

            print(
                f"[VIEWER] GUI shutdown error: "
                f"{error}"
            )

        print()
        print(
            " CLEAN SHUTDOWN COMPLETE"
        )

        print()
        print(
            "JSON output:"
        )

        print(
            OUTPUT_DIR
        )

        print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()