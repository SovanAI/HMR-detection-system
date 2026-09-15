"""
BAS-HMR
Stage 5 - Live Metric Person-Chair Distance

Pipeline:

Camera
    ↓
YOLO11
    ↓
Person + Chair detection
    ↓
Lightweight Person tracking
    ↓
Chair tracking
    ↓
Reference points
    ↓
Metric Depth
    ↓
Camera Geometry
    ↓
Person / Chair 3D coordinates
    ↓
3D + Ground distance
    ↓
Flask MJPEG stream

IMPORTANT:
The current camera intrinsics are DEVELOPMENT values.
Final physical accuracy requires scene calibration and validation.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

from ultralytics import YOLO

from human.depth.metric_depth import MetricDepthEstimator
from human.distance.reference_points import ReferencePointExtractor
from human.distance.id_pair_tracker import IDPairTracker
from human.distance.geometry_estimator import (
    GeometryEstimator,
    PixelPoint,
    MetricPoint,
    euclidean_distance,
    ground_distance,
)
from detection.chair_tracker import (
    ChairDetection,
    ChairTracker,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

CAMERA_DEVICE = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10

YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 416

# COCO classes
PERSON_CLASS = 0
CHAIR_CLASS = 56

# Development camera intrinsics.
# DO NOT treat these as final calibrated values.
FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0

# Metric depth is expensive on CPU.
# Recalculate every N frames and reuse the latest depth map.
DEPTH_INTERVAL = 3

# Flask
HOST = "0.0.0.0"
PORT = 5003


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class TrackedPerson:
    person_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int]
    missed_frames: int = 0


@dataclass
class DistanceRecord:
    person_id: int
    chair_id: int

    person_point: PixelPoint
    chair_point: PixelPoint

    person_3d: MetricPoint
    chair_3d: MetricPoint

    distance_3d: float
    ground_distance: float
    depth_difference: float

    timestamp: float


# =====================================================================
# PERSON TRACKER
# =====================================================================

class PersonTracker:
    """
    Lightweight CPU-friendly person tracker.

    Matching uses image-space center distance.
    """

    def __init__(
        self,
        max_center_distance: float = 120.0,
        max_missed_frames: int = 10,
    ) -> None:

        self.max_center_distance = float(
            max_center_distance
        )

        self.max_missed_frames = int(
            max_missed_frames
        )

        self.next_id = 0
        self.tracks: List[TrackedPerson] = []

    @staticmethod
    def center(
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int]:

        x1, y1, x2, y2 = bbox

        return (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2),
        )

    @staticmethod
    def center_distance(
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> float:

        dx = a[0] - b[0]
        dy = a[1] - b[1]

        return math.sqrt(
            dx * dx + dy * dy
        )

    def update(
        self,
        detections: List[dict],
    ) -> List[TrackedPerson]:

        centers = [
            self.center(d["bbox"])
            for d in detections
        ]

        candidates = []

        for ti, track in enumerate(self.tracks):

            for di, center in enumerate(centers):

                distance = self.center_distance(
                    track.center,
                    center,
                )

                if distance <= self.max_center_distance:

                    candidates.append(
                        (
                            distance,
                            ti,
                            di,
                        )
                    )

        candidates.sort(
            key=lambda x: x[0]
        )

        matched_tracks = set()
        matched_detections = set()

        for (
            distance,
            ti,
            di,
        ) in candidates:

            if ti in matched_tracks:
                continue

            if di in matched_detections:
                continue

            track = self.tracks[ti]
            detection = detections[di]

            matched_tracks.add(ti)
            matched_detections.add(di)

            old_box = track.bbox
            new_box = detection["bbox"]

            alpha = 0.65

            smoothed_box = tuple(
                int(
                    alpha * new_box[i]
                    + (1.0 - alpha) * old_box[i]
                )
                for i in range(4)
            )

            track.bbox = smoothed_box
            track.center = self.center(
                smoothed_box
            )
            track.confidence = (
                detection["confidence"]
            )
            track.missed_frames = 0

        # Unmatched tracks
        for ti, track in enumerate(self.tracks):

            if ti not in matched_tracks:

                track.missed_frames += 1

        # New tracks
        for di, detection in enumerate(detections):

            if di in matched_detections:
                continue

            bbox = detection["bbox"]

            self.tracks.append(
                TrackedPerson(
                    person_id=self.next_id,
                    bbox=bbox,
                    confidence=detection["confidence"],
                    center=self.center(bbox),
                )
            )

            self.next_id += 1

        # Remove stale tracks
        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames
            <= self.max_missed_frames
        ]

        return list(self.tracks)


# =====================================================================
# CAMERA
# =====================================================================

class Camera:

    def __init__(self) -> None:

        self.cap = cv2.VideoCapture(
            CAMERA_DEVICE,
            cv2.CAP_V4L2,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera: "
                f"{CAMERA_DEVICE}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            WIDTH,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            HEIGHT,
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            FPS,
        )

        # MJPG reduces USB bandwidth.
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            ),
        )

    def read(self):

        ok, frame = self.cap.read()

        if not ok:
            return None

        return frame

    def release(self):

        self.cap.release()


# =====================================================================
# DISTANCE SYSTEM
# =====================================================================

class LiveMetricDistanceSystem:

    def __init__(self):

        print()
        print("=" * 70)
        print("INITIALIZING BAS-HMR DISTANCE SYSTEM")
        print("=" * 70)

        print("[1/6] Loading YOLO...")

        self.yolo = YOLO(
            YOLO_MODEL
        )

        print("[YOLO] Loaded.")

        print("[2/6] Loading metric depth...")

        self.depth_estimator = (
            MetricDepthEstimator()
        )

        print("[DEPTH] Loaded.")

        print("[3/6] Creating trackers...")

        self.person_tracker = PersonTracker()

        self.chair_tracker = ChairTracker(
            max_center_distance=120.0,
            min_iou=0.05,
            max_missed_frames=10,
        )

        self.pair_tracker = IDPairTracker(
            max_pair_distance_px=400.0,
            max_missed_frames=15,
        )

        print("[TRACKERS] Ready.")

        print("[4/6] Creating reference extractor...")

        self.reference_extractor = (
            ReferencePointExtractor(
                frame_width=WIDTH,
                frame_height=HEIGHT,
            )
        )

        print("[REFERENCE POINTS] Ready.")

        print("[5/6] Creating geometry estimator...")

        self.geometry = GeometryEstimator(
            fx=FX,
            fy=FY,
            cx=CX,
            cy=CY,
            width=WIDTH,
            height=HEIGHT,
        )

        print("[GEOMETRY] Ready.")

        print("[6/6] Starting camera...")

        self.camera = Camera()

        print("[CAMERA] Ready.")

        self.frame_id = 0
        self.last_depth_map = None
        self.last_depth_frame_id = -1

        self.latest_frame = None
        self.latest_records: List[DistanceRecord] = []

        self.lock = threading.Lock()
        self.running = True

        self.last_process_time = time.time()

        print()
        print("=" * 70)
        print("DISTANCE SYSTEM READY")
        print("=" * 70)
        print()
        print(
            "WARNING: Camera intrinsics are DEVELOPMENT values."
        )
        print(
            "Final accuracy requires scene calibration."
        )
        print()

    # -----------------------------------------------------------------
    # YOLO
    # -----------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
    ):

        results = self.yolo(
            frame,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        persons = []
        chairs = []

        if result.boxes is None:
            return persons, chairs

        boxes = result.boxes

        for i in range(len(boxes)):

            class_id = int(
                boxes.cls[i].item()
            )

            confidence = float(
                boxes.conf[i].item()
            )

            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
            )

            bbox = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )

            if class_id == PERSON_CLASS:

                persons.append(
                    {
                        "bbox": bbox,
                        "confidence": confidence,
                    }
                )

            elif class_id == CHAIR_CLASS:

                chairs.append(
                    ChairDetection(
                        bbox=bbox,
                        confidence=confidence,
                        depth=0.0,
                    )
                )

        return persons, chairs

    # -----------------------------------------------------------------
    # DEPTH
    # -----------------------------------------------------------------

    def update_depth(
        self,
        frame: np.ndarray,
    ):

        if (
            self.last_depth_map is None
            or self.frame_id
            - self.last_depth_frame_id
            >= DEPTH_INTERVAL
        ):

            print(
                f"[DEPTH] Processing frame "
                f"{self.frame_id}..."
            )

            start = time.time()

            self.last_depth_map = (
                self.depth_estimator.estimate(
                    frame
                )
            )

            self.last_depth_frame_id = (
                self.frame_id
            )

            elapsed = time.time() - start

            print(
                f"[DEPTH] Completed in "
                f"{elapsed:.2f}s"
            )

        return self.last_depth_map

    # -----------------------------------------------------------------
    # DEPTH AT REFERENCE POINT
    # -----------------------------------------------------------------

    def get_depth(
        self,
        point: PixelPoint,
    ) -> float:

        if self.last_depth_map is None:
            return float("nan")

        return float(
            self.depth_estimator.depth_at_pixel(
                self.last_depth_map,
                int(point.u),
                int(point.v),
                radius=5,
            )
        )

    # -----------------------------------------------------------------
    # PROCESS FRAME
    # -----------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
    ):

        persons, chairs = self.detect(
            frame
        )

        tracked_persons = (
            self.person_tracker.update(
                persons
            )
        )

        tracked_chairs = (
            self.chair_tracker.update(
                chairs
            )
        )

        depth_map = self.update_depth(
            frame
        )

        # -------------------------------------------------------------
        # Person reference points
        # -------------------------------------------------------------

        person_data = []

        for person in tracked_persons:

            reference = (
                self.reference_extractor
                .person_reference(
                    person.bbox,
                    person.confidence,
                )
            )

            person_data.append(
                {
                    "person_id": person.person_id,
                    "reference_point": (
                        reference.u,
                        reference.v,
                    ),
                    "bbox": person.bbox,
                    "confidence": person.confidence,
                }
            )

        # -------------------------------------------------------------
        # Chair reference points
        # -------------------------------------------------------------

        chair_data = []

        for chair in tracked_chairs:

            reference = (
                self.reference_extractor
                .chair_reference(
                    chair.bbox,
                    chair.confidence,
                )
            )

            chair_data.append(
                {
                    "chair_id": chair.chair_id,
                    "reference_point": (
                        reference.u,
                        reference.v,
                    ),
                    "bbox": chair.bbox,
                    "confidence": chair.confidence,
                }
            )

        # -------------------------------------------------------------
        # Pair persons and chairs
        # -------------------------------------------------------------

        pairs = self.pair_tracker.update(
            person_data,
            chair_data,
        )

        person_lookup = {
            p["person_id"]: p
            for p in person_data
        }

        chair_lookup = {
            c["chair_id"]: c
            for c in chair_data
        }

        records = []

        # -------------------------------------------------------------
        # Metric geometry
        # -------------------------------------------------------------

        for pair in pairs:

            person = person_lookup.get(
                pair.person_id
            )

            chair = chair_lookup.get(
                pair.chair_id
            )

            if person is None:
                continue

            if chair is None:
                continue

            pu, pv = person[
                "reference_point"
            ]

            cu, cv = chair[
                "reference_point"
            ]

            person_pixel = PixelPoint(
                u=pu,
                v=pv,
            )

            chair_pixel = PixelPoint(
                u=cu,
                v=cv,
            )

            person_depth = self.get_depth(
                person_pixel
            )

            chair_depth = self.get_depth(
                chair_pixel
            )

            # Validate depth.
            if not (
                math.isfinite(person_depth)
                and math.isfinite(chair_depth)
            ):
                continue

            if person_depth <= 0:
                continue

            if chair_depth <= 0:
                continue

            # ---------------------------------------------------------
            # Reconstruct 3D points
            # ---------------------------------------------------------

            person_3d = (
                self.geometry.pixel_to_3d(
                    person_pixel,
                    person_depth,
                    confidence=person[
                        "confidence"
                    ],
                    source="metric_depth",
                )
            )

            chair_3d = (
                self.geometry.pixel_to_3d(
                    chair_pixel,
                    chair_depth,
                    confidence=chair[
                        "confidence"
                    ],
                    source="metric_depth",
                )
            )

            # ---------------------------------------------------------
            # Distances
            # ---------------------------------------------------------

            d3 = euclidean_distance(
                person_3d,
                chair_3d,
            )

            dg = ground_distance(
                person_3d,
                chair_3d,
            )

            dz = abs(
                person_3d.z
                - chair_3d.z
            )

            records.append(
                DistanceRecord(
                    person_id=pair.person_id,
                    chair_id=pair.chair_id,
                    person_point=person_pixel,
                    chair_point=chair_pixel,
                    person_3d=person_3d,
                    chair_3d=chair_3d,
                    distance_3d=d3,
                    ground_distance=dg,
                    depth_difference=dz,
                    timestamp=time.time(),
                )
            )

        return tracked_persons, tracked_chairs, records

    # -----------------------------------------------------------------
    # DRAW
    # -----------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        tracked_persons,
        tracked_chairs,
        records,
    ):

        output = frame.copy()

        # -------------------------------------------------------------
        # Persons
        # -------------------------------------------------------------

        for person in tracked_persons:

            x1, y1, x2, y2 = person.bbox

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                f"Person {person.person_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            # Person reference point
            px = int(
                (x1 + x2) / 2
            )

            py = int(y2)

            cv2.circle(
                output,
                (px, py),
                5,
                (0, 255, 255),
                -1,
            )

        # -------------------------------------------------------------
        # Chairs
        # -------------------------------------------------------------

        for chair in tracked_chairs:

            x1, y1, x2, y2 = chair.bbox

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

            cv2.putText(
                output,
                f"Chair {chair.chair_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(y2)

            cv2.circle(
                output,
                (cx, cy),
                5,
                (0, 255, 255),
                -1,
            )

        # -------------------------------------------------------------
        # Distance lines
        # -------------------------------------------------------------

        for record in records:

            px = int(
                record.person_point.u
            )

            py = int(
                record.person_point.v
            )

            cx = int(
                record.chair_point.u
            )

            cy = int(
                record.chair_point.v
            )

            cv2.line(
                output,
                (px, py),
                (cx, cy),
                (0, 255, 255),
                2,
            )

            mid_x = int(
                (px + cx) / 2
            )

            mid_y = int(
                (py + cy) / 2
            )

            cv2.rectangle(
                output,
                (
                    mid_x - 85,
                    mid_y - 45,
                ),
                (
                    mid_x + 85,
                    mid_y + 5,
                ),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                output,
                f"3D: {record.distance_3d:.2f} m",
                (
                    mid_x - 78,
                    mid_y - 25,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output,
                f"Ground: {record.ground_distance:.2f} m",
                (
                    mid_x - 78,
                    mid_y - 7,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )

        # -------------------------------------------------------------
        # Status
        # -------------------------------------------------------------

        cv2.rectangle(
            output,
            (0, 0),
            (WIDTH, 60),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            "BAS-HMR | LIVE METRIC DISTANCE",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            "WARNING: UNCALIBRATED CAMERA GEOMETRY",
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 200, 255),
            1,
        )

        return output

    # -----------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------

    def run(self):

        print()
        print("=" * 70)
        print("STARTING LIVE METRIC DISTANCE")
        print("=" * 70)
        print()
        print(
            f"Open from Windows browser:"
        )
        print(
            f"http://localhost:{PORT}"
        )
        print()

        while self.running:

            frame = self.camera.read()

            if frame is None:

                print(
                    "[CAMERA] Frame read failed."
                )

                time.sleep(0.1)
                continue

            self.frame_id += 1

            try:

                (
                    persons,
                    chairs,
                    records,
                ) = self.process(
                    frame
                )

                output = self.draw(
                    frame,
                    persons,
                    chairs,
                    records,
                )

                with self.lock:

                    self.latest_frame = output

                    self.latest_records = records

            except Exception as exc:

                print(
                    f"[ERROR] {type(exc).__name__}: "
                    f"{exc}"
                )

        self.camera.release()

    # -----------------------------------------------------------------
    # JPEG FRAME
    # -----------------------------------------------------------------

    def get_jpeg(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            frame = self.latest_frame.copy()

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80,
            ],
        )

        if not ok:
            return None

        return encoded.tobytes()


# =====================================================================
# FLASK
# =====================================================================

app = Flask(__name__)

system = None
worker = None


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>BAS-HMR Metric Distance</title>

    <style>
        body {
            margin: 0;
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }

        h1 {
            margin: 15px;
        }

        img {
            width: 90%;
            max-width: 960px;
            border: 2px solid #444;
        }

        .info {
            margin: 15px;
            font-size: 16px;
        }

        .warning {
            color: #ffcc00;
        }
    </style>
</head>

<body>

<h1>BAS-HMR — Live Metric Distance</h1>

<div class="info warning">
    Development geometry — calibration required
</div>

<img src="/video_feed">

</body>
</html>
"""


@app.route("/")
def index():

    return render_template_string(
        HTML
    )


def generate_frames():

    while True:

        if system is None:

            time.sleep(0.2)
            continue

        jpeg = system.get_jpeg()

        if jpeg is None:

            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.route("/api/status")
def status():

    if system is None:

        return jsonify(
            {
                "running": False
            }
        )

    with system.lock:

        records = list(
            system.latest_records
        )

    return jsonify(
        {
            "running": True,
            "frame_id": system.frame_id,
            "pairs": [
                {
                    "person_id": r.person_id,
                    "chair_id": r.chair_id,
                    "distance_3d_m": round(
                        r.distance_3d,
                        3,
                    ),
                    "ground_distance_m": round(
                        r.ground_distance,
                        3,
                    ),
                    "depth_difference_m": round(
                        r.depth_difference,
                        3,
                    ),
                }
                for r in records
            ],
            "geometry": "development_intrinsics",
            "calibrated": False,
        }
    )


# =====================================================================
# START
# =====================================================================

def main():

    global system
    global worker

    system = (
        LiveMetricDistanceSystem()
    )

    worker = threading.Thread(
        target=system.run,
        daemon=True,
    )

    worker.start()

    app.run(
        host=HOST,
        port=PORT,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":

    main()
