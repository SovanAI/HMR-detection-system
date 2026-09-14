from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# HAND KEYPOINT DEFINITIONS
# ============================================================

HAND_KEYPOINT_NAMES = [
    "wrist",

    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",

    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",

    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",

    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",

    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


# ============================================================
# PROCESSOR
# ============================================================

class HandProcessor:

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence: float = 0.20,
        image_size: int = 416,
    ) -> None:

        print("[HAND] Loading hand pose model...")
        print(f"[HAND] Model: {model_path}")

        self.model = YOLO(model_path)

        self.device = device
        self.confidence = confidence
        self.image_size = image_size

        print("[HAND] Model loaded successfully")
        print(f"[HAND] Device: {device}")
        print(f"[HAND] Confidence threshold: {confidence}")

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> dict[str, Any]:

        if frame is None:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "hands": [],
            }

        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        hands: list[dict[str, Any]] = []

        if not results:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "model": "YOLO11n-Pose-Hand",
                "device": self.device,
                "image": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                },
                "hands": hands,
            }

        result = results[0]

        if result.boxes is None:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "model": "YOLO11n-Pose-Hand",
                "device": self.device,
                "image": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                },
                "hands": hands,
            }

        boxes = result.boxes

        keypoints = result.keypoints

        if keypoints is None:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "model": "YOLO11n-Pose-Hand",
                "device": self.device,
                "image": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                },
                "hands": hands,
            }

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        box_conf = boxes.conf.cpu().numpy()

        kp_xy = keypoints.xy.cpu().numpy()

        if keypoints.conf is not None:
            kp_conf = keypoints.conf.cpu().numpy()
        else:
            kp_conf = np.ones(
                (len(kp_xy), 21),
                dtype=np.float32,
            )

        for hand_index in range(len(boxes_xyxy)):

            bbox = boxes_xyxy[hand_index]

            x1, y1, x2, y2 = [
                float(value)
                for value in bbox
            ]

            confidence = float(
                box_conf[hand_index]
            )

            points = []

            for joint_id in range(
                min(21, len(kp_xy[hand_index]))
            ):

                x = float(
                    kp_xy[hand_index][joint_id][0]
                )

                y = float(
                    kp_xy[hand_index][joint_id][1]
                )

                conf = float(
                    kp_conf[hand_index][joint_id]
                )

                points.append(
                    {
                        "joint_id": joint_id,
                        "name": HAND_KEYPOINT_NAMES[joint_id],
                        "x": x,
                        "y": y,
                        "confidence": conf,
                    }
                )

            # ------------------------------------------------
            # Boundary filtering
            #
            # Very weak detections touching the image edge
            # are frequently false duplicates.
            # ------------------------------------------------

            image_height, image_width = frame.shape[:2]

            touches_boundary = (
                x1 <= 2
                or y1 <= 2
                or x2 >= image_width - 2
                or y2 >= image_height - 2
            )

            mean_kp_conf = (
                float(np.mean(kp_conf[hand_index]))
                if len(kp_conf[hand_index]) > 0
                else 0.0
            )

            # Keep boundary detections only if their
            # keypoints are reasonably strong.
            if touches_boundary and mean_kp_conf < 0.65:
                continue

            wrist = points[0] if points else None

            hand_data = {
                "hand_id": len(hands),
                "confidence": confidence,

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "center": {
                    "x": (x1 + x2) / 2.0,
                    "y": (y1 + y2) / 2.0,
                },

                "wrist": wrist,

                "mean_keypoint_confidence":
                    mean_kp_conf,

                "touches_boundary":
                    touches_boundary,

                "keypoints": points,
            }

            hands.append(hand_data)

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,

            "model": "YOLO11n-Pose-Hand",
            "device": self.device,

            "image": {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            },

            "hands": hands,
        }
