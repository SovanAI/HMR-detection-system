from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math


BBox = Tuple[int, int, int, int]


@dataclass
class ChairDetection:
    bbox: BBox
    confidence: float
    depth: float = 0.0


@dataclass
class TrackedChair:
    chair_id: int
    bbox: BBox
    confidence: float
    depth: float
    center: Tuple[int, int]
    missed_frames: int = 0


class ChairTracker:
    """
    Lightweight CPU-friendly chair tracker.

    Tracking is based on:
    1. Bounding-box center distance
    2. Bounding-box IoU
    3. Maximum missed-frame tolerance
    """

    def __init__(
        self,
        max_center_distance: float = 120.0,
        min_iou: float = 0.05,
        max_missed_frames: int = 10,
    ):
        self.max_center_distance = max_center_distance
        self.min_iou = min_iou
        self.max_missed_frames = max_missed_frames

        self.next_id = 0
        self.tracks: List[TrackedChair] = []

    @staticmethod
    def center(bbox: BBox) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox

        return (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2),
        )

    @staticmethod
    def iou(box_a: BBox, box_b: BBox) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        intersection = iw * ih

        if intersection <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def center_distance(
        center_a: Tuple[int, int],
        center_b: Tuple[int, int],
    ) -> float:

        dx = center_a[0] - center_b[0]
        dy = center_a[1] - center_b[1]

        return math.sqrt(dx * dx + dy * dy)

    def suppress_duplicates(
        self,
        detections: List[ChairDetection],
    ) -> List[ChairDetection]:

        """
        Remove duplicate detections that likely refer to
        the same physical chair.

        Higher-confidence detections are kept first.
        """

        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True,
        )

        kept: List[ChairDetection] = []

        for detection in detections:

            is_duplicate = False

            det_center = self.center(detection.bbox)

            for existing in kept:

                existing_center = self.center(existing.bbox)

                overlap = self.iou(
                    detection.bbox,
                    existing.bbox,
                )

                distance = self.center_distance(
                    det_center,
                    existing_center,
                )

                # Duplicate if boxes overlap significantly
                # OR their centers are extremely close.
                if (
                    overlap >= 0.35
                    or distance < 45
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(detection)

        return kept

    def update(
        self,
        detections: List[ChairDetection],
    ) -> List[TrackedChair]:

        """
        Update chair tracks using current-frame detections.
        """

        detections = self.suppress_duplicates(detections)

        detection_centers = [
            self.center(d.bbox)
            for d in detections
        ]

        matched_tracks = set()
        matched_detections = set()

        # --------------------------------------------------
        # Match existing tracks to current detections
        # --------------------------------------------------

        candidates = []

        for track_index, track in enumerate(self.tracks):

            track_center = track.center

            for detection_index, center in enumerate(
                detection_centers
            ):

                distance = self.center_distance(
                    track_center,
                    center,
                )

                overlap = self.iou(
                    track.bbox,
                    detections[detection_index].bbox,
                )

                if (
                    distance <= self.max_center_distance
                    or overlap >= self.min_iou
                ):
                    candidates.append(
                        (
                            distance,
                            -overlap,
                            track_index,
                            detection_index,
                        )
                    )

        # Best matches first.
        candidates.sort()

        for (
            distance,
            negative_iou,
            track_index,
            detection_index,
        ) in candidates:

            if track_index in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            track = self.tracks[track_index]
            detection = detections[detection_index]

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

            center = self.center(detection.bbox)

            # Smooth bounding box position.
            alpha = 0.65

            old_box = track.bbox
            new_box = detection.bbox

            smoothed_box = tuple(
                int(
                    alpha * new_box[i]
                    + (1.0 - alpha) * old_box[i]
                )
                for i in range(4)
            )

            track.bbox = smoothed_box
            track.center = self.center(smoothed_box)

            track.confidence = detection.confidence
            track.depth = detection.depth
            track.missed_frames = 0

        # --------------------------------------------------
        # Increase missed count for unmatched tracks
        # --------------------------------------------------

        for track_index, track in enumerate(self.tracks):

            if track_index not in matched_tracks:
                track.missed_frames += 1

        # --------------------------------------------------
        # Create new tracks
        # --------------------------------------------------

        for detection_index, detection in enumerate(detections):

            if detection_index in matched_detections:
                continue

            center = self.center(detection.bbox)

            new_track = TrackedChair(
                chair_id=self.next_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                depth=detection.depth,
                center=center,
                missed_frames=0,
            )

            self.tracks.append(new_track)

            self.next_id += 1

        # --------------------------------------------------
        # Remove old tracks
        # --------------------------------------------------

        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= self.max_missed_frames
        ]

        return list(self.tracks)


if __name__ == "__main__":

    tracker = ChairTracker()

    test_frames = [
        [
            ChairDetection(
                bbox=(100, 100, 250, 300),
                confidence=0.85,
            )
        ],
        [
            ChairDetection(
                bbox=(105, 103, 255, 303),
                confidence=0.87,
            )
        ],
        [
            ChairDetection(
                bbox=(110, 105, 260, 305),
                confidence=0.89,
            )
        ],
    ]

    for frame_index, detections in enumerate(test_frames):

        tracks = tracker.update(detections)

        print(
            f"Frame {frame_index}: "
            f"{[(t.chair_id, t.bbox, t.center) for t in tracks]}"
        )
