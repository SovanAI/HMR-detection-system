from __future__ import annotations

from typing import Any
import numpy as np


class HandTracker:
    """
    Temporal hand tracker for HandProcessor output.

    Features:
        - Stable hand IDs
        - Temporary detection-loss tolerance
        - Position smoothing
        - Velocity calculation
        - Velocity limiting
        - Track expiration

    Coordinates are image pixels.
    Velocity is pixels/frame, NOT meters/second.
    """

    def __init__(
        self,
        max_distance: float = 180.0,
        max_missed_frames: int = 15,
        smoothing_alpha: float = 0.65,
        max_speed: float = 60.0,
    ):
        self.max_distance = float(max_distance)

        self.max_missed_frames = int(
            max_missed_frames
        )

        self.smoothing_alpha = float(
            smoothing_alpha
        )

        self.max_speed = float(
            max_speed
        )

        self.next_track_id = 0

        self.tracks: dict[int, dict[str, Any]] = {}

    # ========================================================
    # CENTER
    # ========================================================

    @staticmethod
    def get_center(
        hand: dict[str, Any]
    ) -> np.ndarray:

        return np.array(
            [
                float(hand["center"]["x"]),
                float(hand["center"]["y"]),
            ],
            dtype=np.float32,
        )

    # ========================================================
    # CREATE NEW TRACK
    # ========================================================

    def create_track(
        self,
        hand: dict[str, Any],
        frame_id: int,
        timestamp: float,
    ) -> int:

        track_id = self.next_track_id

        self.next_track_id += 1

        center = self.get_center(hand)

        self.tracks[track_id] = {
            "track_id": track_id,

            "last_center": center.copy(),

            "smoothed_center": center.copy(),

            "last_frame_id": int(frame_id),

            "last_timestamp": float(timestamp),

            "missed_frames": 0,

            "velocity": np.zeros(
                2,
                dtype=np.float32,
            ),

            "age": 1,

            "hits": 1,
        }

        return track_id

    # ========================================================
    # FIND CLOSEST TRACK
    # ========================================================

    def find_best_track(
        self,
        hand: dict[str, Any],
        used_tracks: set[int],
    ) -> int | None:

        current_center = self.get_center(hand)

        best_track_id = None

        best_distance = self.max_distance

        for track_id, track in self.tracks.items():

            if track_id in used_tracks:
                continue

            predicted_center = (
                track["smoothed_center"]
                + track["velocity"]
            )

            distance = float(
                np.linalg.norm(
                    current_center
                    - predicted_center
                )
            )

            if distance < best_distance:

                best_distance = distance

                best_track_id = track_id

        return best_track_id

    # ========================================================
    # UPDATE EXISTING TRACK
    # ========================================================

    def update_track(
        self,
        track_id: int,
        hand: dict[str, Any],
        frame_id: int,
        timestamp: float,
    ) -> dict[str, Any]:

        track = self.tracks[track_id]

        current_center = self.get_center(hand)

        previous_center = (
            track["smoothed_center"]
        )

        previous_frame = (
            track["last_frame_id"]
        )

        frame_delta = max(
            1,
            int(frame_id - previous_frame),
        )

        # ----------------------------------------------------
        # RAW MOVEMENT
        # ----------------------------------------------------

        raw_velocity = (
            current_center
            - previous_center
        ) / frame_delta

        raw_speed = float(
            np.linalg.norm(raw_velocity)
        )

        # ----------------------------------------------------
        # LIMIT EXTREME DETECTOR JUMPS
        # ----------------------------------------------------

        if raw_speed > self.max_speed:

            scale = (
                self.max_speed
                / raw_speed
            )

            limited_velocity = (
                raw_velocity * scale
            )

            predicted_center = (
                previous_center
                + limited_velocity
                * frame_delta
            )

            # Use the limited position
            current_center = (
                predicted_center
            )

        # ----------------------------------------------------
        # EXPONENTIAL POSITION SMOOTHING
        # ----------------------------------------------------

        alpha = self.smoothing_alpha

        smoothed_center = (
            alpha * current_center
            + (1.0 - alpha)
            * previous_center
        )

        # ----------------------------------------------------
        # SMOOTHED VELOCITY
        # ----------------------------------------------------

        velocity = (
            smoothed_center
            - previous_center
        ) / frame_delta

        speed = float(
            np.linalg.norm(velocity)
        )

        # ----------------------------------------------------
        # UPDATE TRACK
        # ----------------------------------------------------

        track["last_center"] = (
            current_center.copy()
        )

        track["smoothed_center"] = (
            smoothed_center.copy()
        )

        track["last_frame_id"] = (
            int(frame_id)
        )

        track["last_timestamp"] = (
            float(timestamp)
        )

        track["missed_frames"] = 0

        track["velocity"] = (
            velocity.copy()
        )

        track["age"] += frame_delta

        track["hits"] += 1

        # ----------------------------------------------------
        # UPDATE DETECTION
        # ----------------------------------------------------

        hand["hand_id"] = track_id

        # Replace displayed center with smoothed center
        hand["center"] = {
            "x": float(smoothed_center[0]),
            "y": float(smoothed_center[1]),
        }

        hand["tracking"] = {

            "track_id": track_id,

            "velocity": {
                "x": float(velocity[0]),
                "y": float(velocity[1]),
            },

            "speed": speed,

            "raw_speed": raw_speed,

            "missed_frames": 0,

            "age": track["age"],

            "hits": track["hits"],

            "state": "tracked",
        }

        return hand

    # ========================================================
    # HANDLE TEMPORARY MISSED DETECTION
    # ========================================================

    def handle_missed_tracks(
        self,
        frame_id: int,
    ):

        for track in self.tracks.values():

            track["missed_frames"] += 1

            # Predict where the hand should be
            predicted_center = (
                track["smoothed_center"]
                + track["velocity"]
            )

            track["smoothed_center"] = (
                predicted_center
            )

    # ========================================================
    # REMOVE EXPIRED TRACKS
    # ========================================================

    def remove_old_tracks(self):

        expired = []

        for track_id, track in self.tracks.items():

            if (
                track["missed_frames"]
                > self.max_missed_frames
            ):

                expired.append(track_id)

        for track_id in expired:

            del self.tracks[track_id]

    # ========================================================
    # MAIN UPDATE
    # ========================================================

    def update(
        self,
        hand_result: dict[str, Any],
    ) -> dict[str, Any]:

        frame_id = int(
            hand_result["frame_id"]
        )

        timestamp = float(
            hand_result["timestamp"]
        )

        detections = hand_result.get(
            "hands",
            [],
        )

        # ----------------------------------------------------
        # NO DETECTIONS
        # ----------------------------------------------------

        if len(detections) == 0:

            self.handle_missed_tracks(
                frame_id
            )

            self.remove_old_tracks()

            hand_result["hands"] = []

            hand_result["tracking"] = {

                "active_tracks": len(
                    self.tracks
                ),

                "detected_hands": 0,

                "tracking_status":
                    "prediction_only"
                    if self.tracks
                    else "no_tracks",
            }

            return hand_result

        # ----------------------------------------------------
        # MATCH DETECTIONS
        # ----------------------------------------------------

        used_tracks: set[int] = set()

        updated_hands = []

        for hand in detections:

            track_id = self.find_best_track(
                hand,
                used_tracks,
            )

            # ------------------------------------------------
            # NEW TRACK
            # ------------------------------------------------

            if track_id is None:

                track_id = self.create_track(
                    hand,
                    frame_id,
                    timestamp,
                )

                center = self.get_center(
                    hand
                )

                hand["hand_id"] = track_id

                hand["center"] = {
                    "x": float(center[0]),
                    "y": float(center[1]),
                }

                hand["tracking"] = {

                    "track_id": track_id,

                    "velocity": {
                        "x": 0.0,
                        "y": 0.0,
                    },

                    "speed": 0.0,

                    "raw_speed": 0.0,

                    "missed_frames": 0,

                    "age": 1,

                    "hits": 1,

                    "state": "new",
                }

            # ------------------------------------------------
            # EXISTING TRACK
            # ------------------------------------------------

            else:

                hand = self.update_track(
                    track_id,
                    hand,
                    frame_id,
                    timestamp,
                )

            used_tracks.add(
                track_id
            )

            updated_hands.append(
                hand
            )

        # ----------------------------------------------------
        # MARK UNMATCHED TRACKS
        # ----------------------------------------------------

        for track_id, track in self.tracks.items():

            if track_id not in used_tracks:

                track["missed_frames"] += 1

        # ----------------------------------------------------
        # REMOVE OLD TRACKS
        # ----------------------------------------------------

        self.remove_old_tracks()

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        hand_result["hands"] = (
            updated_hands
        )

        hand_result["tracking"] = {

            "active_tracks": len(
                self.tracks
            ),

            "detected_hands": len(
                updated_hands
            ),

            "tracking_status":
                "tracked"
                if updated_hands
                else "prediction_only",
        }

        return hand_result