from __future__ import annotations

from typing import Any, Optional

import math


BODY_LEFT_WRIST = 7
BODY_RIGHT_WRIST = 4


class HandBodyFusion:

    def __init__(
        self,
        wrist_margin: float = 0.25,
        min_hand_confidence: float = 0.20,
    ) -> None:

        self.wrist_margin = wrist_margin
        self.min_hand_confidence = min_hand_confidence

    # ========================================================
    # BASIC GEOMETRY
    # ========================================================

    @staticmethod
    def point_in_box(
        x: float,
        y: float,
        box: dict[str, float],
        margin: float = 0.0,
    ) -> bool:

        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])

        width = x2 - x1
        height = y2 - y1

        x1 -= width * margin
        x2 += width * margin

        y1 -= height * margin
        y2 += height * margin

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    @staticmethod
    def distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:

        return math.sqrt(
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
        )

    # ========================================================
    # HAND -> PERSON ASSOCIATION
    # ========================================================

    def associate_hand(
        self,
        hand: dict[str, Any],
        person: dict[str, Any],
    ) -> tuple[float, Optional[str]]:

        bbox = person.get("bbox")

        if not bbox:
            return float("inf"), None

        wrist = hand.get("wrist")

        if not wrist:
            center = hand.get("center")

            if not center:
                return float("inf"), None

            hx = float(center["x"])
            hy = float(center["y"])

        else:

            hx = float(wrist["x"])
            hy = float(wrist["y"])

        px = (
            float(bbox["x1"])
            +
            float(bbox["x2"])
        ) / 2.0

        py = (
            float(bbox["y1"])
            +
            float(bbox["y2"])
        ) / 2.0

        person_width = max(
            float(bbox["x2"]) -
            float(bbox["x1"]),
            1.0,
        )

        person_height = max(
            float(bbox["y2"]) -
            float(bbox["y1"]),
            1.0,
        )

        # Normalized distance from person center.
        distance = math.sqrt(
            (
                (hx - px)
                / person_width
            ) ** 2
            +
            (
                (hy - py)
                / person_height
            ) ** 2
        )

        # Determine side using horizontal position
        # relative to person center.
        #
        # NOTE:
        # This is image-space side, not anatomical
        # left/right. Mirrored webcam images may reverse it.

        if hx < px:
            side = "image_left"
        else:
            side = "image_right"

        return distance, side

    # ========================================================
    # FUSE FRAME
    # ========================================================

    def fuse(
        self,
        hmr_result: dict[str, Any],
        hand_result: dict[str, Any],
    ) -> dict[str, Any]:

        hmr_persons = hmr_result.get(
            "persons",
            [],
        )

        hands = hand_result.get(
            "hands",
            [],
        )

        fused_persons = []

        for person in hmr_persons:

            person_id = person.get(
                "person_id",
                -1,
            )

            person_copy = dict(person)

            person_copy["hands"] = []

            fused_persons.append(
                person_copy
            )

        # ----------------------------------------------------
        # Associate every hand with the nearest person.
        # ----------------------------------------------------

        for hand in hands:

            confidence = float(
                hand.get(
                    "confidence",
                    0.0,
                )
            )

            if confidence < self.min_hand_confidence:
                continue

            best_index = None
            best_distance = float("inf")
            best_side = None

            for index, person in enumerate(
                hmr_persons
            ):

                distance, side = (
                    self.associate_hand(
                        hand,
                        person,
                    )
                )

                if distance < best_distance:

                    best_distance = distance
                    best_index = index
                    best_side = side

            if best_index is None:
                continue

            # Reject hands that are too far away
            # from every detected person.
            if best_distance > 1.25:
                continue

            associated_person = (
                fused_persons[best_index]
            )

            hand_copy = dict(hand)

            hand_copy["association"] = {
                "person_id":
                    associated_person.get(
                        "person_id",
                        -1,
                    ),

                "image_side":
                    best_side,

                "normalized_distance":
                    best_distance,
            }

            associated_person[
                "hands"
            ].append(hand_copy)

        return {
            "frame_id": hmr_result.get(
                "frame_id",
                hand_result.get(
                    "frame_id",
                    -1,
                ),
            ),

            "timestamp": hmr_result.get(
                "timestamp",
                hand_result.get(
                    "timestamp",
                    0.0,
                ),
            ),

            "image": hand_result.get(
                "image",
                {},
            ),

            "persons": fused_persons,
        }
