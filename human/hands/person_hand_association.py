from __future__ import annotations

from typing import Any
import math


class PersonHandAssociator:
    """
    Associates detected hands with detected people.

    Association is performed primarily using the hand wrist
    position and the person's bounding box.

    This is an image-space association.

    It does NOT claim 3D hand position.
    """

    def __init__(
        self,
        wrist_margin: float = 0.35,
        center_distance_threshold: float = 0.75,
    ):
        self.wrist_margin = float(
            wrist_margin
        )

        self.center_distance_threshold = float(
            center_distance_threshold
        )

    # ========================================================
    # BASIC GEOMETRY
    # ========================================================

    @staticmethod
    def point_inside_bbox(
        x: float,
        y: float,
        bbox: dict[str, Any],
        margin: float = 0.0,
    ) -> bool:

        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])

        width = x2 - x1
        height = y2 - y1

        x1 -= width * margin
        y1 -= height * margin
        x2 += width * margin
        y2 += height * margin

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    @staticmethod
    def bbox_center(
        bbox: dict[str, Any]
    ) -> tuple[float, float]:

        return (
            (
                float(bbox["x1"])
                +
                float(bbox["x2"])
            ) / 2.0,

            (
                float(bbox["y1"])
                +
                float(bbox["y2"])
            ) / 2.0,
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
    # WRIST EXTRACTION
    # ========================================================

    @staticmethod
    def get_wrist(
        hand: dict[str, Any]
    ) -> tuple[float, float] | None:

        wrist = hand.get("wrist")

        if wrist is None:
            return None

        return (
            float(wrist["x"]),
            float(wrist["y"]),
        )

    # ========================================================
    # FIND BEST PERSON
    # ========================================================

    def find_person_for_hand(
        self,
        hand: dict[str, Any],
        persons: list[dict[str, Any]],
    ) -> tuple[int | None, float]:

        if not persons:
            return None, float("inf")

        wrist = self.get_wrist(hand)

        if wrist is None:

            # Fall back to hand center
            center = hand.get(
                "center"
            )

            if center is None:
                return None, float("inf")

            wrist = (
                float(center["x"]),
                float(center["y"]),
            )

        wrist_x, wrist_y = wrist

        best_person_id = None
        best_score = float("inf")

        for person in persons:

            bbox = person["bbox"]

            # ------------------------------------------------
            # Test whether wrist is inside person's bbox
            # ------------------------------------------------

            inside = self.point_inside_bbox(
                wrist_x,
                wrist_y,
                bbox,
                margin=self.wrist_margin,
            )

            center_x, center_y = (
                self.bbox_center(bbox)
            )

            distance = self.distance(
                wrist_x,
                wrist_y,
                center_x,
                center_y,
            )

            # ------------------------------------------------
            # Normalize distance by person size
            # ------------------------------------------------

            person_width = (
                float(bbox["x2"])
                -
                float(bbox["x1"])
            )

            person_height = (
                float(bbox["y2"])
                -
                float(bbox["y1"])
            )

            person_size = max(
                1.0,
                math.sqrt(
                    person_width ** 2
                    +
                    person_height ** 2
                )
            )

            normalized_distance = (
                distance / person_size
            )

            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            if inside:

                score = (
                    normalized_distance
                    * 0.5
                )

            else:

                score = (
                    normalized_distance
                    + 1.0
                )

            if score < best_score:

                best_score = score

                best_person_id = int(
                    person["person_id"]
                )

        # ----------------------------------------------------
        # Reject extremely distant associations
        # ----------------------------------------------------

        if (
            best_person_id is None
            or
            best_score
            > self.center_distance_threshold
        ):

            return None, best_score

        return (
            best_person_id,
            best_score,
        )

    # ========================================================
    # ASSOCIATE ALL HANDS
    # ========================================================

    def associate(
        self,
        persons: list[dict[str, Any]],
        hands: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        associated_hands = []

        for hand in hands:

            person_id, score = (
                self.find_person_for_hand(
                    hand,
                    persons,
                )
            )

            hand_copy = dict(hand)

            hand_copy["person_id"] = (
                person_id
            )

            hand_copy["association"] = {

                "person_id": person_id,

                "score": (
                    float(score)
                    if math.isfinite(score)
                    else None
                ),

                "method":
                    "wrist_bbox_image_space",
            }

            associated_hands.append(
                hand_copy
            )

        return associated_hands

    # ========================================================
    # COMPLETE FRAME FUSION
    # ========================================================

    def fuse(
        self,
        person_result: dict[str, Any],
        hand_result: dict[str, Any],
    ) -> dict[str, Any]:

        persons = person_result.get(
            "persons",
            [],
        )

        hands = hand_result.get(
            "hands",
            [],
        )

        associated_hands = self.associate(
            persons,
            hands,
        )

        # ----------------------------------------------------
        # Attach hands to people
        # ----------------------------------------------------

        fused_persons = []

        for person in persons:

            person_copy = dict(
                person
            )

            person_id = int(
                person["person_id"]
            )

            person_hands = [

                hand

                for hand in associated_hands

                if hand.get("person_id")
                == person_id

            ]

            person_copy["hands"] = (
                person_hands
            )

            person_copy[
                "hand_count"
            ] = len(person_hands)

            fused_persons.append(
                person_copy
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        return {

            "frame_id":
                person_result.get(
                    "frame_id"
                ),

            "timestamp":
                person_result.get(
                    "timestamp"
                ),

            "persons":
                fused_persons,

            "unassociated_hands": [

                hand

                for hand in associated_hands

                if hand.get("person_id")
                is None

            ],

            "metadata": {

                "association_method":
                    "wrist_bbox_image_space",

                "hand_count":
                    len(associated_hands),

                "person_count":
                    len(persons),
            },
        }
