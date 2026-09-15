"""
BAS-HMR
Person ↔ Chair ID Pair Tracker

Maintains stable relationships between tracked Person IDs and Chair IDs.

Important:
    IDs identify tracked objects.
    They are NOT used to estimate physical distance.

Distance will be calculated later from calibrated geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class PairState:
    """State of one Person ↔ Chair relationship."""

    person_id: int
    chair_id: int

    last_pixel_distance: float
    frames_seen: int = 1
    missed_frames: int = 0

    active: bool = True


class IDPairTracker:
    """
    Tracks Person-ID ↔ Chair-ID relationships.

    Pairing is based on image-space reference-point distance only.
    This is association, NOT physical distance estimation.
    """

    def __init__(
        self,
        max_pair_distance_px: float = 400.0,
        max_missed_frames: int = 15,
    ) -> None:

        if max_pair_distance_px <= 0:
            raise ValueError(
                "max_pair_distance_px must be positive"
            )

        if max_missed_frames < 0:
            raise ValueError(
                "max_missed_frames cannot be negative"
            )

        self.max_pair_distance_px = float(
            max_pair_distance_px
        )

        self.max_missed_frames = int(
            max_missed_frames
        )

        self.pairs: Dict[
            Tuple[int, int],
            PairState
        ] = {}

    # ---------------------------------------------------------------
    # DISTANCE
    # ---------------------------------------------------------------

    @staticmethod
    def pixel_distance(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:

        dx = float(point_a[0]) - float(point_b[0])
        dy = float(point_a[1]) - float(point_b[1])

        return math.sqrt(
            dx * dx + dy * dy
        )

    # ---------------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------------

    def update(
        self,
        persons: List[dict],
        chairs: List[dict],
    ) -> List[PairState]:
        """
        Update Person ↔ Chair associations.

        Expected format:

        persons = [
            {
                "person_id": 1,
                "reference_point": (u, v)
            }
        ]

        chairs = [
            {
                "chair_id": 1,
                "reference_point": (u, v)
            }
        ]
        """

        candidates = []

        for person in persons:

            person_id = int(
                person["person_id"]
            )

            person_point = tuple(
                person["reference_point"]
            )

            for chair in chairs:

                chair_id = int(
                    chair["chair_id"]
                )

                chair_point = tuple(
                    chair["reference_point"]
                )

                distance_px = self.pixel_distance(
                    person_point,
                    chair_point,
                )

                if (
                    distance_px
                    <= self.max_pair_distance_px
                ):
                    candidates.append(
                        (
                            distance_px,
                            person_id,
                            chair_id,
                        )
                    )

        # Closest pairs first.
        candidates.sort(
            key=lambda item: item[0]
        )

        used_persons = set()
        used_chairs = set()
        matched_pairs = set()

        for (
            distance_px,
            person_id,
            chair_id,
        ) in candidates:

            if person_id in used_persons:
                continue

            if chair_id in used_chairs:
                continue

            used_persons.add(person_id)
            used_chairs.add(chair_id)

            pair_key = (
                person_id,
                chair_id,
            )

            matched_pairs.add(pair_key)

            if pair_key in self.pairs:

                state = self.pairs[pair_key]

                state.last_pixel_distance = (
                    distance_px
                )

                state.frames_seen += 1
                state.missed_frames = 0
                state.active = True

            else:

                self.pairs[pair_key] = PairState(
                    person_id=person_id,
                    chair_id=chair_id,
                    last_pixel_distance=distance_px,
                )

        # Update unmatched existing pairs.
        for key, state in list(
            self.pairs.items()
        ):

            if key not in matched_pairs:

                state.missed_frames += 1

                if (
                    state.missed_frames
                    > self.max_missed_frames
                ):
                    state.active = False

        return self.active_pairs()

    # ---------------------------------------------------------------
    # ACTIVE PAIRS
    # ---------------------------------------------------------------

    def active_pairs(self) -> List[PairState]:

        return [
            state
            for state in self.pairs.values()
            if state.active
        ]

    # ---------------------------------------------------------------
    # SPECIFIC PAIR
    # ---------------------------------------------------------------

    def get_pair(
        self,
        person_id: int,
        chair_id: int,
    ) -> Optional[PairState]:

        return self.pairs.get(
            (int(person_id), int(chair_id))
        )

    # ---------------------------------------------------------------
    # RESET
    # ---------------------------------------------------------------

    def reset(self) -> None:

        self.pairs.clear()


# =====================================================================
# SELF TEST
# =====================================================================

if __name__ == "__main__":

    tracker = IDPairTracker(
        max_pair_distance_px=300,
        max_missed_frames=2,
    )

    persons = [
        {
            "person_id": 1,
            "reference_point": (150, 400),
        },
        {
            "person_id": 2,
            "reference_point": (500, 400),
        },
    ]

    chairs = [
        {
            "chair_id": 10,
            "reference_point": (250, 420),
        },
        {
            "chair_id": 20,
            "reference_point": (560, 420),
        },
    ]

    pairs = tracker.update(
        persons,
        chairs,
    )

    print("=" * 60)
    print("ID PAIR TRACKER SELF TEST")
    print("=" * 60)

    for pair in pairs:
        print(
            f"Person {pair.person_id}"
            f" <-> Chair {pair.chair_id}"
            f" | pixel distance="
            f"{pair.last_pixel_distance:.2f}"
            f" | frames="
            f"{pair.frames_seen}"
        )

    assert len(pairs) == 2

    pair_1 = tracker.get_pair(1, 10)
    pair_2 = tracker.get_pair(2, 20)

    assert pair_1 is not None
    assert pair_2 is not None

    assert pair_1.person_id == 1
    assert pair_1.chair_id == 10

    assert pair_2.person_id == 2
    assert pair_2.chair_id == 20

    print()
    print("ID PAIR TRACKER SELF TEST PASSED")
