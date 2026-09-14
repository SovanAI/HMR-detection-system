from dataclasses import dataclass, field
from typing import Dict, Optional
import math
import time


# ============================================================
# TRACKED SPATIAL STATE
# ============================================================

@dataclass
class SpatialState:

    person_id: int
    chair_id: int

    current_distance: float

    previous_distance: Optional[float] = None

    distance_change: float = 0.0

    distance_rate: float = 0.0

    relationship: str = "unknown"

    motion_state: str = "stationary"

    timestamp: float = field(
        default_factory=time.time
    )


# ============================================================
# TEMPORAL SPATIAL TRACKER
# ============================================================

class SpatialTemporalTracker:

    def __init__(
        self,
        movement_threshold=0.05,
        approaching_threshold=-0.05,
        moving_away_threshold=0.05,
        smoothing_alpha=0.5,
    ):

        self.movement_threshold = movement_threshold

        self.approaching_threshold = (
            approaching_threshold
        )

        self.moving_away_threshold = (
            moving_away_threshold
        )

        self.smoothing_alpha = smoothing_alpha

        self.states: Dict[
            tuple,
            SpatialState
        ] = {}

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        person_id: int,
        chair_id: int,
        distance: float,
        relationship: str,
        timestamp: Optional[float] = None,
    ):

        if timestamp is None:

            timestamp = time.time()

        key = (
            person_id,
            chair_id
        )

        previous = self.states.get(key)

        # ----------------------------------------------------
        # FIRST OBSERVATION
        # ----------------------------------------------------

        if previous is None:

            state = SpatialState(

                person_id=person_id,

                chair_id=chair_id,

                current_distance=distance,

                previous_distance=None,

                distance_change=0.0,

                distance_rate=0.0,

                relationship=relationship,

                motion_state="initializing",

                timestamp=timestamp,
            )

            self.states[key] = state

            return state

        # ----------------------------------------------------
        # TIME DIFFERENCE
        # ----------------------------------------------------

        dt = timestamp - previous.timestamp

        if dt <= 0:

            dt = 1e-6

        # ----------------------------------------------------
        # RAW DISTANCE CHANGE
        # ----------------------------------------------------

        raw_change = (
            distance
            - previous.current_distance
        )

        # ----------------------------------------------------
        # SMOOTH DISTANCE CHANGE
        # ----------------------------------------------------

        change = (
            self.smoothing_alpha * raw_change
            +
            (1 - self.smoothing_alpha)
            * previous.distance_change
        )

        # ----------------------------------------------------
        # DISTANCE RATE
        # ----------------------------------------------------

        rate = change / dt

        # ----------------------------------------------------
        # MOTION CLASSIFICATION
        # ----------------------------------------------------

        if change < self.approaching_threshold:

            motion_state = "approaching"

        elif change > self.moving_away_threshold:

            motion_state = "moving_away"

        elif abs(change) <= self.movement_threshold:

            motion_state = "stationary"

        else:

            motion_state = "uncertain"

        # ----------------------------------------------------
        # UPDATE STATE
        # ----------------------------------------------------

        state = SpatialState(

            person_id=person_id,

            chair_id=chair_id,

            current_distance=distance,

            previous_distance=previous.current_distance,

            distance_change=change,

            distance_rate=rate,

            relationship=relationship,

            motion_state=motion_state,

            timestamp=timestamp,
        )

        self.states[key] = state

        return state

    # ========================================================
    # GET STATE
    # ========================================================

    def get_state(
        self,
        person_id: int,
        chair_id: int,
    ):

        return self.states.get(
            (person_id, chair_id)
        )

    # ========================================================
    # REMOVE OLD PAIRS
    # ========================================================

    def remove_pair(
        self,
        person_id: int,
        chair_id: int,
    ):

        self.states.pop(
            (person_id, chair_id),
            None
        )


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print("=" * 70)
    print("SPATIAL TEMPORAL TRACKER SELF TEST")
    print("=" * 70)

    tracker = SpatialTemporalTracker()

    distances = [
        10.0,
        9.5,
        8.8,
        7.9,
        7.0,
        6.4,
        6.0,
    ]

    timestamp = 1000.0

    for i, distance in enumerate(distances):

        state = tracker.update(

            person_id=0,

            chair_id=0,

            distance=distance,

            relationship="medium",

            timestamp=timestamp,
        )

        print()
        print(f"Frame {i}")

        print(
            f"  Distance       : "
            f"{state.current_distance:.3f}"
        )

        print(
            f"  Previous       : "
            f"{state.previous_distance}"
        )

        print(
            f"  ΔDistance      : "
            f"{state.distance_change:.3f}"
        )

        print(
            f"  Distance rate  : "
            f"{state.distance_rate:.3f}"
        )

        print(
            f"  Motion state   : "
            f"{state.motion_state}"
        )

        timestamp += 1.0

    print()
    print("=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    self_test()
