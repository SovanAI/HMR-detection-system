"""
BAS Human Activity Recognition
Distance Filtering and Stabilization

Purpose:
    Stabilize noisy metric distance measurements before
    calculating velocity and motion state.

Pipeline:

    Raw metric distance
            |
            v
       Median filter
            |
            v
       Outlier rejection
            |
            v
          EMA
            |
            v
    Stable metric distance

All distances are in metres.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional


# ============================================================
# Configuration
# ============================================================

DEFAULT_WINDOW_SIZE = 5

# Maximum allowed difference between a new measurement and
# the current filtered value before it is treated as an outlier.
DEFAULT_MAX_JUMP_M = 0.75

# EMA smoothing:
#
# alpha = 1.0 -> no smoothing
# alpha = 0.1 -> heavy smoothing
DEFAULT_ALPHA = 0.35

DEFAULT_MIN_DISTANCE_M = 0.05
DEFAULT_MAX_DISTANCE_M = 20.0


# ============================================================
# Filter result
# ============================================================

@dataclass
class FilterResult:
    """
    Result produced by the distance filter.
    """

    raw_distance_m: float
    median_distance_m: float
    filtered_distance_m: float

    is_valid: bool
    is_outlier: bool

    sample_count: int

    def to_dict(self) -> dict:
        return {
            "raw_distance_m": round(
                self.raw_distance_m,
                4,
            ),

            "median_distance_m": round(
                self.median_distance_m,
                4,
            ),

            "filtered_distance_m": round(
                self.filtered_distance_m,
                4,
            ),

            "is_valid": self.is_valid,
            "is_outlier": self.is_outlier,

            "sample_count": self.sample_count,
        }


# ============================================================
# Per pair history
# ============================================================

@dataclass
class PairHistory:
    """
    Temporal history for one person-chair pair.
    """

    values: deque

    filtered_distance_m: Optional[float] = None


# ============================================================
# Distance Filter
# ============================================================

class DistanceFilter:
    """
    Temporal distance filter.

    Each person-chair pair gets an independent history.

    Processing:

        1. Validate raw distance.
        2. Add measurement to history.
        3. Calculate rolling median.
        4. Reject extreme jumps.
        5. Apply EMA smoothing.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        alpha: float = DEFAULT_ALPHA,
        max_jump_m: float = DEFAULT_MAX_JUMP_M,
        min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    ):

        if window_size < 1:
            raise ValueError(
                "window_size must be >= 1"
            )

        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                "alpha must be > 0 and <= 1"
            )

        if max_jump_m <= 0:
            raise ValueError(
                "max_jump_m must be > 0"
            )

        self.window_size = int(window_size)

        self.alpha = float(alpha)

        self.max_jump_m = float(
            max_jump_m
        )

        self.min_distance_m = float(
            min_distance_m
        )

        self.max_distance_m = float(
            max_distance_m
        )

        self.histories = defaultdict(
            lambda: PairHistory(
                values=deque(
                    maxlen=self.window_size
                )
            )
        )

    # --------------------------------------------------------
    # Validate distance
    # --------------------------------------------------------

    def validate_distance(
        self,
        distance_m: float,
    ) -> bool:

        if not math.isfinite(distance_m):
            return False

        if distance_m < self.min_distance_m:
            return False

        if distance_m > self.max_distance_m:
            return False

        return True

    # --------------------------------------------------------
    # Median
    # --------------------------------------------------------

    @staticmethod
    def calculate_median(
        values,
    ) -> float:

        if not values:
            raise ValueError(
                "Cannot calculate median of empty list"
            )

        return float(
            statistics.median(values)
        )

    # --------------------------------------------------------
    # Update one person-chair pair
    # --------------------------------------------------------

    def update(
        self,
        person_id: int,
        chair_id: int,
        distance_m: float,
    ) -> FilterResult:

        distance_m = float(distance_m)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not self.validate_distance(
            distance_m
        ):

            history = self.histories[
                (int(person_id), int(chair_id))
            ]

            previous = history.filtered_distance_m

            if previous is None:
                previous = 0.0

            return FilterResult(
                raw_distance_m=distance_m,
                median_distance_m=previous,
                filtered_distance_m=previous,

                is_valid=False,
                is_outlier=True,

                sample_count=len(
                    history.values
                ),
            )

        # ----------------------------------------------------
        # Get pair history
        # ----------------------------------------------------

        key = (
            int(person_id),
            int(chair_id),
        )

        history = self.histories[key]

        # ----------------------------------------------------
        # First measurement
        # ----------------------------------------------------

        if history.filtered_distance_m is None:

            history.values.append(
                distance_m
            )

            history.filtered_distance_m = (
                distance_m
            )

            return FilterResult(
                raw_distance_m=distance_m,

                median_distance_m=distance_m,

                filtered_distance_m=distance_m,

                is_valid=True,
                is_outlier=False,

                sample_count=len(
                    history.values
                ),
            )

        # ----------------------------------------------------
        # Outlier detection
        # ----------------------------------------------------

        difference_from_filtered = abs(
            distance_m -
            history.filtered_distance_m
        )

        is_outlier = (
            difference_from_filtered >
            self.max_jump_m
        )

        # ----------------------------------------------------
        # If outlier:
        #
        # Do not allow the spike to directly modify
        # the filtered distance.
        # ----------------------------------------------------

        if is_outlier:

            median_distance = (
                self.calculate_median(
                    history.values
                )
                if history.values
                else history.filtered_distance_m
            )

            return FilterResult(
                raw_distance_m=distance_m,

                median_distance_m=(
                    median_distance
                ),

                filtered_distance_m=(
                    history.filtered_distance_m
                ),

                is_valid=True,
                is_outlier=True,

                sample_count=len(
                    history.values
                ),
            )

        # ----------------------------------------------------
        # Add valid sample
        # ----------------------------------------------------

        history.values.append(
            distance_m
        )

        # ----------------------------------------------------
        # Rolling median
        # ----------------------------------------------------

        median_distance = (
            self.calculate_median(
                history.values
            )
        )

        # ----------------------------------------------------
        # EMA
        #
        # filtered =
        #
        # alpha * median +
        # (1-alpha) * previous
        # ----------------------------------------------------

        previous_filtered = (
            history.filtered_distance_m
        )

        filtered_distance = (
            self.alpha *
            median_distance
            +
            (1.0 - self.alpha) *
            previous_filtered
        )

        history.filtered_distance_m = (
            filtered_distance
        )

        return FilterResult(
            raw_distance_m=distance_m,

            median_distance_m=median_distance,

            filtered_distance_m=filtered_distance,

            is_valid=True,
            is_outlier=False,

            sample_count=len(
                history.values
            ),
        )

    # --------------------------------------------------------
    # Get current filtered distance
    # --------------------------------------------------------

    def get_filtered_distance(
        self,
        person_id: int,
        chair_id: int,
    ) -> Optional[float]:

        history = self.histories.get(
            (
                int(person_id),
                int(chair_id),
            )
        )

        if history is None:
            return None

        return history.filtered_distance_m

    # --------------------------------------------------------
    # Reset one pair
    # --------------------------------------------------------

    def reset_pair(
        self,
        person_id: int,
        chair_id: int,
    ):

        key = (
            int(person_id),
            int(chair_id),
        )

        self.histories.pop(
            key,
            None,
        )

    # --------------------------------------------------------
    # Reset everything
    # --------------------------------------------------------

    def reset(self):

        self.histories.clear()


# ============================================================
# Self Test
# ============================================================

def self_test():

    print("=" * 70)
    print("DISTANCE FILTER SELF TEST")
    print("=" * 70)

    distance_filter = DistanceFilter(
        window_size=5,
        alpha=0.35,
        max_jump_m=0.50,
    )

    # --------------------------------------------------------
    # Simulated stationary object
    #
    # The actual distance is approximately 2.00 m.
    # Measurements contain realistic small fluctuations.
    # --------------------------------------------------------

    measurements = [
        2.00,
        2.08,
        1.94,
        2.04,
        1.97,
        2.03,
        1.99,
        2.06,
    ]

    print()
    print(
        "STATIONARY OBJECT"
    )
    print("-" * 70)

    print(
        f"{'Frame':<8}"
        f"{'Raw':<14}"
        f"{'Median':<14}"
        f"{'Filtered':<14}"
        f"{'Outlier':<10}"
    )

    for index, distance in enumerate(
        measurements,
        start=1,
    ):

        result = distance_filter.update(
            person_id=0,
            chair_id=0,
            distance_m=distance,
        )

        print(
            f"{index:<8}"
            f"{result.raw_distance_m:<14.3f}"
            f"{result.median_distance_m:<14.3f}"
            f"{result.filtered_distance_m:<14.3f}"
            f"{str(result.is_outlier):<10}"
        )

    # --------------------------------------------------------
    # Outlier test
    # --------------------------------------------------------

    print()
    print("OUTLIER TEST")
    print("-" * 70)

    outlier_result = (
        distance_filter.update(
            person_id=0,
            chair_id=0,
            distance_m=8.50,
        )
    )

    print(
        f"Raw distance      : "
        f"{outlier_result.raw_distance_m:.3f} m"
    )

    print(
        f"Filtered distance : "
        f"{outlier_result.filtered_distance_m:.3f} m"
    )

    print(
        f"Outlier detected  : "
        f"{outlier_result.is_outlier}"
    )

    assert outlier_result.is_outlier is True

    # The filtered value should not jump to 8.5 m.
    assert (
        outlier_result.filtered_distance_m
        < 3.0
    )

    # --------------------------------------------------------
    # Independent pair test
    # --------------------------------------------------------

    print()
    print("INDEPENDENT PAIR TEST")
    print("-" * 70)

    pair_1 = distance_filter.update(
        person_id=0,
        chair_id=1,
        distance_m=1.50,
    )

    pair_2 = distance_filter.update(
        person_id=1,
        chair_id=0,
        distance_m=4.00,
    )

    print(
        f"Person 0 -> Chair 1 : "
        f"{pair_1.filtered_distance_m:.3f} m"
    )

    print(
        f"Person 1 -> Chair 0 : "
        f"{pair_2.filtered_distance_m:.3f} m"
    )

    assert (
        pair_1.filtered_distance_m
        != pair_2.filtered_distance_m
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    print()
    print("JSON OUTPUT")
    print("-" * 70)

    print(
        pair_1.to_dict()
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DISTANCE FILTER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
