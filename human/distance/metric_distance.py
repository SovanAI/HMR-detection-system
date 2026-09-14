"""
BAS Human Activity Recognition
Metric Distance Engine

Calculates real-world person <-> chair relationships using
metric 3D camera-space coordinates.

Units:
    Position       -> metres
    Distance       -> metres
    Velocity       -> metres/second

This module does NOT use HMR coordinates.
HMR remains responsible for human pose estimation.
Metric depth + camera geometry are responsible for physical distance.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .camera_model import Point3D
from .metric_3d import MetricObject3D


# ============================================================
# Configuration
# ============================================================

DEFAULT_NEAR_THRESHOLD_M = 0.75
DEFAULT_FAR_THRESHOLD_M = 2.00

DEFAULT_MIN_DT = 0.001
DEFAULT_MAX_VELOCITY_MPS = 5.0


# ============================================================
# Result structure
# ============================================================

@dataclass
class MetricDistanceResult:
    """
    Result of a person <-> chair metric distance calculation.
    """

    person_id: int
    chair_id: int

    distance_m: float

    horizontal_distance_m: float
    vertical_difference_m: float
    depth_difference_m: float

    distance_change_m: float
    velocity_mps: float

    motion_state: str
    relationship: str

    timestamp: float

    person_confidence: float
    chair_confidence: float

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "chair_id": self.chair_id,

            "distance_m": round(self.distance_m, 4),

            "horizontal_distance_m": round(
                self.horizontal_distance_m, 4
            ),

            "vertical_difference_m": round(
                self.vertical_difference_m, 4
            ),

            "depth_difference_m": round(
                self.depth_difference_m, 4
            ),

            "distance_change_m": round(
                self.distance_change_m, 4
            ),

            "velocity_mps": round(
                self.velocity_mps, 4
            ),

            "motion_state": self.motion_state,
            "relationship": self.relationship,

            "timestamp": self.timestamp,

            "person_confidence": round(
                self.person_confidence, 4
            ),

            "chair_confidence": round(
                self.chair_confidence, 4
            ),
        }


# ============================================================
# Previous observation
# ============================================================

@dataclass
class PreviousDistance:
    """
    Stores the previous observation for a person-chair pair.
    """

    distance_m: float
    timestamp: float


# ============================================================
# Metric Distance Engine
# ============================================================

class MetricDistanceEngine:
    """
    Calculates physical person <-> chair distance.

    Coordinate system:

        X = horizontal camera-space position
        Y = vertical camera-space position
        Z = camera depth

    Distance:

        d = sqrt(
            dx² +
            dy² +
            dz²
        )
    """

    def __init__(
        self,
        near_threshold_m: float = DEFAULT_NEAR_THRESHOLD_M,
        far_threshold_m: float = DEFAULT_FAR_THRESHOLD_M,
        min_dt: float = DEFAULT_MIN_DT,
        max_velocity_mps: float = DEFAULT_MAX_VELOCITY_MPS,
    ):
        if near_threshold_m < 0:
            raise ValueError(
                "near_threshold_m must be >= 0"
            )

        if far_threshold_m <= near_threshold_m:
            raise ValueError(
                "far_threshold_m must be greater than "
                "near_threshold_m"
            )

        self.near_threshold_m = float(
            near_threshold_m
        )

        self.far_threshold_m = float(
            far_threshold_m
        )

        self.min_dt = float(min_dt)

        self.max_velocity_mps = float(
            max_velocity_mps
        )

        self.previous = {}

    # --------------------------------------------------------
    # Basic 3D distance
    # --------------------------------------------------------

    @staticmethod
    def euclidean_distance(
        point_a: Point3D,
        point_b: Point3D,
    ) -> float:

        dx = point_a.x - point_b.x
        dy = point_a.y - point_b.y
        dz = point_a.z - point_b.z

        return math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        )

    # --------------------------------------------------------
    # Horizontal distance
    # --------------------------------------------------------

    @staticmethod
    def horizontal_distance(
        point_a: Point3D,
        point_b: Point3D,
    ) -> float:

        dx = point_a.x - point_b.x
        dz = point_a.z - point_b.z

        return math.sqrt(
            dx * dx +
            dz * dz
        )

    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    def classify_relationship(
        self,
        distance_m: float,
    ) -> str:

        if distance_m <= self.near_threshold_m:
            return "near"

        if distance_m <= self.far_threshold_m:
            return "medium"

        return "far"

    # --------------------------------------------------------
    # Motion state
    # --------------------------------------------------------

    @staticmethod
    def classify_motion(
        distance_change_m: float,
        velocity_mps: float,
    ) -> str:

        # Negative change means distance decreased.
        if distance_change_m < -0.01:
            return "approaching"

        # Positive change means distance increased.
        if distance_change_m > 0.01:
            return "moving_away"

        if abs(velocity_mps) <= 0.05:
            return "stationary"

        return "stable"

    # --------------------------------------------------------
    # Calculate one person-chair pair
    # --------------------------------------------------------

    def calculate(
        self,
        person: MetricObject3D,
        chair: MetricObject3D,
        timestamp: Optional[float] = None,
    ) -> MetricDistanceResult:

        if person.object_type != "person":
            raise ValueError(
                "First object must have object_type='person'"
            )

        if chair.object_type != "chair":
            raise ValueError(
                "Second object must have object_type='chair'"
            )

        if timestamp is None:
            timestamp = time.time()

        timestamp = float(timestamp)

        person_position = person.position
        chair_position = chair.position

        # ----------------------------------------------------
        # Coordinate differences
        # ----------------------------------------------------

        dx = (
            person_position.x -
            chair_position.x
        )

        dy = (
            person_position.y -
            chair_position.y
        )

        dz = (
            person_position.z -
            chair_position.z
        )

        # ----------------------------------------------------
        # Euclidean distance
        # ----------------------------------------------------

        distance_m = math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        )

        # ----------------------------------------------------
        # Horizontal camera-space distance
        #
        # X + Z plane
        # ----------------------------------------------------

        horizontal_distance_m = math.sqrt(
            dx * dx +
            dz * dz
        )

        # ----------------------------------------------------
        # Vertical difference
        # ----------------------------------------------------

        vertical_difference_m = abs(dy)

        # ----------------------------------------------------
        # Depth difference
        # ----------------------------------------------------

        depth_difference_m = abs(dz)

        # ----------------------------------------------------
        # Temporal tracking
        # ----------------------------------------------------

        key = (
            int(person.object_id),
            int(chair.object_id),
        )

        previous = self.previous.get(key)

        if previous is None:

            distance_change_m = 0.0
            velocity_mps = 0.0
            motion_state = "initializing"

        else:

            dt = timestamp - previous.timestamp

            distance_change_m = (
                distance_m -
                previous.distance_m
            )

            if dt <= self.min_dt:

                velocity_mps = 0.0

            else:

                velocity_mps = (
                    distance_change_m / dt
                )

            # Protect against depth-estimation spikes.
            velocity_mps = max(
                -self.max_velocity_mps,
                min(
                    self.max_velocity_mps,
                    velocity_mps,
                ),
            )

            motion_state = self.classify_motion(
                distance_change_m,
                velocity_mps,
            )

        # Store current observation.
        self.previous[key] = PreviousDistance(
            distance_m=distance_m,
            timestamp=timestamp,
        )

        relationship = self.classify_relationship(
            distance_m
        )

        return MetricDistanceResult(
            person_id=int(person.object_id),
            chair_id=int(chair.object_id),

            distance_m=distance_m,

            horizontal_distance_m=(
                horizontal_distance_m
            ),

            vertical_difference_m=(
                vertical_difference_m
            ),

            depth_difference_m=(
                depth_difference_m
            ),

            distance_change_m=(
                distance_change_m
            ),

            velocity_mps=velocity_mps,

            motion_state=motion_state,
            relationship=relationship,

            timestamp=timestamp,

            person_confidence=float(
                person.confidence
            ),

            chair_confidence=float(
                chair.confidence
            ),
        )

    # --------------------------------------------------------
    # Calculate all person-chair combinations
    # --------------------------------------------------------

    def calculate_all(
        self,
        persons: list[MetricObject3D],
        chairs: list[MetricObject3D],
        timestamp: Optional[float] = None,
    ) -> list[MetricDistanceResult]:

        if timestamp is None:
            timestamp = time.time()

        results = []

        for person in persons:

            for chair in chairs:

                result = self.calculate(
                    person=person,
                    chair=chair,
                    timestamp=timestamp,
                )

                results.append(result)

        return results

    # --------------------------------------------------------
    # Clear temporal history
    # --------------------------------------------------------

    def reset(self):
        self.previous.clear()


# ============================================================
# Self Test
# ============================================================

def self_test():

    print("=" * 70)
    print("METRIC DISTANCE ENGINE SELF TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create synthetic person
    # --------------------------------------------------------

    person = MetricObject3D(
        object_id=0,
        object_type="person",
        confidence=0.95,

        bbox={
            "x1": 270.0,
            "y1": 180.0,
            "x2": 370.0,
            "y2": 300.0,
        },

        reference_pixel={
            "u": 320.0,
            "v": 240.0,
        },

        depth_m=2.0,

        position=Point3D(
            x=0.0,
            y=0.0,
            z=2.0,
        ),

        timestamp=1.0,
    )

    # --------------------------------------------------------
    # Create synthetic chair
    # --------------------------------------------------------

    chair = MetricObject3D(
        object_id=0,
        object_type="chair",
        confidence=0.92,

        bbox={
            "x1": 370.0,
            "y1": 220.0,
            "x2": 470.0,
            "y2": 340.0,
        },

        reference_pixel={
            "u": 420.0,
            "v": 280.0,
        },

        depth_m=2.0,

        position=Point3D(
            x=0.4,
            y=0.16,
            z=2.0,
        ),

        timestamp=1.0,
    )

    engine = MetricDistanceEngine(
        near_threshold_m=0.75,
        far_threshold_m=2.0,
    )

    # --------------------------------------------------------
    # First frame
    # --------------------------------------------------------

    result_1 = engine.calculate(
        person,
        chair,
        timestamp=1.0,
    )

    print()
    print("FRAME 1")
    print("-" * 70)

    print(
        f"Distance              : "
        f"{result_1.distance_m:.3f} m"
    )

    print(
        f"Horizontal distance   : "
        f"{result_1.horizontal_distance_m:.3f} m"
    )

    print(
        f"Vertical difference   : "
        f"{result_1.vertical_difference_m:.3f} m"
    )

    print(
        f"Depth difference      : "
        f"{result_1.depth_difference_m:.3f} m"
    )

    print(
        f"Velocity              : "
        f"{result_1.velocity_mps:.3f} m/s"
    )

    print(
        f"Motion                : "
        f"{result_1.motion_state}"
    )

    print(
        f"Relationship          : "
        f"{result_1.relationship}"
    )

    # --------------------------------------------------------
    # Second frame - approaching
    # --------------------------------------------------------

    person.position = Point3D(
        x=0.10,
        y=0.0,
        z=1.8,
    )

    result_2 = engine.calculate(
        person,
        chair,
        timestamp=2.0,
    )

    print()
    print("FRAME 2")
    print("-" * 70)

    print(
        f"Distance              : "
        f"{result_2.distance_m:.3f} m"
    )

    print(
        f"Distance change       : "
        f"{result_2.distance_change_m:.3f} m"
    )

    print(
        f"Velocity              : "
        f"{result_2.velocity_mps:.3f} m/s"
    )

    print(
        f"Motion                : "
        f"{result_2.motion_state}"
    )

    print(
        f"Relationship          : "
        f"{result_2.relationship}"
    )

    # --------------------------------------------------------
    # JSON output
    # --------------------------------------------------------

    print()
    print("JSON OUTPUT")
    print("-" * 70)

    print(result_2.to_dict())

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert result_1.distance_m > 0
    assert result_2.distance_m > 0

    assert result_2.distance_change_m < 0
    assert result_2.motion_state == "approaching"

    assert result_1.relationship in (
        "near",
        "medium",
        "far",
    )

    print()
    print("=" * 70)
    print("METRIC DISTANCE ENGINE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
