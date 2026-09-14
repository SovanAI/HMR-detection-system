from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class Point3D:
    """
    3D point in normalized camera coordinates.

    x = horizontal position
    y = vertical position
    z = relative depth
    """

    x: float
    y: float
    z: float


@dataclass
class SpatialRelationship:
    """
    Spatial relationship between a person and a chair.
    """

    person_id: int
    chair_id: int

    person_position: Point3D
    chair_position: Point3D

    distance: float
    relationship: str


class SpatialEngine:
    """
    CPU-friendly person-chair spatial relationship engine.

    IMPORTANT:
    The resulting distance is relative distance.
    It is NOT metres.
    """

    def __init__(
        self,
        near_threshold: float = 0.30,
        far_threshold: float = 0.80,
    ):
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold

    # ========================================================
    # NORMALIZED IMAGE COORDINATES
    # ========================================================

    @staticmethod
    def normalize_image_position(
        center: Tuple[int, int],
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float]:

        cx, cy = center

        if image_width <= 0:
            raise ValueError("image_width must be positive")

        if image_height <= 0:
            raise ValueError("image_height must be positive")

        # Convert image coordinates to [0, 1].
        x = cx / float(image_width)
        y = cy / float(image_height)

        return x, y

    # ========================================================
    # 3D POINT
    # ========================================================

    @classmethod
    def create_point(
        cls,
        center: Tuple[int, int],
        depth: float,
        image_width: int,
        image_height: int,
    ) -> Point3D:

        x, y = cls.normalize_image_position(
            center,
            image_width,
            image_height,
        )

        return Point3D(
            x=x,
            y=y,
            z=float(depth),
        )

    # ========================================================
    # EUCLIDEAN 3D DISTANCE
    # ========================================================

    @staticmethod
    def distance_3d(
        point_a: Point3D,
        point_b: Point3D,
    ) -> float:

        dx = point_a.x - point_b.x
        dy = point_a.y - point_b.y
        dz = point_a.z - point_b.z

        return math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )

    # ========================================================
    # RELATIONSHIP CLASSIFICATION
    # ========================================================

    def classify_distance(
        self,
        distance: float,
    ) -> str:

        if distance < self.near_threshold:
            return "near"

        if distance > self.far_threshold:
            return "far"

        return "medium"

    # ========================================================
    # PERSON ↔ CHAIR
    # ========================================================

    def calculate_relationship(
        self,
        person_id: int,
        person_position: Point3D,
        chair_id: int,
        chair_position: Point3D,
    ) -> SpatialRelationship:

        distance = self.distance_3d(
            person_position,
            chair_position,
        )

        relationship = self.classify_distance(
            distance
        )

        return SpatialRelationship(
            person_id=person_id,
            chair_id=chair_id,
            person_position=person_position,
            chair_position=chair_position,
            distance=distance,
            relationship=relationship,
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    engine = SpatialEngine()

    person = Point3D(
        x=0.50,
        y=0.50,
        z=0.20,
    )

    chair = Point3D(
        x=0.55,
        y=0.55,
        z=0.30,
    )

    relationship = engine.calculate_relationship(
        person_id=0,
        person_position=person,
        chair_id=0,
        chair_position=chair,
    )

    print("=" * 60)
    print("SPATIAL ENGINE TEST")
    print("=" * 60)

    print(
        "Person position:",
        relationship.person_position,
    )

    print(
        "Chair position:",
        relationship.chair_position,
    )

    print(
        "3D distance:",
        f"{relationship.distance:.4f}",
    )

    print(
        "Relationship:",
        relationship.relationship,
    )

    print("=" * 60)
