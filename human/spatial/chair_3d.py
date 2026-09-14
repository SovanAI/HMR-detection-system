from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Chair3D:
    """
    Chair position represented in a normalized image/depth
    coordinate system.

    x:
        Normalized horizontal image position [0, 1]

    y:
        Normalized vertical image position [0, 1]

    z:
        Relative monocular depth [0, 1]

    IMPORTANT:
        z is NOT metres.
    """

    chair_id: int

    x: float
    y: float
    z: float

    confidence: float


class Chair3DExtractor:
    """
    Converts a tracked chair into a normalized 3D
    representation.

    Input:
        chair center
        relative depth
        image dimensions
    """

    @staticmethod
    def normalize_position(
        center: Tuple[int, int],
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float]:

        if image_width <= 0:
            raise ValueError(
                "image_width must be greater than zero"
            )

        if image_height <= 0:
            raise ValueError(
                "image_height must be greater than zero"
            )

        cx, cy = center

        x = cx / float(image_width)
        y = cy / float(image_height)

        return (
            max(0.0, min(1.0, x)),
            max(0.0, min(1.0, y)),
        )

    @classmethod
    def extract(
        cls,
        chair_id: int,
        center: Tuple[int, int],
        relative_depth: float,
        confidence: float,
        image_width: int,
        image_height: int,
    ) -> Chair3D:

        x, y = cls.normalize_position(
            center,
            image_width,
            image_height,
        )

        z = max(
            0.0,
            min(1.0, float(relative_depth)),
        )

        return Chair3D(
            chair_id=int(chair_id),
            x=x,
            y=y,
            z=z,
            confidence=float(confidence),
        )


if __name__ == "__main__":

    extractor = Chair3DExtractor()

    chair = extractor.extract(
        chair_id=0,
        center=(320, 240),
        relative_depth=0.53,
        confidence=0.82,
        image_width=640,
        image_height=480,
    )

    print("=" * 70)
    print("CHAIR 3D EXTRACTION TEST")
    print("=" * 70)

    print(
        f"Chair ID: {chair.chair_id}"
    )

    print(
        f"X: {chair.x:.4f}"
    )

    print(
        f"Y: {chair.y:.4f}"
    )

    print(
        f"Z: {chair.z:.4f}"
    )

    print(
        f"Confidence: {chair.confidence:.4f}"
    )

    print(
        "\nZ is relative depth, NOT metres."
    )

    print("=" * 70)
