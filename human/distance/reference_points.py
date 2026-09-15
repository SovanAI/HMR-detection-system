"""
BAS-HMR
Reference point extraction for person-chair distance estimation.

The reference points are deliberately separated from the bounding-box
centers because distance should represent the physical relationship
between a person and a chair on the floor.

Person:
    Uses the bottom-center of the bounding box as a first approximation
    of the person's floor/contact position.

Chair:
    Uses the bottom-center of the bounding box as a first approximation
    of the chair's floor/contact position.

These are image-space reference points.
They will later be converted into metric 3D coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReferencePoint:
    """A reference point in image coordinates."""

    u: float
    v: float
    confidence: float = 1.0
    source: str = ""

    def as_tuple(self) -> tuple[float, float]:
        return self.u, self.v


class ReferencePointExtractor:
    """
    Extract physical-reference candidates from bounding boxes.

    Bounding box format:
        (x1, y1, x2, y2)

    Coordinates are expressed in pixels.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
    ) -> None:

        if frame_width <= 0:
            raise ValueError("frame_width must be positive")

        if frame_height <= 0:
            raise ValueError("frame_height must be positive")

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)

    # ---------------------------------------------------------------
    # BASIC VALIDATION
    # ---------------------------------------------------------------

    def _validate_bbox(
        self,
        bbox: Sequence[float],
    ) -> tuple[float, float, float, float]:

        if len(bbox) != 4:
            raise ValueError(
                "Bounding box must contain exactly 4 values"
            )

        x1, y1, x2, y2 = map(float, bbox)

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        return x1, y1, x2, y2

    # ---------------------------------------------------------------
    # CLAMP TO IMAGE
    # ---------------------------------------------------------------

    def _clamp(
        self,
        u: float,
        v: float,
    ) -> tuple[float, float]:

        u = max(
            0.0,
            min(float(self.frame_width - 1), u),
        )

        v = max(
            0.0,
            min(float(self.frame_height - 1), v),
        )

        return u, v

    # ---------------------------------------------------------------
    # PERSON REFERENCE
    # ---------------------------------------------------------------

    def person_reference(
        self,
        bbox: Sequence[float],
        confidence: float = 1.0,
    ) -> ReferencePoint:

        x1, y1, x2, y2 = self._validate_bbox(bbox)

        # Bottom-center of person bounding box.
        u = (x1 + x2) / 2.0
        v = y2

        u, v = self._clamp(u, v)

        return ReferencePoint(
            u=u,
            v=v,
            confidence=float(confidence),
            source="person_bottom_center",
        )

    # ---------------------------------------------------------------
    # CHAIR REFERENCE
    # ---------------------------------------------------------------

    def chair_reference(
        self,
        bbox: Sequence[float],
        confidence: float = 1.0,
    ) -> ReferencePoint:

        x1, y1, x2, y2 = self._validate_bbox(bbox)

        # Bottom-center of chair bounding box.
        u = (x1 + x2) / 2.0
        v = y2

        u, v = self._clamp(u, v)

        return ReferencePoint(
            u=u,
            v=v,
            confidence=float(confidence),
            source="chair_bottom_center",
        )

    # ---------------------------------------------------------------
    # GENERIC REFERENCE
    # ---------------------------------------------------------------

    def extract(
        self,
        object_type: str,
        bbox: Sequence[float],
        confidence: float = 1.0,
    ) -> ReferencePoint:

        object_type = object_type.lower().strip()

        if object_type == "person":
            return self.person_reference(
                bbox=bbox,
                confidence=confidence,
            )

        if object_type == "chair":
            return self.chair_reference(
                bbox=bbox,
                confidence=confidence,
            )

        raise ValueError(
            f"Unsupported object type: {object_type}"
        )


# =====================================================================
# SELF TEST
# =====================================================================

if __name__ == "__main__":

    extractor = ReferencePointExtractor(
        frame_width=640,
        frame_height=480,
    )

    person = extractor.person_reference(
        bbox=(100, 100, 200, 400),
        confidence=0.95,
    )

    chair = extractor.chair_reference(
        bbox=(300, 250, 450, 430),
        confidence=0.90,
    )

    print("=" * 60)
    print("REFERENCE POINT SELF TEST")
    print("=" * 60)

    print(
        "Person reference:",
        person.as_tuple(),
        "confidence=",
        person.confidence,
        "source=",
        person.source,
    )

    print(
        "Chair reference:",
        chair.as_tuple(),
        "confidence=",
        chair.confidence,
        "source=",
        chair.source,
    )

    assert person.as_tuple() == (150.0, 400.0)
    assert chair.as_tuple() == (375.0, 430.0)

    print()
    print("REFERENCE POINT SELF TEST PASSED")
