"""
BAS HMR
Metric 3D Object Reconstruction

Purpose
-------
Convert detected objects from image coordinates into
camera-space 3D coordinates measured in metres.

Pipeline:

    Detection
        ↓
    Reference Pixel
        ↓
    Metric Depth
        ↓
    Camera Model
        ↓
    X, Y, Z metres


Supported objects:

    Person
    Chair
    Generic object
"""

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np

from human.spatial.camera_model import (
    CameraModel,
    Point3D,
)


# ============================================================
# 3D OBJECT
# ============================================================

@dataclass
class MetricObject3D:
    """
    Object represented in camera-space 3D.

    All XYZ coordinates are metres.
    """

    object_id: int

    object_type: str

    confidence: float

    bbox: tuple

    reference_pixel: tuple

    depth_m: float

    position: Point3D

    timestamp: float = 0.0

    def to_dict(self):
        """
        Convert object to JSON-compatible dictionary.
        """

        return {
            "object_id": int(
                self.object_id
            ),

            "object_type": str(
                self.object_type
            ),

            "confidence": float(
                self.confidence
            ),

            "bbox": {
                "x1": float(
                    self.bbox[0]
                ),

                "y1": float(
                    self.bbox[1]
                ),

                "x2": float(
                    self.bbox[2]
                ),

                "y2": float(
                    self.bbox[3]
                ),
            },

            "reference_pixel": {
                "u": float(
                    self.reference_pixel[0]
                ),

                "v": float(
                    self.reference_pixel[1]
                ),
            },

            "depth_m": float(
                self.depth_m
            ),

            "position_3d_m": {
                "x": float(
                    self.position.x
                ),

                "y": float(
                    self.position.y
                ),

                "z": float(
                    self.position.z
                ),
            },

            "timestamp": float(
                self.timestamp
            ),
        }


# ============================================================
# METRIC 3D EXTRACTOR
# ============================================================

class Metric3DExtractor:
    """
    Converts image detections + metric depth into
    camera-space 3D coordinates.
    """

    def __init__(
        self,
        camera_model: CameraModel,
    ):

        self.camera_model = (
            camera_model
        )

    # ========================================================
    # BBOX CENTER
    # ========================================================

    @staticmethod
    def bbox_center(
        bbox,
    ):
        """
        Return the center pixel of a bounding box.
        """

        x1, y1, x2, y2 = map(
            float,
            bbox,
        )

        u = (
            x1 + x2
        ) / 2.0

        v = (
            y1 + y2
        ) / 2.0

        return (
            u,
            v,
        )

    # ========================================================
    # VALIDATE DEPTH
    # ========================================================

    @staticmethod
    def validate_depth(
        depth_m: float,
        minimum: float = 0.05,
        maximum: float = 20.0,
    ):
        """
        Validate a metric depth value.

        Returns True if the value is usable.
        """

        if depth_m is None:

            return False

        if not math.isfinite(
            float(depth_m)
        ):

            return False

        if (
            depth_m < minimum
            or depth_m > maximum
        ):

            return False

        return True

    # ========================================================
    # DETECTION → 3D
    # ========================================================

    def extract(
        self,
        object_id: int,
        object_type: str,
        bbox,
        confidence: float,
        depth_map: np.ndarray,
        timestamp: float = 0.0,
        reference_pixel: Optional[tuple] = None,
        depth_radius: int = 5,
    ) -> Optional[MetricObject3D]:
        """
        Convert a detection into a metric 3D object.

        Parameters
        ----------
        object_id:
            Stable object ID.

        object_type:
            "person", "chair", etc.

        bbox:
            (x1, y1, x2, y2)

        confidence:
            Detection confidence.

        depth_map:
            Metric depth map.

        timestamp:
            Measurement timestamp.

        reference_pixel:
            Optional (u, v).

            If omitted, bbox center is used.

        depth_radius:
            Neighborhood radius used when sampling depth.
        """

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        if bbox is None:

            return None

        x1, y1, x2, y2 = map(
            float,
            bbox,
        )

        if x2 <= x1:

            return None

        if y2 <= y1:

            return None

        # ----------------------------------------------------
        # REFERENCE PIXEL
        # ----------------------------------------------------

        if reference_pixel is None:

            reference_pixel = (
                self.bbox_center(
                    bbox
                )
            )

        u, v = map(
            float,
            reference_pixel,
        )

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------

        depth_m = self.depth_at_pixel(
            depth_map=depth_map,
            u=u,
            v=v,
            radius=depth_radius,
        )

        if not self.validate_depth(
            depth_m
        ):

            return None

        # ----------------------------------------------------
        # 3D RECONSTRUCTION
        # ----------------------------------------------------

        position = (
            self.camera_model.pixel_to_3d(
                u=u,
                v=v,
                depth_m=depth_m,
            )
        )

        return MetricObject3D(

            object_id=int(
                object_id
            ),

            object_type=str(
                object_type
            ),

            confidence=float(
                confidence
            ),

            bbox=(
                x1,
                y1,
                x2,
                y2,
            ),

            reference_pixel=(
                u,
                v,
            ),

            depth_m=float(
                depth_m
            ),

            position=position,

            timestamp=float(
                timestamp
            ),
        )

    # ========================================================
    # DEPTH AT PIXEL
    # ========================================================

    @staticmethod
    def depth_at_pixel(
        depth_map: np.ndarray,
        u: float,
        v: float,
        radius: int = 5,
    ) -> float:
        """
        Robustly sample depth around a pixel.
        """

        if depth_map is None:

            return float("nan")

        if depth_map.ndim != 2:

            raise ValueError(
                "Depth map must be 2D."
            )

        height, width = (
            depth_map.shape
        )

        x = int(
            round(u)
        )

        y = int(
            round(v)
        )

        x = max(
            0,
            min(
                x,
                width - 1,
            ),
        )

        y = max(
            0,
            min(
                y,
                height - 1,
            ),
        )

        radius = max(
            0,
            int(radius),
        )

        x1 = max(
            0,
            x - radius,
        )

        y1 = max(
            0,
            y - radius,
        )

        x2 = min(
            width,
            x + radius + 1,
        )

        y2 = min(
            height,
            y + radius + 1,
        )

        crop = depth_map[
            y1:y2,
            x1:x2,
        ]

        valid = crop[
            np.isfinite(crop)
        ]

        if valid.size == 0:

            return float("nan")

        return float(
            np.median(valid)
        )


# ============================================================
# DISTANCE
# ============================================================

def calculate_distance(
    object_a: MetricObject3D,
    object_b: MetricObject3D,
) -> float:
    """
    Calculate Euclidean distance between two metric 3D objects.

    Result is metres.
    """

    return object_a.position.distance_to(
        object_b.position
    )


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print()
    print("=" * 70)
    print("METRIC 3D SELF TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = CameraModel(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    extractor = (
        Metric3DExtractor(
            camera_model=camera
        )
    )

    # --------------------------------------------------------
    # Synthetic metric depth map
    # --------------------------------------------------------

    depth_map = np.full(
        (480, 640),
        2.0,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    person = extractor.extract(

        object_id=0,

        object_type="person",

        bbox=(
            270,
            180,
            370,
            300,
        ),

        confidence=0.95,

        depth_map=depth_map,

        timestamp=1.0,
    )

    # --------------------------------------------------------
    # Chair
    # --------------------------------------------------------

    chair = extractor.extract(

        object_id=0,

        object_type="chair",

        bbox=(
            370,
            220,
            470,
            340,
        ),

        confidence=0.92,

        depth_map=depth_map,

        timestamp=1.0,
    )

    if person is None:

        raise RuntimeError(
            "Person extraction failed."
        )

    if chair is None:

        raise RuntimeError(
            "Chair extraction failed."
        )

    # --------------------------------------------------------
    # Print person
    # --------------------------------------------------------

    print()
    print("PERSON")

    print(
        f"Pixel: "
        f"({person.reference_pixel[0]:.1f}, "
        f"{person.reference_pixel[1]:.1f})"
    )

    print(
        f"Depth: "
        f"{person.depth_m:.3f} m"
    )

    print(
        f"XYZ: "
        f"({person.position.x:.3f}, "
        f"{person.position.y:.3f}, "
        f"{person.position.z:.3f}) m"
    )

    # --------------------------------------------------------
    # Print chair
    # --------------------------------------------------------

    print()
    print("CHAIR")

    print(
        f"Pixel: "
        f"({chair.reference_pixel[0]:.1f}, "
        f"{chair.reference_pixel[1]:.1f})"
    )

    print(
        f"Depth: "
        f"{chair.depth_m:.3f} m"
    )

    print(
        f"XYZ: "
        f"({chair.position.x:.3f}, "
        f"{chair.position.y:.3f}, "
        f"{chair.position.z:.3f}) m"
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance = calculate_distance(
        person,
        chair,
    )

    print()
    print(
        f"Person ↔ Chair distance: "
        f"{distance:.3f} m"
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    print()
    print("JSON OUTPUT")

    print(
        person.to_dict()
    )

    print()
    print(
        chair.to_dict()
    )

    print()
    print(
        "Metric 3D test completed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    self_test()
