"""
BAS-HMR
Stage 4 - Geometry Estimator

Converts image-space reference points into metric 3D coordinates.

The estimator supports two modes:

1. Depth-based reconstruction
   ---------------------------
   Given:
       pixel (u, v)
       depth Z in metres
       camera intrinsics

   reconstruct:
       X = (u - cx) * Z / fx
       Y = (v - cy) * Z / fy
       Z = depth

2. Ground-plane reconstruction
   ----------------------------
   Given a calibrated mapping from image pixels to the floor plane,
   reconstruct:
       X, Z

This module does NOT claim that an arbitrary pixel-to-metre scale
is physically accurate. Real accuracy depends on scene calibration
and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import math


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass(frozen=True)
class MetricPoint:
    """
    Metric 3D point.

    Units:
        metres
    """

    x: float
    y: float
    z: float

    confidence: float = 1.0
    source: str = ""

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def distance_to(
        self,
        other: "MetricPoint",
    ) -> float:

        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def ground_distance_to(
        self,
        other: "MetricPoint",
    ) -> float:

        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass(frozen=True)
class PixelPoint:
    """Image-space point."""

    u: float
    v: float


# =====================================================================
# GEOMETRY ESTIMATOR
# =====================================================================

class GeometryEstimator:
    """
    Converts image coordinates into metric coordinates.

    Camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    All metric values are metres.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> None:

        if fx <= 0:
            raise ValueError("fx must be positive")

        if fy <= 0:
            raise ValueError("fy must be positive")

        if width <= 0:
            raise ValueError("width must be positive")

        if height <= 0:
            raise ValueError("height must be positive")

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        self.width = int(width)
        self.height = int(height)

    # -----------------------------------------------------------------
    # PIXEL VALIDATION
    # -----------------------------------------------------------------

    def validate_pixel(
        self,
        point: PixelPoint,
    ) -> bool:

        return (
            math.isfinite(point.u)
            and math.isfinite(point.v)
            and 0.0 <= point.u < self.width
            and 0.0 <= point.v < self.height
        )

    # -----------------------------------------------------------------
    # DEPTH RECONSTRUCTION
    # -----------------------------------------------------------------

    def pixel_to_3d(
        self,
        point: PixelPoint,
        depth_m: float,
        confidence: float = 1.0,
        source: str = "metric_depth",
    ) -> MetricPoint:

        if not self.validate_pixel(point):
            raise ValueError(
                f"Pixel outside image: {point}"
            )

        if not math.isfinite(depth_m):
            raise ValueError(
                "depth_m must be finite"
            )

        if depth_m <= 0:
            raise ValueError(
                "depth_m must be greater than zero"
            )

        z = float(depth_m)

        x = (
            (point.u - self.cx)
            * z
            / self.fx
        )

        y = (
            (point.v - self.cy)
            * z
            / self.fy
        )

        return MetricPoint(
            x=x,
            y=y,
            z=z,
            confidence=float(confidence),
            source=source,
        )

    # -----------------------------------------------------------------
    # BOUNDING BOX → REFERENCE PIXEL
    # -----------------------------------------------------------------

    @staticmethod
    def bbox_bottom_center(
        bbox,
    ) -> PixelPoint:

        if len(bbox) != 4:
            raise ValueError(
                "bbox must contain x1,y1,x2,y2"
            )

        x1, y1, x2, y2 = map(
            float,
            bbox,
        )

        return PixelPoint(
            u=(x1 + x2) / 2.0,
            v=y2,
        )

    # -----------------------------------------------------------------
    # GROUND PLANE CALLBACK
    # -----------------------------------------------------------------

    def ground_point_to_3d(
        self,
        point: PixelPoint,
        ground_mapper: Callable[
            [float, float],
            Tuple[float, float]
        ],
        confidence: float = 1.0,
        source: str = "scene_calibration",
    ) -> MetricPoint:
        """
        Convert an image pixel to a ground-plane metric position.

        ground_mapper(u, v) must return:

            X, Z

        in metres.

        Y is set to zero because the point is assumed to lie
        on the floor plane.
        """

        if not self.validate_pixel(point):
            raise ValueError(
                f"Pixel outside image: {point}"
            )

        x, z = ground_mapper(
            point.u,
            point.v,
        )

        if not (
            math.isfinite(x)
            and math.isfinite(z)
        ):
            raise ValueError(
                "Ground mapper returned invalid coordinates"
            )

        return MetricPoint(
            x=float(x),
            y=0.0,
            z=float(z),
            confidence=float(confidence),
            source=source,
        )


# =====================================================================
# DISTANCE FUNCTIONS
# =====================================================================

def euclidean_distance(
    point_a: MetricPoint,
    point_b: MetricPoint,
) -> float:

    return point_a.distance_to(point_b)


def ground_distance(
    point_a: MetricPoint,
    point_b: MetricPoint,
) -> float:

    return point_a.ground_distance_to(point_b)


def horizontal_distance(
    point_a: MetricPoint,
    point_b: MetricPoint,
) -> float:

    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.z - point_b.z) ** 2
    )


# =====================================================================
# SELF TEST
# =====================================================================

if __name__ == "__main__":

    estimator = GeometryEstimator(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    # Image centre at 2 metres.
    point = PixelPoint(
        u=320.0,
        v=240.0,
    )

    metric = estimator.pixel_to_3d(
        point,
        depth_m=2.0,
    )

    print("=" * 60)
    print("GEOMETRY ESTIMATOR SELF TEST")
    print("=" * 60)

    print(
        "Pixel:",
        point,
    )

    print(
        "3D:",
        metric.as_tuple(),
    )

    assert abs(metric.x - 0.0) < 1e-9
    assert abs(metric.y - 0.0) < 1e-9
    assert abs(metric.z - 2.0) < 1e-9

    # One metre horizontally from the optical centre
    # at two metres depth.
    point_2 = PixelPoint(
        u=570.0,
        v=240.0,
    )

    metric_2 = estimator.pixel_to_3d(
        point_2,
        depth_m=2.0,
    )

    print(
        "Second 3D:",
        metric_2.as_tuple(),
    )

    assert abs(metric_2.x - 1.0) < 1e-9

    d = euclidean_distance(
        metric,
        metric_2,
    )

    gd = ground_distance(
        metric,
        metric_2,
    )

    print(
        f"3D distance:     {d:.3f} m"
    )

    print(
        f"Ground distance: {gd:.3f} m"
    )

    assert abs(d - 1.0) < 1e-9
    assert abs(gd - 1.0) < 1e-9

    print()
    print("GEOMETRY ESTIMATOR SELF TEST PASSED")
