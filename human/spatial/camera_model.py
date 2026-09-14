"""
BAS HMR
Camera 3D Model

Purpose:
    Convert image pixels + metric depth into camera-space
    3D coordinates measured in metres.

Coordinate system:

        Camera
          │
          │ Z
          │
          ▼

    X → right
    Y → down
    Z → forward

Input:
    pixel (u, v)
    depth Z in metres

Output:
    X, Y, Z in metres
"""

from dataclasses import dataclass

import math


# ============================================================
# 3D POINT
# ============================================================

@dataclass
class Point3D:
    """
    3D point in camera coordinates.

    Units:
        metres
    """

    x: float
    y: float
    z: float

    def distance_to(
        self,
        other: "Point3D",
    ) -> float:

        return math.sqrt(
            (self.x - other.x) ** 2
            +
            (self.y - other.y) ** 2
            +
            (self.z - other.z) ** 2
        )


# ============================================================
# CAMERA MODEL
# ============================================================

class CameraModel:
    """
    Pinhole camera model.

    Intrinsic parameters:

        fx = focal length in pixels, x direction
        fy = focal length in pixels, y direction
        cx = principal point x
        cy = principal point y
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int = 640,
        height: int = 480,
    ):

        if fx <= 0:
            raise ValueError(
                "fx must be greater than zero."
            )

        if fy <= 0:
            raise ValueError(
                "fy must be greater than zero."
            )

        self.fx = float(fx)
        self.fy = float(fy)

        self.cx = float(cx)
        self.cy = float(cy)

        self.width = int(width)
        self.height = int(height)

    # ========================================================
    # PIXEL → 3D
    # ========================================================

    def pixel_to_3d(
        self,
        u: float,
        v: float,
        depth_m: float,
    ) -> Point3D:
        """
        Convert image coordinate + metric depth
        into camera-space XYZ coordinates.

        Parameters
        ----------
        u:
            Pixel x coordinate.

        v:
            Pixel y coordinate.

        depth_m:
            Metric depth in metres.

        Returns
        -------
        Point3D
            Camera-space coordinates in metres.
        """

        depth_m = float(depth_m)

        if not math.isfinite(depth_m):

            raise ValueError(
                "Depth must be finite."
            )

        if depth_m <= 0:

            raise ValueError(
                "Depth must be greater than zero."
            )

        u = float(u)
        v = float(v)

        x = (
            (u - self.cx)
            * depth_m
            / self.fx
        )

        y = (
            (v - self.cy)
            * depth_m
            / self.fy
        )

        z = depth_m

        return Point3D(
            x=x,
            y=y,
            z=z,
        )

    # ========================================================
    # 3D → PIXEL
    # ========================================================

    def point_3d_to_pixel(
        self,
        point: Point3D,
    ):
        """
        Project a camera-space 3D point back onto the image.
        """

        if point.z <= 0:

            raise ValueError(
                "Point must be in front of camera."
            )

        u = (
            self.fx
            * point.x
            / point.z
            + self.cx
        )

        v = (
            self.fy
            * point.y
            / point.z
            + self.cy
        )

        return (
            float(u),
            float(v),
        )

    # ========================================================
    # IMAGE CENTER
    # ========================================================

    def image_center(self):
        """
        Return the principal point.
        """

        return (
            self.cx,
            self.cy,
        )

    # ========================================================
    # DESCRIPTION
    # ========================================================

    def to_dict(self):

        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print()
    print("=" * 70)
    print("CAMERA MODEL SELF TEST")
    print("=" * 70)

    camera = CameraModel(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    # --------------------------------------------------------
    # Center pixel
    # --------------------------------------------------------

    center = camera.pixel_to_3d(
        u=320,
        v=240,
        depth_m=2.0,
    )

    print()
    print("Center pixel:")
    print(
        f"XYZ = "
        f"({center.x:.4f}, "
        f"{center.y:.4f}, "
        f"{center.z:.4f}) m"
    )

    # --------------------------------------------------------
    # Off-center pixel
    # --------------------------------------------------------

    point = camera.pixel_to_3d(
        u=420,
        v=290,
        depth_m=2.0,
    )

    print()
    print("Off-center pixel:")
    print(
        f"XYZ = "
        f"({point.x:.4f}, "
        f"{point.y:.4f}, "
        f"{point.z:.4f}) m"
    )

    # --------------------------------------------------------
    # Back projection
    # --------------------------------------------------------

    pixel = camera.point_3d_to_pixel(
        point
    )

    print()
    print("Back projected pixel:")
    print(
        f"(u, v) = "
        f"({pixel[0]:.2f}, "
        f"{pixel[1]:.2f})"
    )

    # --------------------------------------------------------
    # Distance test
    # --------------------------------------------------------

    p1 = Point3D(
        x=0.0,
        y=0.0,
        z=2.0,
    )

    p2 = Point3D(
        x=0.5,
        y=0.0,
        z=2.0,
    )

    distance = p1.distance_to(
        p2
    )

    print()
    print("Distance test:")
    print(
        f"Distance = "
        f"{distance:.4f} m"
    )

    print()
    print("Camera model test completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    self_test()
