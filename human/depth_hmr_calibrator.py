from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class CalibrationSample:
    """
    One correspondence between:

        Depth Anything relative depth
        HMR2 camera-space Z

    Both values must come from approximately the
    same physical location in the image.
    """

    relative_depth: float
    hmr_z: float


@dataclass
class DepthHMRCalibration:
    """
    Linear mapping:

        hmr_z = scale * relative_depth + offset

    IMPORTANT:
        This maps Depth Anything values into the
        HMR2 Z scale.

        It does NOT automatically make the result metres.
    """

    scale: float
    offset: float
    sample_count: int
    rmse: float


class DepthHMRCalibrator:
    """
    Learns a linear relationship between Depth Anything
    relative depth and HMR2 camera-space Z.

    Minimum recommended samples: 3+
    Better: collect samples across multiple frames and
    multiple person positions.
    """

    def __init__(self):

        self.samples: List[CalibrationSample] = []

    # ========================================================
    # ADD SAMPLE
    # ========================================================

    def add_sample(
        self,
        relative_depth: float,
        hmr_z: float,
    ):

        relative_depth = float(relative_depth)
        hmr_z = float(hmr_z)

        if not np.isfinite(relative_depth):
            raise ValueError(
                "relative_depth must be finite"
            )

        if not np.isfinite(hmr_z):
            raise ValueError(
                "hmr_z must be finite"
            )

        self.samples.append(
            CalibrationSample(
                relative_depth=relative_depth,
                hmr_z=hmr_z,
            )
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(self) -> DepthHMRCalibration:

        if len(self.samples) < 2:

            raise ValueError(
                "At least 2 calibration samples are required"
            )

        x = np.array(
            [
                sample.relative_depth
                for sample in self.samples
            ],
            dtype=np.float64,
        )

        y = np.array(
            [
                sample.hmr_z
                for sample in self.samples
            ],
            dtype=np.float64,
        )

        # Linear least-squares fit:
        #
        # y = scale*x + offset

        A = np.column_stack(
            (
                x,
                np.ones_like(x),
            )
        )

        solution, _, _, _ = np.linalg.lstsq(
            A,
            y,
            rcond=None,
        )

        scale = float(solution[0])
        offset = float(solution[1])

        predicted = (
            scale * x
            + offset
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (predicted - y) ** 2
                )
            )
        )

        return DepthHMRCalibration(
            scale=scale,
            offset=offset,
            sample_count=len(self.samples),
            rmse=rmse,
        )

    # ========================================================
    # CONVERT DEPTH
    # ========================================================

    @staticmethod
    def convert(
        relative_depth: float,
        calibration: DepthHMRCalibration,
    ) -> float:

        return (
            calibration.scale
            * float(relative_depth)
            + calibration.offset
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DEPTH ↔ HMR CALIBRATION TEST")
    print("=" * 70)

    calibrator = DepthHMRCalibrator()

    # Synthetic example only.
    #
    # Imagine Depth Anything produced:
    #
    # 0.20 -> HMR Z 8
    # 0.40 -> HMR Z 12
    # 0.60 -> HMR Z 16
    # 0.80 -> HMR Z 20

    samples = [
        (0.20, 8.0),
        (0.40, 12.0),
        (0.60, 16.0),
        (0.80, 20.0),
    ]

    for depth, hmr_z in samples:

        calibrator.add_sample(
            relative_depth=depth,
            hmr_z=hmr_z,
        )

    calibration = calibrator.fit()

    print()
    print(
        f"Scale:  {calibration.scale:.6f}"
    )

    print(
        f"Offset: {calibration.offset:.6f}"
    )

    print(
        f"Samples: {calibration.sample_count}"
    )

    print(
        f"RMSE:   {calibration.rmse:.6f}"
    )

    test_depth = 0.50

    converted = calibrator.convert(
        test_depth,
        calibration,
    )

    print()
    print(
        f"Relative depth: {test_depth:.3f}"
    )

    print(
        f"Estimated HMR Z: {converted:.3f}"
    )

    print()
    print(
        "NOTE: This is an HMR-compatible relative"
    )

    print(
        "depth estimate, NOT guaranteed metres."
    )

    print("=" * 70)
