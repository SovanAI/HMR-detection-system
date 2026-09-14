"""
BAS Human Activity Recognition
Phase 6 - Physical Distance Validation

Compares:
    Actual real-world distance
against
    Camera measured metric distance.

All distances are in metres.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


# ============================================================
# Configuration
# ============================================================

VALIDATION_DIR = Path(
    "test_results/distance_validation"
)

VALIDATION_FILE = (
    VALIDATION_DIR / "validation_samples.json"
)


# ============================================================
# Validation sample
# ============================================================

@dataclass
class ValidationSample:

    sample_id: int

    actual_distance_m: float
    measured_distance_m: float

    error_m: float
    absolute_error_m: float

    percentage_error: float

    notes: str = ""

    def to_dict(self) -> dict:

        return {
            "sample_id": self.sample_id,

            "actual_distance_m": round(
                self.actual_distance_m,
                4,
            ),

            "measured_distance_m": round(
                self.measured_distance_m,
                4,
            ),

            "error_m": round(
                self.error_m,
                4,
            ),

            "absolute_error_m": round(
                self.absolute_error_m,
                4,
            ),

            "percentage_error": round(
                self.percentage_error,
                4,
            ),

            "notes": self.notes,
        }


# ============================================================
# Validator
# ============================================================

class DistanceValidator:

    def __init__(self):

        self.samples: List[
            ValidationSample
        ] = []

    # --------------------------------------------------------
    # Add measurement
    # --------------------------------------------------------

    def add_sample(
        self,
        actual_distance_m: float,
        measured_distance_m: float,
        notes: str = "",
    ) -> ValidationSample:

        actual_distance_m = float(
            actual_distance_m
        )

        measured_distance_m = float(
            measured_distance_m
        )

        if not math.isfinite(
            actual_distance_m
        ):
            raise ValueError(
                "Actual distance must be finite."
            )

        if not math.isfinite(
            measured_distance_m
        ):
            raise ValueError(
                "Measured distance must be finite."
            )

        if actual_distance_m <= 0:
            raise ValueError(
                "Actual distance must be > 0."
            )

        if measured_distance_m < 0:
            raise ValueError(
                "Measured distance must be >= 0."
            )

        # ----------------------------------------------------
        # Error
        #
        # positive = camera overestimates
        # negative = camera underestimates
        # ----------------------------------------------------

        error_m = (
            measured_distance_m -
            actual_distance_m
        )

        absolute_error_m = abs(
            error_m
        )

        percentage_error = (
            absolute_error_m /
            actual_distance_m
        ) * 100.0

        sample = ValidationSample(
            sample_id=len(self.samples) + 1,

            actual_distance_m=(
                actual_distance_m
            ),

            measured_distance_m=(
                measured_distance_m
            ),

            error_m=error_m,

            absolute_error_m=(
                absolute_error_m
            ),

            percentage_error=(
                percentage_error
            ),

            notes=notes,
        )

        self.samples.append(sample)

        return sample

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def calculate_metrics(self) -> dict:

        if not self.samples:

            return {
                "sample_count": 0,
                "mae_m": 0.0,
                "rmse_m": 0.0,
                "mean_error_m": 0.0,
                "std_error_m": 0.0,
                "mean_percentage_error": 0.0,
                "min_absolute_error_m": 0.0,
                "max_absolute_error_m": 0.0,
            }

        errors = [
            sample.error_m
            for sample in self.samples
        ]

        absolute_errors = [
            sample.absolute_error_m
            for sample in self.samples
        ]

        percentage_errors = [
            sample.percentage_error
            for sample in self.samples
        ]

        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        mae = (
            sum(absolute_errors)
            / len(absolute_errors)
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        mse = (
            sum(
                error * error
                for error in errors
            )
            / len(errors)
        )

        rmse = math.sqrt(mse)

        # ----------------------------------------------------
        # Mean signed error
        # ----------------------------------------------------

        mean_error = (
            sum(errors)
            / len(errors)
        )

        # ----------------------------------------------------
        # Standard deviation
        # ----------------------------------------------------

        if len(errors) > 1:

            std_error = math.sqrt(
                sum(
                    (
                        error -
                        mean_error
                    ) ** 2
                    for error in errors
                )
                / (len(errors) - 1)
            )

        else:

            std_error = 0.0

        return {
            "sample_count": len(
                self.samples
            ),

            "mae_m": mae,

            "rmse_m": rmse,

            "mean_error_m": mean_error,

            "std_error_m": std_error,

            "mean_percentage_error": (
                sum(percentage_errors)
                / len(percentage_errors)
            ),

            "min_absolute_error_m": min(
                absolute_errors
            ),

            "max_absolute_error_m": max(
                absolute_errors
            ),
        }

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    def print_report(self):

        print()
        print("=" * 80)
        print("PHYSICAL DISTANCE VALIDATION REPORT")
        print("=" * 80)

        print()

        print(
            f"{'ID':<5}"
            f"{'Actual':<14}"
            f"{'Measured':<14}"
            f"{'Error':<14}"
            f"{'Abs Error':<14}"
            f"{'Error %':<12}"
        )

        print("-" * 80)

        for sample in self.samples:

            print(
                f"{sample.sample_id:<5}"
                f"{sample.actual_distance_m:<14.3f}"
                f"{sample.measured_distance_m:<14.3f}"
                f"{sample.error_m:<14.3f}"
                f"{sample.absolute_error_m:<14.3f}"
                f"{sample.percentage_error:<12.2f}"
            )

        print("-" * 80)

        metrics = self.calculate_metrics()

        print()

        print(
            f"Samples                : "
            f"{metrics['sample_count']}"
        )

        print(
            f"MAE                    : "
            f"{metrics['mae_m']:.4f} m"
        )

        print(
            f"RMSE                   : "
            f"{metrics['rmse_m']:.4f} m"
        )

        print(
            f"Mean signed error      : "
            f"{metrics['mean_error_m']:.4f} m"
        )

        print(
            f"Standard deviation     : "
            f"{metrics['std_error_m']:.4f} m"
        )

        print(
            f"Mean percentage error  : "
            f"{metrics['mean_percentage_error']:.2f} %"
        )

        print(
            f"Minimum absolute error : "
            f"{metrics['min_absolute_error_m']:.4f} m"
        )

        print(
            f"Maximum absolute error : "
            f"{metrics['max_absolute_error_m']:.4f} m"
        )

        print()

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    def save_json(
        self,
        filepath: Path = VALIDATION_FILE,
    ):

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "samples": [
                sample.to_dict()
                for sample in self.samples
            ],

            "metrics": {
                key: round(value, 6)
                if isinstance(value, float)
                else value

                for key, value
                in self.calculate_metrics().items()
            },
        }

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
            )

        print(
            f"Validation saved to: "
            f"{filepath}"
        )


# ============================================================
# Self Test
# ============================================================

def self_test():

    print("=" * 80)
    print("DISTANCE VALIDATOR SELF TEST")
    print("=" * 80)

    validator = DistanceValidator()

    # --------------------------------------------------------
    # Synthetic measurements
    # --------------------------------------------------------

    test_data = [
        (0.50, 0.52),
        (1.00, 0.96),
        (1.50, 1.48),
        (2.00, 2.07),
        (2.50, 2.43),
    ]

    for actual, measured in test_data:

        validator.add_sample(
            actual_distance_m=actual,
            measured_distance_m=measured,
            notes="synthetic self test",
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    validator.print_report()

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    validator.save_json(
        Path(
            "test_results/"
            "distance_validation/"
            "self_test.json"
        )
    )

    # --------------------------------------------------------
    # Basic assertions
    # --------------------------------------------------------

    metrics = validator.calculate_metrics()

    assert metrics["sample_count"] == 5

    assert metrics["mae_m"] >= 0

    assert metrics["rmse_m"] >= 0

    assert (
        metrics["max_absolute_error_m"]
        >=
        metrics["min_absolute_error_m"]
    )

    print()
    print("=" * 80)
    print("DISTANCE VALIDATOR TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    self_test()
