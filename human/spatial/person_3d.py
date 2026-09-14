from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Person3D:
    """
    HMR2 person position in the HMR2 camera coordinate system.
    """

    person_id: int

    x: float
    y: float
    z: float

    confidence: float | None = None


class Person3DExtractor:
    """
    Extracts a person's 3D camera position from HMR2 output.

    HMR2 provides camera_translation:
        x
        y
        z

    These values remain in the HMR2 coordinate system.
    """

    @staticmethod
    def extract(person: Dict[str, Any]) -> Person3D:

        if "person_id" not in person:
            raise ValueError(
                "HMR person is missing person_id"
            )

        camera_translation = person.get(
            "camera_translation"
        )

        if not camera_translation:
            raise ValueError(
                f"Person {person['person_id']} "
                "is missing camera_translation"
            )

        x = float(
            camera_translation["x"]
        )

        y = float(
            camera_translation["y"]
        )

        z = float(
            camera_translation["z"]
        )

        confidence = person.get(
            "confidence"
        )

        if confidence is not None:
            confidence = float(confidence)

        return Person3D(
            person_id=int(person["person_id"]),
            x=x,
            y=y,
            z=z,
            confidence=confidence,
        )


if __name__ == "__main__":

    import json

    path = (
        "test_results/"
        "person_hand_hmr_fusion/"
        "fused_human_frame.json"
    )

    with open(path, "r") as f:
        data = json.load(f)

    print("=" * 70)
    print("HMR2 PERSON 3D EXTRACTION TEST")
    print("=" * 70)

    extractor = Person3DExtractor()

    for person in data.get("persons", []):

        position = extractor.extract(person)

        print(
            f"\nPerson #{position.person_id}"
        )

        print(
            f"X: {position.x:.4f}"
        )

        print(
            f"Y: {position.y:.4f}"
        )

        print(
            f"Z: {position.z:.4f}"
        )

    print("\n" + "=" * 70)
