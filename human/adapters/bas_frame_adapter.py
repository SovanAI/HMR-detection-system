"""
BAS Frame Adapter

Converts the unified Person + Hand + HMR fusion result into the
canonical schema expected by the existing HumanStateExtractor,
while preserving the unified representation.

Input:
{
    "frame_id": ...,
    "timestamp": ...,
    "image": {...},
    "persons": [
        {
            "person_id": ...,
            "bbox": ...,
            "confidence": ...,
            "camera_translation": ...,
            "pose": {...},
            "hands": [...],
            "hand_count": ...
        }
    ],
    "unassociated_hands": [],
    "metadata": {...}
}

Output:
{
    "frame_id": ...,
    "timestamp": ...,
    "image": {...},

    "yolo": {
        "persons": [...]
    },

    "hmr": {
        "persons": [...]
    },

    "hands": {
        "hands": [...]
    },

    "persons": [
        ...
    ],

    "unassociated_hands": [...],

    "metadata": {...}
}
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


class BASFrameAdapter:
    """
    Converts unified BAS perception output into the canonical
    representation used by downstream human-state modules.
    """

    def adapt(self, fused_data: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(fused_data, dict):
            raise TypeError("fused_data must be a dictionary")

        frame_id = fused_data.get("frame_id")
        timestamp = fused_data.get("timestamp")

        image = deepcopy(
            fused_data.get(
                "image",
                {
                    "width": 0,
                    "height": 0,
                },
            )
        )

        unified_persons: List[Dict[str, Any]] = fused_data.get(
            "persons", []
        )

        unassociated_hands = deepcopy(
            fused_data.get("unassociated_hands", [])
        )

        yolo_persons = []
        hmr_persons = []
        all_hands = []

        for person in unified_persons:

            person_id = int(person.get("person_id", -1))

            bbox = deepcopy(
                person.get(
                    "bbox",
                    {
                        "x1": 0,
                        "y1": 0,
                        "x2": 0,
                        "y2": 0,
                    },
                )
            )

            confidence = float(
                person.get(
                    "confidence",
                    person.get(
                        "detection",
                        {}
                    ).get("confidence", 0.0),
                )
            )

            # --------------------------------------------------
            # YOLO representation
            # --------------------------------------------------

            yolo_person = {
                "person_id": person_id,
                "confidence": confidence,
                "bbox": bbox,
            }

            yolo_persons.append(yolo_person)

            # --------------------------------------------------
            # HMR representation
            # --------------------------------------------------

            hmr_person = {
                "person_id": person_id,
                "bbox": bbox,
                "camera_translation": deepcopy(
                    person.get("camera_translation", {})
                ),
                "pose": deepcopy(
                    person.get(
                        "pose",
                        {
                            "joint_count": 0,
                            "joints_3d": [],
                        },
                    )
                ),
            }

            hmr_persons.append(hmr_person)

            # --------------------------------------------------
            # Hand representation
            # --------------------------------------------------

            person_hands = deepcopy(
                person.get("hands", [])
            )

            for hand in person_hands:
                all_hands.append(hand)

        # ------------------------------------------------------
        # Preserve original unified persons
        # ------------------------------------------------------

        output = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "image": image,

            # Legacy-compatible representation
            "yolo": {
                "persons": yolo_persons,
            },

            "hmr": {
                "persons": hmr_persons,
            },

            "hands": {
                "hands": all_hands,
            },

            # Canonical unified representation
            "persons": deepcopy(unified_persons),

            "unassociated_hands": unassociated_hands,

            "metadata": {
                **deepcopy(
                    fused_data.get("metadata", {})
                ),
                "adapter": "BASFrameAdapter",
                "schema_version": "1.0",
            },
        }

        return output


def adapt_fused_frame(
    fused_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience function.
    """
    return BASFrameAdapter().adapt(fused_data)


if __name__ == "__main__":

    # Small self-test

    sample = {
        "frame_id": 1,
        "timestamp": 123.456,
        "image": {
            "width": 640,
            "height": 480,
        },
        "persons": [
            {
                "person_id": 0,
                "confidence": 0.91,
                "bbox": {
                    "x1": 10,
                    "y1": 20,
                    "x2": 300,
                    "y2": 450,
                },
                "camera_translation": {
                    "x": 0.1,
                    "y": 0.2,
                    "z": 10.0,
                },
                "pose": {
                    "joint_count": 44,
                    "joints_3d": [],
                },
                "hands": [
                    {
                        "hand_id": 0,
                        "confidence": 0.88,
                        "center": {
                            "x": 100,
                            "y": 200,
                        },
                    }
                ],
                "hand_count": 1,
            }
        ],
        "unassociated_hands": [],
        "metadata": {},
    }

    adapter = BASFrameAdapter()

    result = adapter.adapt(sample)

    assert len(result["persons"]) == 1
    assert len(result["yolo"]["persons"]) == 1
    assert len(result["hmr"]["persons"]) == 1
    assert len(result["hands"]["hands"]) == 1

    print("BASFrameAdapter self-test: PASS")
