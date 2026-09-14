import json
from typing import Any, Dict

from human.activity.joint_mapping import get_joint_name


class HumanStateExtractor:
    """
    Converts fused YOLO + HMR JSON into a higher-level
    human state representation.

    This first version uses the verified 25-joint mapping
    and preserves HMR extra joints as extra_0 ... extra_18.
    """

    def extract_from_fused(self, fused_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(fused_data, dict):
            raise TypeError("fused_data must be a dictionary")

        frame_id = fused_data["frame_id"]
        timestamp = fused_data["timestamp"]

        yolo_persons = fused_data.get("yolo", {}).get("persons", [])
        hmr_persons = fused_data.get("hmr", {}).get("persons", [])

        persons = []

        for hmr_person in hmr_persons:

            person_id = int(
                hmr_person["person_id"]
            )

            yolo_person = self._find_yolo_person(
                yolo_persons,
                person_id,
            )

            joints = hmr_person["pose"]["joints_3d"]

            joint_dict = {}

            for joint in joints:

                joint_id = int(
                    joint["joint_id"]
                )

                name = get_joint_name(
                    joint_id
                )

                joint_dict[name] = {
                    "joint_id": joint_id,
                    "x": float(joint["x"]),
                    "y": float(joint["y"]),
                    "z": float(joint["z"]),
                }

            camera_translation = hmr_person.get(
                "camera_translation",
                {},
            )

            person_state = {
                "person_id": person_id,

                "detection": {
                    "confidence": (
                        float(yolo_person["confidence"])
                        if yolo_person is not None
                        else None
                    ),
                    "bbox": (
                        yolo_person["bbox"]
                        if yolo_person is not None
                        else hmr_person.get("bbox")
                    ),
                },

                "camera_translation": {
                    "x": float(
                        camera_translation.get("x", 0.0)
                    ),
                    "y": float(
                        camera_translation.get("y", 0.0)
                    ),
                    "z": float(
                        camera_translation.get("z", 0.0)
                    ),
                },

                "joints_3d": joint_dict,
            }

            persons.append(
                person_state
            )

        return {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "model": "HumanStateExtractor",
            "persons": persons,
        }

    @staticmethod
    def _find_yolo_person(
        yolo_persons,
        person_id,
    ):
        for person in yolo_persons:

            if int(
                person["person_id"]
            ) == int(person_id):

                return person

        return None


def load_fused_json(path: str) -> Dict[str, Any]:
    """
    Load a fused frame JSON file.
    """

    with open(path, "r") as f:
        return json.load(f)


def save_human_state(
    state: Dict[str, Any],
    path: str,
) -> None:
    """
    Save human state as JSON.
    """

    with open(path, "w") as f:
        json.dump(
            state,
            f,
            indent=2,
        )
