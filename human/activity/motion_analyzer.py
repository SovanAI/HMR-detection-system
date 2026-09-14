import math
from typing import Dict, Any


class MotionAnalyzer:
    """
    Computes temporal motion features from consecutive Human State
    representations.

    Position units follow the HMR2 coordinate system.
    Velocity is calculated using the timestamps of the two frames.
    """

    def __init__(self):
        self.previous_state = None

    @staticmethod
    def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.sqrt(
            (a["x"] - b["x"]) ** 2
            + (a["y"] - b["y"]) ** 2
            + (a["z"] - b["z"]) ** 2
        )

    @staticmethod
    def _speed(
        a: Dict[str, float],
        b: Dict[str, float],
        dt: float,
    ) -> float:
        if dt <= 0:
            return 0.0

        return MotionAnalyzer._distance(a, b) / dt

    def analyze(
        self,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:

        current_timestamp = float(
            current_state["timestamp"]
        )

        persons = current_state.get(
            "persons",
            []
        )

        result = {
            "frame_id": int(
                current_state["frame_id"]
            ),
            "timestamp": current_timestamp,
            "persons": [],
        }

        # First frame has no previous frame to compare against.
        if self.previous_state is None:

            for person in persons:

                result["persons"].append({
                    "person_id": int(
                        person["person_id"]
                    ),
                    "motion": {
                        "dt": None,
                        "movement_state": "unknown",
                    },
                })

            self.previous_state = current_state
            return result

        previous_timestamp = float(
            self.previous_state["timestamp"]
        )

        dt = current_timestamp - previous_timestamp

        previous_persons = {
            int(person["person_id"]): person
            for person in self.previous_state.get(
                "persons",
                []
            )
        }

        for current_person in persons:

            person_id = int(
                current_person["person_id"]
            )

            current_joints = current_person.get(
                "joints_3d",
                {}
            )

            previous_person = previous_persons.get(
                person_id
            )

            if previous_person is None:

                result["persons"].append({
                    "person_id": person_id,
                    "motion": {
                        "dt": dt,
                        "movement_state": "new",
                    },
                })

                continue

            previous_joints = previous_person.get(
                "joints_3d",
                {}
            )

            motion = {
                "dt": dt,
                "joints": {},
            }

            # ------------------------------------------------
            # Important human joints
            # ------------------------------------------------

            tracked_joints = [
                "nose",
                "neck",
                "pelvis",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]

            speeds = []

            for joint_name in tracked_joints:

                current_joint = current_joints.get(
                    joint_name
                )

                previous_joint = previous_joints.get(
                    joint_name
                )

                if (
                    current_joint is None
                    or previous_joint is None
                ):
                    continue

                displacement = self._distance(
                    current_joint,
                    previous_joint,
                )

                speed = self._speed(
                    current_joint,
                    previous_joint,
                    dt,
                )

                motion["joints"][joint_name] = {
                    "displacement": displacement,
                    "speed": speed,
                }

                speeds.append(speed)

            # ------------------------------------------------
            # Key body regions
            # ------------------------------------------------

            left_hand_speed = (
                motion["joints"]
                .get("left_wrist", {})
                .get("speed", 0.0)
            )

            right_hand_speed = (
                motion["joints"]
                .get("right_wrist", {})
                .get("speed", 0.0)
            )

            pelvis_speed = (
                motion["joints"]
                .get("pelvis", {})
                .get("speed", 0.0)
            )

            head_speed = (
                motion["joints"]
                .get("nose", {})
                .get("speed", 0.0)
            )

            average_joint_speed = (
                sum(speeds) / len(speeds)
                if speeds
                else 0.0
            )

            # ------------------------------------------------
            # Basic movement classification
            #
            # This is NOT HAR yet.
            # It only describes motion magnitude.
            # ------------------------------------------------

            if average_joint_speed < 0.01:
                movement_state = "stationary"

            elif average_joint_speed < 0.05:
                movement_state = "low_motion"

            else:
                movement_state = "moving"

            motion["left_hand_speed"] = left_hand_speed
            motion["right_hand_speed"] = right_hand_speed
            motion["pelvis_speed"] = pelvis_speed
            motion["head_speed"] = head_speed
            motion["average_joint_speed"] = average_joint_speed
            motion["movement_state"] = movement_state

            result["persons"].append({
                "person_id": person_id,
                "motion": motion,
            })

        self.previous_state = current_state

        return result
