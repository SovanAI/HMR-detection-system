import math
from typing import Dict, Any, Optional


class PostureAnalyzer:
    """
    Computes geometric posture features from 3D human joints.

    This module does not perform machine-learning-based activity
    recognition. It only calculates body geometry and basic posture
    indicators.
    """

    def __init__(self):
        pass

    @staticmethod
    def _vector(a: Dict[str, float], b: Dict[str, float]):
        """
        Vector from point a to point b.
        """
        return (
            b["x"] - a["x"],
            b["y"] - a["y"],
            b["z"] - a["z"],
        )

    @staticmethod
    def _dot(a, b):
        return (
            a[0] * b[0]
            + a[1] * b[1]
            + a[2] * b[2]
        )

    @staticmethod
    def _magnitude(v):
        return math.sqrt(
            v[0] ** 2
            + v[1] ** 2
            + v[2] ** 2
        )

    @staticmethod
    def angle_between_vectors(a, b) -> Optional[float]:
        """
        Return angle between two vectors in degrees.
        """

        mag_a = PostureAnalyzer._magnitude(a)
        mag_b = PostureAnalyzer._magnitude(b)

        if mag_a == 0 or mag_b == 0:
            return None

        cosine = (
            PostureAnalyzer._dot(a, b)
            / (mag_a * mag_b)
        )

        cosine = max(
            -1.0,
            min(1.0, cosine)
        )

        return math.degrees(
            math.acos(cosine)
        )

    @staticmethod
    def joint_angle(
        a: Dict[str, float],
        b: Dict[str, float],
        c: Dict[str, float],
    ) -> Optional[float]:
        """
        Calculate angle ABC in degrees.
        """

        ba = PostureAnalyzer._vector(b, a)
        bc = PostureAnalyzer._vector(b, c)

        return PostureAnalyzer.angle_between_vectors(
            ba,
            bc,
        )

    @staticmethod
    def _midpoint(a, b):
        return {
            "x": (a["x"] + b["x"]) / 2.0,
            "y": (a["y"] + b["y"]) / 2.0,
            "z": (a["z"] + b["z"]) / 2.0,
        }

    def analyze_person(
        self,
        person: Dict[str, Any],
    ) -> Dict[str, Any]:

        joints = person.get(
            "joints_3d",
            {}
        )

        result = {
            "person_id": int(
                person["person_id"]
            ),
            "posture": {},
        }

        # ----------------------------------------------------
        # Required joints
        # ----------------------------------------------------

        required = [
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

        missing = [
            name
            for name in required
            if name not in joints
        ]

        if missing:

            result["posture"]["status"] = (
                "insufficient_data"
            )

            result["posture"]["missing_joints"] = missing

            return result

        # ----------------------------------------------------
        # Knee angles
        # ----------------------------------------------------

        left_knee_angle = self.joint_angle(
            joints["left_hip"],
            joints["left_knee"],
            joints["left_ankle"],
        )

        right_knee_angle = self.joint_angle(
            joints["right_hip"],
            joints["right_knee"],
            joints["right_ankle"],
        )

        # ----------------------------------------------------
        # Elbow angles
        # ----------------------------------------------------

        left_elbow_angle = self.joint_angle(
            joints["left_shoulder"],
            joints["left_elbow"],
            joints["left_wrist"],
        )

        right_elbow_angle = self.joint_angle(
            joints["right_shoulder"],
            joints["right_elbow"],
            joints["right_wrist"],
        )

        # ----------------------------------------------------
        # Shoulder midpoint
        # ----------------------------------------------------

        shoulder_center = self._midpoint(
            joints["left_shoulder"],
            joints["right_shoulder"],
        )

        # ----------------------------------------------------
        # Torso vector
        # ----------------------------------------------------

        torso_vector = self._vector(
            joints["pelvis"],
            shoulder_center,
        )

        # Vertical axis in HMR coordinates.
        #
        # We use the Y direction as a reference only for
        # a geometric verticality estimate.
        vertical_axis = (
            0.0,
            -1.0,
            0.0,
        )

        torso_vertical_angle = (
            self.angle_between_vectors(
                torso_vector,
                vertical_axis,
            )
        )

        # ----------------------------------------------------
        # Hip / body bend proxy
        # ----------------------------------------------------

        hip_center = self._midpoint(
            joints["left_hip"],
            joints["right_hip"],
        )

        hip_to_shoulder = self._vector(
            hip_center,
            shoulder_center,
        )

        body_axis_angle = (
            self.angle_between_vectors(
                hip_to_shoulder,
                vertical_axis,
            )
        )

        # ----------------------------------------------------
        # Arm elevation proxy
        # ----------------------------------------------------

        left_arm_vector = self._vector(
            joints["left_shoulder"],
            joints["left_wrist"],
        )

        right_arm_vector = self._vector(
            joints["right_shoulder"],
            joints["right_wrist"],
        )

        left_arm_angle = (
            self.angle_between_vectors(
                left_arm_vector,
                vertical_axis,
            )
        )

        right_arm_angle = (
            self.angle_between_vectors(
                right_arm_vector,
                vertical_axis,
            )
        )

        # ----------------------------------------------------
        # Basic posture indicators
        # ----------------------------------------------------

        knee_angles = [
            x
            for x in [
                left_knee_angle,
                right_knee_angle,
            ]
            if x is not None
        ]

        average_knee_angle = (
            sum(knee_angles) / len(knee_angles)
            if knee_angles
            else None
        )

        if (
            torso_vertical_angle is not None
            and torso_vertical_angle < 25
        ):
            body_orientation = "upright"

        elif (
            torso_vertical_angle is not None
            and torso_vertical_angle < 55
        ):
            body_orientation = "inclined"

        else:
            body_orientation = "horizontal_or_bent"

        if (
            average_knee_angle is not None
            and average_knee_angle > 150
            and body_orientation == "upright"
        ):
            posture_state = "standing_like"

        elif (
            average_knee_angle is not None
            and average_knee_angle < 120
        ):
            posture_state = "knees_bent"

        else:
            posture_state = "undetermined"

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result["posture"] = {

            "status": "ok",

            "body_orientation": body_orientation,

            "posture_state": posture_state,

            "angles": {
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle,
                "left_elbow": left_elbow_angle,
                "right_elbow": right_elbow_angle,
                "torso_vertical": torso_vertical_angle,
                "body_axis": body_axis_angle,
                "left_arm_vertical": left_arm_angle,
                "right_arm_vertical": right_arm_angle,
            },

            "average_knee_angle": average_knee_angle,
        }

        return result

    def analyze(
        self,
        human_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze all people in a Human State frame.
        """

        result = {
            "frame_id": int(
                human_state["frame_id"]
            ),
            "timestamp": float(
                human_state["timestamp"]
            ),
            "persons": [],
        }

        for person in human_state.get(
            "persons",
            []
        ):

            result["persons"].append(
                self.analyze_person(
                    person
                )
            )

        return result
