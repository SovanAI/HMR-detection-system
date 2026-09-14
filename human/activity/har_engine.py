from typing import Any, Dict, List


class HAREngine:
    """
    First deterministic baseline HAR engine.

    This version combines temporal motion and posture
    features. It is intentionally rule-based so that the
    reasoning is transparent and easy to validate.

    It is NOT the final production HAR classifier.
    """

    def __init__(self):
        pass

    def classify_person(
        self,
        sequence: List[Dict[str, Any]],
        person_id: int,
    ) -> Dict[str, Any]:
        """
        Classify one person's activity from a temporal sequence.
        """

        if not sequence:
            return {
                "person_id": person_id,
                "activity": "unknown",
                "confidence": 0.0,
                "reason": "empty_sequence",
            }

        motion_frames = []

        left_hand_speeds = []
        right_hand_speeds = []
        pelvis_speeds = []

        posture_states = []
        body_orientations = []

        for frame in sequence:

            for person in frame.get(
                "persons",
                []
            ):

                if int(
                    person["person_id"]
                ) != int(person_id):
                    continue

                motion = person.get(
                    "motion",
                    {}
                )

                posture = person.get(
                    "posture",
                    {}
                )

                movement_state = motion.get(
                    "movement_state"
                )

                if movement_state:
                    motion_frames.append(
                        movement_state
                    )

                left_speed = motion.get(
                    "left_hand_speed"
                )

                right_speed = motion.get(
                    "right_hand_speed"
                )

                pelvis_speed = motion.get(
                    "pelvis_speed"
                )

                if left_speed is not None:
                    left_hand_speeds.append(
                        float(left_speed)
                    )

                if right_speed is not None:
                    right_hand_speeds.append(
                        float(right_speed)
                    )

                if pelvis_speed is not None:
                    pelvis_speeds.append(
                        float(pelvis_speed)
                    )

                posture_state = posture.get(
                    "posture_state"
                )

                if posture_state:
                    posture_states.append(
                        posture_state
                    )

                orientation = posture.get(
                    "body_orientation"
                )

                if orientation:
                    body_orientations.append(
                        orientation
                    )

        if not motion_frames:
            return {
                "person_id": person_id,
                "activity": "unknown",
                "confidence": 0.0,
                "reason": "person_not_found",
            }

        # ----------------------------------------------------
        # Aggregate features
        # ----------------------------------------------------

        moving_count = sum(
            1
            for state in motion_frames
            if state == "moving"
        )

        low_motion_count = sum(
            1
            for state in motion_frames
            if state == "low_motion"
        )

        stationary_count = sum(
            1
            for state in motion_frames
            if state == "stationary"
        )

        max_left_hand_speed = (
            max(left_hand_speeds)
            if left_hand_speeds
            else 0.0
        )

        max_right_hand_speed = (
            max(right_hand_speeds)
            if right_hand_speeds
            else 0.0
        )

        max_hand_speed = max(
            max_left_hand_speed,
            max_right_hand_speed,
        )

        average_pelvis_speed = (
            sum(pelvis_speeds)
            / len(pelvis_speeds)
            if pelvis_speeds
            else 0.0
        )

        # ----------------------------------------------------
        # Rule 1: Strong hand movement + stable pelvis
        # ----------------------------------------------------

        if (
            max_hand_speed >= 0.12
            and average_pelvis_speed < 0.01
        ):

            return {
                "person_id": person_id,
                "activity": "possible_reaching_or_arm_motion",
                "confidence": 0.70,
                "reason": (
                    "high hand speed with "
                    "low pelvis movement"
                ),
                "features": {
                    "max_hand_speed": max_hand_speed,
                    "average_pelvis_speed":
                        average_pelvis_speed,
                },
            }

        # ----------------------------------------------------
        # Rule 2: Predominantly moving
        # ----------------------------------------------------

        if moving_count >= 2:

            confidence = min(
                0.50
                + 0.10 * moving_count,
                0.90,
            )

            return {
                "person_id": person_id,
                "activity": "moving",
                "confidence": confidence,
                "reason": (
                    "multiple temporal frames "
                    "show significant motion"
                ),
                "features": {
                    "moving_frames": moving_count,
                    "low_motion_frames":
                        low_motion_count,
                    "stationary_frames":
                        stationary_count,
                },
            }

        # ----------------------------------------------------
        # Rule 3: Mostly stationary
        # ----------------------------------------------------

        if (
            stationary_count + low_motion_count
            >= max(
                1,
                len(motion_frames) - moving_count,
            )
        ):

            return {
                "person_id": person_id,
                "activity": "stationary_or_low_motion",
                "confidence": 0.65,
                "reason": (
                    "temporal sequence is "
                    "predominantly low motion"
                ),
                "features": {
                    "moving_frames": moving_count,
                    "low_motion_frames":
                        low_motion_count,
                    "stationary_frames":
                        stationary_count,
                },
            }

        return {
            "person_id": person_id,
            "activity": "unknown",
            "confidence": 0.20,
            "reason": "no_rule_matched",
        }

    def classify(
        self,
        sequence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Classify all people present in the temporal window.
        """

        person_ids = set()

        for frame in sequence:

            for person in frame.get(
                "persons",
                []
            ):

                person_ids.add(
                    int(person["person_id"])
                )

        results = []

        for person_id in sorted(
            person_ids
        ):

            results.append(
                self.classify_person(
                    sequence,
                    person_id,
                )
            )

        return {
            "start_frame": (
                sequence[0]["frame_id"]
                if sequence
                else None
            ),

            "end_frame": (
                sequence[-1]["frame_id"]
                if sequence
                else None
            ),

            "sequence_length": len(
                sequence
            ),

            "persons": results,
        }
