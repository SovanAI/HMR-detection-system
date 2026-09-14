"""
BAS Human Feature Analyzer

Pipeline:

BAS fused frame
    ↓
Human State
    ↓
Motion
    ↓
Posture
    ↓
Unified Human Features

This module does NOT classify activities.
HAR classification is handled separately by HAREngine.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from human.activity.state_extractor import HumanStateExtractor
from human.activity.motion_analyzer import MotionAnalyzer
from human.activity.posture_analyzer import PostureAnalyzer


class HumanFeatureAnalyzer:
    """
    Combines human state, motion, posture, and hand information
    into a single temporal feature representation.
    """

    def __init__(self):

        self.state_extractor = HumanStateExtractor()

        self.motion_analyzer = MotionAnalyzer()

        self.posture_analyzer = PostureAnalyzer()

    # ----------------------------------------------------------
    # Find hands belonging to a person
    # ----------------------------------------------------------

    @staticmethod
    def _get_person_hands(
        fused_data: Dict[str, Any],
        person_id: int,
    ):

        persons = fused_data.get(
            "persons",
            [],
        )

        for person in persons:

            if int(
                person.get("person_id", -1)
            ) == int(person_id):

                return person.get(
                    "hands",
                    [],
                )

        return []

    # ----------------------------------------------------------
    # Process one frame
    # ----------------------------------------------------------

    def process_frame(
        self,
        fused_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process one canonical BAS fused frame.

        Returns a feature frame suitable for the
        TemporalFeatureBuffer and HAREngine.
        """

        # ======================================================
        # 1. HUMAN STATE
        # ======================================================

        human_state = (
            self.state_extractor.extract_from_fused(
                fused_data
            )
        )

        # ======================================================
        # 2. MOTION
        # ======================================================

        motion = self.motion_analyzer.analyze(
            human_state
        )

        # ======================================================
        # 3. POSTURE
        # ======================================================

        posture = self.posture_analyzer.analyze(
            human_state
        )

        # ======================================================
        # 4. INDEX RESULTS BY PERSON ID
        # ======================================================

        motion_persons = {
            int(person["person_id"]): person
            for person in motion.get(
                "persons",
                [],
            )
        }

        posture_persons = {
            int(person["person_id"]): person
            for person in posture.get(
                "persons",
                [],
            )
        }

        # ======================================================
        # 5. COMBINE
        # ======================================================

        persons = []

        for person in human_state.get(
            "persons",
            [],
        ):

            person_id = int(
                person["person_id"]
            )

            motion_data = motion_persons.get(
                person_id,
                {},
            )

            posture_data = posture_persons.get(
                person_id,
                {},
            )

            # --------------------------------------------------
            # Hand information
            # --------------------------------------------------

            hands = self._get_person_hands(
                fused_data,
                person_id,
            )

            # --------------------------------------------------
            # Detection
            # --------------------------------------------------

            detection = person.get(
                "detection",
                {},
            )

            # --------------------------------------------------
            # Position
            # --------------------------------------------------

            joints = person.get(
                "joints_3d",
                {},
            )

            position = {
                "camera_translation": person.get(
                    "camera_translation",
                    {},
                ),
                "pelvis": joints.get(
                    "pelvis"
                ),
            }

            # --------------------------------------------------
            # Final person feature representation
            # --------------------------------------------------

            persons.append(
                {
                    "person_id": person_id,

                    "detection": detection,

                    "position": position,

                    "motion": motion_data.get(
                        "motion",
                        {},
                    ),

                    "posture": posture_data.get(
                        "posture",
                        {},
                    ),

                    "joints_3d": joints,

                    "hands": hands,

                    "hand_count": len(hands),
                }
            )

        # ======================================================
        # 6. FRAME OUTPUT
        # ======================================================

        return {
            "frame_id": int(
                human_state["frame_id"]
            ),

            "timestamp": float(
                human_state["timestamp"]
            ),

            "model": "HumanFeatureAnalyzer",

            "persons": persons,

            "metadata": {
                "feature_version": "1.1",

                "features": [
                    "detection",
                    "position",
                    "joints_3d",
                    "motion",
                    "posture",
                    "hands",
                ],
            },
        }


# ==============================================================
# FILE HELPERS
# ==============================================================

def load_fused_file(
    path: str,
) -> Dict[str, Any]:
    """
    Load fused JSON.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_features(
    features: Dict[str, Any],
    path: str,
) -> None:
    """
    Save combined HAR features.
    """

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            features,
            f,
            indent=2,
        )