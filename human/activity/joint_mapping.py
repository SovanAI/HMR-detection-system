# ============================================================
# HMR2 JOINT MAPPING
# ============================================================

# Verified 25-joint OpenPose-style portion produced by the
# SMPL wrapper used by this HMR2 installation.

JOINT_NAMES_25 = {
    0: "nose",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "pelvis",
    9: "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    15: "right_eye",
    16: "left_eye",
    17: "right_ear",
    18: "left_ear",
    19: "left_big_toe",
    20: "left_small_toe",
    21: "left_heel",
    22: "right_big_toe",
    23: "right_small_toe",
    24: "right_heel",
}


def get_joint_name(joint_id: int) -> str:
    """
    Return the semantic name for a joint.

    Joints 0-24 are the verified named joints.
    Joints 25-43 are kept as extra joints until their
    exact semantic definitions are verified.
    """

    joint_id = int(joint_id)

    if joint_id in JOINT_NAMES_25:
        return JOINT_NAMES_25[joint_id]

    if 25 <= joint_id <= 43:
        return f"extra_{joint_id - 25}"

    raise ValueError(
        f"Invalid HMR2 joint_id: {joint_id}. "
        f"Expected 0-43."
    )
