import glob
import json
import os
import time

import matplotlib.pyplot as plt


# ============================================================
# SKELETON CONNECTIONS
# ============================================================

SKELETON_EDGES = [
    ("nose", "neck"),

    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("neck", "pelvis"),

    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    ("nose", "right_eye"),
    ("nose", "left_eye"),

    ("right_eye", "right_ear"),
    ("left_eye", "left_ear"),
]


# ============================================================
# LOAD JSON
# ============================================================

def load_frame(path):

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# EXTRACT FIRST PERSON
# ============================================================

def get_person(data, person_id=0):

    persons = data.get("hmr", {}).get("persons", [])

    for person in persons:

        if int(person["person_id"]) == int(person_id):
            return person

    return None


# ============================================================
# VISUALIZE
# ============================================================

def visualize(
    files,
    person_id=0,
    interval=0.5,
):

    if not files:
        raise RuntimeError(
            "No fused JSON files found."
        )

    plt.ion()

    fig = plt.figure(
        figsize=(9, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    for path in files:

        data = load_frame(path)

        person = get_person(
            data,
            person_id,
        )

        if person is None:
            continue

        joints = {
            j["joint_id"]: j
            for j in person["pose"]["joints_3d"]
        }

        named_joints = {}

        # ----------------------------------------------------
        # Verified first 25 joint mapping
        # ----------------------------------------------------

        names = {
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

        for joint_id, name in names.items():

            if joint_id in joints:
                named_joints[name] = joints[joint_id]

        # ----------------------------------------------------
        # Clear previous frame
        # ----------------------------------------------------

        ax.clear()

        # ----------------------------------------------------
        # Plot joints
        # ----------------------------------------------------

        xs = []
        ys = []
        zs = []

        for name, joint in named_joints.items():

            x = joint["x"]
            y = joint["y"]
            z = joint["z"]

            xs.append(x)
            ys.append(y)
            zs.append(z)

        if xs:

            ax.scatter(
                xs,
                ys,
                zs,
                s=40,
            )

        # ----------------------------------------------------
        # Plot skeleton
        # ----------------------------------------------------

        for a, b in SKELETON_EDGES:

            if (
                a not in named_joints
                or b not in named_joints
            ):
                continue

            ja = named_joints[a]
            jb = named_joints[b]

            ax.plot(
                [
                    ja["x"],
                    jb["x"],
                ],
                [
                    ja["y"],
                    jb["y"],
                ],
                [
                    ja["z"],
                    jb["z"],
                ],
                linewidth=2,
            )

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        frame_id = data["frame_id"]
        timestamp = data["timestamp"]

        ax.set_title(
            f"BAS-HMR 3D Skeleton | "
            f"Frame {frame_id} | "
            f"Timestamp {timestamp:.3f}"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ----------------------------------------------------
        # Equal-ish axis limits
        # ----------------------------------------------------

        if xs:

            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            cz = sum(zs) / len(zs)

            radius = 1.2

            ax.set_xlim(
                cx - radius,
                cx + radius,
            )

            ax.set_ylim(
                cy - radius,
                cy + radius,
            )

            ax.set_zlim(
                cz - radius,
                cz + radius,
            )

        plt.draw()
        plt.pause(0.001)

        time.sleep(interval)

    print(
        "3D visualization complete."
    )

    plt.ioff()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        glob.glob(
            os.path.join(
                "output",
                "*_fused.json",
            )
        )
    )

    print(
        f"Found {len(files)} fused frames."
    )

    visualize(
        files,
        person_id=0,
        interval=0.5,
    )


if __name__ == "__main__":
    main()
