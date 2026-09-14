import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

json_file = ROOT / "output" / "frame_000000.json"

with open(json_file, "r") as f:
    data = json.load(f)

print()
print("====================================")
print(" FUSED FRAME INSPECTION")
print("====================================")

print(f"Frame ID       : {data['frame_id']}")
print(f"Timestamp      : {data['timestamp']}")
print(
    f"Sync status    : "
    f"{data['synchronization']['status']}"
)

print()

yolo_persons = data["yolo"]["persons"]
hmr_persons = data["hmr"]["persons"]

print(f"YOLO persons   : {len(yolo_persons)}")
print(f"HMR persons    : {len(hmr_persons)}")

print()

for i, (yolo_person, hmr_person) in enumerate(
    zip(yolo_persons, hmr_persons)
):

    joints = hmr_person["pose"]["joints_3d"]

    print("------------------------------------")
    print(f"Person {i}")
    print("------------------------------------")

    print(
        f"YOLO confidence : "
        f"{yolo_person['confidence']}"
    )

    print(
        f"YOLO bbox       : "
        f"{yolo_person['bbox']}"
    )

    print(
        f"HMR joint count : "
        f"{len(joints)}"
    )

    # Print first joint
    joint0 = joints[0]

    print(
        f"Joint 0        : "
        f"x={joint0['x']:.4f}, "
        f"y={joint0['y']:.4f}, "
        f"z={joint0['z']:.4f}"
    )

print()
print("====================================")
print(" INSPECTION COMPLETE")
print("====================================")
