import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
HMR2_ROOT = ROOT / "4D-Humans"

sys.path.insert(0, str(HMR2_ROOT))

from hmr2.models import load_hmr2
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils import recursive_to


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHECKPOINT = (
    Path.home()
    / ".cache"
    / "4DHumans"
    / "logs"
    / "train"
    / "multiruns"
    / "hmr2"
    / "0"
    / "checkpoints"
    / "epoch=35-step=1000000.ckpt"
)

OUTPUT_DIR = ROOT / "output"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Get image from command line
    # -----------------------------------------------------

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python hmr2_image.py <image_path>")
        print()
        print("Example:")
        print("  python hmr2_image.py input/astronaut.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1]).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"HMR2 checkpoint not found:\n{CHECKPOINT}"
        )

    print("=" * 60)
    print("BAS HUMAN 3D POSE PIPELINE")
    print("=" * 60)

    print(f"[1] Image      : {image_path}")
    print(f"[2] Device     : {DEVICE}")

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    img_cv2 = cv2.imread(str(image_path))

    if img_cv2 is None:
        raise RuntimeError(
            f"OpenCV could not read:\n{image_path}"
        )

    print(f"[3] Image shape: {img_cv2.shape}")

    # -----------------------------------------------------
    # YOLO
    # -----------------------------------------------------

    print("\n[4] Loading YOLO...")

    yolo = YOLO("yolo11n.pt")

    print("[5] Detecting people...")

    results = yolo(
        img_cv2,
        device="cpu",
        classes=[0],
        verbose=False
    )

    boxes = []
    confidences = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            if confidence < 0.5:
                continue

            xyxy = box.xyxy[0].cpu().numpy()

            boxes.append(xyxy)
            confidences.append(confidence)

            print(
                f"    Person | "
                f"confidence={confidence:.3f} | "
                f"box={xyxy.astype(int)}"
            )

    if not boxes:
        print("\nNo person detected.")
        return

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    confidences = np.asarray(
        confidences,
        dtype=np.float32
    )

    print(
        f"\n[6] Persons detected: {len(boxes)}"
    )

    # -----------------------------------------------------
    # Load HMR2
    # -----------------------------------------------------

    print("\n[7] Loading HMR2...")

    model, model_cfg = load_hmr2(
        str(CHECKPOINT)
    )

    model = model.to(DEVICE)
    model.eval()

    print("[8] HMR2 loaded.")

    # -----------------------------------------------------
    # HMR2 preprocessing
    # -----------------------------------------------------

    print("\n[9] Preparing person crops...")

    dataset = ViTDetDataset(
        model_cfg,
        img_cv2,
        boxes
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # HMR2 inference
    # -----------------------------------------------------

    print("\n[10] Running 3D pose estimation...")

    all_joints = []
    all_vertices = []
    all_camera = []

    with torch.no_grad():

        for person_id, batch in enumerate(dataloader):

            batch = recursive_to(
                batch,
                DEVICE
            )

            output = model(batch)

            joints = (
                output["pred_keypoints_3d"]
                .cpu()
                .numpy()
            )

            vertices = (
                output["pred_vertices"]
                .cpu()
                .numpy()
            )

            camera = (
                output["pred_cam_t"]
                .cpu()
                .numpy()
            )

            all_joints.append(joints)
            all_vertices.append(vertices)
            all_camera.append(camera)

            print(
                f"    Person {person_id}: "
                f"joints={joints.shape}, "
                f"vertices={vertices.shape}"
            )

    # -----------------------------------------------------
    # Combine HMR2 results
    # -----------------------------------------------------

    joints = np.concatenate(
        all_joints,
        axis=0
    )

    vertices = np.concatenate(
        all_vertices,
        axis=0
    )

    camera = np.concatenate(
        all_camera,
        axis=0
    )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save NPZ
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR / "hmr2_result.npz"
    )

    np.savez(
        output_file,
        boxes=boxes,
        confidences=confidences,
        joints_3d=joints,
        vertices=vertices,
        camera_translation=camera
    )

    # -----------------------------------------------------
    # Create JSON
    # -----------------------------------------------------

    json_data = {

        "system": {
            "name": "BAS Human 3D Pose Pipeline",
            "device": "cpu",
            "person_detector": "YOLO11n",
            "pose_estimator": "HMR2"
        },

        "image": {
            "filename": image_path.name,
            "width": int(img_cv2.shape[1]),
            "height": int(img_cv2.shape[0]),
            "channels": int(img_cv2.shape[2])
        },

        "persons": []
    }

    # -----------------------------------------------------
    # Convert each person's data to JSON
    # -----------------------------------------------------

    for person_id in range(len(boxes)):

        x1, y1, x2, y2 = boxes[person_id]

        person_data = {

            "person_id": person_id,

            "detection": {

                "confidence": float(
                    confidences[person_id]
                ),

                "bbox": {

                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            },

            "camera_translation": {

                "x": float(
                    camera[person_id][0]
                ),

                "y": float(
                    camera[person_id][1]
                ),

                "z": float(
                    camera[person_id][2]
                )
            },

            "pose": {

                "joint_count": int(
                    joints.shape[1]
                ),

                "joints_3d": []
            }
        }

        # -------------------------------------------------
        # Add all 44 joints
        # -------------------------------------------------

        for joint_id in range(
            joints.shape[1]
        ):

            joint = (
                joints[person_id][joint_id]
            )

            person_data["pose"]["joints_3d"].append({

                "joint_id": joint_id,

                "x": float(joint[0]),

                "y": float(joint[1]),

                "z": float(joint[2])
            })

        json_data["persons"].append(
            person_data
        )

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    json_output_file = (
        OUTPUT_DIR / "hmr2_result.json"
    )

    with open(
        json_output_file,
        "w"
    ) as f:

        json.dump(
            json_data,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(f"Persons       : {len(boxes)}")
    print(f"3D joints     : {joints.shape}")
    print(f"SMPL vertices : {vertices.shape}")
    print(f"Camera        : {camera.shape}")

    print("\nSaved NPZ:")
    print(output_file)

    print("\nSaved JSON:")
    print(json_output_file)


# ---------------------------------------------------------
# Run program
# ---------------------------------------------------------

if __name__ == "__main__":
    main()