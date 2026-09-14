import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
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

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# HMR Processor
# ---------------------------------------------------------

class HMRProcessor:

    def __init__(self):

        print("[HMR] Loading HMR2 model...")

        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"HMR2 checkpoint not found:\n{CHECKPOINT}"
            )

        # Load HMR2 ONLY ONCE
        self.model, self.model_cfg = load_hmr2(
            str(CHECKPOINT)
        )

        self.model = self.model.to(DEVICE)

        self.model.eval()

        print("[HMR] HMR2 loaded successfully.")
        print(f"[HMR] Device: {DEVICE}")

    # -----------------------------------------------------
    # Process one camera frame
    # -----------------------------------------------------

    def process_frame(
        self,
        frame,
        frame_id,
        timestamp,
        boxes
    ):

        if frame is None:
            raise ValueError(
                "[HMR] Received empty frame."
            )

        if boxes is None or len(boxes) == 0:

            print(
                f"[HMR] Frame {frame_id}: "
                "No persons detected."
            )

            return {
                "frame_id": int(frame_id),

                "timestamp": float(timestamp),

                "model": "HMR2",

                "persons": []
            }

        boxes = np.asarray(
            boxes,
            dtype=np.float32
        )

        print(
            f"[HMR] Processing frame "
            f"{frame_id} | "
            f"persons={len(boxes)}"
        )

        # -------------------------------------------------
        # Prepare person crops
        # -------------------------------------------------

        dataset = ViTDetDataset(
            self.model_cfg,
            frame,
            boxes
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        # -------------------------------------------------
        # HMR inference
        # -------------------------------------------------

        all_joints = []
        all_camera = []

        with torch.no_grad():

            for person_id, batch in enumerate(
                dataloader
            ):

                batch = recursive_to(
                    batch,
                    DEVICE
                )

                output = self.model(batch)

                joints = (
                    output[
                        "pred_keypoints_3d"
                    ]
                    .cpu()
                    .numpy()
                )

                camera = (
                    output[
                        "pred_cam_t"
                    ]
                    .cpu()
                    .numpy()
                )

                all_joints.append(joints)
                all_camera.append(camera)

        # -------------------------------------------------
        # Combine results
        # -------------------------------------------------

        joints = np.concatenate(
            all_joints,
            axis=0
        )

        camera = np.concatenate(
            all_camera,
            axis=0
        )

        # -------------------------------------------------
        # Build JSON-compatible result
        # -------------------------------------------------

        result = {

            "frame_id": int(frame_id),

            "timestamp": float(timestamp),

            "model": "HMR2",

            "device": "cpu",

            "persons": []
        }

        # -------------------------------------------------
        # Create person data
        # -------------------------------------------------

        for person_id in range(
            len(boxes)
        ):

            person_data = {

                "person_id": int(
                    person_id
                ),

                "bbox": {

                    "x1": float(
                        boxes[person_id][0]
                    ),

                    "y1": float(
                        boxes[person_id][1]
                    ),

                    "x2": float(
                        boxes[person_id][2]
                    ),

                    "y2": float(
                        boxes[person_id][3]
                    )
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

            # ---------------------------------------------
            # Add all 44 joints
            # ---------------------------------------------

            for joint_id in range(
                joints.shape[1]
            ):

                joint = (
                    joints[
                        person_id
                    ][joint_id]
                )

                person_data[
                    "pose"
                ][
                    "joints_3d"
                ].append({

                    "joint_id": int(
                        joint_id
                    ),

                    "x": float(
                        joint[0]
                    ),

                    "y": float(
                        joint[1]
                    ),

                    "z": float(
                        joint[2]
                    )
                })

            result[
                "persons"
            ].append(
                person_data
            )

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return result


# ---------------------------------------------------------
# Test HMR Processor
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("HMR PROCESSOR TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Test image
    # -----------------------------------------------------

    image_path = (
        ROOT
        / "input"
        / "astronaut_image_2.jpg"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Test image not found:\n{image_path}"
        )

    frame = cv2.imread(
        str(image_path)
    )

    # -----------------------------------------------------
    # TEST BOX
    #
    # For now we use the YOLO boxes from your
    # previous test.
    #
    # Later these will come directly from YOLO.
    # -----------------------------------------------------

    boxes = np.array(
        [
            [170, 127, 347, 359],
            [292, 77, 435, 335],
            [81, 54, 359, 343]
        ],
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Create HMR processor
    # -----------------------------------------------------

    processor = HMRProcessor()

    # -----------------------------------------------------
    # Process frame
    # -----------------------------------------------------

    result = processor.process_frame(
        frame=frame,

        frame_id=0,

        timestamp=0.0,

        boxes=boxes
    )

    # -----------------------------------------------------
    # Save test JSON
    # -----------------------------------------------------

    output_dir = (
        ROOT
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "hmr_test_frame_000000.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("=" * 60)
    print("HMR PROCESSOR TEST COMPLETE")
    print("=" * 60)

    print(
        f"Frame ID : "
        f"{result['frame_id']}"
    )

    print(
        f"Persons  : "
        f"{len(result['persons'])}"
    )

    print(
        f"Joints   : "
        f"{result['persons'][0]['pose']['joint_count']}"
    )

    print()
    print("JSON saved:")
    print(output_file)
