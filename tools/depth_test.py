from pathlib import Path

import cv2
import torch
import numpy as np

from transformers import pipeline


ROOT = Path(__file__).resolve().parent

DEVICE = -1  # CPU


print("=" * 60)
print("BAS-HMR DEPTH ESTIMATION TEST")
print("=" * 60)

print()
print("Loading depth model...")
print("Device: CPU")
print()


depth_pipeline = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=DEVICE,
)


print("Depth model loaded.")
print()


CAMERA = "/dev/video0"

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_FPS,
    10
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        *"MJPG"
    )
)


# Warm-up
for _ in range(5):
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError(
            "Could not read webcam frame."
        )


print("Camera ready.")
print()
print("Capturing one frame...")
print()


ret, frame = cap.read()

if not ret:
    cap.release()

    raise RuntimeError(
        "Could not capture frame."
    )


# OpenCV BGR → RGB → PIL Image
rgb = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
)
from PIL import Image

pil_image = Image.fromarray(rgb)

print("Running depth estimation...")
print()


result = depth_pipeline(pil_image)


depth = result["depth"]

depth_array = np.array(depth)


print("Depth estimation completed.")
print()


print(
    "Depth image shape:",
    depth_array.shape
)

print(
    "Depth minimum:",
    depth_array.min()
)

print(
    "Depth maximum:",
    depth_array.max()
)

print(
    "Depth mean:",
    depth_array.mean()
)


# --------------------------------------------------
# Normalize depth for visualization
# --------------------------------------------------

depth_visual = cv2.normalize(
    depth_array,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

depth_visual = depth_visual.astype(
    np.uint8
)


# Resize to camera resolution
depth_visual = cv2.resize(
    depth_visual,
    (
        frame.shape[1],
        frame.shape[0]
    )
)


# Apply OpenCV colormap
depth_color = cv2.applyColorMap(
    depth_visual,
    cv2.COLORMAP_INFERNO
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

output_dir = (
    ROOT /
    "test_results" /
    "depth_test"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


rgb_output = (
    output_dir /
    "camera.jpg"
)

depth_output = (
    output_dir /
    "depth.jpg"
)


cv2.imwrite(
    str(rgb_output),
    frame
)

cv2.imwrite(
    str(depth_output),
    depth_color
)


print("=" * 60)
print("RESULTS")
print("=" * 60)

print()
print(
    "RGB image:",
    rgb_output
)

print(
    "Depth image:",
    depth_output
)

print()
print("Test complete.")

cap.release()
