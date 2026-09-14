from __future__ import annotations

import time
from pathlib import Path

import cv2


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "datasets"
    / "cardboard_webcam"
    / "images"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10


# ============================================================
# DATASET SETTINGS
# ============================================================

TARGET_IMAGES = 200

CAPTURE_INTERVAL = 1.0


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BAS REAL-WORLD CARDBOARD BOX DATASET CAPTURE")
    print("=" * 70)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print("The camera will capture one image every")
    print(f"{CAPTURE_INTERVAL} second.")

    print()
    print("IMPORTANT:")
    print("Move/change the cardboard box during capture.")
    print()
    print("Capture situations such as:")
    print("  1. Box close to camera")
    print("  2. Box far from camera")
    print("  3. Box on table")
    print("  4. Box on chair")
    print("  5. Box tilted")
    print("  6. Box from different angles")
    print("  7. Hand touching box")
    print("  8. Person standing near box")
    print("  9. Different lighting")
    print(" 10. WITHOUT box")
    print()
    print("The terminal will show the capture progress.")
    print()
    print("Press Ctrl+C to stop.")
    print()

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():

        print("ERROR: Could not open camera.")
        return

    # --------------------------------------------------------
    # Stable MJPEG configuration
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS,
    )

    time.sleep(1.0)

    # --------------------------------------------------------
    # Determine existing images
    # --------------------------------------------------------

    existing_images = sorted(
        OUTPUT_DIR.glob("*.jpg")
    )

    image_count = len(existing_images)

    print(
        f"Existing images: {image_count}"
    )

    if image_count >= TARGET_IMAGES:

        print()
        print(
            "Target number of images already exists."
        )

        cap.release()

        return

    print()
    print(
        f"Starting capture..."
    )

    print()

    # --------------------------------------------------------
    # Capture loop
    # --------------------------------------------------------

    last_capture = 0.0

    try:

        while image_count < TARGET_IMAGES:

            ret, frame = cap.read()

            if not ret:

                print(
                    "WARNING: Failed to read camera frame."
                )

                time.sleep(0.2)

                continue

            now = time.time()

            # ------------------------------------------------
            # Capture every second
            # ------------------------------------------------

            if (
                now - last_capture
                >= CAPTURE_INTERVAL
            ):

                image_count += 1

                filename = (
                    OUTPUT_DIR
                    / f"webcam_{image_count:04d}.jpg"
                )

                success = cv2.imwrite(
                    str(filename),
                    frame,
                )

                if success:

                    print(
                        f"[{image_count:03d}/"
                        f"{TARGET_IMAGES}] "
                        f"Saved {filename.name}"
                    )

                else:

                    print(
                        f"ERROR: Could not save "
                        f"{filename.name}"
                    )

                last_capture = now

            # Small delay to prevent unnecessary CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:

        print()
        print("Capture stopped by user.")

    finally:

        cap.release()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CAPTURE COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Images captured: {image_count}"
    )

    print()
    print("Dataset directory:")
    print(OUTPUT_DIR)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
