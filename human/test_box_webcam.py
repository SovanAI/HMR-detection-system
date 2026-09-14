"""
BAS Cardboard Box Webcam Test
Headless WSL version.

Pipeline:

    Camera
       ↓
    YOLO Box Detector
       ↓
    Bounding Boxes
       ↓
    Saved Annotated Images

No cv2.imshow() is used because this runs inside WSL.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL = (
    ROOT
    / "runs"
    / "cardboard_yolo11n"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "test_results"
    / "box_webcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CAMERA
# ============================================================

CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10


# ============================================================
# YOLO
# ============================================================

CONFIDENCE = 0.20
IMAGE_SIZE = 416


# ============================================================
# TEST
# ============================================================

TEST_SECONDS = 30

# Save one annotated frame every N frames
SAVE_EVERY = 10


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BAS CARDBOARD BOX WEBCAM TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    print()
    print("Model:")
    print(MODEL)

    if not MODEL.exists():

        print()
        print("ERROR: Model not found.")
        print()

        print("Available best.pt files:")

        for path in sorted(
            ROOT.glob("runs/**/best.pt")
        ):
            print(path)

        return

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Loading YOLO model...")

    model = YOLO(str(MODEL))

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    print()
    print("Opening camera:")
    print(CAMERA)

    cap = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():

        print()
        print("ERROR: Could not open camera.")
        print()

        return

    # --------------------------------------------------------
    # CAMERA CONFIGURATION
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

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print()
    print("Camera configuration:")
    print(
        f"  Resolution: "
        f"{actual_width}x{actual_height}"
    )

    print(
        f"  FPS: "
        f"{actual_fps:.2f}"
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    frame_count = 0
    frames_with_box = 0
    total_boxes = 0

    confidence_sum = 0.0

    start_time = time.time()

    next_report = start_time + 5.0

    # --------------------------------------------------------
    # TEST START
    # --------------------------------------------------------

    print()
    print(
        f"Running test for "
        f"{TEST_SECONDS} seconds..."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Keep the cardboard box clearly "
        "visible in front of the camera."
    )

    print()
    print(
        "Annotated frames will be saved to:"
    )

    print(OUTPUT_DIR)

    print()

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    while True:

        elapsed = (
            time.time()
            - start_time
        )

        if elapsed >= TEST_SECONDS:
            break

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "WARNING: Failed to read frame."
            )

            continue

        frame_count += 1

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        boxes = result.boxes

        box_count = 0

        if boxes is not None:

            box_count = len(boxes)

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        if box_count > 0:

            frames_with_box += 1

            total_boxes += box_count

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        annotated = frame.copy()

        if boxes is not None:

            for i, box in enumerate(boxes):

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy,
                )

                confidence = float(
                    box.conf[0]
                    .cpu()
                    .item()
                )

                confidence_sum += confidence

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # Center
                # ------------------------------------------------

                cv2.circle(
                    annotated,
                    (
                        center_x,
                        center_y,
                    ),
                    5,
                    (0, 0, 255),
                    -1,
                )

                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                label = (
                    f"BOX {i} "
                    f"conf={confidence:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        x1,
                        max(20, y1 - 8),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # Center coordinates
                # ------------------------------------------------

                center_text = (
                    f"Center "
                    f"({center_x},{center_y})"
                )

                cv2.putText(
                    annotated,
                    center_text,
                    (
                        x1,
                        min(
                            HEIGHT - 10,
                            y2 + 20,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status = (
            f"Boxes: {box_count} | "
            f"Frame: {frame_count}"
        )

        cv2.putText(
            annotated,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        if frame_count % SAVE_EVERY == 0:

            output_file = (
                OUTPUT_DIR
                / f"frame_{frame_count:05d}.jpg"
            )

            cv2.imwrite(
                str(output_file),
                annotated,
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        now = time.time()

        if now >= next_report:

            current_fps = (
                frame_count
                / max(
                    elapsed,
                    0.001,
                )
            )

            detection_rate = (
                frames_with_box
                / frame_count
                * 100
                if frame_count > 0
                else 0
            )

            print(
                f"[{elapsed:5.1f}s] "
                f"Frames={frame_count} | "
                f"BoxFrames={frames_with_box} | "
                f"DetectionRate="
                f"{detection_rate:.1f}% | "
                f"FPS={current_fps:.2f}"
            )

            next_report += 5.0

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()

    # IMPORTANT:
    # No cv2.destroyAllWindows()
    # because this is a headless WSL test.

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    processing_fps = (
        frame_count
        / max(
            elapsed,
            0.001,
        )
    )

    detection_rate = (
        frames_with_box
        / frame_count
        * 100
        if frame_count > 0
        else 0
    )

    average_confidence = (
        confidence_sum
        / total_boxes
        if total_boxes > 0
        else 0.0
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BOX DETECTION TEST RESULT")
    print("=" * 70)

    print()

    print(
        f"Frames processed:       "
        f"{frame_count}"
    )

    print(
        f"Frames with box:        "
        f"{frames_with_box}"
    )

    print(
        f"Total box detections:   "
        f"{total_boxes}"
    )

    print(
        f"Detection rate:         "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Average confidence:     "
        f"{average_confidence:.3f}"
    )

    print(
        f"Processing FPS:         "
        f"{processing_fps:.2f}"
    )

    print()

    print(
        "Saved annotated frames:"
    )

    print(OUTPUT_DIR)

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()