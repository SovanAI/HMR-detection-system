"""
BAS Cardboard Box Visual Webcam Test - V2

Uses the newly trained cardboard YOLO model.

Pipeline:

    Webcam
       ↓
    YOLO11n V2
       ↓
    Cardboard Box Detection
       ↓
    Bounding Box + Confidence + Center
       ↓
    Saved Visual Results

This version is headless and WSL compatible.
It does NOT use cv2.imshow().
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# V2 MODEL
# ============================================================

MODEL = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "cardboard_yolo11n_v2"
    / "weights"
    / "best.pt"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "test_results"
    / "box_webcam_v2"
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
# YOLO SETTINGS
# ============================================================

CONFIDENCE = 0.25

IMAGE_SIZE = 416


# ============================================================
# TEST SETTINGS
# ============================================================

TEST_SECONDS = 30

SAVE_EVERY = 5


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BAS CARDBOARD BOX VISUAL DETECTION - V2")
    print("=" * 70)

    # ========================================================
    # MODEL CHECK
    # ========================================================

    print()
    print("Model:")
    print(MODEL)

    if not MODEL.exists():

        print()
        print("ERROR: V2 model not found.")
        print()

        print("Available best.pt files:")

        for path in sorted(
            ROOT.glob("runs/**/best.pt")
        ):
            print(path)

        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading YOLO V2 model...")

    model = YOLO(
        str(MODEL)
    )

    print("Model loaded successfully.")

    print()
    print("Classes:")
    print(model.names)

    # ========================================================
    # CAMERA
    # ========================================================

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

        return

    # ========================================================
    # CAMERA CONFIGURATION
    # ========================================================

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
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
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

    # ========================================================
    # STATISTICS
    # ========================================================

    frame_count = 0

    frames_with_box = 0

    total_boxes = 0

    confidence_sum = 0.0

    start_time = time.time()

    next_report = (
        start_time + 5.0
    )

    # ========================================================
    # START TEST
    # ========================================================

    print()
    print(
        f"Running V2 visual detection "
        f"for {TEST_SECONDS} seconds..."
    )

    print()
    print(
        "Keep the REAL cardboard box "
        "clearly visible."
    )

    print()
    print(
        "Annotated frames will be saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        elapsed = (
            time.time()
            - start_time
        )

        if elapsed >= TEST_SECONDS:
            break

        # ----------------------------------------------------
        # Read frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "WARNING: Failed to read frame."
            )

            continue

        frame_count += 1

        # ----------------------------------------------------
        # YOLO inference
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

        if boxes is None:

            box_count = 0

        else:

            box_count = len(boxes)

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if box_count > 0:

            frames_with_box += 1

            total_boxes += box_count

        # ----------------------------------------------------
        # Make annotated frame
        # ----------------------------------------------------

        annotated = frame.copy()

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        if boxes is not None:

            for index, box in enumerate(boxes):

                # Bounding box
                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy,
                )

                # Confidence
                confidence = float(
                    box.conf[0]
                    .cpu()
                    .item()
                )

                confidence_sum += (
                    confidence
                )

                # Center
                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2

                # ------------------------------------------------
                # Draw bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                )

                # ------------------------------------------------
                # Draw center
                # ------------------------------------------------

                cv2.circle(
                    annotated,
                    (
                        center_x,
                        center_y,
                    ),
                    6,
                    (0, 0, 255),
                    -1,
                )

                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                label = (
                    f"CARDBOARD BOX "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 10,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # Center coordinates
                # ------------------------------------------------

                center_text = (
                    f"Center: "
                    f"{center_x},"
                    f"{center_y}"
                )

                cv2.putText(
                    annotated,
                    center_text,
                    (
                        x1,
                        min(
                            HEIGHT - 10,
                            y2 + 25,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                # ------------------------------------------------
                # Terminal output
                # ------------------------------------------------

                print(
                    f"Frame {frame_count}: "
                    f"BOX {index} | "
                    f"conf={confidence:.3f} | "
                    f"bbox=("
                    f"{x1},{y1},"
                    f"{x2},{y2}) | "
                    f"center=("
                    f"{center_x},"
                    f"{center_y})"
                )

        # ----------------------------------------------------
        # Status text
        # ----------------------------------------------------

        status = (
            f"CARDBOARD BOXES: "
            f"{box_count}"
        )

        cv2.putText(
            annotated,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Save annotated frame
        # ----------------------------------------------------

        if (
            frame_count
            % SAVE_EVERY
            == 0
        ):

            output_file = (
                OUTPUT_DIR
                / f"frame_"
                f"{frame_count:05d}.jpg"
            )

            cv2.imwrite(
                str(output_file),
                annotated,
            )

        # ----------------------------------------------------
        # Progress report
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

            print()

            print(
                f"[{elapsed:5.1f}s] "
                f"Frames={frame_count} | "
                f"BoxFrames="
                f"{frames_with_box} | "
                f"DetectionRate="
                f"{detection_rate:.1f}% | "
                f"FPS="
                f"{current_fps:.2f}"
            )

            print()

            next_report += 5.0

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

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

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("V2 VISUAL BOX DETECTION RESULT")
    print("=" * 70)

    print()

    print(
        f"Frames processed:     "
        f"{frame_count}"
    )

    print(
        f"Frames with box:      "
        f"{frames_with_box}"
    )

    print(
        f"Total detections:     "
        f"{total_boxes}"
    )

    print(
        f"Detection rate:       "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Average confidence:   "
        f"{average_confidence:.3f}"
    )

    print(
        f"Processing FPS:       "
        f"{processing_fps:.2f}"
    )

    print()

    print(
        "Annotated images:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
