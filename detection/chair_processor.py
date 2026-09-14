from pathlib import Path
import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "yolo11n.pt"

CAMERA = "/dev/video0"

IMG_SIZE = 416
CONFIDENCE = 0.35

# COCO classes
PERSON_CLASS = 0
CHAIR_CLASS = 56


def main():
    print("=" * 60)
    print("BAS - PERSON + CHAIR DETECTION")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"Camera: {CAMERA}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA}"
        )

    # Stable webcam configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)

    # MJPEG
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    frame_id = 0

    person_count_total = 0
    chair_count_total = 0

    print("Camera started.")
    print("Press Ctrl+C to stop.")
    print()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame.")
                continue

            frame_id += 1

            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            persons = []
            chairs = []

            if result.boxes is not None:

                for box in result.boxes:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    if cls == PERSON_CLASS:

                        persons.append({
                            "bbox": [
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2),
                            ],
                            "confidence": conf,
                        })

                    elif cls == CHAIR_CLASS:

                        chairs.append({
                            "bbox": [
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2),
                            ],
                            "confidence": conf,
                        })

            person_count_total += len(persons)
            chair_count_total += len(chairs)

            # --------------------------------------------------
            # PRINT DETECTIONS
            # --------------------------------------------------

            print(
                f"\rFrame {frame_id:5d} | "
                f"Persons: {len(persons):2d} | "
                f"Chairs: {len(chairs):2d}",
                end="",
                flush=True,
            )

            # --------------------------------------------------
            # DRAW DETECTIONS
            # --------------------------------------------------

            annotated = frame.copy()

            # Persons
            for i, person in enumerate(persons):

                x1, y1, x2, y2 = person["bbox"]
                conf = person["confidence"]

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"PERSON {i} "
                    f"{conf:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            # Chairs
            for i, chair in enumerate(chairs):

                x1, y1, x2, y2 = chair["bbox"]
                conf = chair["confidence"]

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )

                label = (
                    f"CHAIR {i} "
                    f"{conf:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2,
                )

            # Information
            cv2.putText(
                annotated,
                f"Persons: {len(persons)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Chairs: {len(chairs)}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            # Save every 30th frame
            if frame_id % 30 == 0:

                output_dir = ROOT / "test_results" / "chair_detection"

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                output_path = (
                    output_dir /
                    f"frame_{frame_id:06d}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    annotated,
                )

    except KeyboardInterrupt:

        print("\n")
        print("Stopping camera...")

    finally:

        cap.release()

    print()
    print("=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    print(f"Frames processed: {frame_id}")
    print(f"Total person detections: {person_count_total}")
    print(f"Total chair detections : {chair_count_total}")

    if frame_id > 0:

        print(
            f"Average persons/frame: "
            f"{person_count_total / frame_id:.2f}"
        )

        print(
            f"Average chairs/frame : "
            f"{chair_count_total / frame_id:.2f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
