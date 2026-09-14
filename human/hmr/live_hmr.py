import cv2
import time
import sys
import os

# Allow importing from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from hmr.hmr_processor import HMRProcessor


CAMERA = "/dev/video0"

# Process HMR every Nth frame.
# Start with 10 because your system is CPU-only.
HMR_INTERVAL = 10


def main():

    print("=" * 60)
    print("LIVE HMR2")
    print("=" * 60)

    # --------------------------------------------------
    # Load HMR2 once
    # --------------------------------------------------

    print("[HMR] Loading HMR2...")

    processor = HMRProcessor()

    print("[HMR] HMR2 loaded successfully")
    print("[HMR] Device: CPU")

    # --------------------------------------------------
    # Open camera
    # --------------------------------------------------

    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(
            f"[CAMERA] Could not open {CAMERA}"
        )

    # Force MJPEG
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        f"[CAMERA] {width}x{height} @ {fps:.1f} FPS"
    )

    print("=" * 60)
    print("[SYSTEM] Starting live HMR")
    print("[SYSTEM] Press Ctrl+C to stop")
    print("=" * 60)

    frame_id = 0
    hmr_count = 0

    try:

        while True:

            # --------------------------------------------------
            # Capture frame
            # --------------------------------------------------

            ret, frame = cap.read()

            if not ret:
                print("[CAMERA] Failed to read frame")
                continue

            timestamp = time.time()

            # --------------------------------------------------
            # Only run HMR every Nth frame
            # --------------------------------------------------

            if frame_id % HMR_INTERVAL != 0:

                frame_id += 1
                continue

            print(
                f"\n[HMR] Processing frame {frame_id}..."
            )

            start_time = time.time()

            # --------------------------------------------------
            # Run YOLO first to obtain person bounding boxes
            #
            # For this first test, we use YOLO directly here.
            # Later the pipeline will provide these boxes.
            # --------------------------------------------------

            from ultralytics import YOLO

            # IMPORTANT:
            # This should eventually be loaded only once.
            # For this simple test we load it lazily below.
            if not hasattr(main, "yolo_model"):

                print("[YOLO] Loading YOLO11n...")

                main.yolo_model = YOLO(
                    os.path.join(
                        PROJECT_ROOT,
                        "yolo11n.pt"
                    )
                )

                print("[YOLO] YOLO11n loaded")

            yolo_results = main.yolo_model(
                frame,
                device="cpu",
                classes=[0],
                verbose=False
            )

            # --------------------------------------------------
            # Extract person bounding boxes
            # --------------------------------------------------

            boxes = []

            for result in yolo_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    # Ignore very weak detections
                    if confidence < 0.30:
                        continue

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    boxes.append([
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ])

            print(
                f"[YOLO] Persons detected: {len(boxes)}"
            )

            # --------------------------------------------------
            # No people -> don't run HMR
            # --------------------------------------------------

            if len(boxes) == 0:

                print(
                    f"[HMR] No person detected "
                    f"in frame {frame_id}"
                )

                frame_id += 1
                continue

            # --------------------------------------------------
            # Run HMR2
            # --------------------------------------------------

            try:

                result = processor.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    boxes=boxes
                )

                elapsed = time.time() - start_time

                print(
                    f"[HMR] Frame {frame_id} completed "
                    f"in {elapsed:.2f}s"
                )

                print(
                    f"[HMR] Persons processed: "
                    f"{len(result['persons'])}"
                )

                # --------------------------------------------------
                # Print 3D position + joint information
                # --------------------------------------------------

                for person in result["persons"]:

                    person_id = person["person_id"]

                    camera_translation = (
                        person["camera_translation"]
                    )

                    joint_count = (
                        person["pose"]["joint_count"]
                    )

                    print(
                        f"  Person {person_id}: "
                        f"{joint_count} joints"
                    )

                    print(
                        f"  Camera translation: "
                        f"{camera_translation}"
                    )

            except Exception as e:

                print(
                    f"[HMR ERROR] Frame "
                    f"{frame_id}: {e}"
                )

            hmr_count += 1
            frame_id += 1

    except KeyboardInterrupt:

        print("\n")
        print("=" * 60)
        print("[SYSTEM] Stopping live HMR...")
        print(f"[SYSTEM] HMR frames processed: {hmr_count}")
        print("=" * 60)

    finally:

        cap.release()


if __name__ == "__main__":
    main()
