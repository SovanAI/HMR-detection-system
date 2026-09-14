import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.frame_manager import FrameManager
from yolo.yolo_processor import YOLOProcessor


def main():

    print("====================================")
    print(" LIVE CAMERA + YOLO TEST")
    print("====================================")

    # -----------------------------
    # Initialize camera
    # -----------------------------

    camera = FrameManager(camera_index=0)

    # -----------------------------
    # Initialize YOLO
    # -----------------------------

    yolo = YOLOProcessor(
        model_path="yolo11n.pt",
        device="cpu"
    )

    print()
    print("[SYSTEM] Camera + YOLO initialized")
    print("[SYSTEM] Press Ctrl+C to stop")
    print()

    try:

        while True:

            # Get frame
            result = camera.read_frame()

            if result is None:

                print("[CAMERA] Failed to read frame")
                break

            frame_id, timestamp, frame = result

            # -----------------------------
            # Run YOLO
            # -----------------------------

            yolo_result = yolo.process_frame(
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp
            )

            # -----------------------------
            # Print result
            # -----------------------------

            persons = yolo_result["persons"]

            print(
                f"[YOLO] Frame {frame_id:06d} | "
                f"Persons detected: {len(persons)}"
            )

            for person in persons:

                print(
                    f"        Person {person['person_id']} | "
                    f"confidence={person['confidence']:.3f} | "
                    f"bbox={person['bbox']}"
                )

    except KeyboardInterrupt:

        print()
        print("[SYSTEM] Test interrupted.")

    finally:

        camera.release()

        print("[SYSTEM] Camera released.")
        print("[SYSTEM] Test completed.")


if __name__ == "__main__":
    main()
