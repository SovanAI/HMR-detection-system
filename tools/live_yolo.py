import cv2
import json
import time
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
CAMERA = "/dev/video0"


def main():

    print("[SYSTEM] Starting live YOLO...")

    # -------------------------
    # Load YOLO
    # -------------------------
    model = YOLO(MODEL_PATH)

    print("[YOLO] Model loaded")
    print("[YOLO] Device: CPU")

    # -------------------------
    # Open camera
    # -------------------------
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("[CAMERA] Cannot open camera")

    # Force MJPEG
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[CAMERA] Camera opened")

    frame_id = 0

    try:

        while True:

            # -------------------------
            # Capture frame
            # -------------------------
            ret, frame = cap.read()

            if not ret:
                print("[CAMERA] Failed to read frame")
                continue

            timestamp = time.time()

            # -------------------------
            # YOLO inference
            # -------------------------
            results = model(
                frame,
                device="cpu",
                classes=[0],
                verbose=False
            )

            persons = []

            for result in results:

                if result.boxes is None:
                    continue

                for i, box in enumerate(result.boxes):

                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    persons.append({
                        "person_id": i,
                        "confidence": confidence,
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }
                    })

            # -------------------------
            # Create YOLO JSON
            # -------------------------
            output = {

                "frame_id": frame_id,

                "timestamp": timestamp,

                "model": "YOLO11n",

                "device": "cpu",

                "image": {
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                },

                "persons": persons
            }

            # -------------------------
            # Print result
            # -------------------------
            print(
                f"[FRAME {frame_id}] "
                f"Persons detected: {len(persons)}"
            )

            print(json.dumps(output, indent=2))

            frame_id += 1

    except KeyboardInterrupt:

        print("\n[SYSTEM] Stopping...")

    finally:

        cap.release()


if __name__ == "__main__":
    main()
