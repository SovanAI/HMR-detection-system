import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


class YOLOProcessor:

    def __init__(self, model_path="yolo11n.pt", device="cpu"):
        """
        Initialize YOLO once.

        The model is loaded only once and reused for every frame.
        """

        self.device = device

        print("[YOLO] Loading model...")
        self.model = YOLO(model_path)

        print(f"[YOLO] Model loaded: {model_path}")
        print(f"[YOLO] Device: {self.device}")

    def process_frame(self, frame, frame_id, timestamp):
        """
        Process one camera frame.

        Parameters
        ----------
        frame : numpy.ndarray
            OpenCV BGR image.

        frame_id : int
            Unique ID assigned by the camera/frame manager.

        timestamp : float
            Timestamp assigned to the frame.

        Returns
        -------
        dict
            YOLO detection result in JSON-compatible format.
        """

        if frame is None:
            raise ValueError("[YOLO] Frame is None")

        # Run YOLO
        results = self.model(
            frame,
            device=self.device,
            verbose=False
        )

        persons = []

        # YOLO returns a list of results.
        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                # Class ID
                class_id = int(boxes.cls[i].item())

                # COCO class 0 = person
                if class_id != 0:
                    continue

                # Confidence
                confidence = float(boxes.conf[i].item())

                # Bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                person = {
                    "person_id": len(persons),

                    "confidence": round(confidence, 4),

                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                }

                persons.append(person)

        # Final JSON-compatible result
        output = {
            "frame_id": int(frame_id),

            "timestamp": float(timestamp),

            "model": "YOLO11n",

            "device": self.device,

            "image": {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0])
            },

            "persons": persons
        }

        return output


def save_json(data, output_path):
    """
    Save detection result as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print()
    print("====================================")
    print(" YOLO PROCESSOR TEST")
    print("====================================")

    # Project root
    ROOT = Path(__file__).resolve().parent.parent

    # Test image
    image_path = ROOT / "input" / "astronaut_image_2.jpg"

    print(f"[TEST] Image: {image_path}")

    # Load image
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    print(
        f"[TEST] Image shape: {frame.shape}"
    )

    # Create YOLO processor
    yolo = YOLOProcessor(
        model_path="yolo11n.pt",
        device="cpu"
    )

    # Example frame information
    frame_id = 0
    timestamp = 0.0

    # Process image
    result = yolo.process_frame(
        frame=frame,
        frame_id=frame_id,
        timestamp=timestamp
    )

    # Print result
    print()
    print("====================================")
    print(" YOLO RESULT")
    print("====================================")

    print(
        f"Frame ID: {result['frame_id']}"
    )

    print(
        f"Persons detected: {len(result['persons'])}"
    )

    for person in result["persons"]:

        print(
            f"Person {person['person_id']}: "
            f"confidence={person['confidence']} "
            f"bbox={person['bbox']}"
        )

    # Save JSON
    output_path = (
        ROOT /
        "output" /
        "yolo_test_frame_000000.json"
    )

    save_json(
        result,
        output_path
    )

    print()
    print(f"[TEST] JSON saved to:")
    print(output_path)

    print()
    print("YOLO processor test completed.")
