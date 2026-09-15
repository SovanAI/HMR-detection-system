import cv2
from ultralytics import YOLO

MODEL = "yolo11n.pt"

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 10

CONFIDENCE = 0.25
IMAGE_SIZE = 640

PERSON_CLASS = 0
CHAIR_CLASS = 56


model = YOLO(MODEL)

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2,
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH,
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT,
)

camera.set(
    cv2.CAP_PROP_FPS,
    FPS,
)

camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG"),
)

print("=" * 60)
print("PERSON + CHAIR DETECTION TEST")
print("=" * 60)

print("Camera opened:", camera.isOpened())
print()


try:

    while True:

        ok, frame = camera.read()

        if not ok:
            print("Camera frame failed.")
            continue

        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            classes=[
                PERSON_CLASS,
                CHAIR_CLASS,
            ],
            device="cpu",
            verbose=False,
        )

        persons = 0
        chairs = 0

        if results:

            result = results[0]

            if result.boxes is not None:

                for box in result.boxes:

                    cls = int(
                        box.cls[0].item()
                    )

                    conf = float(
                        box.conf[0].item()
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    if cls == PERSON_CLASS:

                        persons += 1

                        label = (
                            f"PERSON "
                            f"{conf:.2f}"
                        )

                    elif cls == CHAIR_CLASS:

                        chairs += 1

                        label = (
                            f"CHAIR "
                            f"{conf:.2f}"
                        )

                    else:
                        continue

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

        print(
            f"\rPersons: {persons} | "
            f"Chairs: {chairs}",
            end="",
            flush=True,
        )

        cv2.imshow(
            "Person + Chair Detection",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("Detection test stopped.")
