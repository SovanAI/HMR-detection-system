import cv2
from ultralytics import YOLO

CAMERA = "/dev/video0"
MODEL = "yolo11n.pt"

print("=" * 70)
print("BAS-HMR LAYER 2 — YOLO11 CAMERA TEST")
print("=" * 70)

print("\n[1] Loading YOLO11n...")

model = YOLO(MODEL)

print("[OK] YOLO11n loaded.")
print("[OK] Device: CPU")

print("\n[2] Opening camera...")

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    raise SystemExit(1)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)

print("[OK] Camera opened.")

frame_count = 0

print("\n[3] Starting detection...")
print("Press CTRL+C to stop.\n")

try:

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("[ERROR] Camera frame failed.")
            continue

        frame_count += 1

        results = model.predict(
            source=frame,
            conf=0.15,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        persons = 0
        chairs = 0

        if result.boxes is not None:

            for i in range(len(result.boxes)):

                class_id = int(
                    result.boxes.cls[i].item()
                )

                confidence = float(
                    result.boxes.conf[i].item()
                )

                if class_id == 0:
                    persons += 1

                elif class_id == 56:
                    chairs += 1

        if frame_count % 10 == 0:

            print(
                f"[FRAME {frame_count}] "
                f"Persons={persons} "
                f"Chairs={chairs}"
            )

            if persons > 0:
                print("  [OK] Person detected.")

            if chairs > 0:
                print("  [OK] Chair detected.")

        annotated = result.plot()

        cv2.imwrite(
            "/tmp/bas_yolo_latest.jpg",
            annotated
        )

except KeyboardInterrupt:

    print("\n[STOP] CTRL+C received.")

finally:

    cap.release()

    print("[OK] Camera released.")

print("\n" + "=" * 70)
print("YOLO11 CAMERA TEST FINISHED")
print("=" * 70)
