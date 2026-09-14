import cv2
import numpy as np
import os
import glob
import time


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_DEVICE = "/dev/video0"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Checkerboard:
# Number of INNER corners
CHECKERBOARD = (9, 6)

# Physical size of one square.
# Measure your printed checkerboard.
# Example: 25 mm = 0.025 m
SQUARE_SIZE = 0.025

CALIBRATION_DIR = "calibration/images"
OUTPUT_FILE = "calibration/camera_intrinsics.npz"

os.makedirs(CALIBRATION_DIR, exist_ok=True)


# ============================================================
# PREPARE OBJECT POINTS
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


object_points = []
image_points = []

image_size = None


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA_DEVICE}"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

print("=" * 70)
print("WEBCAM CAMERA CALIBRATION")
print("=" * 70)
print()
print("Checkerboard:")
print(f"  Inner corners: {CHECKERBOARD}")
print(f"  Square size:   {SQUARE_SIZE} meters")
print()
print("Controls:")
print("  SPACE = capture checkerboard")
print("  Q     = finish calibration")
print()
print("Collect around 15-25 GOOD views.")
print()


# ============================================================
# CAPTURE LOOP
# ============================================================

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    display = frame.copy()

    if found:

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        cv2.drawChessboardCorners(
            display,
            CHECKERBOARD,
            corners_refined,
            found
        )

        cv2.putText(
            display,
            "CHECKERBOARD DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            display,
            "CHECKERBOARD NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        display,
        f"Samples: {count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Camera Calibration",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        if found:

            object_points.append(
                objp.copy()
            )

            image_points.append(
                corners_refined.copy()
            )

            image_size = gray.shape[::-1]

            filename = os.path.join(
                CALIBRATION_DIR,
                f"calibration_{count:02d}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            count += 1

            print(
                f"[+] Sample {count} captured"
            )

        else:

            print(
                "[-] Checkerboard not detected. "
                "Move/rotate the board."
            )

    # Q
    elif key == ord("q"):

        break


cap.release()
cv2.destroyAllWindows()


# ============================================================
# CHECK SAMPLE COUNT
# ============================================================

if count < 10:

    raise RuntimeError(
        f"Only {count} samples collected. "
        "Collect at least 10, preferably 15-25."
    )


# ============================================================
# CAMERA CALIBRATION
# ============================================================

print()
print("=" * 70)
print("CALIBRATING CAMERA")
print("=" * 70)

ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)


# ============================================================
# REPROJECTION ERROR
# ============================================================

total_error = 0

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion
    )

    error = cv2.norm(
        image_points[i],
        projected_points,
        cv2.NORM_L2
    ) / len(projected_points)

    total_error += error


mean_error = total_error / len(object_points)


# ============================================================
# SAVE
# ============================================================

np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    checkerboard_width=CHECKERBOARD[0],
    checkerboard_height=CHECKERBOARD[1],
    square_size=SQUARE_SIZE,
    reprojection_error=mean_error
)


# ============================================================
# PRINT RESULTS
# ============================================================

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print()
print("=" * 70)
print("CAMERA CALIBRATION COMPLETE")
print("=" * 70)

print()
print("Camera Matrix:")
print(camera_matrix)

print()
print("Distortion:")
print(distortion.ravel())

print()
print("Intrinsic Parameters:")
print(f"fx = {fx:.6f}")
print(f"fy = {fy:.6f}")
print(f"cx = {cx:.6f}")
print(f"cy = {cy:.6f}")

print()
print(f"Samples: {count}")
print(f"Mean reprojection error: {mean_error:.6f}")

print()
print(f"Saved to:")
print(f"  {OUTPUT_FILE}")

print("=" * 70)
