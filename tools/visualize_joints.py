import numpy as np
import matplotlib.pyplot as plt

RESULT_FILE = "output/hmr2_result.npz"

# Load HMR2 result
data = np.load(RESULT_FILE)

joints = data["joints_3d"][0]

print("3D joints shape:", joints.shape)

# Create 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot joints
ax.scatter(
    joints[:, 0],
    joints[:, 1],
    joints[:, 2],
    s=30
)

# Label joints
for i, (x, y, z) in enumerate(joints):
    ax.text(x, y, z, str(i), fontsize=8)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("HMR2 3D Human Joints")

# Equal-ish aspect ratio
x_range = joints[:, 0].max() - joints[:, 0].min()
y_range = joints[:, 1].max() - joints[:, 1].min()
z_range = joints[:, 2].max() - joints[:, 2].min()

max_range = max(x_range, y_range, z_range)

x_mid = (joints[:, 0].max() + joints[:, 0].min()) / 2
y_mid = (joints[:, 1].max() + joints[:, 1].min()) / 2
z_mid = (joints[:, 2].max() + joints[:, 2].min()) / 2

ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

plt.tight_layout()
plt.show()
