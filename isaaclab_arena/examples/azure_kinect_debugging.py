# %%


# NOTE! THIS CODE SHOULD BE RUN IN THE NVBLOX DOCKER!
# I just copied it here for completeness.

import open3d as o3d
import torch

from nvblox_torch.visualization import get_open3d_coordinate_frame

print("hello world")


# %%


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion into a 3x3 rotation matrix.

    Args:
        q (torch.Tensor): Tensor of shape (4,) in (w, x, y, z) format.

    Returns:
        torch.Tensor: 3x3 rotation matrix.
    """
    assert q.shape == (4,) or q.shape == (4, 1)
    q = q.flatten()
    w, x, y, z = q

    # Normalize the quaternion to avoid scaling distortion
    norm = torch.sqrt(w * w + x * x + y * y + z * z)
    w = w / norm
    x = x / norm
    y = y / norm
    z = z / norm

    R = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=q.dtype,
        device=q.device,
    )

    return R


# %%

import numpy as np
import pathlib

data_root = pathlib.Path("/datasets/2025_11_12_azure")
pose_dir = data_root / "pose"

idx = 1

pose_path = pose_dir / f"{idx:05d}.txt"

pose = np.loadtxt(pose_path)

t_W_C = torch.tensor(pose[:3])
q_W_C = torch.tensor(pose[3:])
print(f"t_W_C: {t_W_C}")
print(f"q_W_C: {q_W_C}")

R_W_C = quaternion_to_rotation_matrix(q_W_C)
print(f"R_W_C: {R_W_C}")

T_W_C = torch.eye(4)
T_W_C[:3, :3] = R_W_C
T_W_C[:3, 3] = t_W_C
print(f"T_W_C: {T_W_C}")


axis_W = get_open3d_coordinate_frame(torch.eye(4))
axis_W_C = get_open3d_coordinate_frame(T_W_C)

o3d.visualization.draw_geometries([axis_W, axis_W_C])

# %%

import glob
import tqdm

T_W_C_vec = []
axis_W_C_vec = []
for pose_path in tqdm.tqdm(sorted(glob.glob(str(pose_dir / "*.txt")))):

    pose = np.loadtxt(pose_path)
    t_W_C = torch.tensor(pose[:3])
    q_W_C = torch.tensor(pose[3:])
    R_W_C = quaternion_to_rotation_matrix(q_W_C)
    T_W_C = torch.eye(4)
    T_W_C[:3, :3] = R_W_C
    T_W_C[:3, 3] = t_W_C
    axis_W_C = get_open3d_coordinate_frame(T_W_C)
    T_W_C_vec.append(T_W_C)
    axis_W_C_vec.append(axis_W_C)

o3d.visualization.draw_geometries(axis_W_C_vec[::50] + [axis_W])


# %%
