# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

print("Launching simulation app once in notebook")
args_cli = get_isaaclab_arena_cli_parser().parse_args(["--enable_cameras"])
simulation_app = AppLauncher(args_cli)

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.dummy_task import DummyTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("gr1_pink")(enable_cameras=True)
cracker_box = asset_registry.get_asset_by_name("cracker_box")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

scene = Scene(assets=[background, cracker_box])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=DummyTask(),
    teleop_device=None,
)

env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

from isaaclab_arena.examples.azure_kinect import set_azure_camera_properties

camera_prim_path = "/World/envs/env_0/Robot/head_yaw_link/RobotPOVCam"
set_azure_camera_properties(camera_prim_path)


# %%


# Run some zero actions.
NUM_STEPS = 300
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)


# %%

camera = env.scene["robot_pov_cam"]
print(f"pos_w:{camera.data.pos_w}")
print(f"quat_w_world:{camera.data.quat_w_world}")


# %%

import imageio.v3 as iio

image_tensor = camera.data.output["rgba"]

# write to png
image_path = "/datasets/azure.png"
iio.imwrite(image_path, image_tensor.cpu().numpy())

import numpy as np

# %%
import imageio.v3 as iio

depth_tensor = camera.data.output["distance_to_image_plane"]
depth_tensor = depth_tensor.cpu().squeeze().numpy()
depth_tensor = (depth_tensor * 1000).astype(np.uint16)
image_path = "/datasets/azure_depth.png"
iio.imwrite(image_path, depth_tensor)


# %%

from matplotlib import pyplot as plt

rgb_tensor = camera.data.output["rgb"]
semantic_image = camera.data.output["semantic_segmentation"]

plt.subplot(1, 2, 1)
plt.imshow(rgb_tensor.squeeze().cpu().numpy())
plt.subplot(1, 2, 2)
plt.imshow(semantic_image.squeeze().cpu().numpy())
plt.show()


# %%
