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

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.scene.scene import Scene
# from isaac_arena.tasks.dummy_task import DummyTask
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaac_arena.utils.pose import Pose
from isaac_arena.assets.object_reference import ObjectReference

asset_registry = AssetRegistry()


background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

destination_location = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
)

scene = Scene(assets=[background, cracker_box])
isaac_arena_environment = IsaacArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=PickAndPlaceTask(cracker_box, destination_location, background),
    teleop_device=None,
)

# args_cli = get_isaac_arena_cli_parser().parse_args([])
# env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
# env = env_builder.make_registered()
# env.reset()


args_cli = get_isaac_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env_cfg = env_builder.compose_manager_cfg()


#%%

from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaac_arena.utils.pose import Pose
from isaac_arena.assets.asset import Asset


# Poses for envs 1 and 2.
pose_list = [
    Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    Pose(position_xyz=(-0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
]


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
):
    if env_ids is None:
        return

    # Grab the object
    asset = env.scene[asset_cfg.name]

    print(f"env_ids: {env_ids}")
    assert env_ids.ndim == 1
    assert env_ids.shape[0] == len(pose_list)
    for pose_list, cur_env in zip(pose_list, env_ids.tolist()):
        print(f"cur_env: {cur_env}")
        # Convert the pose to the env frame
        # pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device).repeat(num_envs, 1)
        print(f"Writing pose: {pose} to env: {cur_env}")
        pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device)
        print(f"pose_t_xyz_q_wxyz.shape: {pose_t_xyz_q_wxyz.shape}")
        pose_t_xyz_q_wxyz[:3] += env.scene.env_origins[env_ids].squeeze()
        print(f"env_origins.shape: {env.scene.env_origins[env_ids].shape}")
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)


initial_pose = cracker_box.get_initial_pose()
env_cfg.events.reset_pick_up_object_pose = EventTermCfg(
    func=set_object_pose,
    mode="reset",
    params={
        # "pose": initial_pose,
        "pose_list": pose_list,
        "asset_cfg": SceneEntityCfg(cracker_box.name),
    },
)

#%%

env_cfg.num_envs = 2
env = env_builder.make_registered(env_cfg)
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%
