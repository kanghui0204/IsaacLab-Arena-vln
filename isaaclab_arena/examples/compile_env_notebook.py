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

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.dummy_task import DummyTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

# %%

from lightwheel_sdk.loader import object_loader

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType


def get_lightwheel_object(name: str, scale: float, file_name: str | None = None):
    if file_name is None:
        file_path, object_name, metadata = object_loader.acquire_by_registry(
            registry_type="objects", registry_name=[name], file_type="USD"
        )
        print(f"Loaded {object_name} from registry")
    else:
        file_path, _, _ = object_loader.acquire_by_registry(
            registry_type="objects", file_name=file_name, file_type="USD"
        )
    scale_tuple = (scale, scale, scale)
    return Object(name=name, usd_path=file_path, object_type=ObjectType.RIGID, scale=scale_tuple)


idx = 3
object_names = ["banana", "apple", "broccoli", "carrot"]
file_names = ["Banana034", "Apple025", "broccoli_14", "Carrot016"]
object_scales = [1.0, 1.0, 3.0, 1.5]
object_name = object_names[idx]
object_scale = object_scales[idx]
file_name = file_names[idx]
manipulated_object = get_lightwheel_object(object_name, object_scale)
manipulated_object = get_lightwheel_object(object_name, object_scale, file_name=file_name)
# bowl = get_lightwheel_object("bowl", 1.0)
bowl = get_lightwheel_object("bowl", 1.2, file_name="Bowl033")

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
# embodiment = asset_registry.get_asset_by_name("gr1_pink")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
# banana = asset_registry.get_asset_by_name("banana")()
# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
manipulated_object.set_initial_pose(Pose(position_xyz=(0.4, 0.05, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
bowl.set_initial_pose(Pose(position_xyz=(0.3, -0.35, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

scene = Scene(assets=[background, manipulated_object, bowl])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=DummyTask(),
    teleop_device=None,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%

from isaaclab.sim import SimulationContext

simulation_context = SimulationContext.instance()
simulation_context._disable_app_control_on_stop_handle = True
simulation_context.stop()
simulation_context.clear_instance()
env.close()
import omni.timeline

omni.timeline.get_timeline_interface().stop()

# %%

# Only required when using Lightwheel SDK
from lightwheel_sdk.loader import object_loader

object_names = []
for object_dict in object_loader.list_registry():
    object_names.append(object_dict["name"])

object_names.sort()
print(object_names)

# %%

file_path, object_name, metadata = object_loader.acquire_by_registry(
    registry_type="objects", registry_name=["bowl"], file_type="USD"
)

# %%

print(object_name)


# %%


# %%

apple
tomato
banana
carrot
pineapple
broccoli


# %%
