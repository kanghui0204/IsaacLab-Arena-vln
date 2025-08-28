# %%


import gymnasium as gym
import torch
import tqdm

import pinocchio  # noqa: F401

# Launching the simulation app
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()


from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.embodiments.franka import FrankaEmbodiment
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.geometry.pose import Pose

# from isaac_arena.fixures.fixtures import Microwave
from isaac_arena.scene.open_door_scene import OpenDoorScene
from isaac_arena.tasks.open_door import OpenDoorTask

args_parser = get_isaac_arena_cli_parser()
args_cli = args_parser.parse_args([])


asset_registry = AssetRegistry()
background = asset_registry.get_asset_by_name("packing_table_pick_and_place")()
microwave = asset_registry.get_asset_by_name("microwave")()

# Put the microwave on the packing table.
microwave.set_initial_pose(
    Pose(
        position_xyz=(0.6, -0.00586, 0.22773),
        rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
    )
)

isaac_arena_environment = IsaacArenaEnvironment(
    name="open_door",
    embodiment=FrankaEmbodiment(),
    scene=OpenDoorScene(background, microwave),
    task=OpenDoorTask(),
)

env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
name, cfg = env_builder.build_registered()
env = gym.make(name, cfg=cfg)
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 10

from collections.abc import Callable

from isaaclab.envs.manager_based_env import ManagerBasedEnv


def step_zeros_and_call(env: ManagerBasedEnv, function: Callable[[ManagerBasedEnv], None], num_steps: int):
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
            function(env)


def assert_closed(env: ManagerBasedEnv):
    is_open = microwave.is_open(env, "interactable_object")
    print(f"assert_closed: {is_open.item()}")
    assert is_open.shape == torch.Size([1])
    assert not is_open.item()


def assert_open(env: ManagerBasedEnv):
    is_open = microwave.is_open(env, "interactable_object")
    print(f"assert_open: {is_open.item()}")
    assert is_open.shape == torch.Size([1])
    assert is_open.item()


microwave.close(env, "interactable_object")
step_zeros_and_call(env, assert_closed, NUM_STEPS)
microwave.open(env, "interactable_object")
step_zeros_and_call(env, assert_open, NUM_STEPS)
