



#%%



import pinocchio  # noqa: F401


import gymnasium as gym

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser

# Launching the simulation app
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()


from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
# from isaac_arena.embodiments.franka import FrankaEmbodiment
from isaac_arena.embodiments.gr1t2 import GR1T2Embodiment
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.open_door_task import OpenDoorTask

args_parser = get_isaac_arena_cli_parser()
args_cli = args_parser.parse_args([])

asset_registry = AssetRegistry()
background = asset_registry.get_asset_by_name("packing_table_pick_and_place")()
microwave = asset_registry.get_asset_by_name("microwave")()

# Put the microwave on the packing table.
microwave.set_initial_pose(
    Pose(
        position_xyz=(0.6 + 0.1, -0.00586, 0.22773),
        rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
    )
)

scene = Scene(assets=[background, microwave])

isaac_arena_environment = IsaacArenaEnvironment(
    name="open_door",
    embodiment=GR1T2Embodiment(),
    scene=scene,
    task=OpenDoorTask(microwave),
)

env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
name, cfg = env_builder.build_registered()
env = gym.make(name, cfg=cfg).unwrapped
env.reset()


#%%

import tqdm
import torch

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)


#%%