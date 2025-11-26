# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()


# NEW Background
from isaaclab_arena.assets.background import Background


class ComposedKitchenBackground(Background):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            name="kitchen",
            tags=["background", "pick_and_place"],
            usd_path="/workspaces/isaaclab_arena/saved_kitchen_with_cracker_box.usd",
            object_min_z=-0.2,
        )


# background = ComposedKitchenBackground()
# scene = Scene(assets=[background])
background_initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
background.set_initial_pose(background_initial_pose)

cracker_box_initial_pose = Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
cracker_box.set_initial_pose(cracker_box_initial_pose)


scene = Scene(assets=[background, cracker_box])

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

import pathlib

# Write the scene to a USD file.
output_root = pathlib.Path("/workspaces/isaaclab_arena")
output_path = output_root / "saved_kitchen_with_cracker_box.usd"
scene.export_to_usd(output_path)


# %%

import pathlib

import pxr

# Roload the stage and check that it was saved correctly.

output_root = pathlib.Path("/workspaces/isaaclab_arena")
output_path = output_root / "saved_kitchen_with_cracker_box.usd"
stage = pxr.Usd.Stage.Open(output_path.as_posix())

# %%

from isaaclab_arena.assets.asset_registry import AssetRegistry

asset_registry = AssetRegistry()
kitchen = asset_registry.get_asset_by_name("kitchen")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()

# assets = {
#     kitchen.name: kitchen,
#     cracker_box.name: cracker_box,
# }

poses = {
    "kitchen": background_initial_pose,
    "cracker_box": cracker_box_initial_pose,
}


# Get the root prim.
root_prim = stage.GetDefaultPrim()
print(root_prim.GetPath() == "/World")

test_prim_paths = [
    "/World/kitchen",
    "/World/cracker_box",
]

import numpy as np

EPS = 1e-6

from pxr import Gf


def to_numpy_q_wxyz(q_wxyz: Gf.Quatf) -> np.ndarray:
    return np.array([q_wxyz.GetReal(), *q_wxyz.GetImaginary()])


# Loop through the children of the root prim.
for child in root_prim.GetChildren():
    # Asset name
    prim_name = child.GetName()
    print(prim_name)
    asset = assets[prim_name]

    print(child.GetPath())
    print(child.GetPath() in test_prim_paths)
    # Get the pose of the prim.
    expected_pose = poses[prim_name]
    prim_position = child.GetAttribute("xformOp:translate").Get()
    prim_orientation = child.GetAttribute("xformOp:orient").Get()
    print(np.linalg.norm(prim_position - expected_pose.position_xyz) < EPS)
    print(np.linalg.norm(to_numpy_q_wxyz(prim_orientation) - np.array(expected_pose.rotation_wxyz)) < EPS)
    print(expected_pose)
    print(prim_position)
    print(prim_orientation)


# %%


# from pxr import Usd, UsdGeom, Gf

# from isaaclab_arena.assets.object import Object


# def create_prim_from_asset(stage: Usd.Stage, asset: Object):
#     assert isinstance(asset, Object)
#     # Get the default prim path
#     default_prim_path = stage.GetDefaultPrim().GetPath()
#     asset_path = str(default_prim_path) + "/" + asset.name
#     prim = stage.DefinePrim(asset_path, "Xform")
#     prim.GetReferences().AddReference(asset.usd_path)
#     # Add the transform
#     prim_xform = UsdGeom.Xform(prim)
#     prim_xform.ClearXformOpOrder()
#     if asset.initial_pose is not None:
#         prim_xform.AddTranslateOp().Set(Gf.Vec3f(asset.initial_pose.position_xyz))
#         prim_xform.AddOrientOp().Set(Gf.Quatf(*asset.initial_pose.rotation_wxyz))
#     prim_xform.AddScaleOp().Set(Gf.Vec3f(asset.scale))


# def export_scene_to_usd(scene: Scene, output_path: pathlib.Path, root_prim_path: str = "/World"):
#     # Create a new stage for composition
#     stage_out = Usd.Stage.CreateInMemory()
#     # Add the root/default prim
#     world = stage_out.DefinePrim(root_prim_path, "Xform")
#     stage_out.SetDefaultPrim(world)
#     # Add each asset to the stage, under the root prim
#     for asset in scene.assets.values():
#         create_prim_from_asset(stage_out, asset)
#     # Flatten
#     flattened_layer = stage_out.Flatten()
#     # Save to a file
#     flattened_layer.Export(output_path.as_posix())
# export_scene_to_usd(scene, output_path)

# %%

# # stage_out = Usd.Stage.CreateNew(new_usd_path.as_posix())
# stage_out = Usd.Stage.CreateInMemory()

# # Reference background into /World
# world = stage_out.DefinePrim("/World", "Xform")
# stage_out.SetDefaultPrim(world)

# for asset in scene.assets.values():
#     create_prim_from_asset(stage_out, asset)

# flattened_layer = stage_out.Flatten()

# # Save it
# flattened_layer.Export(flattened_usd_path.as_posix())

# %%


# stage_out.GetRootLayer().Save()
# stage = Usd.Stage.Open(new_usd_path.as_posix())
# flattened_layer = stage.Flatten()


# def set_prim_pose(prim: Usd.Prim, pose: Pose | None, scale: tuple[float, float, float]):
#     prim_xform = UsdGeom.Xform(prim)
#     prim_xform.ClearXformOpOrder()
#     if pose is not None:
#         prim_xform.AddTranslateOp().Set(Gf.Vec3f(pose.position_xyz))
#         prim_xform.AddOrientOp().Set(Gf.Quatf(*pose.rotation_wxyz))
#     prim_xform.AddScaleOp().Set(Gf.Vec3f(scale))
#     return prim_xform


# def create_prim_from_asset(stage: Usd.Stage, asset: Asset, path: str):
#     prim = stage.DefinePrim(path, "Xform")
#     prim.GetReferences().AddReference(asset.usd_path)
#     set_prim_pose(prim, asset.initial_pose, asset.scale)
#     return prim


# # Create a prim from the background
# background_prim = stage_out.DefinePrim(default_prim_path + "/Background", "Xform")
# background_prim.GetReferences().AddReference(background.usd_path)
# set_prim_pose(background_prim, background.initial_pose, background.scale)

# def export_scene_to_usd(scene: Scene, output_path: pathlib.Path):
# TDB


# # Reference the background into /World/Background
# background_prim = stage_out.DefinePrim("/World/Background", "Xform")
# background_prim.GetReferences().AddReference(background.usd_path)
# set_prim_pose(background_prim, background.initial_pose, background.scale)


# # Reference cracker box into /World/CrackerBox
# cracker_prim = stage_out.DefinePrim("/World/CrackerBox", "Xform")
# cracker_prim.GetReferences().AddReference(cracker_box.usd_path)
# set_prim_pose(cracker_prim, cracker_box.initial_pose, cracker_box.scale)


# %%

# inputs = [background.usd_path, cracker_box.usd_path]


# for path in inputs:
#     src_stage = Usd.Stage.Open(path)
#     UsdUtils.CopyLayerMetadata(src_stage.GetRootLayer(), stage_out.GetRootLayer())

#     UsdUtils.CopyPrim(
#         sourcePrim = src_stage.GetPseudoRoot(),
#         destLayer  = stage_out.GetEditTarget().GetLayer(),
#         destPath   = "/"
#     )


# %%


# %%

# print(cracker_box.usd_path)


# %%
