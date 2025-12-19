# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import (
    ExampleEnvironmentBase,
)

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class CloseDrawerEnvironment(ExampleEnvironmentBase):
    """
    A task environment for closing a cabinet drawer using a Franka robot.
    The drawer starts half-open and the goal is to close it.
    """

    name = "close_drawer"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask

        # Get assets from registry
        cabinet = self.asset_registry.get_asset_by_name("cabinet")()
        cabinet.set_initial_pose(Pose(position_xyz=(1.0, 0.0, 0.4), rotation_wxyz=(0.0, 0.0, 0.0, 1.0)))

        # Get embodiment based on selection
        if args_cli.embodiment == "franka":
            embodiment = self.asset_registry.get_asset_by_name("franka")(enable_cameras=args_cli.enable_cameras)
            # Set custom initial joint positions for close drawer task
            # Position the arm closer to the drawer handle
            embodiment.event_config.init_franka_arm_pose.params["default_pose"] = [
                0.0, -1.309, 0.0, -2.793, 0.0, 3.037, -0.830, 0.04, 0.04
            ]
        else:
            raise NotImplementedError(f"Embodiment {args_cli.embodiment} not supported")

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set robot initial pose
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Scene with cabinet, ground plane, and light
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        scene = Scene(assets=[cabinet, ground_plane, light])

        # Create close drawer task
        # The task will start with the drawer half-open (reset_openness=0.5)
        # and the goal is to close it (closedness_threshold=0.05)
        close_drawer_task = CloseDoorTask(
            openable_object=cabinet,
            closedness_threshold=0.05,
            reset_openness=0.7,
            episode_length_s=30.0,
            task_description="Close the cabinet drawer.",
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=close_drawer_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment to use")
        parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")


class OpenAndCloseDrawerEnvironment(ExampleEnvironmentBase):
    """
    A sequential task environment for opening then closing a cabinet drawer using a Franka robot.
    The task consists of two subtasks:
    1. Open the drawer (starts closed, goal is to open it to 80%)
    2. Close the drawer (goal is to close it to 5% or less)
    """

    name = "open_and_close_drawer"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

        # Get assets from registry
        cabinet = self.asset_registry.get_asset_by_name("cabinet")()
        cabinet.set_initial_pose(Pose(position_xyz=(1.0, 0.0, 0.4), rotation_wxyz=(0.0, 0.0, 0.0, 1.0)))

        # Get embodiment based on selection
        if args_cli.embodiment == "franka":
            embodiment = self.asset_registry.get_asset_by_name("franka")(enable_cameras=args_cli.enable_cameras)
            # Set custom initial joint positions for close drawer task
            # Position the arm closer to the drawer handle
            embodiment.event_config.init_franka_arm_pose.params["default_pose"] = [
                0.0, -1.309, 0.0, -2.793, 0.0, 3.037, -0.830, 0.04, 0.04
            ]
        else:
            raise NotImplementedError(f"Embodiment {args_cli.embodiment} not supported")

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set robot initial pose
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Scene with cabinet, ground plane, and light
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        scene = Scene(assets=[cabinet, ground_plane, light])

        # Create sequential task: first open the drawer, then close it
        # Subtask 1: Open drawer (starts closed, goal is to open it)
        open_drawer_task = OpenDoorTask(
            openable_object=cabinet,
            openness_threshold=0.5,
            reset_openness=0.0,  # Start with drawer fully closed
            task_description="Open the cabinet drawer.",
        )

        # Subtask 2: Close drawer (after opening, goal is to close it)
        close_drawer_task = CloseDoorTask(
            openable_object=cabinet,
            closedness_threshold=0.05,
            task_description="Close the cabinet drawer.",
        )
        # Set this to none to not overwrite open drawer events cfg
        close_drawer_task.events_cfg = None

        # Create a sequential task wrapper class
        class SequentialOpenCloseDrawerTask(SequentialTaskBase):
            def __init__(self, subtasks, episode_length_s=None):
                super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

            def get_metrics(self):
                return []

            def get_mimic_env_cfg(self, arm_mode):
                return None

        # Create the sequential task
        sequential_task = SequentialOpenCloseDrawerTask(
            subtasks=[open_drawer_task, close_drawer_task],
            episode_length_s=60.0,  # Longer episode for two subtasks
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=sequential_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object", type=str, default="apple", help="(Unused) Object argument for compatibility"
        )
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment to use")
        parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
