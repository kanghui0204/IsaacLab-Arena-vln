# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True


def get_test_environment(remove_reset_door_state_event: bool, num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.open_door_task import OpenDoorTask
    from isaaclab_arena.tasks.close_door_task import CloseDoorTask
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
    from isaaclab_arena.utils.pose import Pose

    class SequentialOpenCloseDoorTask(SequentialTaskBase):
        def __init__(
            self,
            subtasks,
            episode_length_s=None,
        ):
            super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

        def get_metrics(self):
            return []

        def get_prompt(self):
            return ""

        def get_mimic_env_cfg(self, embodiment_name: str):
            return None

        def get_events_cfg(self):
            """Override to use only the first subtask's events configuration."""
            from isaaclab_arena.utils.configclass import combine_configclass_instances

            # Get the first subtask's events config
            first_subtask_events_cfg = self.subtasks[0].get_events_cfg()

            # Combine with the sequential task event (reset subtask success state)
            events_cfg = combine_configclass_instances(
                "EventsCfg", first_subtask_events_cfg, self.make_sequential_task_events_cfg()
            )

            return events_cfg

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    # Put the microwave on the packing table.
    microwave.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )

    # Create sequential task: open door first, then close door
    open_task = OpenDoorTask(microwave, reset_openness=0.0)
    close_task = CloseDoorTask(microwave, reset_openness=1.0)

    scene = Scene(assets=[background, microwave])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="sequential_open_close_door",
        embodiment=FrankaEmbodiment(),
        scene=scene,
        task=SequentialOpenCloseDoorTask([open_task, close_task]),
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    if remove_reset_door_state_event:
        # Remove the reset door and subtask state events to allow us to inspect the scene without having it reset.
        # Since we override get_events_cfg to use only first subtask's events, the name doesn't have _subtask_0 suffix
        cfg.events.reset_openable_object_revolute_joint_percentage = None
        cfg.events.reset_subtask_success_state = None
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, microwave


def _test_sequential_open_close_door_microwave(simulation_app) -> bool:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave = get_test_environment(remove_reset_door_state_event=True, num_envs=1)

    def assert_composite_task_incomplete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([1])
        assert not terminated.item()
        if not terminated.item():
            print("Composite task is not completed")

    def assert_composite_task_complete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([1])
        assert terminated.item()
        if terminated.item():
            print("Composite task is completed")

    try:
        print("Starting with microwave closed")
        microwave.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave (completing subtask 0: open door)")
        microwave.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Closing microwave (completing subtask 1: close door, composite task should be complete)")
        microwave.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_complete)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_sequential_open_close_door_microwave_reset_condition(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave = get_test_environment(remove_reset_door_state_event=False, num_envs=2)

    try:
        print("Starting with microwave closed")
        microwave.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS)
        is_closed = microwave.is_closed(env)
        print(f"expected: [True, True]: got: {is_closed}")
        assert torch.all(is_closed == torch.tensor([True, True], device=env.device))

        print("Opening microwave (completing subtask 0: open door)")
        microwave.open(env, None)
        step_zeros_and_call(env, NUM_STEPS)
        is_open = microwave.is_open(env)
        print(f"expected: [True, True]: got: {is_open}")
        assert torch.all(is_open == torch.tensor([True, True], device=env.device))

        # Check that envs automatically reset when composite task completes
        print("Closing microwave (completing subtask 1: close door)")
        microwave.close(env, None)
        step_zeros_and_call(env, NUM_STEPS)
        # After composite task completion and reset, the door should be in initial state (closed from reset)
        is_closed = microwave.is_closed(env)
        print(f"After composite task completion, expected: [True, True]: got: {is_closed}")
        assert torch.all(is_closed == torch.tensor([True, True], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_sequential_open_close_door_microwave():
    result = run_simulation_app_function(
        _test_sequential_open_close_door_microwave,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_open_close_door_microwave.__name__} failed"


def test_sequential_open_close_door_microwave_reset_condition():
    result = run_simulation_app_function(
        _test_sequential_open_close_door_microwave_reset_condition,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_open_close_door_microwave_reset_condition.__name__} failed"


if __name__ == "__main__":
    test_sequential_open_close_door_microwave()
    test_sequential_open_close_door_microwave_reset_condition()
