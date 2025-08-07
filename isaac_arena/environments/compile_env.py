# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import argparse
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_tasks.utils import parse_env_cfg

from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.environments.isaac_arena_manager_based_env import IsaacArenaManagerBasedRLEnvCfg
from isaac_arena.utils.configclass import combine_configclass_instances


def compile_environment_config(
    isaac_arena_environment: IsaacArenaEnvironment, args_cli: argparse.Namespace
) -> ManagerBasedRLEnvCfg:
    """Compile the arena environment configuration to a gymnasium environment.

    Args:
        isaac_arena_environment (IsaacArenaEnvironment): The arena environment configuration.
        args_cli (argparse.Namespace): The command line arguments.

    Returns:
        gym.Env: The compiled gymnasium environment.
    """
    # Get the manager-based environment configuration.
    arena_env_cfg = compile_manager_based_env_cfg(isaac_arena_environment=isaac_arena_environment)
    if args_cli.mimic:
        # We compile the mimic env configuration now. This is a combination of the arena env cfg and the task mimic env cfg.
        env_cfg = _compile_mimic_env_cfg(arena_env_cfg=arena_env_cfg, isaac_arena_environment=isaac_arena_environment)
        # We also point to the mimic env entry point which is a inherited class of ManagerBasedRLEnv.
        entry_point = _combine_mimic_env_from_embodiment_and_task(isaac_arena_environment=isaac_arena_environment)
    else:
        env_cfg = arena_env_cfg
        entry_point = "isaaclab.envs:ManagerBasedRLEnv"

    env_cfg = compile_gym_env_cfg(
        name=isaac_arena_environment.name, entry_point=entry_point, arena_env_cfg=env_cfg, args_cli=args_cli
    )

    return env_cfg


def compile_manager_based_env_cfg(isaac_arena_environment: IsaacArenaEnvironment) -> IsaacArenaManagerBasedRLEnvCfg:
    """Get the manager-based environment configuration.

    Args:
        isaac_arena_environment (IsaacArenaEnvironment): The arena environment configuration.

    Returns:
        IsaacArenaManagerBasedRLEnvCfg: The manager-based environment configuration.
    """

    # Set the robot position
    isaac_arena_environment.embodiment.set_robot_initial_pose(isaac_arena_environment.scene.get_robot_initial_pose())

    # Scene composition - The scene is composed of:
    # - Base IsaacLab config
    # - Contributions from the (background) scene
    # - Contributions from the embodiment
    scene_cfg = combine_configclass_instances(
        "SceneCfg",
        InteractiveSceneCfg(
            num_envs=4096,
            env_spacing=30.0,
            replicate_physics=False,
        ),
        isaac_arena_environment.scene.get_scene_cfg(),
        isaac_arena_environment.embodiment.get_scene_cfg(),
    )

    events_cfg = combine_configclass_instances(
        "EventsCfg",
        isaac_arena_environment.embodiment.get_event_cfg(),
        isaac_arena_environment.scene.get_events_cfg(),
    )

    termination_cfg = combine_configclass_instances(
        "TerminationCfg",
        isaac_arena_environment.task.get_termination_cfg(),
        isaac_arena_environment.scene.get_termination_cfg(),
    )

    # Build the manager-based environment configuration.
    arena_env_cfg = IsaacArenaManagerBasedRLEnvCfg(
        observations=isaac_arena_environment.embodiment.get_observation_cfg(),
        actions=isaac_arena_environment.embodiment.get_action_cfg(),
        events=events_cfg,
        scene=scene_cfg,
        terminations=termination_cfg,
    )
    return arena_env_cfg


def _compile_mimic_env_cfg(
    arena_env_cfg: IsaacArenaManagerBasedRLEnvCfg, isaac_arena_environment: IsaacArenaEnvironment
) -> IsaacArenaManagerBasedRLEnvCfg:
    """Compile the mimic env configuration.

    Args:
        arena_env_cfg (IsaacArenaManagerBasedRLEnvCfg): The manager-based environment configuration.
        isaac_arena_environment (IsaacArenaEnvironment): The arena environment configuration.

    Returns:
        IsaacArenaManagerBasedRLEnvCfg: The mimic env configuration.
    """
    # We combine the mimic env and the arena env together
    task_mimic_env_cfg = isaac_arena_environment.task.get_mimic_env_cfg()
    mimic_env_cfg = combine_configclass_instances(
        "MimicEnvCfg",
        arena_env_cfg,
        task_mimic_env_cfg,
    )

    return mimic_env_cfg


def _combine_mimic_env_from_embodiment_and_task(
    isaac_arena_environment: IsaacArenaEnvironment,
) -> ManagerBasedRLMimicEnv:
    """Combine the mimic env from the embodiment and the task.

    Args:
        isaac_arena_environment (IsaacArenaEnvironment): The arena environment configuration.

    Returns:
        ManagerBasedRLMimicEnv: The combined mimic env.
    """
    embodiment_mimic_env = isaac_arena_environment.embodiment.get_mimic_env()
    task_mimic_env = isaac_arena_environment.task.get_mimic_env()

    # We combine the mimic env and the arena env together
    mimic_env = type("CombinedMimicEnv", (embodiment_mimic_env, task_mimic_env), {})
    return mimic_env


def compile_gym_env_cfg(
    name: str,
    entry_point: str | ManagerBasedRLMimicEnv,
    arena_env_cfg: IsaacArenaManagerBasedRLEnvCfg,
    args_cli: argparse.Namespace,
) -> ManagerBasedRLEnvCfg:
    """Compile the arena environment configuration to a gymnasium environment.

    Args:
        name (str): The name of the environment.
        arena_env_cfg (IsaacArenaManagerBasedRLEnvCfg): The manager-based environment configuration.
        args_cli (argparse.Namespace): The command line arguments.

    Returns:
        gym.Env: The compiled gymnasium environment.
    """
    gym.register(
        id=name,
        entry_point=entry_point,
        kwargs={
            "env_cfg_entry_point": arena_env_cfg,
        },
        disable_env_checker=True,
    )
    env_cfg = parse_env_cfg(
        name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    return env_cfg


def make_gym_env(name: str, env_cfg: ManagerBasedRLEnvCfg) -> gym.Env:
    """Make the gymnasium environment.

    Args:
        name (str): The name of the environment.
        env_cfg (ManagerBasedRLEnvCfg): The environment configuration.

    Returns:
        gym.Env: The gymnasium environment.
    """

    env = gym.make(name, cfg=env_cfg)
    # Reset for good measure.
    env.reset()

    return env
