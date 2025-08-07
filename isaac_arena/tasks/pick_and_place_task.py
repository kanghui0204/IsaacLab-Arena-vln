# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import torch
from collections.abc import Sequence

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaac_arena.tasks.task import TaskBase

# from isaac_arena.tasks.terminations.object_in_drawer import object_in_drawer
from isaac_arena.tasks.terminations import object_on_destination


class PickAndPlaceTask(TaskBase):
    def __init__(self):
        super().__init__()

    def get_termination_cfg(self):
        return TerminationsCfg()

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self):
        return PickPlaceMimicEnvCfg()

    def get_mimic_env(self):
        return PickPlaceMimicEnv


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # TODO(cvolk): Make this config generic and move instance out.
    # time_out: TerminationTermCfg = MISSING
    # termination_terms: TerminationTermCfg = MISSING
    # success: TerminationTermCfg = MISSING
    time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=False)

    success = TerminationTermCfg(
        func=object_on_destination,
        params={
            "object_cfg": SceneEntityCfg("pick_up_object"),
            "contact_sensor_cfg": SceneEntityCfg("pick_up_object_contact_sensor"),
            "force_threshold": 1.0,
            "velocity_threshold": 0.01,
        },
    )


@configclass
class PickPlaceMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Pick and Place env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()
        # # TODO: Figure out how we can move this to the MimicEnvCfg class
        # # The __post_init__() above only calls the init for FrankaCubeStackEnvCfg and not MimicEnvCfg
        # # https://stackoverflow.com/questions/59986413/achieving-multiple-inheritance-using-python-dataclasses

        # Override the existing values
        self.datagen_config.name = "demo_src_pickplace_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the pick and place task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="pick_up_object",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_1",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="destination_object",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["robot"] = subtask_configs


class PickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """Configuration for Pick and Place Mimic."""

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.
        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.
        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
        return signals
