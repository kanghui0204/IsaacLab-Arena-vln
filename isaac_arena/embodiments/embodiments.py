from abc import ABC
from dataclasses import MISSING
from typing import Any
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ActionTermCfg
from isaac_arena.embodiments.mdp import franka_stack_events
from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.sensors import CameraCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
from isaaclab.utils import configclass
from isaac_arena.embodiments import mdp
import isaaclab.envs.mdp as mdp_isaac_lab

@configclass
class FrankaActionsCfg:
    """Action specifications for the MDP."""

    arm_action: type[ActionTermCfg] = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

    gripper_action: type[ActionTermCfg] = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
    
@configclass
class FrankaObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()



@configclass
class FrankaEventCfg:
    """Configuration for Franek."""
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        # We changed the mode from startup to reset as the default pose got reset after it was
        # set by the startup event.
        # TODO(remos): find out why this happened and fix it
        mode="reset",
        params={
            "default_pose": [0.0, -0.785, -0.1107, -1.1775, 0.0, 0.785, 0.785, 0.0400, 0.0400],
        },
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


class EmbodimentBase(ABC):

    # TODO(cvolk): Add types again correectly
    #robot: type[ArticulationCfg] = MISSING
    #action_config: type[ActionsCfg] = MISSING
    #observation_config = MISSING
    #event_config: type[FrankaEventCfg] = MISSING

    robot = MISSING
    action_config = MISSING
    observation_config = MISSING
    event_config = MISSING

    def __init__(self, params: dict[str, Any]):
        pass

    def get_robot_cfg(self) -> Any:
        return self.robot

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> Any:
        return self.observation_config

    def get_event_cfg(self) -> Any:
        return self.event_config


# Here it should become instances
class FrankaEmbodiment(EmbodimentBase):
    def __init__(self):
        self.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config = FrankaActionsCfg()
        self.observation_config = FrankaObservationsCfg()
        self.event_config = FrankaEventCfg()

    def __post_init__(self):
        self.robot.spawn.semantic_tags = [("class", "robot_arm")]
