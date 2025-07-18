from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaac_arena.embodiments import mdp
from isaac_arena.embodiments.mdp import franka_arrange_events, franka_stack_events
from isaac_arena.embodiments.embodiments import FrankaEmbodiment
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# class ArenaEnv(ABC):
#    isaac_lab_env_cfg: ManagerBasedRLEnvCfg = MISSING
#
#    metrics_cfg: MISSING


@configclass
class KitchenTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    # TODO(cvolk): Not anymore in scene
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    # TODO(cvolk): Where is this now? Robot?
    ee_frame: FrameTransformerCfg = MISSING

    # Add the kitchen scene here
    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        # These positions are hardcoded for the kitchen scene. Its important to keep them.
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]),
        spawn=UsdFileCfg(
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/kitchen_scene_teleop_v3.usd"
        ),
    )

    # HELPER OBJECTS

    # Add a plate right below the bottom of the drawer were the mugs are placed.
    # This will be useful to have a fixed reference to the mugs drawer in mimicgen
    bottom_of_drawer_with_mugs = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_mugs",
        spawn=sim_utils.CuboidCfg(
            size=[0.4, 0.65, 0.01],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
    )
    # Add a plate right below the bottom of the drawer were the boxes are placed.
    # This will be useful to have a fixed reference to the boxes drawer in mimicgen
    bottom_of_drawer_with_boxes = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_boxes",
        spawn=sim_utils.CuboidCfg(
            size=[0.4, 0.65, 0.01],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
    )

    # OBJECTS ON TABLE

    mac_n_cheese_on_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mac_n_cheese_on_table",
        spawn=UsdFileCfg(
            usd_path=(
                "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mac_n_cheese_physics.usd"
            ),
            scale=(1.0, 1.0, 1.0),
        ),
    )
    tomato_soup_on_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_on_table",
        spawn=UsdFileCfg(
            usd_path=(
                "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/tomato_soup_physics.usd"
            ),
            scale=(1.0, 1.0, 1.0),
        ),
    )

    # To have a fixed reference frame for mimicgen
    mug1_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug1_in_drawer",
        spawn=UsdFileCfg(
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Isaac/Props/Mugs/SM_Mug_A2.usd",
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug2_physics.usd",
            scale=(0.0125, 0.0125, 0.0125),
            activate_contact_sensors=True,
        ),
    )
    mug2_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug2_in_drawer",
        spawn=UsdFileCfg(
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug3_physics.usd",
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Isaac/Props/Mugs/SM_Mug_D1.usd",
            scale=(0.0125, 0.0125, 0.0125),
        ),
    )
    sugar_box_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_in_drawer",
        spawn=UsdFileCfg(
            usd_path=(
                "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/sugar_box_physics.usd"
            ),
            scale=(1.0, 1.0, 1.0),
        ),
    )
    pudding_box_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pudding_box_in_drawer",
        spawn=UsdFileCfg(
            usd_path=(
                "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/pudding_box_physics.usd"
            ),
            scale=(1.0, 1.0, 1.0),
        ),
    )
    gelatin_box_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/gelatin_box_in_drawer",
        spawn=UsdFileCfg(
            usd_path=(
                "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/gelatin_box_physics.usd"
            ),
            scale=(1.0, 1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("target_mug")},
    )

    success = DoneTerm(func=mdp.object_in_drawer)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy = FrankaEmbodiment().get_observation_cfg().policy 


@configclass
class EventCfg:
    """Configuration for events."""
    pass

@configclass
class ArrangeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    # TODO(cvolk) Move this out.
    scene: KitchenTableSceneCfg = KitchenTableSceneCfg(num_envs=4096, env_spacing=30, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()

    for key, value in vars(FrankaEmbodiment().get_action_cfg()).items():
        if not key.startswith("__"):
            setattr(ActionsCfg, key, value)

    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()


    for key, value in vars(FrankaEmbodiment().get_event_cfg()).items():
        if not key.startswith("__"):
            setattr(EventCfg, key, value)

    # Set events
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625



@configclass
class ArrangeKitchenObjectEnvCfg(ArrangeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

            # Set Franka as robot
        self.scene.robot = FrankaEmbodiment().get_robot_cfg()

        self.scene.target_mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target_mug",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
            spawn=UsdFileCfg(
                usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug_physics.usd",
                scale=(0.0125, 0.0125, 0.0125),
                activate_contact_sensors=True,
            ),
        )

        self.scene.contact_forces_target_mug = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/target_mug", history_length=3, track_air_time=True
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
