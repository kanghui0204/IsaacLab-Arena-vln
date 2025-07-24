from dataclasses import MISSING, dataclass

import isaaclab.sim as sim_utils
from isaac_arena.scene.scene import SceneBase
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass


from typing import List, Tuple, Any
import dataclasses

from isaac_arena.utils.configclass import make_configclass, get_field_info


# Change to cls
@staticmethod
def extend_configclass(config_class: configclass, name: str, field_info: List[Tuple[str, type, Any]]) -> configclass:
    base_field_info = get_field_info(config_class)
    combined_field_info = base_field_info + field_info
    return make_configclass(name, combined_field_info)



@configclass
class PickAndPlaceSceneCfg(InteractiveSceneCfg):
    background_scene: AssetBaseCfg = MISSING
    pick_up_object: RigidObjectCfg = MISSING
    destination_object: RigidObjectCfg = MISSING

    # TODO(cvolk): It seems like the scene needs to hold a robot
    robot: ArticulationCfg = MISSING


@configclass
class PickAndPlaceSceneCfg2:
    background_scene: AssetBaseCfg = MISSING
    pick_up_object: RigidObjectCfg = MISSING
    destination_object: RigidObjectCfg = MISSING

    # REMOVE
    # robot: ArticulationCfg = MISSING


class KitchenPickAndPlaceScene(SceneBase):
    def __init__(
        self, pick_up_object: RigidObjectCfg, destination_object: RigidObjectCfg, robot: ArticulationCfg = None
    ):
        super().__init__()
        # The background scene
        self.background_scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            # These positions are hardcoded for the kitchen scene. Its important to keep them.
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]),
            spawn=UsdFileCfg(
                usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/kitchen_scene_teleop_v3.usd"
            ),
        )
        # An object, which has to be placed on/into the target object
        self.pick_up_object: RigidObjectCfg = pick_up_object

        # An object, which has to be placed on/into the target object
        self.destination_object: RigidObjectCfg = destination_object

        # Robot configuration
        self.robot: ArticulationCfg = robot


    # # Change to cls
    # @staticmethod
    # def get_field_info(config_class: configclass) -> List[Tuple[str, type, Any]]:
    #     field_info_list = []
    #     for f in dataclasses.fields(config_class):
    #         field_info = (f.name, f.type)
    #         if f.default is not dataclasses.MISSING:
    #             field_info += (f.default,)
    #         elif f.default_factory is not dataclasses.MISSING:
    #                 field_info += (f.default_factory,)
    #         field_info_list.append(field_info)
    #     return field_info_list



    def get_scene_cfg(self) -> PickAndPlaceSceneCfg:
        # return make_configclass(
        #     "PickAndPlaceSceneCfg",
        #     [
        #         ("num_envs", int, 4096),
        #         ("env_spacing", float, 30.0),
        #         ("replicate_physics", bool, False),
        #         ("background_scene", AssetBaseCfg, self.background_scene),
        #         ("pick_up_object", RigidObjectCfg, self.pick_up_object),
        #         ("destination_object", RigidObjectCfg, self.destination_object),
        #         ("robot", ArticulationCfg, self.robot),
        #     ],
        # )

        # field_info = self.get_field_info(PickAndPlaceSceneCfg)
        # base_field_info = self.get_field_info(InteractiveSceneCfg)

        # new_field_info = [
        #     ("background_scene", AssetBaseCfg, self.background_scene),
        #     ("pick_up_object", RigidObjectCfg, self.pick_up_object),
        #     ("destination_object", RigidObjectCfg, self.destination_object),
        #     ("robot", ArticulationCfg, self.robot),
        # ]

        # field_info = base_field_info + new_field_info
        # PickAndPlaceSceneCfg2 = make_configclass("PickAndPlaceSceneCfg2", field_info)
        # PickAndPlaceSceneCfg2 = extend_configclass(InteractiveSceneCfg, "PickAndPlaceSceneCfg2", new_field_info)
        # return PickAndPlaceSceneCfg2(

        # compose_configclass(
        #     "PickAndPlaceSceneCfg2",
        #     InteractiveSceneCfg,
        #     PickAndPlaceSceneCfg,
        # )

        # ATTEMPT 3 - combine types and initialize
        # from isaac_arena.utils.configclass import combine_configclasses
        # PickAndPlaceSceneCfg3 = combine_configclasses(
        #     "PickAndPlaceSceneCfg3",
        #     InteractiveSceneCfg,
        #     PickAndPlaceSceneCfg2,
        # )
        
        # return PickAndPlaceSceneCfg3(
        #     num_envs=4096,
        #     env_spacing=30.0,
        #     replicate_physics=False,
        #     background_scene=self.background_scene,
        #     pick_up_object=self.pick_up_object,
        #     destination_object=self.destination_object,
        #     robot=self.robot,
        # )

        # ATTEMPT 4 - combine instances
        # interactive_scene_cfg = InteractiveSceneCfg(
        #     num_envs=4096,
        #     env_spacing=30.0,
        #     replicate_physics=False,
        # )

        # pick_and_place_scene_cfg2 = PickAndPlaceSceneCfg2(
        #     background_scene=self.background_scene,
        #     pick_up_object=self.pick_up_object,
        #     destination_object=self.destination_object,
        #     robot=self.robot,
        # )

        # from isaac_arena.utils.configclass import combine_configclass_instances
        # pick_and_place_scene_cfg3 = combine_configclass_instances(
        #     "PickAndPlaceSceneCfg3",
        #     interactive_scene_cfg,
        #     pick_and_place_scene_cfg2,
        # )
        # return pick_and_place_scene_cfg3

        # ATTEMPT 5 - only return scene-relavent elements, compose in compiler
        return PickAndPlaceSceneCfg2(
            background_scene=self.background_scene,
            pick_up_object=self.pick_up_object,
            destination_object=self.destination_object,
            # robot=self.robot,
        )


class MugInDrawerKitchenPickAndPlaceScene(KitchenPickAndPlaceScene):
    def __init__(self, robot: ArticulationCfg = None):
        super().__init__(
            pick_up_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_mug",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            ),
            destination_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_mugs",
                spawn=sim_utils.CuboidCfg(
                    size=[0.4, 0.65, 0.01],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    activate_contact_sensors=True,
                ),
            ),
            robot=robot,
        )
