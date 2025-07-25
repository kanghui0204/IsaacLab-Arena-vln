from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


class PickUpObjects:
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self, pick_up_object: RigidObjectCfg):
        self.pick_up_object = pick_up_object

    def get_pick_up_object(self) -> RigidObjectCfg:
        """Return the configured pick-up object asset."""
        return self.pick_up_object


class Mug(PickUpObjects):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_mug",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            )
        )

class GelatinBox(PickUpObjects):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_gelatin_box",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/gelatin_box_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            )
        )

class MacandCheeseBox(PickUpObjects):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_mac_and_cheese_box",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mac_n_cheese_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            )
        )

class SugarBox(PickUpObjects):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_sugar_box",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/sugar_box_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            )
        )

class TomatoSoupCan(PickUpObjects):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_tomato_soup_can",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/tomato_soup_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            )
        )
