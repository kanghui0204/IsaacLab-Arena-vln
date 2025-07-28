from isaac_arena.scene.asset import Asset
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


class PickUpObject(Asset):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self, prim_path: str, usd_path: str, scale: tuple[float, float, float], name: str):
        super().__init__(name, ["pick_up_object"])
        self.pick_up_object_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=UsdFileCfg(
                usd_path=usd_path,
                scale=scale,
                activate_contact_sensors=True,
            ),
        )

    def get_pick_up_object_cfg(self) -> RigidObjectCfg:
        """Return the configured pick-up object asset."""
        return self.pick_up_object_cfg


class Mug(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            prim_path="{ENV_REGEX_NS}/target_mug",
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mugs/SM_Mug_A2.usd",
            scale=(1.0, 1.0, 1.0),
            name="mug",
        )


class GelatinBox(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            prim_path="{ENV_REGEX_NS}/target_gelatin_box",
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
            scale=(1.0, 1.0, 1.0),
            name="gelatin_box",
        )


class MustardBottle(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            prim_path="{ENV_REGEX_NS}/target_mustard_bottle",
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            scale=(1.0, 1.0, 1.0),
            name="mustard_bottle",
        )


class SugarBox(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            prim_path="{ENV_REGEX_NS}/target_sugar_box",
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            scale=(1.0, 1.0, 1.0),
            name="sugar_box",
        )


class TomatoSoupCan(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            prim_path="{ENV_REGEX_NS}/target_tomato_soup_can",
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(1.0, 1.0, 1.0),
            name="tomato_soup_can",
        )
