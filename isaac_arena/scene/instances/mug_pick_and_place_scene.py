



from typing import Dict, Any

from isaaclab.assets import AssetBaseCfg

from isaac_arena.core.object import Object
from isaac_arena.scene.pick_and_place_scene import PickAndPlaceSceneBase



class KitchenPickAndPlaceScene(PickAndPlaceSceneBase):

    background_scene: AssetBaseCfg = UsdSceneCfg(
        asset_path='Background/background.usd',
    )
    pick_up_object: Object = ObjectCfg(
        asset_path='Object/mug.usd',
    )
    destination_object: Object = ObjectCfg(
        prim_path='Background/background/drawer',
    )

    def __init__(self):
        super().__init__(
            background_scene=self.background_scene,
            pick_up_object=self.pick_up_object,
            destination_object=self.destination_object,
        )
    
    def get_scene_cfg(self) -> Dict[str, Any]:
        class SceneCfg:
            background_scene = self.background_scene
            object = self.destination_object
        return SceneCfg()

    def get_events_cfg(self) -> Dict[str, Any]:
        class EventsCfg:
            randomize_pick_up_object = RandomizeObjectCfg(
                object=self.pick_up_object,
                assets_list=[
                    'Object/mug.usd',
                    'Object/cup.usd',
                    'Object/bowl.usd',
                ],
            )
            randomize_pick_up_object_position = RandomizeObjectPositionCfg(
                object=self.pick_up_object,
                position_std=[0.1, 0.1, 0.1],
                orientation_std=[0.1, 0.1, 0.1],
            )
        return EventsCfg()
