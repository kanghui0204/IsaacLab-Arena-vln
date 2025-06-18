


from typing import Dict, Any

from isaaclab.assets import AssetBaseCfg

from isaac_arena.core.object import Object
from isaac_arena.scene.scene import SceneBase


class PickAndPlaceSceneBase(SceneBase):

    def __init__(self, background_scene: AssetBaseCfg, pick_up_object: Object, destination_object: Object):
        # The background scene
        self.background_scene = background_scene
        # An object, which has to be placed on/into the target object
        self.pick_up_object = pick_up_object
        # An object, which has to be placed on/into the target object
        self.destination_object = destination_object
    
    def get_scene_cfg(self) -> Dict[str, Any]:
        pass

    def get_observation_cfg(self) -> Dict[str, Any]:
        class ObservationCfg:
            pick_up_object_position = ObjectPositionObservationCfg(
                object=self.pick_up_object,
            )
        return ObservationCfg()


