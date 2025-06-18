


from abc import ABC
from typing import Dict, Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from isaac_arena.core.object import Object



class SceneBase(ABC):
    def __init__(self):
        pass

    def get_scene_cfg(self) -> Dict[str, Any]:
        pass

    def get_observation_cfg(self) -> Dict[str, Any]:
        pass

    def get_events_cfg(self) -> Dict[str, Any]:
        pass

