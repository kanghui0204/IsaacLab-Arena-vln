from abc import ABC
from typing import Any

from isaac_arena.utils.configclass import configclass


class SceneBase(ABC):
    def __init__(self):
        pass

    def get_scene_cfg(self) -> Any: 
        pass

    def get_observation_cfg(self) -> Any:
        pass

    def get_events_cfg(self) -> Any:
        pass
