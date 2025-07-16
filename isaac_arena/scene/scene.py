from abc import ABC
from typing import Any


class SceneBase(ABC):
    def __init__(self):
        pass

    def get_scene_cfg(self) -> dict[str, Any]:
        pass

    def get_observation_cfg(self) -> dict[str, Any]:
        pass

    def get_events_cfg(self) -> dict[str, Any]:
        pass
