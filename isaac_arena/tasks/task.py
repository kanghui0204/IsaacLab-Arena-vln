



from typing import Dict, Any
from abc import ABC

from isaac_arena.scene.scene import SceneBase, PickAndPlaceScene


class TaskBase(ABC):

    def __init__(self, scene: SceneBase):
        self.scene = scene

    def get_termination_cfg(self):
        pass

    def get_prompt(self) -> str:
        pass


