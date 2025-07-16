from abc import ABC
from typing import Any


class EmbodimentBase(ABC):
    def __init__(self, params: dict[str, Any]):
        pass

    def get_action_cfg(self) -> dict[str, Any]:
        pass

    def get_observation_cfg(self) -> dict[str, Any]:
        pass
