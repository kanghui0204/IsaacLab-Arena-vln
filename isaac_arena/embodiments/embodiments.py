

from abc import ABC
from typing import Dict, Any


class EmbodimentBase(ABC):
    def __init__(self, params: Dict[str, Any]):
        pass

    def get_action_cfg(self) -> Dict[str, Any]:
        pass

    def get_observation_cfg(self) -> Dict[str, Any]:
        pass
    
    

    

