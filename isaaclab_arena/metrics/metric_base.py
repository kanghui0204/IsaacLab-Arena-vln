import numpy as np
from abc import ABC, abstractmethod

from isaaclab.managers.recorder_manager import RecorderTermCfg


class MetricBase(ABC):

    name: str
    recorder_term_name: str

    @abstractmethod
    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        raise NotImplementedError("Function not implemented yet.")
