from abc import ABC, abstractmethod


class TeleopDeviceBase(ABC):

    name: str | None = None

    def __init__(self, sim_device: str | None = None):
        self.sim_device = sim_device

    @abstractmethod
    def get_teleop_device_cfg(self, embodiment: object | None = None):
        raise NotImplementedError
