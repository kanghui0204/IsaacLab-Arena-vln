from abc import ABC, abstractmethod
from typing import Any


class BaseStreamer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def start_streaming(self):
        pass

    @abstractmethod
    def get(self) -> Any:
        pass

    @abstractmethod
    def stop_streaming(self):
        pass
