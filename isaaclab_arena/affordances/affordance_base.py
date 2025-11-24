from abc import ABC, abstractmethod


class AffordanceBase(ABC):
    """Base class for affordances."""

    @property
    @abstractmethod
    def name(self) -> str:
        # NOTE(alexmillane, 2025.09.19) Affordances always have be combined with
        # an Asset which has a "name" property. By declaring this property
        # abstract here, we enforce this.
        pass
