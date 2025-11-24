import gymnasium as gym
import torch
from abc import ABC, abstractmethod
from gymnasium.spaces.dict import Dict as GymSpacesDict


class PolicyBase(ABC):
    def __init__(self):
        """
        Base class for policies.
        """

    @abstractmethod
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Compute an action given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        pass

    def reset(self, env: gym.Env) -> None:
        """
        Reset the policy.
        """
        pass
