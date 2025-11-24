import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.policy_base import PolicyBase


class ZeroActionPolicy(PolicyBase):
    def __init__(self):
        super().__init__()

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Always returns a zero action.
        """
        return torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))
