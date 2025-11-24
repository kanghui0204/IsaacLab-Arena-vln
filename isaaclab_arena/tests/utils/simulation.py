import torch
import tqdm
from collections.abc import Callable

from isaaclab.envs.manager_based_env import ManagerBasedEnv


def step_zeros_and_call(
    env: ManagerBasedEnv, num_steps: int, function: Callable[[ManagerBasedEnv, torch.Tensor], None] | None = None
) -> None:
    """Step through the environment with zero actions for a specified number of steps."""
    for _ in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.device)
            _, _, terminated, _, _ = env.step(actions)
            if function is not None:
                function(env, terminated)
