# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to run an IsaacLab Arena environment with a policy by deserializing from YAML config."""

import gymnasium as gym
import numpy as np
import random
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
        

def main():
    """Script to run an IsaacLab Arena environment with a policy from serialized config."""
    
    # Parse CLI arguments
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        # Add policy-related arguments to the parser
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()

        # Load environment configuration from YAML (metrics are reconstructed from YAML)
        print("[INFO] Loading environment configuration from YAML...")
        from isaaclab_arena.utils.config_serialization import load_env_cfg_from_yaml
        cfg = load_env_cfg_from_yaml("/datasets/cfg_entry.yaml")
        
        arena_builder = get_arena_builder_from_cli(args_cli)
        cli_cfg = arena_builder.return_cfg()
        print("[DEBUG] Expected cfg.metrics", cli_cfg.metrics)
        print("[DEBUG] Actual cfg.metrics", cfg.metrics)
        

        # Register environment with gymnasium
        name = "kitchen_pick_and_place"
        entry_point = "isaaclab.envs:ManagerBasedRLEnv"
        
        gym.register(
            id=name,
            entry_point=entry_point,
            kwargs={"env_cfg_entry_point": cfg},
            disable_env_checker=True,
        )
        
        # Parse environment config for runtime
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        cfg = parse_env_cfg(
            name,
            device="cuda:0",
            num_envs=1,
            use_fabric=False,
        )
        
        # Create environment
        print("[INFO] Creating environment...")
        env = gym.make(name, cfg=cfg).unwrapped

        # Set random seeds
        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        # Reset environment
        obs, _ = env.reset()

        # Create policy
        print("[INFO] Creating policy...")
        policy, num_steps = create_policy(args_cli)
        
        # Lazy import to prevent app stalling
        from isaaclab_arena.metrics.metrics import compute_metrics

        # Run policy
        print(f"[INFO] Running policy for {num_steps} steps...")
        for _ in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)

        # Compute and print metrics
        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")

        # Close the environment
        env.close()


if __name__ == "__main__":
    main()

