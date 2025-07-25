# %%
import argparse
import torch
import tqdm

# Global simulation app, initialized only once in notebook
simulation_app = None
first_run = True

# Only launch sim app once
if simulation_app is None:
    from isaaclab.app import AppLauncher

    print("Launching simulation app once in notebook")
    simulation_app = AppLauncher()

# %%
for i in range(5):
    print("Running in notebook mode")
    args_parser = argparse.ArgumentParser(description="Isaac Arena CLI parser.")
    args = args_parser.parse_args([])
    args.background = "kitchen"
    args.pick_up_object = None
    args.num_steps = 3
    args.device = "cuda:0"
    args.num_envs = 1
    args.disable_fabric = True
    # Imports have to follow simulation startup.
    from isaac_arena.embodiments.franka.franka_embodiment import FrankaEmbodiment
    from isaac_arena.environments.compile_env import run_environment
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.scene.get_scene_details import get_scene_details
    from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTaskCfg
    import omni.usd

    omni.usd.get_context().new_stage()

    from isaacsim.core.utils.viewports import set_camera_view


    scene_details = get_scene_details(args.background, args.pick_up_object)

    # Arena Environment
    isaac_arena_environment = IsaacArenaEnvironment(
        name="kitchen_pick_and_place",
        embodiment=FrankaEmbodiment(),
        scene=PickAndPlaceScene(
            scene_details["background"], scene_details["pick_up_object"], scene_details["destination_object"]
        ),
        task=PickAndPlaceTaskCfg(),
    )

    # Compile an IsaacLab compatible arena environment configuration
    env = run_environment(isaac_arena_environment, args)

    # disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    env.reset()

    set_camera_view([-0.65481, 1.3153, 0.779], [90.8, -80.0, -30.462])

    # Run some zero actions.
    for _ in tqdm.tqdm(range(args.num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    # Close the environment.
    env.close()

# %%
