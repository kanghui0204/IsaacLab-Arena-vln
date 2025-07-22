import argparse

from isaacsim import SimulationApp

from isaaclab.app import AppLauncher


def app_launcher(args: argparse.Namespace) -> SimulationApp:
    """Launch the Isaac Sim app and return the simulation app.

    Note: The simulation app should be launched before importing most IsaacLab modules.

    This is required because:
    1. Omniverse extensions are hot-loaded when the simulator starts
    2. Many IsaacLab and Isaac Sim modules become available only after the app is launched
    3. Importing IsaacLab modules before app launch may cause missing module errors
    """
    # AppLauncher.add_app_launcher_args(args)
    # args_cli = args.parse_args()

    if args.enable_pinocchio:
        # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
        # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
        import pinocchio  # noqa: F401

    # launch omniverse app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    return simulation_app
