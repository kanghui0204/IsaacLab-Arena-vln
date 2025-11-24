from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.tasks.task_base import TaskBase


class DummyTask(TaskBase):
    def __init__(self):
        super().__init__()

    def get_scene_cfg(self):
        pass

    def get_termination_cfg(self):
        pass

    def get_events_cfg(self):
        pass

    def get_prompt(self):
        pass

    def get_mimic_env_cfg(self, embodiment_name: str):
        pass

    def get_metrics(self):
        pass

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))
