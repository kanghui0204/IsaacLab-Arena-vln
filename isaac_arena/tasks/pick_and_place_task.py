from enum import Enum

from isaac_arena.scene.scene import PickAndPlaceScene
from isaac_arena.tasks.task import TaskBase


class PickAndPlaceCompletionType(Enum):
    ON = "on"
    IN = "in"


class PickAndPlaceTaskCfg(TaskBase):
    def __init__(self, completion_type: PickAndPlaceCompletionType):
        self.completion_type = completion_type

    def get_termination_cfg(self, scene: PickAndPlaceScene):
        # TODO(cvolk): Implement proper termination configuration using IsaacLab mdp functions
        # For now, return None to avoid undefined class errors here and below on pre-commit checks.
        return None

        # Original implementation (commented out due to undefined classes):
        # return [
        #     self._get_success_termination_cfg(scene),
        #     self._get_failure_termination_cfg(scene),
        # ]

    def get_prompt(self):
        return (
            f"Pick {self.scene.target_object.name} and place"
            f" {self.completion_type.name} {self.scene.destination_object.name}"
        )

    # def _get_success_termination_cfg(self, scene: PickAndPlaceScene):
    #     if self.completion_type == PickAndPlaceCompletionType.OBJECT_ON:
    #         return ObjectInTerminationCfg(
    #             target_object=scene.place_object,
    #             destination_object=scene.pick_up_object,
    #         )
    #     elif self.completion_type == PickAndPlaceCompletionType.OBJECT_IN:
    #         return ObjectOnTerminationCfg(
    #             target_object=scene.place_object,
    #             destination_object=scene.pick_up_object,
    #         )
    #     else:
    #         raise ValueError(f"Invalid completion type: {self.completion_type}")

    # def _get_failure_termination_cfg(self, scene: PickAndPlaceScene):
    #     return ObjectFallenTerminationCfg(
    #         target_object=scene.pick_up_object,
    #     )
