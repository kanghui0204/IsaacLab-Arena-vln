
from typing import Dict, Any

import torch

import isaac_arena



# Define the policy under test
class MyStupidPolicy(isaac_arena.policies.Policy):

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        super().__init__()

    def step(self, observation: Dict[str, Any]) -> torch.Tensor:
        return isaac_arena.Action.random()

my_stupid_policy = MyStupidPolicy(params={})

# Describe the tasks
embodiment = isaac_arena.embodiments.HumanoidEmbodiment()
tasks = isaac_arena.tasks.get_all_tasks_of_type(
    task_type=isaac_arena.tasks.TaskType.PICK_AND_PLACE,
    randomization=True,
)
envs = isaac_arena.environments.compile_env(
    scene=isaac_arena.scene.instances.KitchenPickAndPlaceScene(),
    embodiment=embodiment,
    task=isaac_arena.tasks.PickAndPlaceTaskCfg(),
)

# Metrics
metrics = isaac_arena.metrics.get_default_metrics_for_type(
    task_type=isaac_arena.tasks.TaskType.PICK_AND_PLACE,
)

# Run the policy
runner = isaac_arena.EvaluationRunner(
    envs=envs,
    metrics=metrics,
)
report = runner.run(my_stupid_policy, repeats_per_task=10)

print(f'Success rate: {report.success_rate}')
