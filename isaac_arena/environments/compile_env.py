# def compile_env(
#    scene: SceneBase,
#    embodiment: EmbodimentBase,
#    task: TaskBase,
#    metrics: MetricsBase,
# ) -> ArenaEnv:
#    # Compose embodiment and scene observation cfg
#    class ObservationCfg:
#        embodiment_observation = embodiment.get_observation_cfg()
#        # scene_observation = scene.get_observation_cfg()
#
#    # Compose embodiment and scene events cfg
#    # class EventsCfg:
#    #     embodiment_events = embodiment.get_events_cfg()
#    #     scene_events = scene.get_events_cfg()
#
#    class IsaacLabEnvCfg(ManagerBasedRLEnvCfg):
#        scene_cfg = scene.get_scene_cfg()
#        observations_cfg = ObservationCfg()
#        actions_cfg = embodiment.get_action_cfg()
#        terminations_cfg = None
#        events_cfg = None
#
#        def __post_init__(self):
#            self._add_robot_to_scene_cfg()
#
#        def _add_robot_to_scene_cfg(self):
#            self.scene_cfg.robot = embodiment.get_robot_cfg()
#
#    return ArenaEnv(
#        isaac_lab_env_cfg=IsaacLabEnvCfg(),
#        metrics_cfg=metrics.get_metrics_cfg(),
#    )
#
#
# franka_global = Franka()
# scene_global = KitchenPickAndPlaceScene()
# Compose embodiment and scene observation cfg
#
# class ObservationsCfg:
#    embodiment_observation = franka_global.get_observation_cfg()
#    # scene_observation = scene.get_observation_cfg()
#
# Compose embodiment and scene events cfg
# class EventsCfg:
#     embodiment_events = embodiment.get_events_cfg()
#     scene_events = scene.get_events_cfg()