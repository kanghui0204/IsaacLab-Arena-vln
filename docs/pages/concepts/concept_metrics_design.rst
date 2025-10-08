Metrics Design
==============

THIS IS A WORK IN PROGRESS.*****VIK*****

Metrics in Isaac Arena provide a systematic framework for evaluating robot performance and task completion across different simulation scenarios. The metrics system integrates seamlessly with Isaac Lab's recorder manager to capture simulation data and compute meaningful performance indicators that support research, development, and benchmarking efforts.

Core Architecture
-----------------

The metrics system is built around a two-component architecture that separates data collection from metric computation:

.. code-block:: python

   class MetricBase(ABC):
       """Base class for all metrics."""

       name: str
       recorder_term_name: str

       @abstractmethod
       def get_recorder_term_cfg(self) -> RecorderTermCfg:
           # Define what data to record during simulation

       @abstractmethod
       def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
           # Compute final metric from recorded data

Each metric consists of a **RecorderTerm** that collects data during simulation and a **MetricBase** implementation that processes the recorded data into meaningful performance indicators.

Data Collection Pipeline
------------------------

**RecorderTerm Components**
   Handle real-time data collection during simulation:

   - **Recording Triggers**: Define when data is captured (pre-reset, post-step, etc.)
   - **Data Extraction**: Access simulation state and extract relevant measurements
   - **Storage Format**: Structure data for efficient storage and retrieval
   - **Episode Tracking**: Organize data by simulation episodes and environments

**Recording Modes**
   Support different data collection patterns:

   - **Pre-Reset Recording**: Capture final episode state before environment reset
   - **Post-Step Recording**: Collect data after each simulation step
   - **Event-Triggered Recording**: Record data when specific conditions occur
   - **Continuous Monitoring**: Track quantities throughout episode duration

Available Metrics
-----------------

Isaac Arena provides several built-in metrics for common evaluation scenarios:

**Success Rate Metric**
   Measures task completion across episodes:

   - **Purpose**: Fundamental measure of task achievement
   - **Data Source**: Environment termination conditions
   - **Computation**: Proportion of episodes ending in success
   - **Use Cases**: All task types requiring binary success evaluation

**Door Moved Rate Metric**
   Evaluates interaction with openable objects:

   - **Purpose**: Measure physical interaction progress with affordances
   - **Data Source**: Object joint positions throughout episodes
   - **Computation**: Proportion of episodes with significant object movement
   - **Use Cases**: Affordance-based tasks (doors, drawers, appliances)

**Object Moved Rate Metric**
   Tracks manipulation of objects in the environment:

   - **Purpose**: Assess robot's ability to influence object states
   - **Data Source**: Object linear velocities during simulation
   - **Computation**: Proportion of episodes with object displacement
   - **Use Cases**: Manipulation tasks, object interaction evaluation

Metric Integration
------------------

Metrics integrate into the Isaac Arena environment system through task definitions:

.. code-block:: python

   class OpenDoorTask(TaskBase):
       def get_metrics(self) -> list[MetricBase]:
           return [
               SuccessRateMetric(),
               DoorMovedRateMetric(
                   self.openable_object,
                   reset_openness=self.reset_openness,
               ),
           ]

Tasks specify which metrics apply to their objectives, enabling automatic performance evaluation across different scenarios and embodiments.

Recording System Integration
----------------------------

The metrics system leverages Isaac Lab's recorder manager for data persistence:

.. code-block:: python

   def metrics_to_recorder_manager_cfg(metrics: list[MetricBase]) -> RecorderManagerBaseCfg:
       """Convert metrics to recorder manager configuration."""
       configclass_fields = []
       for metric in metrics:
           recorder_cfg = metric.get_recorder_term_cfg()
           configclass_fields.append((metric.name, type(recorder_cfg), recorder_cfg))

       recorder_cfg_cls = make_configclass("RecorderManagerCfg", configclass_fields)
       recorder_cfg = recorder_cfg_cls()
       recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
       return recorder_cfg

This automatic configuration generation ensures that all required data is captured without manual recorder setup.

Data Storage and Retrieval
---------------------------

Recorded data is stored in HDF5 format for efficient access and processing:

**Storage Structure**
   - **Episodes**: Data organized by simulation episode
   - **Environments**: Support for parallel environment execution
   - **Time Series**: Sequential data within episodes
   - **Metadata**: Episode information and configuration details

**Data Access Patterns**
   - **Batch Processing**: Load all episodes for metric computation
   - **Streaming Access**: Process large datasets without memory constraints
   - **Selective Loading**: Access specific metric data or episode ranges

Metric Computation Workflow
----------------------------

The system follows a standardized workflow for performance evaluation:

1. **Data Recording**
   RecorderTerms capture relevant simulation data throughout execution.

2. **Episode Completion**
   Data is organized and stored when episodes terminate.

3. **Metric Computation**
   Post-simulation processing computes final performance indicators.

4. **Result Aggregation**
   Multiple metrics are combined into comprehensive evaluation reports.

.. code-block:: python

   def compute_metrics(env: ManagerBasedRLEnv) -> dict[str, float]:
       """Compute all registered metrics from recorded data."""
       dataset_path = get_metric_recorder_dataset_path(env)
       metrics_data = {}

       for metric in env.cfg.metrics:
           recorded_data = get_recorded_metric_data(dataset_path, metric.recorder_term_name)
           metrics_data[metric.name] = metric.compute_metric_from_recording(recorded_data)

       metrics_data["num_episodes"] = get_num_episodes(dataset_path)
       return metrics_data

Performance Indicators
----------------------

Metrics provide various types of performance assessment:

**Binary Success Metrics**
   Simple pass/fail evaluation based on task completion criteria.

**Continuous Progress Metrics**
   Measure partial progress toward objectives (e.g., distance moved, openness achieved).

**Efficiency Metrics**
   Evaluate resource utilization (time to completion, energy consumption).

**Robustness Metrics**
   Assess consistency across different conditions and random seeds.

Creating Custom Metrics
------------------------

New metrics can be created by implementing the two-component pattern:

1. **Define RecorderTerm**
   Specify what data to collect and when:

.. code-block:: python

   class CustomRecorder(RecorderTerm):
       name = "custom_data"

       def record_post_step(self):
           # Extract custom measurements from simulation
           custom_data = self._extract_custom_measurement()
           return self.name, custom_data

2. **Implement MetricBase**
   Define how to compute the final metric:

.. code-block:: python

   class CustomMetric(MetricBase):
       name = "custom_metric"
       recorder_term_name = CustomRecorder.name

       def get_recorder_term_cfg(self) -> RecorderTermCfg:
           return CustomRecorderCfg()

       def compute_metric_from_recording(self, recorded_data: list[np.ndarray]) -> float:
           # Process recorded data into final metric value
           return self._compute_custom_metric(recorded_data)

3. **Register with Tasks**
   Add the metric to relevant task definitions:

.. code-block:: python

   def get_metrics(self) -> list[MetricBase]:
       return [CustomMetric(), SuccessRateMetric()]

Usage Examples
--------------

**Environment Execution with Metrics**:

.. code-block:: python

   from isaac_arena.metrics.metrics import compute_metrics

   # Run environment with metric collection enabled
   env_builder = ArenaEnvBuilder(arena_environment, args)
   env = env_builder.make_registered()

   # Execute episodes
   for episode in range(100):
       obs, _ = env.reset()
       done = False
       while not done:
           actions = policy(obs)
           obs, _, terminated, truncated, _ = env.step(actions)
           done = terminated or truncated

   # Compute final metrics
   metrics_results = compute_metrics(env)
   print(f"Success Rate: {metrics_results['success_rate']:.2%}")
   print(f"Door Moved Rate: {metrics_results['door_moved_rate']:.2%}")

**Task-Specific Metrics**:

.. code-block:: python

   # Pick and place task with object interaction metrics
   task = PickAndPlaceTask(pick_object, destination, background)
   metrics = task.get_metrics()  # [SuccessRateMetric(), ObjectMovedRateMetric()]

   # Door opening task with affordance metrics
   task = OpenDoorTask(microwave, openness_threshold=0.8)
   metrics = task.get_metrics()  # [SuccessRateMetric(), DoorMovedRateMetric()]

Evaluation Workflows
--------------------

The metrics system supports different evaluation patterns:

**Development Iteration**
   Quick metrics computation during development cycles for rapid feedback.

**Benchmarking Studies**
   Comprehensive evaluation across multiple embodiments, scenes, and conditions.

**Performance Monitoring**
   Continuous tracking of system performance across training iterations.

**Comparative Analysis**
   Standardized metrics enable fair comparison between different approaches.

Integration with Environment System
-----------------------------------

Metrics seamlessly integrate with Isaac Arena's environment composition:

**Task Integration**
   Tasks specify relevant metrics for their objectives and constraints.

**Environment Builder**
   Automatically configures recorder managers based on task metrics.

**Configuration Management**
   Metrics configurations merge with other environment settings transparently.

**Data Management**
   Recorded data is automatically organized and made available for analysis.

Benefits and Applications
-------------------------

**Research Applications**
   - Algorithm development and validation
   - Comparative studies across different approaches
   - Performance characterization and analysis

**Development Support**
   - Rapid iteration feedback during development
   - Regression detection and quality assurance
   - Progress tracking toward project objectives

**Benchmarking**
   - Standardized evaluation across different systems
   - Fair comparison between approaches and implementations
   - Performance baseline establishment

**Production Monitoring**
   - System performance tracking in deployment
   - Quality assurance and validation
   - Long-term performance trend analysis

The metrics system in Isaac Arena provides a robust foundation for quantitative evaluation of robot performance. By separating data collection from metric computation and integrating seamlessly with the environment system, it enables comprehensive performance assessment while maintaining flexibility for custom evaluation requirements.
