# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py
import numpy as np
import pathlib

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.datasets import HDF5DatasetFileHandler


def compute_metrics(env: ManagerBasedRLEnv) -> dict[str, float]:
    assert hasattr(env.cfg, "metrics")
    # Get the path where the recorded data is stored
    dataset_path = get_metric_recorder_dataset_path(env)
    metrics_data = {}
    for metric in env.cfg.metrics:
        recorded_metric_data = get_recorded_metric_data(dataset_path, metric.recorder_term_name)
        metrics_data[metric.name] = metric.compute_metric_from_recording(recorded_metric_data)
    metrics_data["num_episodes"] = get_num_episodes(dataset_path)
    return metrics_data


def get_recorded_metric_data(dataset_path: pathlib.Path, recorder_term_name: str) -> list[np.ndarray]:
    """Gets the recorded metric data for a given metric name."""
    recorded_metric_data_per_demo: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as f:
        demos = f["data"]
        for demo in demos:
            recorded_metric_data_per_demo.append(demos[demo][recorder_term_name][:])
    return recorded_metric_data_per_demo


def get_num_episodes(dataset_path: pathlib.Path) -> int:
    with h5py.File(dataset_path, "r") as f:
        return len(f["data"])


def get_metric_recorder_dataset_path(env: ManagerBasedRLEnv) -> pathlib.Path:
    assert env.cfg.recorders.dataset_file_handler_class_type == HDF5DatasetFileHandler
    return pathlib.Path(env.cfg.recorders.dataset_export_dir_path) / pathlib.Path(
        env.cfg.recorders.dataset_filename + ".hdf5"
    )
