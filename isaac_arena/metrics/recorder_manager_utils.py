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

from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaac_arena.metrics.metric_base import MetricBase
from isaac_arena.utils.configclass import make_configclass


def metrics_to_recorder_manager_cfg(metrics: list[MetricBase] | None) -> RecorderManagerBaseCfg | None:
    if metrics is None:
        return None
    configclass_fields: list[tuple[str, type, object]] = []
    for metric in metrics:
        configclass_fields.append((metric.name, type(metric.get_recorder_term_cfg()), metric.get_recorder_term_cfg()))
    recorder_cfg_cls = make_configclass("RecorderManagerCfg", configclass_fields, bases=(RecorderManagerBaseCfg,))
    recorder_cfg = recorder_cfg_cls()
    recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    return recorder_cfg
