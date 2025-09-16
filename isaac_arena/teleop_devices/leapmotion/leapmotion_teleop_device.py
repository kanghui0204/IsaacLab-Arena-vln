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

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaac_arena.embodiments.g1.robot_model import RobotModel, ReducedRobotModel
from isaac_arena.teleop_devices.leapmotion.leapmotion_streamer import LeapMotionStreamer

from isaac_arena.teleop_devices.leapmotion.preprocesors.fingers import FingersPreProcessor
from isaac_arena.teleop_devices.leapmotion.preprocesors.wrists import WristsPreProcessor

class LeapmotionTeleopDevice:
    """
    Robot-agnostic teleop retargeting inverse kinematics code.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        left_hand_ik_solver,
        right_hand_ik_solver,
        body_control_device: str,
        hand_control_device: str | None = None,
        enable_real_device=True,
        body_active_joint_groups: list[str] | None = None,
    ):
        # initialize the body
        if body_active_joint_groups is not None:
            self.body = ReducedRobotModel.from_active_groups(robot_model, body_active_joint_groups)
            self.full_robot = self.body.full_robot
            self.using_reduced_robot_model = True
        else:
            self.body = robot_model
            self.full_robot = self.body
            self.using_reduced_robot_model = False

        # initialize pre_proccessors
        self.body_control_device = body_control_device
        if body_control_device:
            self.body_pre_processor = WristsPreProcessor()
            self.body_pre_processor.register(self.body)
        else:
            self.body_pre_processor = None

        # initialize hand pre-processors and post-processors
        self.hand_control_device = hand_control_device
        if hand_control_device:
            self.left_hand_pre_processor = FingersPreProcessor(side="left")
            self.right_hand_pre_processor = FingersPreProcessor(side="right")

        else:
            self.left_hand_pre_processor = None
            self.right_hand_pre_processor = None


        # enable real robot and devices
        self.enable_real_device = enable_real_device
        if self.enable_real_device:

            self.body_streamer = LeapMotionStreamer()
            self.body_streamer.start_streaming()

            self.hand_streamer = None

        self.raw_data = {}

    def get_streamer_raw_data(self):
        # print("+++++++++++get_streamer_raw++++++++")
        if self.body_streamer:
            body_data_ = self.body_streamer.get()
            self.raw_data.update(body_data_)
        if self.hand_streamer:
            self.raw_data.update(self.hand_streamer.get())
        return self.raw_data

    def calibrate(self):
        """Calibrate the pre-processors."""
        assert (
            self.body_streamer or self.hand_streamer
        ), "Real device is enabled, but no streamer is initialized."
        raw_data = self.get_streamer_raw_data()

        if self.body_pre_processor:
            self.body_pre_processor.calibrate(raw_data)
        if self.left_hand_pre_processor:
            self.left_hand_pre_processor.calibrate(raw_data)
        if self.right_hand_pre_processor:
            self.right_hand_pre_processor.calibrate(raw_data)

    def pre_process(self, raw_data):
        """Pre-process the raw data."""
        assert (
            self.body_pre_processor or self.left_hand_pre_processor or self.right_hand_pre_processor
        ), "Pre-processors are not initialized."
        if self.body_pre_processor:
            body_data = self.body_pre_processor(raw_data)
            if self.left_hand_pre_processor and self.right_hand_pre_processor:
                left_hand_data = self.left_hand_pre_processor(raw_data)
                right_hand_data = self.right_hand_pre_processor(raw_data)
                return body_data, left_hand_data, right_hand_data
            else:
                return body_data, None, None
        else:  # only hands
            if self.left_hand_pre_processor and self.right_hand_pre_processor:
                left_hand_data = self.left_hand_pre_processor(raw_data)
                right_hand_data = self.right_hand_pre_processor(raw_data)
                return None, left_hand_data, right_hand_data

    def get_leapmotion_action(self):
        raw_action = self.get_streamer_raw_data()
        body_data, left_hand_data, right_hand_data = self.pre_process(raw_action)

        # Original variables
        left_wrist_action = np.array(body_data['left_wrist_yaw_link'])
        right_wrist_action = np.array(body_data['right_wrist_yaw_link'])
        left_fingers_position = np.array(left_hand_data['position'])
        right_fingers_position = np.array(right_hand_data['position'])

        # Extract position and quaternion from left wrist 4x4 pose matrix
        left_wrist_pos = torch.from_numpy(left_wrist_action[:3, 3])
        left_wrist_rot_matrix = left_wrist_action[:3, :3]
        left_wrist_quat = R.from_matrix(left_wrist_rot_matrix).as_quat()
        
        # Extract position and quaternion from right wrist 4x4 pose matrix
        right_wrist_pos = torch.from_numpy(right_wrist_action[:3, 3])
        right_wrist_rot_matrix = right_wrist_action[:3, :3]
        right_wrist_quat = R.from_matrix(right_wrist_rot_matrix).as_quat()

        # Convert quaternions from (x, y, z, w) to IsaacLab (w, x, y, z) format
        left_wrist_quat_wxyz = torch.from_numpy(np.roll(left_wrist_quat, 1))
        right_wrist_quat_wxyz = torch.from_numpy(np.roll(right_wrist_quat, 1))

        # Flattened variables
        left_fingers_position_flat = torch.from_numpy(left_fingers_position.flatten())
        right_fingers_position_flat = torch.from_numpy(right_fingers_position.flatten())

        # Assemble into upper body env action
        upperbody_action = torch.cat([left_wrist_pos,
                            left_wrist_quat_wxyz,
                            right_wrist_pos,
                            right_wrist_quat_wxyz,
                            left_fingers_position_flat,
                            right_fingers_position_flat])

        return upperbody_action