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


from typing import List

import isaacsim.core.utils.prims as prims_utils
from pxr import Usd, UsdGeom


def set_lens_distortion_properties(
    prim: Usd.Prim,
    distortion_model: str,
    distortion_model_attr: str,
    coefficient_map: list[str],
    coefficients: list[float],
    **kwargs,
) -> None:
    """Sets lens distortion model parameters if camera prim is using lens distortion model."""
    prim.ApplyAPI(f"OmniLensDistortion{distortion_model}API")
    prim.GetAttribute("omni:lensdistortion:model").Set(distortion_model_attr)
    for coefficient_name, coefficient_value in zip(coefficient_map or [], coefficients or []):
        if coefficient_value is None:
            continue
        prim.GetAttribute(f"omni:lensdistortion:{distortion_model_attr}:{coefficient_name}").Set(coefficient_value)
    for attr_name, attr_value in kwargs.items():
        if attr_value is None:
            continue
        tokens = attr_name.split("_")
        updated_attr_name = tokens[0]
        for i in tokens[1:]:
            updated_attr_name += i.capitalize()
        prim.GetAttribute(f"omni:lensdistortion:{distortion_model_attr}:{updated_attr_name}").Set(attr_value)


def set_azure_camera_properties(camera_prim_path: str):
    """Sets Azure camera properties."""
    # Get the prim
    prim = prims_utils.get_prim_at_path(camera_prim_path)

    # Azure camera properties
    width = 1024
    height = 1024
    cx = 522.327
    cy = 512.519
    fx = 504.563
    fy = 504.501

    # Get the camera
    camera = UsdGeom.Camera(prim)

    # Print current camera info
    print(f"focal length: {camera.GetFocalLengthAttr().Get()}")
    print(f"horizontal aperture: {camera.GetHorizontalApertureAttr().Get()}")
    print(f"vertical aperture: {camera.GetVerticalApertureAttr().Get()}")
    print(f"horizontal aperture offset: {camera.GetHorizontalApertureOffsetAttr().Get()}")
    print(f"vertical aperture offset: {camera.GetVerticalApertureOffsetAttr().Get()}")
    print(prim.GetAttribute("omni:lensdistortion:model").Get())

    # Calculate the desired focal length (in mm)
    horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
    desired_focal_length_mm = fx * (horizontal_aperture / width)
    print(f"desired focal length: {desired_focal_length_mm}")

    # Set the desired focal length
    camera.GetFocalLengthAttr().Set(desired_focal_length_mm)

    # Set the lens distortion properties
    OPENCV_PINHOLE_ATTRIBUTE_MAP = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]
    set_lens_distortion_properties(
        prim=prim,
        distortion_model="OpenCvPinhole",
        distortion_model_attr="opencvPinhole",
        coefficient_map=OPENCV_PINHOLE_ATTRIBUTE_MAP,
        coefficients=[
            9.51684,
            4.65586,
            0.000100116,
            4.73081e-05,
            0.167834,
            9.84895,
            7.84502,
            1.066,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
