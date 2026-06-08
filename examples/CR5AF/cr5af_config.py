# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modality configuration for Doosan CR5AF 6-axis arm with TopHand dexterous hand.

Single-arm setup:
- 2 cameras: D405 (hand_view, wrist-mounted) + D455 (table_view, fixed third-person)
- 9D EEF state: xyz(3) + rotation_6d(6) from TCP pose + quaternion
- 6D joint state: QActual[6] joint angles
- 1D gripper state: 0.0=closed (holding), 1.0=open (release)
- 40-step action horizon (5 seconds at 8Hz control)
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


cr5af_config = {
    # Video: 2 frames at delta [-20, 0] (captures ~2.5s of history at 8Hz)
    "video": ModalityConfig(
        delta_indices=[-20, 0],
        modality_keys=[
            "hand_view",   # D405 wrist-mounted
            "table_view",  # D455 fixed third-person
        ],
    ),
    # State: current proprioceptive reading
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "eef_9d",       # TCP xyz(3) + rotation_6d(6)
            "joint_pos",    # QActual[6] joint angles
            "gripper_pos",  # 0.0=close, 1.0=open
        ],
    ),
    # Action: 40-step prediction horizon at 8Hz control (5 seconds)
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "eef_9d",
            "joint_pos",
            "gripper_pos",
        ],
        action_configs=[
            # EEF: relative delta from current TCP pose (better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_9d",
            ),
            # Joints: relative delta from current joint angles
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Gripper: absolute target (binary open/close works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(cr5af_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
