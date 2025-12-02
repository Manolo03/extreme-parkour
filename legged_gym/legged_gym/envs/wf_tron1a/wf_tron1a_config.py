# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class WfTron1aCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 8
         
        n_proprio = 3 + 2 + 2 + 3 + (6+8+8) + 2 # 3 base ang vel + 2 imu (roll, pitch) + 2 yaw deltas (current and next) + 3 commmand and terrain flags + 6 joint pos + 8 joint vel + 8 action history + 2 contact filt (2 wheels)
        n_scan = 132
        n_priv = 3  # 3 base lin vel
        n_priv_latent = 4 + 1 + 8 + 8 
        history_len = 10
        num_observations = n_proprio + n_scan + n_priv +  n_priv_latent + history_len*n_proprio  #34 + 132 + 3 + 21  + 340  = 530
        
        # Observation index mapping for tron1a (indices may differ from base config due to removed padding)
        # TODO: Update these indices to match your actual compute_observations() structure
        # Current structure (with removed command padding): 
        # 0-2: base_ang_vel, 3-4: imu, 5: padding_yaw, 6: delta_yaw, 7: delta_next_yaw,
        # 8: command, 9-10: env_class_flags, 11-22: dof_pos (reindexed), 23-34: dof_vel, 
        # 35-46: action_history, 47-48: contact_filt
        obs_indices = {
            "base_ang_vel": (0, 3),
            "imu": (3, 2),
            "yaw": (5, 2),
            "command": (7, 1),
            "env_class_flags": (8, 2),
            "dof_pos": (10, 6),  # Will be reindexed to 8
            "dof_vel": (16, 8),  # Will be reindexed to 8
            "action_history": (24, 8),  # Will be reindexed to 8
            "contact_filt": (32, 2),  # Will be reindexed to 2
        }
        
        
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8 + 0.1664]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {
            "abad_L_Joint": 42.0,
            "hip_L_Joint": 42.0,
            "knee_L_Joint": 42.0,
            "abad_R_Joint": 42.0,
            "hip_R_Joint": 42.0,
            "knee_R_Joint": 42.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,
            "hip_L_Joint": 2.5,
            "knee_L_Joint": 2.5,
            "abad_R_Joint": 2.5,
            "hip_R_Joint": 2.5,
            "knee_R_Joint": 2.5,
            "wheel_L_Joint": 0.8,
            "wheel_R_Joint": 0.8,
        }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wf_tron1a/urdf/robot.urdf"
        foot_name = "wheel"
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.6 + 0.1664


class WfTron1aCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "wf_tron1a"

