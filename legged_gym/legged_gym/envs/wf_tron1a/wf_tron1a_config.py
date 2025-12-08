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
         
        n_proprio = 3 + 2 + 2 + 3 + (6+8+8) + 2 # 3 base ang vel + 2 imu (roll, pitch) + 2 yaw deltas (current and next) + 3 command and terrain flags + 6 joint pos + 8 joint vel (6 leg + 2 wheel) + 8 action history + 2 contact filt (2 wheels) = 34
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
            "dof_pos": (10, 6),  
            "dof_vel": (16, 8), 
            "action_history": (24, 8),  
            "contact_filt": (32, 2),  
        }
        
        
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8 + 0.1664]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {
            "abad_L_Joint": 42.0,
            "hip_L_Joint": 42.0,
            "knee_L_Joint": 42.0,
            "wheel_L_Joint": 0.0,
            "abad_R_Joint": 42.0,
            "hip_R_Joint": 42.0,
            "knee_R_Joint": 42.0,            
            "wheel_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,
            "hip_L_Joint": 2.5,
            "knee_L_Joint": 2.5,
            "wheel_L_Joint": 0.8,
            "abad_R_Joint": 2.5,
            "hip_R_Joint": 2.5,
            "knee_R_Joint": 2.5,            
            "wheel_R_Joint": 0.8,
        }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wf_tron1a/urdf/robot.urdf"
        foot_name = "wheel"
        foot_radius = 0.127
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # Keep visual frames as-authored and avoid collapsing fixed joints to preserve assembly
        flip_visual_attachments = False
        collapse_fixed_joints = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.6 + 0.1664
        class scales:
            # tracking rewards
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            #hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1
            feet_edge = -1
            
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        max_contact_force = 40. # forces above this value are penalized


class WfTron1aCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "wf_tron1a"

    class depth_encoder:
        if_depth = WfTron1aCfg.depth.use_camera
        depth_shape = WfTron1aCfg.depth.resized
        buffer_len = WfTron1aCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = WfTron1aCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = WfTron1aCfg.env.n_priv
        num_prop = WfTron1aCfg.env.n_proprio
        num_scan = WfTron1aCfg.env.n_scan

