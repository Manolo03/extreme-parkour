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
import numpy as np

class WfTron1aCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 8
         
        # 3 base ang vel + 2 imu (roll, pitch) +
        # 2 commands related (1 lin_vel_x command + 1 yaw error computed from heading target command) +
        # 6 joint pos (legs only) + 8 joint vel (6 leg + 2 wheel) +
        # 8 action history + 2 contact filt (2 wheels) = 29
        n_proprio = 3 + 2 + 2 + (6+8+8)
        n_scan = 132
        n_priv = 3  # 3 base lin vel
        n_priv_latent = 4 + 1 + 8 + 8 
        history_len = 10
        num_observations = n_proprio + n_scan + n_priv +  n_priv_latent + history_len*n_proprio  #29 + 132 + 3 + 21  + 290  = 475
        
        # Observation index mapping for tron1a (indices may differ from base config due to removed padding)
        # Current proprio structure:
        # 0-2: base_ang_vel
        # 3-4: imu (roll, pitch)
        # 5:   commands (lin_vel_x)
        # 6:   yaw error (delta_yaw)
        # 7-12:  dof_pos (6 leg DOFs)
        # 13-20: dof_vel (all 8 DOFs)
        # 21-28: action_history
        # 29-30: contact_filt
        obs_indices = {
            "base_ang_vel": (0, 3),
            "imu": (3, 2),
            "commands": (5, 1),
            "yaw_error": (6, 1),
            "dof_pos": (7, 6),  
            "dof_vel": (13, 8), 
            "action_history": (21, 8),  
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

    class depth(LeggedRobotCfg.depth):
        """
        Depth camera configuration for RealSense D435.
        
        Coordinate System:
        - Both URDF and Isaac Gym use: x forward, y left, z UP (standard ROS convention)
        - The position is relative to the root body (base_Link) frame
        - Negative z means below the base origin, positive z means above
        """
        use_camera = False  # Set to True to enable depth camera
        
        # Camera position relative to root body (base_Link)
        # From URDF d435_joint: xyz="0.13223 0.0222 -0.26826"
        # z = -0.26826 means camera is 0.26826m below base origin (camera looks down)
        position = [0.13223, 0.0222, -0.26826]  # [x forward, y left, z UP] in meters
        
        # Camera pitch angle (rotation around y-axis)
        # From URDF: rpy="0 1.063778179 0" (pitch = 1.063778179 rad ≈ 60.9°)
        # Positive pitch = camera looks down
        # Range allows domain randomization (±5° variation)
        angle = [55.9, 65.9]  # degrees, positive pitch = down
        
        # Camera update frequency
        # update_interval = 5 means update every 5 simulation steps
        # Lower = more frequent updates (more compute), Higher = less frequent (less compute)
        update_interval = 5  # Update every 5 steps (5 works without retraining, 8 worse)
        
        # Image resolution
        # RealSense D435 depth: 640×480 native, but we use smaller for efficiency
        original = (106, 60)  # Original depth image resolution (width, height) in pixels
        resized = (87, 58)   # Resized resolution for neural network input (width, height)
        
        # Field of view
        # RealSense D435 depth camera: 85.2° × 58° FOV (H × V)
        horizontal_fov = 85.2  # Horizontal FOV in degrees (matches D435 depth FOV)
        #vertical fox computed from horizontal fov and aspect ratio



        # Depth buffer (temporal history)
        buffer_len = 2       # Number of consecutive frames stored (for temporal features)
        
        # Depth clipping planes
        near_clip = 0        # Near clipping plane in meters (closer objects are clipped)
        far_clip = 2         # Far clipping plane in meters (farther objects are clipped)
        
        # Noise and processing
        dis_noise = 0.0      # Distance noise standard deviation (for domain randomization)
        scale = 1           # Depth scaling factor (typically 1.0)
        invert = True       # Whether to invert depth values (True = closer = brighter)
        
        # Terrain generation for camera (if using camera-specific terrain)
        camera_num_envs = 192      # Number of environments with cameras enabled
        camera_terrain_num_rows = 10  # Terrain rows for camera environments
        camera_terrain_num_cols = 20  # Terrain cols for camera environments
    
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        # Use a fixed 10x40 grid of selected sloped-wall terrains (16 robots per tile with 6400 envs)
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        # Selected sloped-wall terrain on a 10x40 grid (6400 envs / 400 tiles = 16 robots per tile)
        selected = True
        terrain_kwargs = {
            "type": "sloped_wall_terrain",
            "terrain_kwargs": {
                "min_height": 0.3,
                "max_height": 1.0,
                "slope_len_range": (1.0, 2.0),
                "platform_len": 0.5,
            }
        }
        max_init_terrain_level = 0  # Not used when curriculum=False
        # terrain_length matches sloped_wall_terrain feature size:
        # start platform (1.5m) + gap (1.0m) + feature (2×slope + 0.5m platform, ~6m worst case) + buffer
        terrain_length = 10.0  # meters (forward direction, x-axis)
        terrain_width = 30.0   # meters (lateral direction, y-axis) - matches visualize_terrain.py
        num_rows = 8  # number of terrain rows (levels)
        num_cols = 8  # number of terrain cols (types)
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8
        
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = False # Disabled: we handle yaw error ourselves via heading_target (fixed per env)
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4
        # curriculum ranges
        class ranges:
            lin_vel_x = [0., 1.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            target_heading = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Easy ranges
        class max_ranges:
            lin_vel_x = [0.3, 0.8] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
            target_heading = [-0, 0]    # min max [rad/s]
            heading = [-np.pi/3, np.pi/3]  # -60° to +60° in radians

        class crclm_incremnt:
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            target_heading = 0.1    # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7


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

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = False
        push_interval_s = 5.0  # Push every 5 seconds (perpendicular to wall)
        max_push_vel_xy = 2.0  # Max push velocity in m/s

    # class rewards(LeggedRobotCfg.rewards):
    #     soft_dof_pos_limit = 0.95
    #     base_height_target = 0.6 + 0.1664
    #     class scales:
    #         # tracking rewards
    #         tracking_goal_vel = 1.5
    #         tracking_yaw = 0.5
    #         # regularization rewards
    #         lin_vel_z = -1.0
    #         ang_vel_xy = -0.05
    #         orientation = -1.
    #         dof_acc = -2.5e-7
    #         collision = -50.
    #         action_rate = -0.1
    #         delta_torques = -1.0e-7
    #         torques = -0.00001
    #         # termination penalty (only for base contact falls)
    #         base_contact_termination = -50.0
    #         #hip_pos = -0.5
    #         #dof_error = -0.04
    #         #feet_stumble = -1
    #         #feet_edge = -1
            
    #     only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    #     tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_vel_limit = 1
    #     soft_torque_limit = 0.4
    #     max_contact_force = 40. # forces above this value are penalized

    class rewards:
        class scales:
            # termination related rewards
            keep_balance = 1.0

            # tracking related rewards
            tracking_lin_vel = 4.0
            tracking_yaw = 2.0
            tracking_lin_vel_pb = 1.0
            tracking_yaw_pb = 0.2

            # regulation related rewards
            #nominal_foot_position = 4.0
            #leg_symmetry = 0.5
            #same_foot_x_position = -50 # 0.5
            #same_foot_z_position = -100
            # lin_vel_z = -0.3
            # ang_vel_xy = -0.3
            torques = -0.0016
            dof_acc = -1.5e-6
            action_rate = -0.05
            #dof_pos_limits = -2.0
            collision = -50
            action_smooth = -0.03
            orientation = -12.0
            #feet_distance = -100
            base_height = -20

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.6 + 0.1664
        feet_height_target = 0.10
        min_feet_distance = 0.32
        max_feet_distance = 0.35
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005
        feet_height_tracking_sigma = 0.005

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

