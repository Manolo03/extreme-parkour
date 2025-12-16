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

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .wf_tron1a_config import WfTron1aCfg
from legged_gym.envs.base.legged_robot import euler_from_quaternion
import torch, torchvision
import numpy as np
import os
import cv2
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from tqdm import tqdm


class WfTron1a(LeggedRobot):
    """
    Wheeled-foot bipedal robot (Tron1a) environment.
    Inherits from LeggedRobot and overrides methods specific to bipedal wheeled locomotion.
    """
    cfg: WfTron1aCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def _get_non_wheel_dof_indices(self):
        """Helper function to get indices of DOFs that are not wheels."""
        non_wheel_indices = []
        for i, name in enumerate(self.dof_names):
            if "wheel" not in name.lower():
                non_wheel_indices.append(i)
        return non_wheel_indices
    
    def _get_non_wheel_dof_mask(self):
        """Helper function to get a boolean mask for non-wheel DOFs."""
        wheel_indices = []
        for i, name in enumerate(self.dof_names):
            if "wheel" in name.lower():
                wheel_indices.append(i)
        mask = torch.ones(self.num_dofs, dtype=torch.bool, device=self.device)
        if wheel_indices:
            mask[wheel_indices] = False
        return mask
    
    def _create_envs(self):
        """Override to skip quadruped-specific force sensors and joint indexing."""
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        # Skip quadruped-specific force sensor creation - tron1a doesn't have FR_foot, etc.
        # (Force sensors are optional and not used in rewards anyway)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # Ensure goal buffers exist even when terrain has no goals (e.g., plane)
        if not hasattr(self, "cur_goal_idx"):
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        if not hasattr(self, "env_goals"):
            # Dummy goals: origin for current and future
            num_goals = getattr(self.cfg.terrain, "num_goals", 1)
            num_future = getattr(self.cfg.env, "num_future_goal_obs", 1)
            total = num_goals + num_future
            self.env_goals = torch.zeros(self.num_envs, total, 3, device=self.device, requires_grad=False)
        if not hasattr(self, "cur_goals"):
            self.cur_goals = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not hasattr(self, "next_goals"):
            self.next_goals = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            
            self.attach_camera(i, env_handle, anymal_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # Skip quadruped-specific joint indexing (hip/thigh/calf) - not needed for tron1a
        # Create dummy tensors to avoid AttributeError if something tries to access them
        self.hip_indices = torch.zeros(0, dtype=torch.long, device=self.device, requires_grad=False)
        self.thigh_indices = torch.zeros(0, dtype=torch.long, device=self.device, requires_grad=False)
        self.calf_indices = torch.zeros(0, dtype=torch.long, device=self.device, requires_grad=False)
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # Skip force sensors - tron1a doesn't use them
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Skip force sensor refresh - tron1a doesn't use them
            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # Skip force_sensor_tensor - not used for tron1a (create dummy to avoid AttributeError)
        n_feet = len(self.feet_indices) if hasattr(self, 'feet_indices') else 2
        self.force_sensor_tensor = torch.zeros(self.num_envs, n_feet, 6, device=self.device, dtype=torch.float)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_actions, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)

    def _reset_dofs(self, env_ids):
        """Override DOF reset to use exactly the default joint angles (no randomization).

        This keeps tron1a's joints at the configuration specified in
        WfTron1aCfg.init_state.default_joint_angles when environments are reset.
        """
        # Positions: copy default_dof_pos_all for the selected envs
        self.dof_pos[env_ids] = self.default_dof_pos_all[env_ids]
        # Velocities: zero
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )



    def post_physics_step(self):
        """Override to skip force sensor refresh (tron1a doesn't use force sensors)."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Skip force sensor refresh - tron1a doesn't use them
        
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        
        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            self._draw_goals()
            self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

    def check_termination(self):
        """Check if environments need to be reset, including termination contacts."""
        # Call parent method to get standard terminations (roll, pitch, height, timeout)
        super().check_termination()
        
        # Add termination contact check (base class doesn't implement this!)
        if hasattr(self, 'termination_contact_indices') and len(self.termination_contact_indices) > 0:
            # Check if any termination contact bodies are touching the ground
            termination_contacts = torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0.1
            termination_contact_cutoff = torch.any(termination_contacts, dim=-1)
            self.reset_buf |= termination_contact_cutoff

    def reindex(self, vec):
        return vec  # if your URDF order is already what you want

    def _compute_torques(self, actions):
        """
        Tron1a-specific torque computation copied from the legacy wheelfoot controller.
        Maps an 8D action into 10 DOF torques:
        - Leg joints get position targets (PD position control)
        - Wheel joints get velocity targets (PD velocity control)
        """
        # Position commands for leg joints (expand 8 actions to 10 DOFs)
        pos_scale = getattr(self.cfg.control, "action_scale_pos", self.cfg.control.action_scale)
        vel_scale = getattr(self.cfg.control, "action_scale_vel", self.cfg.control.action_scale)
        torques_scale = getattr(self.cfg.control, "torques_scale", 1.0)

        pos_action = (
            torch.cat(
                (
                    actions[:, 0:3], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                    actions[:, 4:7], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                ),
                axis=1,
            )
            * pos_scale
        )
        # Velocity commands for wheel joints
        vel_action = (
            torch.cat(
                (
                    torch.zeros_like(actions[:, 0:3]), actions[:, 3].view(self.num_envs, 1),
                    torch.zeros_like(actions[:, 0:3]), actions[:, 7].view(self.num_envs, 1),
                ),
                axis=1,
            )
            * vel_scale
        )
        # PD controller (position + velocity components)
        torques = self.p_gains * (pos_action + self.default_dof_pos_all - self.dof_pos) + \
                  self.d_gains * (vel_action - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques * torques_scale

    def _draw_feet(self):
        """
        Overridden feet/wheel drawing for tron1a.
        The base implementation assumes 4 feet; tron1a has 2 wheels.
        We reuse the same visualization idea but iterate over the actual number
        of contact bodies in `self.feet_indices`.
        """
        if hasattr(self, "feet_at_edge"):
            non_edge_geom = gymutil.WireframeSphereGeometry(
                0.02, 16, 16, None, color=(0, 1, 0)
            )
            edge_geom = gymutil.WireframeSphereGeometry(
                0.02, 16, 16, None, color=(1, 0, 0)
            )

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            n_feet = self.feet_indices.shape[0]
            for i in range(n_feet):
                pose = gymapi.Transform(
                    gymapi.Vec3(
                        feet_pos[self.lookat_id, i, 0],
                        feet_pos[self.lookat_id, i, 1],
                        feet_pos[self.lookat_id, i, 2],
                    ),
                    r=None,
                )
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(
                        edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
                    )
                else:
                    gymutil.draw_lines(
                        non_edge_geom,
                        self.gym,
                        self.viewer,
                        self.envs[self.lookat_id],
                        pose,
                    )

    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw

        # Filter DOF positions to exclude wheels (6 leg DOFs only)
        # Keep all DOF velocities (all 8 DOFs including wheels)
        non_wheel_mask = self._get_non_wheel_dof_mask()
        dof_pos = self.dof_pos[:, non_wheel_mask]
        default_dof_pos = self.default_dof_pos_all[:, non_wheel_mask]
        
        obs_buf = torch.cat((#skill_vector, 
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            self.delta_yaw[:, None],
                            self.delta_next_yaw[:, None],
                            self.commands[:, 0:1],  #[1,1]
                            (self.env_class != 17).float()[:, None], 
                            (self.env_class == 17).float()[:, None],
                            ((dof_pos - default_dof_pos) * self.obs_scales.dof_pos),
                            (self.dof_vel * self.obs_scales.dof_vel),
                            (self.action_history_buf[:, -1]),
                            (self.contact_filt.float()-0.5),
                            ),dim=-1)
        
        # Debug: verify observation size
        if self.common_step_counter == 1:
            print(f"WfTron1a.compute_observations: obs_buf.shape = {obs_buf.shape}, expected n_proprio = {self.cfg.env.n_proprio}")
            print(f"  dof_pos.shape = {dof_pos.shape}, dof_vel.shape = {self.dof_vel.shape}")
            print(f"  obs_history_buf.shape = {self.obs_history_buf.shape}, expected (num_envs, {self.cfg.env.history_len}, {self.cfg.env.n_proprio})")
        
        priv_explicit = (self.base_lin_vel * self.obs_scales.lin_vel)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        # Debug: check final observation size
        if self.common_step_counter == 1:
            print(f"  Final self.obs_buf.shape = {self.obs_buf.shape}, first 34 dims shape = {self.obs_buf[:, :34].shape}")
        # Mask yaw in proprioceptive history using config-based indices
        start_idx, length = self.cfg.env.obs_indices.get("yaw")
        obs_buf[:, start_idx:start_idx + length] = 0 #is it useful to mask yaw error in history ?
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
    
    ################## parkour rewards ##################

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        return rew
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        rew[self.env_class != 17] *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew[self.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        """Penalize DOF positions deviating from default, excluding wheel joints."""
        # Use helper function to get non-wheel DOF mask
        non_wheel_mask = self._get_non_wheel_dof_mask()
        
        # Only penalize non-wheel DOFs
        leg_dof_pos = self.dof_pos[:, non_wheel_mask]
        leg_default_dof_pos = self.default_dof_pos[:, non_wheel_mask]
        dof_error = torch.sum(torch.square(leg_dof_pos - leg_default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        """Penalize wheels/feet at terrain edges, checking forward direction with radius offset."""
        # Skip if terrain doesn't exist or required attributes are missing (e.g., plane terrain)
        if not hasattr(self, 'terrain') or self.terrain is None or \
           not hasattr(self, 'x_edge_mask') or not hasattr(self, 'terrain_levels'):
            if not hasattr(self, 'feet_at_edge'):
                self.feet_at_edge = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool, device=self.device)
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # Get forward direction from robot heading
        forward_vec = torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], dim=1)  # (num_envs, 2)
        
        # Offset wheel center forward by radius to check forward edge
        foot_radius_pixels = self.cfg.asset.foot_radius / self.cfg.terrain.horizontal_scale
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale)  # (num_envs, n_feet, 2)
        # Offset forward by wheel radius
        forward_offset = forward_vec.unsqueeze(1) * foot_radius_pixels  # (num_envs, 1, 2)
        check_pos_xy = feet_pos_xy + forward_offset  # (num_envs, n_feet, 2)
        
        check_pos_xy = check_pos_xy.round().long()
        check_pos_xy[..., 0] = torch.clip(check_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        check_pos_xy[..., 1] = torch.clip(check_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[check_pos_xy[..., 0], check_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
