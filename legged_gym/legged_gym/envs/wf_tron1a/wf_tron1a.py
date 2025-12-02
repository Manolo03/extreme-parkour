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
from .wf_tron1a_config import WfTron1aCfg
import torch, torchvision


class WfTron1a(LeggedRobot):
    """
    Wheeled-foot bipedal robot (Tron1a) environment.
    Inherits from LeggedRobot and overrides methods specific to bipedal wheeled locomotion.
    """
    cfg: WfTron1aCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def reindex(self, vec):
        """
        Reindex DOF vectors from default order to tron1a-specific order.
        Tron1a has 8 DOFs: [abad_L, hip_L, knee_L, foot_L, abad_R, hip_R, knee_R, foot_R, wheel_L, wheel_R]
        This method should map the default order to the desired order for observations/actions.
        
        Args:
            vec: Tensor of shape (num_envs, num_dofs) to reindex
            
        Returns:
            Reindexed tensor
        """
        # TODO: Implement reindexing based on your DOF order
        # For now, return as-is. You'll need to determine the correct permutation
        # based on how your URDF defines joint order vs. how you want them ordered
        return vec
    
    def reindex_feet(self, vec):
        """
        Reindex contact/feet vectors for bipedal robot.
        Tron1a has 2 contact points (wheels/feet) instead of 4.
        
        Args:
            vec: Tensor of shape (num_envs, num_feet) to reindex
            
        Returns:
            Reindexed tensor for 2 feet/wheels
        """
        # TODO: Implement reindexing for 2 contact points
        # For now, return first 2 elements if vec has 4, or return as-is if already 2
        if vec.shape[1] == 4:
            return vec[:, [0, 1]]  # Take first 2 (adjust indices as needed)
        return vec
    
    def check_termination(self):
        """
        Check if environments need to be reset.
        Override to add bipedal-specific termination conditions.
        """
        super().check_termination()
        # Add any tron1a-specific termination conditions here
        # e.g., different roll/pitch thresholds, wheel slip conditions, etc.
    
    def compute_reward(self):
        """
        Compute rewards.
        Override to customize reward structure for bipedal wheeled locomotion.
        """
        super().compute_reward()
        # Add any tron1a-specific reward modifications here
        # The base class already calls all reward functions from config,
        # but you can add additional logic or modify existing rewards
    
    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw
        obs_buf = torch.cat((#skill_vector, 
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            self.delta_yaw[:, None],
                            self.delta_next_yaw[:, None],
                            self.commands[:, 0:1],  #[1,1]
                            (self.env_class != 17).float()[:, None], 
                            (self.env_class == 17).float()[:, None],
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt.float()-0.5),
                            ),dim=-1)
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
        # Mask yaw in proprioceptive history using config-based indices
        start_idx, length = self.cfg.env.obs_indices.get("yaw")
        obs_buf[:, start_idx:start_idx + length] = 0
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

