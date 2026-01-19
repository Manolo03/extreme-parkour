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

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        
        # Add spacing between terrains on x-axis (2m buffer)
        self.terrain_spacing = self.env_length + 2.0  # 12m total spacing per terrain
        self.spacing_pixels = int(2.0 / cfg.horizontal_scale)  # 2m buffer in pixels

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        # Account for spacing between terrains on x-axis
        self.tot_rows = int(cfg.num_rows * (self.length_per_env_pixels + self.spacing_pixels)) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
            # self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows-1)
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        terrain_params = self.cfg.terrain_kwargs.pop('terrain_kwargs', {})
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.length_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **terrain_params)
            # Set idx attribute (required by add_terrain_to_map)
            # Use 21 as a special index for custom/selected terrains
            if not hasattr(terrain, 'idx'):
                terrain.idx = 21
            # Ensure goals attribute exists (some terrain functions may not set it)
            if not hasattr(terrain, 'goals'):
                terrain.goals = np.zeros((self.num_goals, 2))
            else:
                # If the terrain function returned a different number of goals,
                # pad or truncate to match self.num_goals to avoid shape errors.
                g = np.asarray(terrain.goals)
                if g.ndim == 1:
                    g = g.reshape(1, -1)
                if g.shape[0] < self.num_goals:
                    pad = np.zeros((self.num_goals - g.shape[0], g.shape[1]), dtype=g.dtype)
                    pad[:] = g[-1]  # repeat last goal
                    g = np.concatenate([g, pad], axis=0)
                elif g.shape[0] > self.num_goals:
                    g = g[:self.num_goals]
                terrain.goals = g[:, :2]
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0]/ 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # self.add_roughness(terrain)
        elif choice < self.proportions[2]:
            idx = 2
            if choice<self.proportions[1]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            idx = 4
            if choice<self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2*difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(terrain, stone_size=1.5-0.2*difficulty, stone_distance=0.0+0.4*difficulty, max_height=0.2*difficulty, platform_size=1.2)
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            idx = 9
            self.add_roughness(terrain)
            # pass
        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty )
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            x_range = [-0.1, 0.1+0.3*difficulty]  # offset to stone_len
            y_range = [0.2, 0.3+0.1*difficulty]
            stone_len = [0.9 - 0.3*difficulty, 1 - 0.2*difficulty]#2 * round((0.6) / 2.0, 1)
            incline_height = 0.25*difficulty
            last_incline_height = incline_height + 0.1 - 0.1*difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range, 
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0, 
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            idx = 15
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[15]:
            idx = 16
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.25*difficulty],
                                   pad_height=0,
                                   x_range=[1.2, 2.2],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.4, 0.8],
                                   )
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[16]:
            idx = 17
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.15*difficulty],
                                   pad_height=0,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   step_height=0.1 + 0.35*difficulty,
                                   x_range=[0.3,1.5],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.5, 1],
                                   pad_height=0,
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            idx = 19
            parkour_gap_terrain(terrain,
                                num_gaps=self.num_goals - 2,
                                gap_size=0.1 + 0.7 * difficulty,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2],
                                # flat=True
                                )
            self.add_roughness(terrain)
        elif choice < self.proportions[19]:
            idx = 20
            demo_terrain(terrain)
            self.add_roughness(terrain)
        # np.set_printoptions(precision=2)
        # print(np.array(self.proportions), choice)
        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # Separate terrains on x-axis to prevent robots from getting stuck on adjacent terrain walls
        # Add spacing between terrains: each terrain gets its own space with buffer
        # map coordinate system - add spacing between terrains on x-axis
        start_x = self.border + i * (self.length_per_env_pixels + self.spacing_pixels)
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        
        # Ensure we don't exceed height field bounds
        end_x = min(end_x, self.height_field_raw.shape[0])
        end_y = min(end_y, self.height_field_raw.shape[1])
        
        if start_x < self.height_field_raw.shape[0] and start_y < self.height_field_raw.shape[1]:
            terrain_end_x = min(end_x - start_x, terrain.height_field_raw.shape[0])
            terrain_end_y = min(end_y - start_y, terrain.height_field_raw.shape[1])
            self.height_field_raw[start_x:start_x + terrain_end_x, start_y:start_y + terrain_end_y] = \
                terrain.height_field_raw[:terrain_end_x, :terrain_end_y]

        # Set environment origin: maintain same absolute position as before (ignoring border for origin calc)
        # Old: env_origin_x = i * env_length + 1.0
        # New: use same pattern with terrain_spacing to keep wall at same distance from spawn
        # Note: border_size affects terrain mesh but origins use this simpler calculation
        env_origin_x = i * self.terrain_spacing + 1.0  # Same as old: i*10+1.0, now i*12+1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        # Goals are relative to terrain start in world coordinates
        # Terrain starts at: border_size + i * terrain_spacing in world coords
        terrain_start_world_x = self.cfg.border_size + i * self.terrain_spacing
        self.goals[i, j, :, :2] = terrain.goals + [terrain_start_world_x, j * self.env_width]
        # self.env_slope_vec[i, j] = terrain.slope_vector

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -400
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty#10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)
    
    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)
    
    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_step_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                        #    x_range=[1.5, 2.4],
                            x_range=[0.2, 0.4],
                           y_range=[-0.15, 0.15],
                           half_valid_width=[0.45, 0.5],
                           step_height = 0.2,
                           pad_width=0.1,
                           pad_height=0.5):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def bump_field_terrain(terrain, 
                       bump_size_range=[0.2, 0.4],
                       bump_spacing=0.35,
                       max_height=0.2,
                       platform_len=1.5,
                       platform_height=0.):
    """
    Generate a dense bump field (uniform grid of bumps) across the terrain.
    
    Parameters:
        terrain (SubTerrain): the terrain object to modify
        bump_size_range (list): [min, max] radius of bumps in meters
        bump_spacing (float): center-to-center spacing between bumps in meters
        max_height (float): maximum height of bumps (similar to stepping_stones) [meters]
        platform_len (float): length of starting platform [meters]
        platform_height (float): height of starting platform [meters]
    Returns:
        terrain (SubTerrain): updated terrain with goals set
    """
    mid_y = terrain.length // 2  # lateral center
    
    # Convert to pixel units
    max_height_int = int(max_height / terrain.vertical_scale)
    platform_len_px = max(1, round(platform_len / terrain.horizontal_scale))
    platform_height_px = round(platform_height / terrain.vertical_scale)
    bump_spacing_px = round(bump_spacing / terrain.horizontal_scale)
    
    # Initialize terrain to ground level
    terrain.height_field_raw[:, :] = 0
    
    # Create starting platform centered around the environment origin (tile center)
    # height_field_raw is indexed as [x (forward), y (lateral)]
    mid_x = terrain.width // 2
    x_start = max(0, mid_x - platform_len_px // 2)
    x_end = min(terrain.width, x_start + platform_len_px)
    terrain.height_field_raw[x_start:x_end, :] = platform_height_px
    
    # Create a dense grid of bumps across the entire terrain (after platform)
    # Use coordinate grids for efficient computation
    x_coords, y_coords = np.ogrid[:terrain.width, :terrain.length]
    
    # Create bump centers in a grid pattern
    bump_centers_x = []
    bump_centers_y = []
    
    # Start bumps after the platform end
    start_x = x_end
    # Create grid of bumps
    x_pos = start_x
    while x_pos < terrain.width:
        y_pos = bump_spacing_px // 2  # Start slightly offset from edge
        while y_pos < terrain.length:
            bump_centers_x.append(x_pos)
            bump_centers_y.append(y_pos)
            y_pos += bump_spacing_px
        x_pos += bump_spacing_px
    
    # Create all bumps
    for bump_x, bump_y in zip(bump_centers_x, bump_centers_y):
        # Random bump size and height for variety
        bump_radius = np.random.uniform(bump_size_range[0], bump_size_range[1])
        bump_radius_px = round(bump_radius / terrain.horizontal_scale)
        bump_height = np.random.randint(0, max_height_int + 1)
        
        # Calculate distance from this bump center
        x_dist = x_coords - bump_x
        y_dist = y_coords - bump_y
        distance_squared = x_dist**2 + y_dist**2
        
        # Create bump shape (circular base with height falloff)
        bump_mask = distance_squared <= bump_radius_px**2
        
        # Create height profile: higher at center, lower at edges
        # Use a smooth falloff (cosine-like) for more natural bumps
        normalized_dist = np.sqrt(distance_squared) / bump_radius_px
        normalized_dist = np.clip(normalized_dist, 0, 1)
        # Cosine falloff: height = max_height * (1 - normalized_dist^2)
        height_profile = bump_height * (1 - normalized_dist**2)
        height_profile = height_profile.astype(np.int16)
        
        # Apply bump height only where mask is True (use maximum to handle overlapping bumps)
        terrain.height_field_raw[bump_mask] = np.maximum(
            terrain.height_field_raw[bump_mask], 
            height_profile[bump_mask]
        )
    
    # Add some roughness to the surface (like other terrains)
    # Small random height variations across the entire terrain
    roughness = np.random.randint(-1, 2, size=terrain.height_field_raw.shape, dtype=np.int16)
    terrain.height_field_raw = np.clip(terrain.height_field_raw + roughness, 0, None)
    
    # Set simple goals: start, middle, end
    goals = np.zeros((3, 2))
    # Center of platform in meters
    platform_center_x = (x_start + x_end) * terrain.horizontal_scale * 0.5
    goals[0] = [platform_center_x, mid_y * terrain.horizontal_scale]
    goals[1] = [platform_center_x + 1.0, mid_y * terrain.horizontal_scale]
    goals[2] = [terrain.width * terrain.horizontal_scale - round(1.0 / terrain.horizontal_scale), mid_y * terrain.horizontal_scale]
    terrain.goals = goals
    
    return terrain

def wall_terrain(terrain,
                 min_height=0.3,
                 max_height=1.0,
                 max_yaw_deg=90.0,
                 thickness=0.6,
                 platform_len=1.5,
                 platform_height=0.0,
                 yaw_deg=None):
    """
    Generate a single wall of varying height and yaw.

    The wall:
    - Has height sampled uniformly between [min_height, max_height] (meters)
    - Has yaw sampled uniformly in [-max_yaw_deg, max_yaw_deg] degrees
      (0° = perpendicular to robot forward direction; ±90° = parallel)
    - Has a finite thickness (meters) around the wall center line
    - Is placed after a flat start platform of length platform_len
    """
    # Convert to discrete units
    min_h_int = int(min_height / terrain.vertical_scale)
    max_h_int = max(int(max_height / terrain.vertical_scale), min_h_int + 1)
    wall_height_int = np.random.randint(min_h_int, max_h_int + 1)

    platform_len_px = max(1, int(platform_len / terrain.horizontal_scale))
    platform_height_px = int(platform_height / terrain.vertical_scale)
    half_thickness_px = max(1, int(thickness / terrain.horizontal_scale / 2.0))

    # Initialize flat ground
    terrain.height_field_raw[:, :] = 0

    # Start platform (flat), centered around tile center so robot stands in the middle
    mid_x = terrain.width // 2
    platform_start = max(0, mid_x - platform_len_px // 2)
    platform_end = min(terrain.width, platform_start + platform_len_px)
    terrain.height_field_raw[platform_start:platform_end, :] = platform_height_px

    # Wall center further after platform, in the middle laterally (distance scaled down)
    extra_offset_px = max(int(1.0 / terrain.horizontal_scale), half_thickness_px * 2)
    cx = platform_end + extra_offset_px
    cx = min(cx, terrain.width - half_thickness_px - 2)
    cy = terrain.length // 2

    # Sample or fix yaw angle in radians
    if yaw_deg is None:
        yaw_rad = np.deg2rad(np.random.uniform(-max_yaw_deg, max_yaw_deg))
    else:
        yaw_rad = np.deg2rad(yaw_deg)

    # Special case: yaw ≈ 0 => axis-aligned wall (two parallel faces) with no quirks
    if abs(np.sin(yaw_rad)) < 1e-3:
        # Make a rectangular slab [x0:x1] × all y
        x0 = max(platform_end + half_thickness_px, 0)
        x1 = min(x0 + 2 * half_thickness_px, terrain.width)
        terrain.height_field_raw[x0:x1, :] = wall_height_int
    else:
        # General yaw: use distance-to-line band
        # Direction vector along the wall
        dx = np.cos(yaw_rad)
        dy = np.sin(yaw_rad)

        # Coordinate grids (x forward, y lateral)
        x_coords, y_coords = np.meshgrid(np.arange(terrain.width),
                                         np.arange(terrain.length),
                                         indexing='ij')

        # Vector from wall center to each point
        vx = x_coords - cx
        vy = y_coords - cy

        # Signed distance to infinite line through (cx, cy) with direction (dx, dy)
        # In 2D, distance magnitude = |v x d| / ||d||, but here ||d|| = 1
        # v x d (scalar in 2D) = vx * dy - vy * dx
        cross = vx * dy - vy * dx
        dist = np.abs(cross)

        # Limit wall extent roughly to the usable terrain region (avoid painting over start platform)
        wall_mask = (dist <= half_thickness_px) & (x_coords >= platform_len_px)

        # Raise the wall; keep maximum in case of future modifications
        terrain.height_field_raw[wall_mask] = np.maximum(
            terrain.height_field_raw[wall_mask],
            wall_height_int
        )

    # Define simple goals: before wall and after wall
    goals = np.zeros((2, 2))
    mid_y = cy
    # approximate platform center in meters
    platform_center_x = (platform_start + platform_end) * terrain.horizontal_scale * 0.5
    goals[0] = [platform_center_x, mid_y * terrain.horizontal_scale]
    goals[1] = [(terrain.width - 1) * terrain.horizontal_scale, mid_y * terrain.horizontal_scale]
    terrain.goals = goals

    return terrain

def sloped_wall_terrain(terrain,
                        min_height=0.3,
                        max_height=1.0,
                        slope_len_range=(1.0, 2.0),
                        platform_len=0.5,
                        platform_height=None):
    """
    Generate a "wall" whose faces are slopes:
      - one up-slope,
      - followed by a flat platform (0.5 m by default),
      - followed by a down-slope.

    All heights (slopes and platform) are randomized in the same range as wall_terrain.
    """
    # Convert to discrete units
    min_h_int = int(min_height / terrain.vertical_scale)
    max_h_int = max(int(max_height / terrain.vertical_scale), min_h_int + 1)

    # Sample platform height in the same range as wall_terrain
    platform_h_int = np.random.randint(min_h_int, max_h_int + 1)

    # Choose a single slope angle between 20° and 90° and derive the slope length
    # from the platform height so both faces have the same angle.
    angle_deg = np.random.uniform(20.0, 90.0)
    angle_rad = np.deg2rad(angle_deg)
    height_m = platform_h_int * terrain.vertical_scale
    # For very steep walls (angle → 90°), avoid division by ~0; cap the minimum tan.
    tan_theta = max(np.tan(angle_rad), 1e-3)
    slope_len_m = height_m / tan_theta
    slope_len_px = max(1, int(slope_len_m / terrain.horizontal_scale))

    # Force the top platform to always be exactly 0.5 m wide (in meters),
    # regardless of the platform_len argument.
    platform_len_m = 0.5
    platform_len_px = max(1, int(platform_len_m / terrain.horizontal_scale))

    # Initialize flat ground
    terrain.height_field_raw[:, :] = 0

    # Start platform for the robot, centered around tile center (as in wall_terrain)
    mid_x = terrain.width // 2
    start_platform_len_px = max(1, int(1.5 / terrain.horizontal_scale))  # 1.5 m start platform
    start_platform_start = max(0, mid_x - start_platform_len_px // 2)
    start_platform_end = min(terrain.width, start_platform_start + start_platform_len_px)
    start_platform_h_int = int(0.0 / terrain.vertical_scale)  # flat at 0 height
    terrain.height_field_raw[start_platform_start:start_platform_end, :] = start_platform_h_int

    # Place sloped "wall" feature after the start platform, along x.
    # Ensure the full [up-slope][platform][down-slope] fits so both slopes
    # have exactly the same number of cells.
    gap_px = max(1, int(1.0 / terrain.horizontal_scale))  # small gap after platform
    total_feature_px = 2 * slope_len_px + platform_len_px
    max_x0 = max(0, terrain.width - total_feature_px)
    x0 = min(start_platform_end + gap_px, max_x0)
    x1 = x0 + slope_len_px
    x2 = x1 + platform_len_px
    x3 = x2 + slope_len_px

    # Up-slope: ramp from 0 up to platform_h_int
    if x1 > x0:
        ramp = np.linspace(0, platform_h_int, x1 - x0, endpoint=False).astype(np.int16)
        ramp = ramp[:, None]  # shape (len,1)
        terrain.height_field_raw[x0:x1, :] = np.maximum(terrain.height_field_raw[x0:x1, :], ramp)

    # Flat platform at platform_h_int
    terrain.height_field_raw[x1:x2, :] = platform_h_int

    # Down-slope: ramp back down to 0
    if x3 > x2:
        ramp_down = np.linspace(platform_h_int, 0, x3 - x2, endpoint=False).astype(np.int16)
        ramp_down = ramp_down[:, None]
        terrain.height_field_raw[x2:x3, :] = np.maximum(terrain.height_field_raw[x2:x3, :], ramp_down)

    # Goals: before feature and after feature, in meters
    goals = np.zeros((2, 2))
    mid_y = terrain.length // 2
    # Just before the first slope
    goals[0] = [(start_platform_end * terrain.horizontal_scale), mid_y * terrain.horizontal_scale]
    # Just after the second slope
    goals[1] = [(x3 * terrain.horizontal_scale), mid_y * terrain.horizontal_scale]
    terrain.goals = goals

    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0