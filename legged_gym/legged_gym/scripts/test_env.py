#!/usr/bin/env python3
"""
Simple script to test if an environment loads correctly without training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def test_env():
    # Parse arguments
    args = get_args()
    
    # Override task if not specified
    if not hasattr(args, 'task') or args.task is None:
        args.task = 'wf_tron1a'
    
    print(f"Testing environment: {args.task}")
    print("=" * 60)
    
    # Pull default cfgs and apply the same terrain settings as play.py
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Apply terrain settings from play.py
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
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
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    # Ensure optional attrs exist to avoid AttributeError in Terrain
    if not hasattr(env_cfg.terrain, "flat_wall"):
        env_cfg.terrain.flat_wall = False
    if not hasattr(env_cfg.terrain, "max_init_terrain_level"):
        env_cfg.terrain.max_init_terrain_level = 0
    
    # Create environment
    try:
        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        print(f"✓ Environment created successfully!")
        print(f"  - Number of environments: {env.num_envs}")
        print(f"  - Number of observations: {env.num_obs}")
        print(f"  - Number of privileged observations: {env.num_privileged_obs}")
        print(f"  - Number of actions: {env.num_actions}")
        print(f"  - Device: {env.device}")
        
        # Test getting observations
        obs = env.get_observations()
        print(f"✓ Observations retrieved successfully!")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Expected shape: ({env.num_envs}, {env.num_obs})")
        
        # Test reset
        env.reset()
        print(f"✓ Environment reset successfully!")
        
        # Run a short rollout for visualization (default ~3 seconds at 60 Hz sim)
        rollout_steps = getattr(args, "rollout_steps", 200)
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for i in range(rollout_steps):
            obs, privileged_obs, rewards, dones, infos = env.step(actions)
        print(f"✓ Environment rollout executed ({rollout_steps} steps)")
        print(f"  - Rewards shape: {rewards.shape}")
        print(f"  - Dones shape: {dones.shape}")
        
        print("=" * 60)
        print("✓ All tests passed! Environment is working correctly.")
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"✗ Error loading environment:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_env()
    sys.exit(0 if success else 1)

