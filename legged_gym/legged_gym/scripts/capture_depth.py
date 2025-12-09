#!/usr/bin/env python3
"""
Capture a sample depth frame for wf_tron1a.

By default uses tron1a camera config, saves to logs/depth_samples/tron1a_config/
Use --use-legged-config to use legged_robot_config camera settings, saves to logs/depth_samples/legged_config/
"""
import os
import sys
import argparse
import numpy as np
import imageio

# Import isaacgym before any torch imports (torch comes in via legged_gym.utils)
import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
import torch

# Ensure repo root on path and then import legged_gym
repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, repo_root)

# Import envs first to register tasks and avoid circular import
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
from legged_gym import LEGGED_GYM_ROOT_DIR


def capture_depth_images(env, output_dir, config_name="custom", num_images=20, time_between_images=0.1, save_initial_still=False):
    """
    Capture multiple depth images from the environment with time intervals.
    
    The depth buffer is updated every cfg.depth.update_interval simulation steps
    in post_physics_step(). We need to step enough times to trigger at least one update.
    
    Args:
        env: The environment
        output_dir: Directory to save images
        config_name: Name of the config for display
        num_images: Number of images to capture
        time_between_images: Time in seconds between captures
    """
    update_interval = env.cfg.depth.update_interval
    buffer_len = env.cfg.depth.buffer_len
    dt = env.cfg.sim.dt  # Simulation timestep
    
    # Calculate steps between captures
    steps_between_captures = max(int(time_between_images / dt), update_interval)
    
    print(f"\n=== Capturing {num_images} depth images with {config_name} config ===")
    print(f"  Camera position: {env.cfg.depth.position}")
    print(f"  Camera angle: {env.cfg.depth.angle} degrees")
    print(f"  Update interval: {update_interval} steps (updates every {update_interval} simulation steps)")
    print(f"  Buffer length: {buffer_len} frames")
    print(f"  Original resolution: {env.cfg.depth.original}")
    print(f"  Resized resolution: {env.cfg.depth.resized}")
    print(f"  Horizontal FOV: {env.cfg.depth.horizontal_fov} degrees")
    # Calculate vertical FOV from horizontal FOV and aspect ratio
    aspect_ratio = env.cfg.depth.original[1] / env.cfg.depth.original[0]  # height/width
    vertical_fov_deg = 2 * np.degrees(np.arctan(np.tan(np.radians(env.cfg.depth.horizontal_fov / 2)) * aspect_ratio))
    print(f"  Vertical FOV: {vertical_fov_deg:.1f} degrees (calculated from aspect ratio)")
    print(f"  Near/Far clip: {env.cfg.depth.near_clip}/{env.cfg.depth.far_clip} m")
    print(f"    Note: Depth range is controlled by near_clip/far_clip in post-processing.")
    print(f"    Isaac Gym renders depth, then values are clipped and normalized to [-0.5, 0.5].")
    print(f"  Simulation timestep: {dt} s")
    print(f"  Time between captures: {time_between_images} s ({steps_between_captures} steps)")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step enough times initially to populate depth buffer
    initial_steps = max(update_interval * 2, 10)
    print(f"\n  Initial stepping ({initial_steps} steps) to populate depth buffer...")
    print(f"  Applying actions to maintain default joint positions (keep robot upright)...")
    
    # Actions of zero should maintain default joint angles (for tron1a, default angles are all 0)
    # This keeps the robot in its initial pose
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    
    # Reset environment to ensure robot starts in initial upright position
    env.reset()
    
    for step_idx in range(initial_steps):
        env.step(actions)
    
    # Optionally capture initial still images (robot perfectly still, no stepping)
    if save_initial_still:
        initial_still_dir = os.path.join(output_dir, "initial_still")
        os.makedirs(initial_still_dir, exist_ok=True)
        print(f"\n  Capturing 5 initial still images (robot perfectly still, no simulation stepping)...")
        
        # Capture 5 images from the current depth buffer without stepping
        # They will be identical since we're not stepping the simulation
        for still_idx in range(5):
            depth = env.depth_buffer[0, -1].cpu().numpy()
            filename = f"initial_still_{still_idx:03d}.png"
            output_path = os.path.join(initial_still_dir, filename)
            
            # Normalize depth image for saving (depth_buffer is in [-0.5, 0.5], convert to [0, 255])
            depth_normalized = (depth + 0.5) * 255
            depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)
            imageio.imwrite(output_path, depth_normalized)
        
        print(f"  Saved 5 initial still images to: {initial_still_dir}")
    
    # Capture multiple images
    print(f"\n  Capturing {num_images} images...")
    for img_idx in range(num_images):
        # Reset robot to initial position before each capture to keep it upright
        if img_idx > 0:
            env.reset()
            # Step a few times after reset to stabilize
            for _ in range(update_interval):
                env.step(actions)
        
        # Step the environment for the time interval
        # Continue using zero actions to maintain default pose
        for _ in range(steps_between_captures):
            env.step(actions)
        
        # Grab latest depth frame for env 0
        # depth_buffer shape: (num_envs, buffer_len, height, width)
        depth = env.depth_buffer[0, -1].cpu().numpy()
        
        # Generate filename with zero-padded index
        filename = f"depth_sample_{img_idx:03d}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Normalize depth image for saving (depth_buffer is in [-0.5, 0.5], convert to [0, 255])
        depth_normalized = (depth + 0.5) * 255
        depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)
        imageio.imwrite(output_path, depth_normalized)
        
        if (img_idx + 1) % 5 == 0 or img_idx == 0:
            print(f"    Image {img_idx + 1}/{num_images}: Saved to {filename} "
                  f"(step {env.global_counter}, depth range: [{depth.min():.3f}, {depth.max():.3f}])")
    
    print(f"\n  All {num_images} images saved to: {output_dir}\n")


def main():
    # Parse custom output argument first
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use-legged-config", action="store_true",
                       help="Use legged_robot_config camera settings instead of tron1a config")
    parser.add_argument("--num-images", type=int, default=20,
                       help="Number of images to capture (default: 20)")
    parser.add_argument("--time-between", type=float, default=0.1,
                       help="Time in seconds between captures (default: 0.1)")
    parser.add_argument("--save-initial-still", action="store_true",
                       help="Save first 5 images while robot is perfectly still (no stepping)")
    custom_args, remaining = parser.parse_known_args()
    use_legged_config = custom_args.use_legged_config
    num_images = custom_args.num_images
    time_between = custom_args.time_between
    save_initial_still = custom_args.save_initial_still
    
    # Use get_args() to get all standard arguments with proper defaults
    # Temporarily replace sys.argv to pass remaining args
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    args = get_args()
    sys.argv = old_argv
    
    # Override task if not specified
    if not hasattr(args, 'task') or args.task is None:
        args.task = 'wf_tron1a'

    # Load cfgs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Enable depth camera
    env_cfg.depth.use_camera = True
    env_cfg.env.num_envs = args.num_envs
    
    # Set camera config and output folder based on flag
    if use_legged_config:
        from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
        # Use legged_robot_config camera settings
        env_cfg.depth.position = list(LeggedRobotCfg.depth.position)
        env_cfg.depth.angle = list(LeggedRobotCfg.depth.angle)
        env_cfg.depth.original = LeggedRobotCfg.depth.original
        env_cfg.depth.resized = LeggedRobotCfg.depth.resized
        env_cfg.depth.horizontal_fov = LeggedRobotCfg.depth.horizontal_fov
        config_name = "legged_robot_config"
        output_folder = "legged_config"
    else:
        # Use wf_tron1a config (default)
        config_name = "wf_tron1a"
        output_folder = "tron1a_config"
    
    # Set output path to side-by-side folders
    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "depth_samples", output_folder)

    # Match play terrain tweaks (optional: keep defaults to reflect training)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True

    # Create env
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Capture multiple depth images
    capture_depth_images(env, output_dir, config_name=config_name, 
                        num_images=num_images, time_between_images=time_between,
                        save_initial_still=save_initial_still)


if __name__ == "__main__":
    main()

