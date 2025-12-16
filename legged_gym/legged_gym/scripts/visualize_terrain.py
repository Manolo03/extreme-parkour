#!/usr/bin/env python3
"""
Minimal visualization: spawn wf_tron1a in its default pose on a chosen wall-like terrain,
make it gravity-less and static, and render without stepping physics.

Usage:
    python visualize_terrain.py --task wf_tron1a --num-envs 1 --terrain-func wall_terrain
    python visualize_terrain.py --task wf_tron1a --num-envs 1 --terrain-func sloped_wall_terrain
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args


def main():
    # Parse custom terrain options first
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument(
        "--terrain-func",
        type=str,
        default="wall_terrain",
        choices=["wall_terrain", "sloped_wall_terrain"],
        help="Terrain generator to use for each tile.",
    )
    custom_parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of terrain rows (forward direction).",
    )
    custom_parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of terrain columns (lateral direction).",
    )
    custom_args, remaining = custom_parser.parse_known_args()

    # Use standard args so device/sim settings match the rest of the codebase
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    args = get_args()
    sys.argv = old_argv

    if not hasattr(args, "task") or args.task is None:
        args.task = "wf_tron1a"
    if not hasattr(args, "num_envs") or args.num_envs is None:
        args.num_envs = 1

    # Load configs and force number of envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False

    # Force use of selected wall-like terrain function
    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        rows = custom_args.rows if custom_args.rows is not None else 1
        cols = custom_args.cols if custom_args.cols is not None else 1
        env_cfg.terrain.num_rows = rows
        env_cfg.terrain.num_cols = cols
        # Set tile size so wall centers are spaced ≈10 m apart in x, 30 m in y
        if hasattr(env_cfg.terrain, "terrain_length"):
            env_cfg.terrain.terrain_length = 10.0
        if hasattr(env_cfg.terrain, "terrain_width"):
            env_cfg.terrain.terrain_width = 30.0

        terrain_kwargs = {}
        if custom_args.terrain_func == "sloped_wall_terrain":
            terrain_kwargs["min_height"] = 0.3
            terrain_kwargs["max_height"] = 1.0
            terrain_kwargs["slope_len_range"] = (1.0, 2.0)
            terrain_kwargs["platform_len"] = 0.5

        env_cfg.terrain.terrain_kwargs = {
            "type": custom_args.terrain_func,
            "terrain_kwargs": terrain_kwargs,
        }
        # Disable slope_threshold smoothing for this viz
        if hasattr(env_cfg.terrain, "slope_treshold"):
            env_cfg.terrain.slope_treshold = 1e9

    print("=" * 80)
    print(f"Spawning task '{args.task}' with {env_cfg.env.num_envs} env(s)")
    print(f"Terrain function: {custom_args.terrain_func}")
    print(f"Grid: {env_cfg.terrain.num_rows} rows x {env_cfg.terrain.num_cols} cols")
    print("=" * 80)

    # Create environment as usual (this builds terrain according to cfg.terrain)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.debug_viz = False

    # Make robot gravity-less, static, and centered at (0,0,z_init) in each env frame
    try:
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)

        # Identity quaternion for base orientation
        base_quat_identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)

        for i in range(env.num_envs):
            origin = env.env_origins[i].to(env.device)
            z_init = env.cfg.init_state.pos[2]

            # Center base in env frame at z_init -> origin + [0,0,z_init]
            base_pos_world = origin.clone()
            base_pos_world[2] = z_init
            env.root_states[i, 0:3] = base_pos_world
            env.root_states[i, 3:7] = base_quat_identity

        # Push updated root states
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))

        # Zero DOFs for all envs (tron1a defaults are 0 in config)
        env.dof_pos[:] = 0.0
        env.dof_vel[:] = 0.0
        env_ids_int32 = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # Disable gravity for all rigid bodies of all actors
        for i in range(env.num_envs):
            props = env.gym.get_actor_rigid_body_properties(env.envs[i], env.actor_handles[i])
            for p in props:
                p.flags |= gymapi.RIGID_BODY_DISABLE_GRAVITY
            env.gym.set_actor_rigid_body_properties(env.envs[i], env.actor_handles[i], props, recomputeInertia=False)

        # Debug print for env 0
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        base_pos_world = env.root_states[0, 0:3].cpu().numpy()
        origin_np = env.env_origins[0].cpu().numpy()
        base_pos_rel = base_pos_world - origin_np
        dof_pos = env.dof_pos[0].cpu().numpy()
        print("========== Robot state (env 0) ==========")
        print(f"Base position (world):        {base_pos_world}")
        print(f"Env origin (world):           {origin_np}")
        print(f"Base position (env-relative): {base_pos_rel}")
        print(f"DOF positions:                {dof_pos}")
        print("=========================================")
    except Exception as e:
        print(f"[warn] kinematic/gravity-less setup failed: {e}")

    # Set up viewer
    if env.viewer is None:
        print("No viewer created (likely headless).")
        return

    cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

    print("\n" + "=" * 80)
    print("Static visualization (no physics stepping)")
    print("Controls:")
    print("  - ESC: Exit")
    print("  - Mouse: Orbit/pan")
    print("=" * 80 + "\n")

    # Render loop without physics stepping
    try:
        while not env.gym.query_viewer_has_closed(env.viewer):
            env.gym.poll_viewer_events(env.viewer)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            env.gym.sync_frame_time(env.sim)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()


