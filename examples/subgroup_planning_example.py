"""Test motion planning for all subgroup configurations.

Iterates through every planning subgroup (single arm, dual arm, torso+arm,
whole body), plans a path from the home configuration to a random valid goal,
and animates the result in PyBullet.

Usage:
    # Interactive visualization (press 'n' to advance, 'q' to quit)
    pixi run python examples/subgroup_planning_example.py

    # Use a different planner
    pixi run python examples/subgroup_planning_example.py --planner_name prm
"""

from __future__ import annotations

import time

import numpy as np
from fire import Fire

from autolife_planning.config.robot_config import (
    PLANNING_SUBGROUPS,
    VIZ_URDF_PATH,
    autolife_robot_config,
    subgroup_base_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

# Subgroup categories for display
CATEGORIES = {
    "Single arm (7 DOF)": [
        "autolife_left_high",
        "autolife_left_mid",
        "autolife_left_low",
        "autolife_right_high",
        "autolife_right_mid",
        "autolife_right_low",
    ],
    "Dual arm (14 DOF)": [
        "autolife_dual_high",
        "autolife_dual_mid",
        "autolife_dual_low",
    ],
    "Torso + arm (9 DOF)": [
        "autolife_torso_left_high",
        "autolife_torso_left_mid",
        "autolife_torso_left_low",
        "autolife_torso_right_high",
        "autolife_torso_right_mid",
        "autolife_torso_right_low",
    ],
    "Whole body (21 DOF)": [
        "autolife_body",
    ],
}


def wait_key(env, key, msg):
    """Block until *key* is pressed in the PyBullet GUI window."""
    import pybullet as pb

    client = env.sim.client
    text_id = client.addUserDebugText(
        msg, [0, 0, 1.5], textColorRGB=[0, 0, 0], textSize=1.5
    )
    print(msg)
    while True:
        keys = client.getKeyboardEvents()
        if key in keys and keys[key] & pb.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.01)
    client.removeUserDebugItem(text_id)


def test_subgroup(env, robot_name, planner_name="rrtc"):
    """Plan and animate one subgroup in PyBullet."""
    print(f"\n{'=' * 60}")
    print(f"  {robot_name}  ({PLANNING_SUBGROUPS[robot_name]['dof']} DOF)")
    print(f"{'=' * 60}")

    planner = create_planner(
        robot_name,
        config=PlannerConfig(planner_name=planner_name),
    )

    # Base config with correct frozen stance for this subgroup
    base_cfg = subgroup_base_config(robot_name)

    # Start from the subgroup's base config (uses correct waist preset for torso groups)
    start = planner.extract_config(base_cfg)
    env.set_configuration(base_cfg)
    print(f"  Start (home): {np.round(start, 3)}")

    # Sample a collision-free goal
    goal = planner.sample_valid()
    goal_full = planner.embed_config(goal, base_config=base_cfg)
    env.set_configuration(goal_full)
    print(f"  Goal:          {np.round(goal, 3)}")

    wait_key(env, ord("n"), f"[{robot_name}] Goal shown. Press 'n' to plan.")

    # Plan
    result = planner.plan(start, goal)
    print(f"  Status: {result.status.value}")

    if result.success:
        print(
            f"  Path: {result.path.shape[0]} waypoints, "
            f"{result.planning_time_ns / 1e6:.1f} ms, "
            f"cost = {result.path_cost:.4f}"
        )

        # Show start briefly
        env.set_configuration(planner.embed_config(start, base_config=base_cfg))
        time.sleep(0.3)

        # Animate the planned path
        full_path = planner.embed_path(result.path, base_config=base_cfg)
        for i in range(full_path.shape[0]):
            env.set_configuration(full_path[i])
            time.sleep(1.0 / 120.0)

        wait_key(env, ord("n"), f"[{robot_name}] Path done. Press 'n' for next.")
    else:
        wait_key(env, ord("n"), f"[{robot_name}] Planning failed. Press 'n' for next.")


def main(planner_name: str = "rrtc"):
    """Test motion planning for all 16 subgroups.

    Args:
        planner_name: Planning algorithm (rrtc, prm, fcit, aorrtc).
    """
    env = PyBulletEnv(
        autolife_robot_config, visualize=True, viz_urdf_path=VIZ_URDF_PATH
    )

    for category, names in CATEGORIES.items():
        print(f"\n\n{'#' * 60}")
        print(f"  {category}")
        print(f"{'#' * 60}")
        for robot_name in names:
            try:
                test_subgroup(env, robot_name, planner_name)
            except Exception as e:
                print(f"  ERROR on {robot_name}: {e}")
                import traceback

                traceback.print_exc()
                wait_key(env, ord("n"), f"[{robot_name}] Error. Press 'n' for next.")

    wait_key(env, ord("q"), "All subgroups tested. Press 'q' to quit.")
    print("\nDone.")


if __name__ == "__main__":
    Fire(main)
