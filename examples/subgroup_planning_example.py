"""Test motion planning for every kinematic subgroup at multiple stances.

Subgroup names describe *only* the active joints (e.g. ``autolife_left_arm``).
The values for the inactive joints are supplied separately as ``base_config``,
so the same kinematic group can be planned around any pose the caller wants
without baking that pose into the planner name.

This example demonstrates the flexibility by sweeping the arm subgroups
over three different leg + waist stances ("high"/"mid"/"low") that are
defined *here in the example* — they are not part of the planning API.
You can replace ``STANCE_PRESETS`` with any 24-DOF base config you like
(for example the live state read from your env).

Usage:
    # Interactive visualization (press 'n' to advance, 'q' to quit)
    pixi run python examples/subgroup_planning_example.py

    # Use a different OMPL planner
    pixi run python examples/subgroup_planning_example.py --planner_name prm
"""

from __future__ import annotations

import time

import numpy as np
from fire import Fire

from autolife_planning.config.robot_config import (
    HOME_JOINTS,
    PLANNING_SUBGROUPS,
    VIZ_URDF_PATH,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

# ── Example-local stance presets ─────────────────────────────────────
#
# These joint values are NOT part of the planning API — they live here
# purely so this example can show planning at three different leg +
# waist poses.  Replace this dict (or build a base_config from scratch)
# to plan around any pose you want.
STANCE_PRESETS: dict[str, dict[str, float]] = {
    "high": {
        "Joint_Ankle": 0.0,
        "Joint_Knee": 0.0,
        "Joint_Waist_Pitch": 0.00,
        "Joint_Waist_Yaw": -0.14,
    },
    "mid": {
        "Joint_Ankle": 0.78,
        "Joint_Knee": 1.60,
        "Joint_Waist_Pitch": 0.89,
        "Joint_Waist_Yaw": -0.14,
    },
    "low": {
        "Joint_Ankle": 1.41,
        "Joint_Knee": 2.38,
        "Joint_Waist_Pitch": 0.95,
        "Joint_Waist_Yaw": -0.14,
    },
}

# Subgroups whose active joints don't include the legs/waist — they're
# meaningful to plan at different stances since the legs/waist stay frozen.
ARM_SUBGROUPS = [
    "autolife_left_arm",
    "autolife_right_arm",
    "autolife_dual_arm",
    "autolife_torso_left_arm",
    "autolife_torso_right_arm",
]


def make_base_config(stance: str) -> np.ndarray:
    """Build a 24-DOF base config from HOME_JOINTS + a stance override."""
    base = HOME_JOINTS.copy()
    joint_names = autolife_robot_config.joint_names
    for joint_name, value in STANCE_PRESETS[stance].items():
        base[joint_names.index(joint_name)] = value
    return base


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


def run_subgroup(
    env,
    robot_name: str,
    base_config: np.ndarray,
    label: str,
    planner_name: str,
):
    """Plan one (subgroup, base_config) case and animate it."""
    print(f"\n{'=' * 60}")
    print(f"  {label}  ({PLANNING_SUBGROUPS[robot_name]['dof']} DOF)")
    print(f"{'=' * 60}")

    # base_config carries the values for joints not in the subgroup.
    # The planner stores it and the C++ collision checker injects it
    # for inactive joints on every state validity / motion validity
    # query — and embed_config / embed_path use it as the default base.
    planner = create_planner(
        robot_name,
        config=PlannerConfig(planner_name=planner_name),
        base_config=base_config,
    )

    # Start from the supplied base, projected into the subgroup's joints.
    start = planner.extract_config(base_config)
    env.set_configuration(base_config)
    print(f"  Start: {np.round(start, 3)}")

    # Sample a collision-free goal in the reduced subspace.
    goal = planner.sample_valid()
    env.set_configuration(planner.embed_config(goal))
    print(f"  Goal:  {np.round(goal, 3)}")

    wait_key(env, ord("n"), f"[{label}] Goal shown. Press 'n' to plan.")

    result = planner.plan(start, goal)
    print(f"  Status: {result.status.value}")

    if not result.success or result.path is None:
        wait_key(env, ord("n"), f"[{label}] Planning failed. Press 'n' for next.")
        return

    print(
        f"  Path: {result.path.shape[0]} waypoints, "
        f"{result.planning_time_ns / 1e6:.1f} ms, "
        f"cost = {result.path_cost:.4f}"
    )

    # embed_path defaults to the planner's stored base_config, so the
    # animated full-body path uses the same configuration the C++ side
    # validated.
    full_path = planner.embed_path(result.path)

    env.set_configuration(planner.embed_config(start))
    time.sleep(0.3)
    for i in range(full_path.shape[0]):
        env.set_configuration(full_path[i])
        time.sleep(1.0 / 120.0)

    wait_key(env, ord("n"), f"[{label}] Path done. Press 'n' for next.")


def main(planner_name: str = "rrtc"):
    """Sweep arm subgroups across three stances + run the whole-body planner.

    Args:
        planner_name: OMPL planner to use (rrtc, rrt, prm, bitstar, ...).
    """
    env = PyBulletEnv(
        autolife_robot_config, visualize=True, viz_urdf_path=VIZ_URDF_PATH
    )

    # Same kinematic subgroups, three different frozen leg/waist
    # poses.  Identical planner code; the difference is just the
    # base_config we hand in.
    for stance in ("high", "mid", "low"):
        base = make_base_config(stance)
        print(f"\n\n{'#' * 60}")
        print(f"  Stance: {stance}")
        print(f"{'#' * 60}")
        for robot_name in ARM_SUBGROUPS:
            label = f"{robot_name} @ {stance}"
            try:
                run_subgroup(env, robot_name, base, label, planner_name)
            except Exception as e:
                print(f"  ERROR on {label}: {e}")
                import traceback

                traceback.print_exc()
                wait_key(env, ord("n"), f"[{label}] Error. Press 'n' for next.")

    # Whole-body planner — its active set already covers the legs and
    # waist, so the stance choice doesn't affect anything; HOME_JOINTS
    # is the natural base.
    print(f"\n\n{'#' * 60}")
    print("  Whole body")
    print(f"{'#' * 60}")
    try:
        run_subgroup(
            env, "autolife_body", HOME_JOINTS.copy(), "autolife_body", planner_name
        )
    except Exception as e:
        print(f"  ERROR on autolife_body: {e}")
        import traceback

        traceback.print_exc()
        wait_key(env, ord("n"), "[autolife_body] Error. Press 'n' for next.")

    wait_key(env, ord("q"), "All subgroups tested. Press 'q' to quit.")
    print("\nDone.")


if __name__ == "__main__":
    Fire(main)
