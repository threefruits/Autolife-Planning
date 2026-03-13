"""Test the coupled-leg body planner (autolife_body_coupled, 20 DOF).

Demonstrates that Joint_Knee is driven by the coupling knee = 2 * ankle,
reducing the body planner from 21 to 20 independent DOF.  Plans a path
from home to a random valid goal, verifies the coupling holds along
the entire trajectory, and animates the result in PyBullet.

Usage:
    pixi run python examples/coupled_planning_example.py
"""

from __future__ import annotations

import time

import numpy as np
import pybullet as pb

from autolife_planning.config.robot_config import (
    VIZ_URDF_PATH,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner

MULTIPLIER = 2.0
OFFSET = 0.0
ANKLE_IDX = autolife_robot_config.joint_names.index("Joint_Ankle")
KNEE_IDX = autolife_robot_config.joint_names.index("Joint_Knee")


def wait_key(env, key, msg):
    """Block until *key* is pressed in the PyBullet GUI window."""
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


def sample_and_choose(env, coupled, label):
    """Randomly sample valid configs until user accepts one ('y') or rejects ('n')."""
    print(f"Press 'y' to accept {label}, 'n' to resample.")
    client = env.sim.client
    while True:
        config = coupled.sample_valid()
        full = coupled.embed_path(config.reshape(1, -1))[0]
        env.set_configuration(full)
        text_id = client.addUserDebugText(
            f"{label}: press 'y' accept / 'n' resample",
            [0, 0, 1.5],
            textColorRGB=[0, 0, 0],
            textSize=1.5,
        )
        while True:
            keys = client.getKeyboardEvents()
            if ord("y") in keys and keys[ord("y")] & pb.KEY_WAS_TRIGGERED:
                client.removeUserDebugItem(text_id)
                print(f"  Accepted {label}: {np.round(config, 3)}")
                return config, full
            if ord("n") in keys and keys[ord("n")] & pb.KEY_WAS_TRIGGERED:
                print(f"  Rejected, resampling {label}...")
                break
            time.sleep(0.01)
        client.removeUserDebugItem(text_id)


def check_coupling(full_config: np.ndarray, label: str) -> bool:
    """Verify knee = 2 * ankle in a full 24-DOF config."""
    ankle = full_config[ANKLE_IDX]
    knee = full_config[KNEE_IDX]
    expected = MULTIPLIER * ankle + OFFSET
    ok = np.isclose(knee, expected, atol=1e-6)
    status = "OK" if ok else "FAIL"
    print(
        f"  [{status}] {label}: ankle={ankle:.4f}, knee={knee:.4f}, expected={expected:.4f}"
    )
    return ok


def main():
    # --- Create coupled planner (20 DOF) and regular body planner (21 DOF) ---
    coupled = create_planner("autolife_body_coupled")
    body = create_planner("autolife_body")

    print(f"autolife_body DOF:          {body.num_dof}")
    print(f"autolife_body_coupled DOF:  {coupled.num_dof}")
    print(f"Coupled joint names:        {coupled.joint_names}")
    assert coupled.num_dof == 20, f"Expected 20 DOF, got {coupled.num_dof}"
    assert (
        "Joint_Knee" not in coupled.joint_names
    ), "Joint_Knee should not be independent"
    print()

    # --- Setup visualization ---
    env = PyBulletEnv(
        autolife_robot_config, visualize=True, viz_urdf_path=VIZ_URDF_PATH
    )

    # --- Interactively sample start and goal ---
    start, start_full = sample_and_choose(env, coupled, "start")
    goal, _ = sample_and_choose(env, coupled, "goal")
    print()

    # --- Plan ---
    result = coupled.plan(start, goal)
    print(f"Status: {result.status.value}")

    if not result.success:
        print("Planning failed.")
        wait_key(env, ord("q"), "Planning failed. Press 'q' to quit.")
        return

    print(
        f"Path: {result.path.shape[0]} waypoints, "
        f"{result.planning_time_ns / 1e6:.1f} ms, "
        f"cost = {result.path_cost:.4f}"
    )
    print()

    # --- Embed path to full 24 DOF and verify coupling ---
    full_path = coupled.embed_path(result.path)
    print(f"Full path shape: {full_path.shape}")

    all_ok = True
    all_ok &= check_coupling(full_path[0], "first waypoint")
    all_ok &= check_coupling(full_path[len(full_path) // 2], "middle waypoint")
    all_ok &= check_coupling(full_path[-1], "last waypoint")

    # Check every waypoint
    for i in range(full_path.shape[0]):
        ankle = full_path[i, ANKLE_IDX]
        knee = full_path[i, KNEE_IDX]
        if not np.isclose(knee, MULTIPLIER * ankle + OFFSET, atol=1e-6):
            print(f"  [FAIL] waypoint {i}: ankle={ankle:.4f}, knee={knee:.4f}")
            all_ok = False

    print()
    if all_ok:
        print("All coupling constraints satisfied.")
    else:
        print("ERROR: Some waypoints violate the coupling constraint!")

    # --- Animate path ---
    env.set_configuration(start_full)
    time.sleep(0.3)
    for i in range(full_path.shape[0]):
        env.set_configuration(full_path[i])
        time.sleep(1.0 / 120.0)

    wait_key(env, ord("q"), "Path done. Press 'q' to quit.")


if __name__ == "__main__":
    main()
