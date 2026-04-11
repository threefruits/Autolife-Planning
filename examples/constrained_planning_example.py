"""Constrained motion planning: end effector on a manifold, around an obstacle.

The left gripper is constrained to slide across a horizontal plane
(one holonomic equation, ``z = z0``), written inline as a CasADi
expression over the planner's joint vector.  An obstacle sphere sits
in the gripper's straight-line sweep, so OMPL's ProjectedStateSpace
has to bend the arm around it while keeping the gripper on the plane.

    pixi run python examples/constrained_planning_example.py
"""

import numpy as np
import pybullet as pb
from fire import Fire

from autolife_planning.config.robot_config import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import Constraint, SymbolicContext, create_planner
from autolife_planning.types import PlannerConfig


def main(planner_name: str = "rrtc", time_limit: float = 10.0):
    env = PyBulletEnv(autolife_robot_config, visualize=True)

    # 1. Symbolic context for the 7-DOF left-arm subgroup.
    ctx = SymbolicContext("autolife_left_arm")
    start = HOME_JOINTS[ctx.active_indices].copy()

    # 2. Write the constraint as a CasADi residual over ctx.q.
    #    "left gripper z-coordinate must stay at its home value"
    z_surface = float(ctx.evaluate_link_pose("Link_Left_Gripper", start)[2, 3])
    left_plane = Constraint(
        residual=ctx.link_translation("Link_Left_Gripper")[2] - z_surface,
        q_sym=ctx.q,
        name="left_gripper_on_plane",
    )

    # 3. Pick a goal seed and project it onto the manifold.  Any seed
    #    within the basin of attraction works; ctx.project runs a
    #    Gauss-Newton pass using the constraint's own Jacobian.
    goal_seed = np.array([-1.26, 0.07, -1.06, 0.10, -2.42, 0.50, -0.29])
    goal = ctx.project(goal_seed, left_plane.residual)

    # 4. Drop an obstacle sphere in the gripper's sweep, discretised
    #    as a point cloud for the VAMP collision checker, and add a
    #    translucent visual for the PyBullet viewer.
    left_start_xyz = ctx.evaluate_link_pose("Link_Left_Gripper", start)[:3, 3]
    left_goal_xyz = ctx.evaluate_link_pose("Link_Left_Gripper", goal)[:3, 3]
    sphere_center = 0.5 * (left_start_xyz + left_goal_xyz) + np.array([0, 0, -0.03])
    sphere_radius = 0.09

    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(800, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = sphere_radius * rng.random(800) ** (1.0 / 3.0)
    cloud = (dirs * radii[:, None] + sphere_center).astype(np.float32)

    env.sim.client.createMultiBody(
        baseVisualShapeIndex=env.sim.client.createVisualShape(
            pb.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[0.95, 0.3, 0.3, 0.55]
        ),
        basePosition=sphere_center.tolist(),
    )

    # 5. Plan on the constraint manifold with the sphere as an obstacle.
    planner = create_planner(
        "autolife_left_arm",
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            point_radius=0.012,
        ),
        pointcloud=cloud,
        constraints=[left_plane],
    )
    result = planner.plan(start, goal)

    n = result.path.shape[0] if result.path is not None else 0
    print(
        f"{result.status.value}: {n} waypoints in "
        f"{result.planning_time_ns / 1e6:.0f} ms, cost {result.path_cost:.2f}"
    )

    if result.success and result.path is not None:
        env.animate_path(planner.embed_path(result.path), fps=60)
    env.wait_key("q", "press 'q' to quit")


if __name__ == "__main__":
    Fire(main)
