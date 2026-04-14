"""Soft cost planning: push the gripper away from a virtual hot-spot.

Motion planning with a **soft** cost rather than a hard constraint.
We place a "hot point" in the workspace, write a CasADi cost that
penalises proximity of the left gripper to that point, and ask an
asymptotically-optimal planner (RRT*) to minimise it.  The planner
is free to approach the hot-spot — it's not an obstacle — but the
optimiser prefers paths that stay further away.

The comparison is run twice:

    * with no cost — default path-length objective (straight-line),
    * with the cost  — RRT* bends the path away from the hot-spot.

Both solutions are collision-free against the table pointcloud from
:mod:`motion_planning_example`.

    pixi run python examples/cost_planning_example.py
    pixi run python examples/cost_planning_example.py --time_limit 5
"""

from pathlib import Path

import casadi as ca
import numpy as np
import trimesh
from fire import Fire

import autolife_planning
from autolife_planning.config.robot_config import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import Cost, SymbolicContext, create_planner
from autolife_planning.types import PlannerConfig

SUBGROUP = "autolife_left_arm"
LEFT_FINGER_LINK = "Link_Left_Gripper_Left_Finger"
RIGHT_FINGER_LINK = "Link_Left_Gripper_Right_Finger"


def load_table(distance: float = 0.85, height: float = 0.35) -> np.ndarray:
    """Same pointcloud as motion_planning_example — a table in front of the robot."""
    pkg_root = Path(autolife_planning.__file__).parent
    pcd = trimesh.load(str(pkg_root / "resources" / "envs" / "pcd" / "table.ply"))
    pts = np.asarray(pcd.vertices, dtype=np.float32)
    pts = pts - pts.mean(axis=0)
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    pts = pts @ rot.T
    pts[:, 0] += float(distance)
    pts[:, 2] += float(height)
    return pts


def ee_translation(ctx: SymbolicContext):
    """Symbolic TCP — midpoint between the two finger link origins."""
    return 0.5 * (
        ctx.link_translation(LEFT_FINGER_LINK) + ctx.link_translation(RIGHT_FINGER_LINK)
    )


def main(
    planner_name: str = "rrtstar",
    time_limit: float = 5.0,
    weight: float = 40.0,
    sigma: float = 0.18,
) -> None:
    env = PyBulletEnv(autolife_robot_config, visualize=True)

    cloud = load_table()
    env.add_pointcloud(cloud, pointsize=3)

    ctx = SymbolicContext(SUBGROUP)

    # Virtual hot-spot: a point in the workspace we'd prefer the
    # gripper to avoid.  Not an obstacle — the planner is free to
    # cross it — but the cost makes it expensive.
    hot_spot = np.array([0.55, 0.10, 0.60], dtype=np.float64)

    # Gaussian repulsion: exp(-d^2 / (2 sigma^2)), scaled by weight.
    # Smooth, bounded, autodifferentiable — RRT*'s rewiring converges
    # cleanly on paths that skirt the peak.
    tcp = ee_translation(ctx)
    d2 = ca.sumsqr(tcp - ca.DM(hot_spot))
    expr = ca.exp(-d2 / (2.0 * sigma * sigma))
    cost = Cost(expression=expr, q_sym=ctx.q, name="hot_spot", weight=weight)

    env.draw_sphere(
        center=hot_spot.tolist(),
        radius=float(sigma),
        color=(1.0, 0.3, 0.2, 0.35),
    )

    # Fixed start/goal so the two runs are directly comparable.
    start = HOME_JOINTS[ctx.active_indices].copy()

    # Build a temporary planner just to sample a collision-free goal
    # that isn't coincident with the hot spot.
    sampler = create_planner(
        SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
        base_config=HOME_JOINTS.copy(),
        pointcloud=cloud,
    )
    np.random.seed(0)
    goal = sampler.sample_valid()

    def run(label: str, costs: list | None):
        planner = create_planner(
            SUBGROUP,
            config=PlannerConfig(planner_name=planner_name, time_limit=time_limit),
            base_config=HOME_JOINTS.copy(),
            pointcloud=cloud,
            costs=costs,
        )
        result = planner.plan(start, goal)
        n = result.path.shape[0] if result.path is not None else 0
        print(
            f"── {label} ──\n"
            f"  planner={planner_name}  {result.status.value}: {n} waypoints in "
            f"{result.planning_time_ns / 1e6:.0f} ms, cost {result.path_cost:.3f}"
        )
        if result.success and result.path is not None:
            env.animate_path(planner.embed_path(result.path), fps=60)
        return result

    run("baseline: path-length objective", costs=None)
    run("with hot-spot repulsion cost", costs=[cost])

    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
