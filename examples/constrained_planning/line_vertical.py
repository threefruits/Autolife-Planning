"""Vertical-line constraint: the left gripper moves straight up and down.

Two holonomic equations stacked into one residual — the gripper's
x and y coordinates are pinned to their home values, leaving only
z free.  The planner has to reconfigure a 7-DOF arm so the gripper
slides along a purely vertical rail without drifting sideways.

A bold yellow cylinder marks the rail.

    pixi run python examples/constrained_planning/line_vertical.py
"""

import casadi as ca
from _shared import EE_LINK, SUBGROUP, find_goal, run_demo, setup
from fire import Fire

from autolife_planning.planning import Constraint, create_planner
from autolife_planning.types import PlannerConfig


def main(time_limit: float = 5.0):
    env, ctx, start = setup()
    p0 = ctx.evaluate_link_pose(EE_LINK, start)[:3, 3]
    left_pos = ctx.link_translation(EE_LINK)

    # The manifold: x and y are pinned; z is free.
    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        left_pos[0] - float(p0[0]),
        left_pos[1] - float(p0[1]),
    )
    line = Constraint(residual=residual, q_sym=ctx.q, name="line_v")

    planner = create_planner(
        SUBGROUP, config=PlannerConfig(time_limit=time_limit), constraints=[line]
    )
    # Push the gripper as high as the manifold allows.
    goal = find_goal(
        ctx, line.residual, start, planner, score=lambda xyz: xyz[2] - p0[2]
    )

    # Draw the rail — extend a bit above the goal and below the start.
    p_goal = ctx.evaluate_link_pose(EE_LINK, goal)[:3, 3]
    env.draw_rod(
        p1=[float(p0[0]), float(p0[1]), float(p0[2]) - 0.10],
        p2=[float(p0[0]), float(p0[1]), float(p_goal[2]) + 0.10],
        radius=0.008,
        color=(0.95, 0.85, 0.10, 1.0),
    )

    run_demo(env, planner, start, goal, "line_v: gripper slides up a vertical rail")


if __name__ == "__main__":
    Fire(main)
