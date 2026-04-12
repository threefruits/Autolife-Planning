"""Long-horizon pick-and-place inside the full RLS-env room.

A single end-to-end showcase of the project's planning stack:

* **Subgroup planning** — navigation uses ``autolife_base`` with
  BiT\\* (asymptotically optimal), grasping uses
  ``autolife_torso_left_arm`` with RRTConnect (feasibility).
* **Collision avoidance** — every planning call is handed the
  151k-point cloud concatenated from all seven ``pcd/*.ply`` files in
  ``assets/envs/rls_env``.  VAMP's SIMD checker keeps it real-time.
* **IK for world-frame grasps** — grasp poses are hard-coded top-down
  SE(3) transforms in the world frame.  TRAC-IK on the
  ``whole_body_base_left`` chain (base_link = ``Link_Zero_Point``,
  world origin) turns them into arm joint goals; the base/legs/waist
  are tight-clamped to their current values so only the arm moves.
* **Constrained planning** — every pregrasp→grasp and grasp→lift
  motion is planned on a CasADi straight-line manifold (TCP pinned
  to the approach axis) via ``ProjectedStateSpace``.
* **Leg stability** — the legs never move.  This is enforced
  *structurally* by the subgroup choices: ``autolife_base`` (3 DOF)
  and ``autolife_torso_left_arm`` (9 DOF) do not include ankle/knee,
  so those joints are pinned at the caller's ``base_config`` by the
  C++ collision checker on every query.  A runtime assert verifies
  the invariant after every phase.

Storyline (one PyBullet window, one concatenated path):

    1. robot spawns inside the room next to the big table
    2. pick apple from the table top
    3. place apple back on the table (different spot)
    4. base-navigate across the room to the sofa (BiT\\*)
    5. pick bottle from the sofa
    6. base-navigate back to the table
    7. place bottle beside the apple

Usage::

    pixi run python examples/rls_pick_place_demo.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pybullet as pb
import trimesh
from fire import Fire

from autolife_planning.config.robot_config import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.kinematics.trac_ik_solver import TracIKSolver
from autolife_planning.planning import Constraint, SymbolicContext, create_planner
from autolife_planning.types import (
    ChainConfig,
    IKConfig,
    PlannerConfig,
    SE3Pose,
    SolveType,
)

# ── Constants ──────────────────────────────────────────────────────
# The four planning subgroups this demo exercises — see
# ``autolife_planning.config.robot_config.PLANNING_SUBGROUPS`` for the
# joint lists.
BASE_SUBGROUP = "autolife_base"  # 3 DOF: virtual x/y/yaw
HEIGHT_SUBGROUP = "autolife_height"  # 3 DOF: ankle, knee, waist_pitch
ARM_SUBGROUP = "autolife_torso_left_arm"  # 9 DOF: 2 waist + 7 left arm
BODY_SUBGROUP = "autolife_body"  # 21 DOF: legs + waist + arms + neck

# Full-DOF layout reminder:
#   [0:3]   virtual base (x, y, theta)
#   [3:5]   legs (ankle, knee)
#   [5:7]   waist (pitch, yaw)
#   [7:14]  left arm
#   [14:17] neck
#   [17:24] right arm
#
# Subgroup indices inside the 24-DOF body vector.
TORSO_ARM_IDX = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13])  # waist_pitch,yaw + left arm
HEIGHT_IDX = np.array([3, 4, 5])  # ankle, knee, waist_pitch
BODY_IDX = np.array(
    [  # everything except base
        3,
        4,
        5,
        6,  # ankle, knee, waist_pitch, waist_yaw
        7,
        8,
        9,
        10,
        11,
        12,
        13,  # left arm
        14,
        15,
        16,  # neck
        17,
        18,
        19,
        20,
        21,
        22,
        23,  # right arm
    ]
)
# Inside the 21-element autolife_body active vector, ankle and knee
# are joints 0 and 1 — used by the leg-pin constraint below.
BODY_ANKLE_IDX = 0
BODY_KNEE_IDX = 1

# The rigid gripper body frame.  We use this link for both the IK
# target and the line constraint so the two stay consistent — the
# finger tips live at a fixed rigid offset from it.
GRIPPER_LINK = "Link_Left_Gripper"

# Offset from Link_Left_Gripper origin to the finger midpoint,
# expressed in the gripper's local frame.  Constant rigid-body
# property of the URDF (measured once via pinocchio FK at HOME).
# Used to convert a desired finger midpoint target into a gripper
# link target for the IK solver.
FINGER_MIDPOINT_IN_GRIPPER = np.array([-0.031, -0.064, 0.0])

# Asset paths.
RLS_ROOT = "assets/envs/rls_env"
MESH_DIR = f"{RLS_ROOT}/meshes"
PCD_DIR = f"{RLS_ROOT}/pcd"

# Scene props — these seven meshes and pcds share a single room frame
# so they load at identity.  The tea_table mesh matches the pcd named
# "coffee_table" (same bounding box, different filename).
SCENE_PROPS: list[tuple[str, str]] = [
    # (mesh name, pcd name)
    ("rls_2", "rls_2"),
    ("open_kitchen", "open_kitchen"),
    ("wall", "wall"),
    ("workstation", "workstation"),
    ("table", "table"),
    ("sofa", "sofa"),
    ("tea_table", "coffee_table"),
]

# ── Hard-coded poses ────────────────────────────────────────────────
# Robot base poses (Virtual_X, Virtual_Y, Virtual_Theta) — each one
# picked after probing FK+IK so the left-shoulder-at-HOME can reach
# the target grasp pose on the corresponding surface.
# Squat stance for the under-the-table pick.  The 3 height joints
# (ankle, knee, waist_pitch) bend the robot down before the arm
# reaches in between the table legs.  Picked after IK+self-collision
# probing: a deeper squat forces the upper body's waist_pitch to
# fold further forward to reach the apple, and the arm collides with
# the legs.  This "half-squat" leaves enough room for the arm to
# snake under the table without self-colliding.
SQUAT_ANKLE = 0.6
SQUAT_KNEE = 1.2
SQUAT_WAIST_PITCH = 0.4
STANDING_STANCE = np.array([0.0, 0.0, 0.0])  # [ankle, knee, waist_pitch]
SQUAT_STANCE = np.array([SQUAT_ANKLE, SQUAT_KNEE, SQUAT_WAIST_PITCH])

BASE_NEAR_TABLE_EAST = np.array([-1.30, 1.67, np.pi])  # east of dining table, facing -x
BASE_NEAR_SOFA = np.array([2.00, 1.30, -np.pi / 2])  # north of sofa, facing -y
BASE_NEAR_COFFEE = np.array(
    [3.60, 1.30, -np.pi / 2]
)  # north of coffee table, facing -y
BASE_NEAR_KITCHEN = np.array(
    [-2.50, -1.85, -np.pi / 2]
)  # north of kitchen counter, facing -y
BASE_NORTH_OF_TABLE = np.array(
    [-2.30, 2.95, -np.pi / 2]
)  # north of dining table, facing -y

# Surface tops (world z) — measured from the pcd z-histograms.
Z_TABLE_TOP = 0.75
Z_COFFEE_TOP = 0.58
Z_KITCHEN_TOP = 0.88

# Graspable meshes.  For each object we store:
#   *_MESH_INIT    — where the mesh is placed at scene start
#   *_MESH_PLACE   — where the mesh should end up after the place action
#   *_GRASP_Z      — vertical offset from the mesh base to the grasp point
#                    (roughly half the object's height)
#   *_APPROACH     — world-frame approach direction (unit vector) used
#                    for BOTH pick and place, so the object's offset
#                    relative to the gripper stays identical through
#                    the whole motion
# GRASP_Z gives enough vertical clearance from the gripper body to
# the destination surface — the gripper body is ~8 cm wide, so at
# least 0.12 m of headroom above the target surface is needed for
# the planner's collision check to pass.
APPLE_MESH_INIT = np.array([-2.05, 1.67, 0.32])  # under the dining table, 32 cm up
APPLE_MESH_PLACE = np.array([-2.05, 1.67, 0.75])  # on the east edge of the dining table
APPLE_GRASP_Z = 0.13
APPLE_APPROACH = np.array([-1.0, 0.0, 0.0])  # horizontal -x (reach in from east)

BOTTLE_MESH_INIT = np.array([1.95, 0.30, 0.69])  # on the sofa top
BOTTLE_MESH_PLACE = np.array([3.60, 0.30, 0.58])  # on the coffee table top
BOTTLE_GRASP_Z = 0.14
BOTTLE_APPROACH = np.array([0.0, -1.0, 0.0])  # horizontal -y (reach from north)

CHIPBOX_MESH_INIT = np.array([-2.50, -2.40, 0.88])  # on the kitchen counter (near edge)
CHIPBOX_MESH_PLACE = np.array(
    [-2.30, 2.20, 0.75]
)  # on the north side of the dining table
CHIPBOX_GRASP_Z = 0.16
CHIPBOX_APPROACH = np.array([0.0, -1.0, 0.0])  # horizontal -y (reach from north)

# Distance from pregrasp to grasp along the approach axis.
PREGRASP_OFFSET = 0.12
# Under-the-table pick uses a shorter offset — the 21-DOF body IK
# has a hard time finding self-collision-free branches when the
# gripper sweeps >10 cm at a bent-torso squat stance.
PREGRASP_OFFSET_UNDER_TABLE = 0.05

# Planning budgets.
ARM_FREE_TIME = 4.0
ARM_LINE_TIME = 8.0
BASE_NAV_TIME = 6.0


# ── Scene setup ────────────────────────────────────────────────────


def load_room_meshes(env: PyBulletEnv) -> None:
    """Load every rls_env scene prop at identity — they already share a frame."""
    for mesh_name, _ in SCENE_PROPS:
        path = os.path.abspath(f"{MESH_DIR}/{mesh_name}/{mesh_name}.obj")
        env.add_mesh(path, position=np.zeros(3))


def load_room_pointcloud(stride: int = 1) -> np.ndarray:
    """Concatenate every rls_env pcd into one (N, 3) float32 array."""
    chunks: list[np.ndarray] = []
    for _, pcd_name in SCENE_PROPS:
        p = os.path.abspath(f"{PCD_DIR}/{pcd_name}.ply")
        pc = trimesh.load(p)
        v = np.asarray(pc.vertices, dtype=np.float32)  # type: ignore[union-attr]
        if stride > 1:
            v = v[::stride]
        chunks.append(v)
    return np.concatenate(chunks, axis=0)


def place_graspable(env: PyBulletEnv, mesh_name: str, xyz: np.ndarray) -> int:
    """Add a movable mesh at *xyz* (world).  Returns the PyBullet body id."""
    path = os.path.abspath(f"{MESH_DIR}/{mesh_name}/{mesh_name}.obj")
    return env.add_mesh(path, position=np.asarray(xyz, dtype=float))


# ── CasADi / SymbolicContext helpers ───────────────────────────────


def gripper_translation(ctx: SymbolicContext) -> ca.SX:
    """Symbolic position of the rigid left-gripper body frame."""
    return ctx.link_translation(GRIPPER_LINK)


def make_line_constraint(
    ctx: SymbolicContext,
    p_from: np.ndarray,
    p_to: np.ndarray,
    name: str,
) -> Constraint:
    """Pin the gripper link to the line through *p_from* → *p_to*.

    Residual has two rows — the two coordinates orthogonal to the line
    direction — so the gripper is free to slide along the line but
    cannot leave it.  Projection is handled by OMPL's
    ProjectedStateSpace.
    """
    p_from = np.asarray(p_from, dtype=float)
    p_to = np.asarray(p_to, dtype=float)
    d = p_to - p_from
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-9:
        raise ValueError("make_line_constraint: p_from and p_to coincide")
    d = d / d_norm
    # Pick any world axis that isn't (anti)parallel to d.
    seed = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, seed)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    gripper = gripper_translation(ctx)
    diff = gripper - ca.DM(p_from.tolist())
    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        ca.dot(diff, ca.DM(u.tolist())),
        ca.dot(diff, ca.DM(v.tolist())),
    )
    return Constraint(residual=residual, q_sym=ctx.q, name=name)


# ── World-frame IK ─────────────────────────────────────────────────


@dataclass
class IKBundle:
    """Cached TRAC-IK solver for the whole_body_base_left chain."""

    solver: TracIKSolver
    lower: np.ndarray
    upper: np.ndarray


def build_ik_bundle() -> IKBundle:
    """TRAC-IK solver from ``Link_Zero_Point`` (world origin) to
    ``Link_Left_Gripper`` (the same link used by the line constraint).

    The stock ``whole_body_base_left`` chain targets the wrist, which
    is ~18 cm behind the gripper body — so we bypass the factory and
    build a local :class:`ChainConfig` that targets the gripper link
    directly.  Nothing in the library is modified.
    """
    urdf_path = CHAIN_CONFIGS["whole_body_base_left"].urdf_path
    chain = ChainConfig(
        base_link="Link_Zero_Point",
        ee_link=GRIPPER_LINK,
        num_joints=14,
        urdf_path=urdf_path,
    )
    cfg = IKConfig(
        timeout=0.1,
        epsilon=1e-5,
        solve_type=SolveType.DISTANCE,
        max_attempts=10,
    )
    solver = TracIKSolver(chain, cfg)
    lo, hi = solver.joint_limits
    # Shrink the IK solver's bounds by a margin so solutions stay
    # comfortably inside the downstream OMPL/VAMP bounds.  CasADi
    # projection can nudge joints by up to ~0.03 rad, so a 0.05 rad
    # inset guarantees the projected configs stay in-bounds.
    margin = 0.05
    lo_tight = lo + margin
    hi_tight = hi - margin
    solver.set_joint_limits(lo_tight, hi_tight)
    return IKBundle(solver=solver, lower=lo_tight.copy(), upper=hi_tight.copy())


def solve_world_arm_goal(
    ik: IKBundle,
    current_full: np.ndarray,
    world_pose: SE3Pose,
    *,
    clamp_count: int = 5,
) -> np.ndarray:
    """Solve torso+left-arm joint values for a world-frame gripper pose.

    * ``current_full`` is the live 24-DOF body configuration.
    * The IK chain has 14 DOF: ``[vx, vy, vtheta, ankle, knee,
      waist_pitch, waist_yaw, left_arm×7]`` — exactly ``current_full[:14]``.
    * ``clamp_count`` — number of leading joints to tight-clamp so they
      stay at their seed values.  ``5`` (default) pins base + legs,
      leaving waist + arm free for the ``autolife_torso_left_arm``
      subgroup.  ``5`` also works for the ``autolife_body`` subgroup,
      because the leg-pin CasADi constraint handles ankle/knee there.
    """
    seed = np.array(current_full[:14], dtype=np.float64)
    eps = 1e-4
    lower = ik.lower.copy()
    upper = ik.upper.copy()
    for i in range(clamp_count):
        lower[i] = seed[i] - eps
        upper[i] = seed[i] + eps
    ik.solver.set_joint_limits(lower, upper)
    try:
        result = ik.solver.solve(world_pose, seed=seed)
    finally:
        ik.solver.set_joint_limits(ik.lower, ik.upper)

    if not result.success or result.joint_positions is None:
        raise RuntimeError(
            f"IK failed for target {world_pose.position.tolist()}: "
            f"status={result.status.value} pos_err={result.position_error:.4f}"
        )
    # TRAC-IK occasionally returns joint values a hair outside its
    # own joint limits — clamp them back inside with a small inset
    # so the downstream OMPL planner doesn't reject the goal.
    clamped = np.clip(
        result.joint_positions,
        ik.lower + 1e-4,
        ik.upper - 1e-4,
    )
    new_full = current_full.copy()
    new_full[:14] = clamped
    # Re-force clamped joints exactly — the solver is within ±eps.
    new_full[:clamp_count] = current_full[:clamp_count]
    return new_full


# ── Planning wrappers ──────────────────────────────────────────────


def _extract_torso_arm(full: np.ndarray) -> np.ndarray:
    return full[TORSO_ARM_IDX].copy()


def plan_arm_free(
    current_full: np.ndarray,
    goal_full: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """RRTConnect over the 9-DOF torso+arm subgroup.  Returns (N, 24)."""
    planner = create_planner(
        ARM_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_FREE_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
    )
    start = _extract_torso_arm(current_full)
    goal = _extract_torso_arm(goal_full)
    result = planner.plan(start, goal)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def plan_arm_line(
    current_full: np.ndarray,
    goal_full: np.ndarray,
    p_from: np.ndarray,
    p_to: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """Line-constrained plan on the torso+arm subgroup.  Returns (N, 24)."""
    ctx = SymbolicContext(ARM_SUBGROUP, base_config=current_full)
    # CasADi function names must be valid C identifiers — letters,
    # digits, or NON-consecutive underscores.  Sanitise the label.
    safe_chars = [c if c.isalnum() else "_" for c in label]
    safe = "".join(safe_chars)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    constraint = make_line_constraint(ctx, p_from, p_to, name=f"line_{safe}")

    raw_start = _extract_torso_arm(current_full)
    raw_goal = _extract_torso_arm(goal_full)

    planner = create_planner(
        ARM_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_LINE_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
        constraints=[constraint],
    )
    lo = np.array(planner._planner.lower_bounds())
    hi = np.array(planner._planner.upper_bounds())
    # Iterative project→clamp: projection lands exactly on the
    # manifold but may violate joint bounds; clamping respects bounds
    # but moves off the manifold.  A few rounds converge to both.
    start = _project_and_clamp(ctx, constraint.residual, raw_start, lo, hi)
    goal = _project_and_clamp(ctx, constraint.residual, raw_goal, lo, hi)
    result = planner.plan(start, goal)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def plan_base(
    current_full: np.ndarray,
    base_goal: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """BiT\\* base navigation (3 DOF: x, y, theta).  Returns (N, 24)."""
    planner = create_planner(
        BASE_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=BASE_NAV_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
    )
    start = current_full[:3].copy()
    result = planner.plan(start, np.asarray(base_goal, dtype=np.float64))
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


# ── Height (squat/stand) + upper-body planners ────────────────────


def plan_height(
    current_full: np.ndarray,
    target_stance: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """Plan the 3-DOF height chain (ankle + knee + waist_pitch).

    ``target_stance`` is a 3-vector ``[ankle, knee, waist_pitch]``.
    Used to squat the robot down before the under-table pick and to
    stand it back up afterwards.  Returns a full ``(N, 24)`` path.
    """
    planner = create_planner(
        HEIGHT_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_FREE_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
    )
    start = current_full[HEIGHT_IDX].copy()
    result = planner.plan(start, np.asarray(target_stance, dtype=np.float64))
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def make_leg_pin_constraint(
    ctx: SymbolicContext,
    ankle_value: float,
    knee_value: float,
    name: str,
) -> Constraint:
    """Pin the ankle and knee to given stance values.

    The constraint lives on the ``autolife_body`` 21-DOF subgroup —
    inside that active vector ankle is index 0 and knee is index 1.
    Combined with the gripper line constraint during the pregrasp →
    grasp sweep, this says "hold the legs at the squat stance while
    the whole upper body (waist + arms + neck) reaches in under the
    table".
    """
    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        ctx.q[BODY_ANKLE_IDX] - ca.DM(float(ankle_value)),
        ctx.q[BODY_KNEE_IDX] - ca.DM(float(knee_value)),
    )
    return Constraint(residual=residual, q_sym=ctx.q, name=name)


def _project_and_clamp(
    ctx: SymbolicContext,
    residual: ca.SX,
    q_init: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iters: int = 5,
) -> np.ndarray:
    """Project ``q_init`` onto the manifold ``residual(q)=0``, then
    clamp to ``[lo, hi]``, and re-project.  A few rounds converge to
    a state that is both on-manifold AND within bounds."""
    q = q_init.copy()
    for _ in range(max_iters):
        q = ctx.project(q, residual)
        q = np.clip(q, lo + 1e-5, hi - 1e-5)
    return q


def _sanitize(label: str) -> str:
    """CasADi function names must be valid C identifiers — letters,
    digits, or NON-consecutive underscores."""
    safe = "".join(c if c.isalnum() else "_" for c in label)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def plan_body_free(
    current_full: np.ndarray,
    goal_full: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """RRTConnect on the 21-DOF ``autolife_body`` subgroup with only the
    leg-pin constraint (ankle + knee locked at the squat stance).
    Used for free-plan current → pregrasp during the under-table pick.
    """
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    leg_pin = make_leg_pin_constraint(
        ctx,
        float(current_full[3]),
        float(current_full[4]),
        name=f"legpin_{_sanitize(label)}",
    )
    raw_start = current_full[BODY_IDX].copy()
    raw_goal = goal_full[BODY_IDX].copy()

    planner = create_planner(
        BODY_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_FREE_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
        constraints=[leg_pin],
    )
    lo = np.array(planner._planner.lower_bounds())
    hi = np.array(planner._planner.upper_bounds())
    start = _project_and_clamp(ctx, leg_pin.residual, raw_start, lo, hi)
    goal = _project_and_clamp(ctx, leg_pin.residual, raw_goal, lo, hi)
    result = planner.plan(start, goal)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def plan_body_line(
    current_full: np.ndarray,
    goal_full: np.ndarray,
    p_from: np.ndarray,
    p_to: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """Line-constrained plan on the 21-DOF ``autolife_body`` subgroup.

    Combines the leg-pin constraint (ankle + knee frozen at the squat
    stance) with a 2-equation gripper line constraint along
    ``p_from`` → ``p_to``.  Four CasADi equations on 21 DOF — the
    planner has 17 effective degrees of freedom to bend the upper
    body while the TCP slides along a straight line.
    """
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    safe = _sanitize(label)

    # Build the combined residual manually so both constraints share
    # the same CasADi symbol ``ctx.q`` and can be compiled as one
    # compiled SX Function.
    leg_pin_res: ca.SX = ca.vertcat(  # type: ignore[assignment]
        ctx.q[BODY_ANKLE_IDX] - ca.DM(float(current_full[3])),
        ctx.q[BODY_KNEE_IDX] - ca.DM(float(current_full[4])),
    )
    # Re-derive the same u,v basis used by make_line_constraint.
    p_from_np = np.asarray(p_from, dtype=float)
    p_to_np = np.asarray(p_to, dtype=float)
    d = p_to_np - p_from_np
    d = d / float(np.linalg.norm(d))
    seed_vec = (
        np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(d, seed_vec)
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)

    gripper = ctx.link_translation(GRIPPER_LINK)
    diff = gripper - ca.DM(p_from_np.tolist())
    line_res: ca.SX = ca.vertcat(  # type: ignore[assignment]
        ca.dot(diff, ca.DM(u.tolist())),
        ca.dot(diff, ca.DM(v.tolist())),
    )
    combined: ca.SX = ca.vertcat(leg_pin_res, line_res)  # type: ignore[assignment]
    constraint = Constraint(residual=combined, q_sym=ctx.q, name=f"body_legline_{safe}")

    raw_start = current_full[BODY_IDX].copy()
    raw_goal = goal_full[BODY_IDX].copy()

    planner = create_planner(
        BODY_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_LINE_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
        constraints=[constraint],
    )
    lo = np.array(planner._planner.lower_bounds())
    hi = np.array(planner._planner.upper_bounds())
    start = _project_and_clamp(ctx, constraint.residual, raw_start, lo, hi)
    goal = _project_and_clamp(ctx, constraint.residual, raw_goal, lo, hi)
    result = planner.plan(start, goal)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def _solve_validated(
    ik: IKBundle,
    current_full: np.ndarray,
    world_pose: SE3Pose,
    validator,
    label: str,
    clamp_count: int = 5,
    max_attempts: int = 200,
) -> np.ndarray:
    """Solve IK and retry with new random seeds until the planner
    accepts the resulting config as collision-free.

    TRAC-IK is non-deterministic — especially for the 21-DOF
    ``autolife_body`` subgroup with the torso already bent, it lands
    on different kinematic branches across runs and the solver's
    SolveType.DISTANCE option only guarantees proximity to the seed,
    not self-collision safety.  We re-roll until ``validator(full)``
    returns ``True`` or we run out of attempts.
    """
    original = current_full.copy()
    for attempt in range(max_attempts):
        try:
            full = solve_world_arm_goal(
                ik, current_full, world_pose, clamp_count=clamp_count
            )
        except RuntimeError:
            # IK itself failed — jitter and try again.
            jitter = np.random.uniform(-0.2, 0.2, size=original.shape[0])
            jitter[:clamp_count] = 0.0
            current_full = original + jitter
            continue
        if validator(full):
            return full
        # Randomize the non-clamped joints slightly to land on a
        # different kinematic branch next call.  Small jitter
        # encourages nearby branches; bigger jitter every few tries
        # helps escape local minima.
        magnitude = 0.1 if attempt < 50 else 0.3
        jitter = np.random.uniform(-magnitude, magnitude, size=original.shape[0])
        jitter[:clamp_count] = 0.0
        current_full = original + jitter
    raise RuntimeError(
        f"{label}: no collision-free IK solution found after {max_attempts} attempts"
    )


def pick_upper_body(
    ik: IKBundle,
    current_full: np.ndarray,
    finger_target_xyz: np.ndarray,
    approach_world: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Like :func:`pick`, but plans on the 21-DOF ``autolife_body``
    subgroup with a leg-pin CasADi constraint holding ankle + knee
    frozen.  The robot must already be in its squat stance before
    this function is called — pass the squatted ``current_full``."""
    grasp_pose, pregrasp_pose, grasp_pos, pregrasp_pos = build_grasp_pose(
        finger_target_xyz,
        approach_world,
        pregrasp_offset=PREGRASP_OFFSET_UNDER_TABLE,
    )

    # Build a throwaway planner to validate candidate IK solutions
    # against the same leg-pin constraint the real planner will use.
    val_ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    val_leg_pin = make_leg_pin_constraint(
        val_ctx,
        float(current_full[3]),
        float(current_full[4]),
        name=f"legpin_{_sanitize(label)}_val",
    )
    val_planner = create_planner(
        BODY_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=0.1),
        pointcloud=pointcloud,
        base_config=current_full,
        constraints=[val_leg_pin],
    )
    val_lo = np.array(val_planner._planner.lower_bounds())
    val_hi = np.array(val_planner._planner.upper_bounds())
    val_margin = 1e-5

    def _validate(full: np.ndarray) -> bool:
        reduced = full[BODY_IDX]
        projected = val_ctx.project(reduced, val_leg_pin.residual)
        if np.any(projected < val_lo + val_margin) or np.any(
            projected > val_hi - val_margin
        ):
            return False
        return val_planner.validate(projected)

    # IK clamps base + legs (0..4), leaves waist + arm free — the
    # leg-pin constraint in the planner handles ankle/knee pinning.
    #
    # We need BOTH the grasp and the pregrasp (derived by projection)
    # to be collision-free.  Even when TRAC-IK finds a valid grasp
    # branch, the 5 cm projection to the pregrasp can wobble the arm
    # through a forbidden zone.  Re-roll the whole pair until both
    # configs validate.
    original_seed = current_full.copy()
    grasp_full = None
    pregrasp_full = None
    for pair_attempt in range(80):
        jitter = np.random.uniform(-0.15, 0.15, size=original_seed.shape[0])
        jitter[:5] = 0.0
        seed = original_seed + jitter
        try:
            grasp_full = _solve_validated(
                ik,
                seed,
                grasp_pose,
                _validate,
                f"{label} grasp (pair {pair_attempt})",
                clamp_count=5,
                max_attempts=20,
            )
        except RuntimeError:
            continue
        pregrasp_full = _derive_pregrasp_from_grasp(
            grasp_full,
            BODY_SUBGROUP,
            BODY_IDX,
            pregrasp_pos,
            base_config_override=original_seed,
            pin_legs=True,
        )
        if _validate(pregrasp_full):
            break
    else:
        raise RuntimeError(
            f"{label}: no grasp+pregrasp pair was collision-free after 80 attempts"
        )
    assert grasp_full is not None and pregrasp_full is not None

    free_path = plan_body_free(
        current_full, pregrasp_full, pointcloud, f"{label} free→pregrasp"
    )
    _assert_legs_frozen(free_path, f"{label} free")

    approach_path = plan_body_line(
        pregrasp_full,
        grasp_full,
        pregrasp_pos,
        grasp_pos,
        pointcloud,
        f"{label} approach",
    )
    _assert_legs_frozen(approach_path, f"{label} approach")

    lift_path = plan_body_line(
        grasp_full,
        pregrasp_full,
        grasp_pos,
        pregrasp_pos,
        pointcloud,
        f"{label} lift",
    )
    _assert_legs_frozen(lift_path, f"{label} lift")

    return [free_path, approach_path], [lift_path], pregrasp_full


def _report(label: str, result) -> None:
    n = result.path.shape[0] if result.path is not None else 0
    ms = result.planning_time_ns / 1e6
    print(
        f"  [{label}] {result.status.value}: {n} wp in {ms:.0f} ms"
        + (f", cost {result.path_cost:.2f}" if result.success else "")
    )


# ── Grasp pose builder ─────────────────────────────────────────────


def grasp_rotation_from_approach(approach_world: np.ndarray) -> np.ndarray:
    """Rotation matrix for a grasp whose approach axis is ``approach_world``.

    Convention: the gripper's local ``-Y`` axis (the finger-pointing
    direction) is aligned with the approach direction, and the local
    ``+Z`` axis is kept as close to world ``+Z`` as possible — so the
    gripper stays "upright" for horizontal approaches, and picks an
    arbitrary horizontal reference for vertical (top-down) approaches.

    This single convention handles every surface in the demo: side
    grasps on the sofa and under the dining table, plus top-down
    descents onto the dining / coffee / kitchen tops.
    """
    a = np.asarray(approach_world, dtype=float)
    a = a / float(np.linalg.norm(a))
    y_axis = -a  # gripper +Y = opposite of approach
    # Fallback reference for the gripper's +Z axis — world +Z if the
    # approach is mostly horizontal, world +X otherwise (top/bottom).
    z_ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    z_axis = z_ref - float(np.dot(z_ref, y_axis)) * y_axis
    z_axis /= float(np.linalg.norm(z_axis))
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def build_grasp_pose(
    finger_target_xyz: np.ndarray,
    approach_world: np.ndarray,
    pregrasp_offset: float = PREGRASP_OFFSET,
) -> tuple[SE3Pose, SE3Pose, np.ndarray, np.ndarray]:
    """Return ``(grasp_pose, pregrasp_pose, grasp_pos, pregrasp_pos)``.

    ``finger_target_xyz`` is the point we want the gripper's finger
    midpoint to occupy at grasp time, in world coordinates.  We
    convert that to a gripper-link target for the IK solver by
    subtracting the fixed rigid-body offset
    ``R @ FINGER_MIDPOINT_IN_GRIPPER``.

    The pregrasp position is a fixed distance ``pregrasp_offset``
    backed off along ``approach_world``.  Both poses share the same
    orientation ``R`` so the line constraint only needs to pin the
    two degrees of freedom orthogonal to the line direction.
    """
    R = grasp_rotation_from_approach(approach_world)
    approach = np.asarray(approach_world, dtype=float)
    approach /= float(np.linalg.norm(approach))

    finger_target = np.asarray(finger_target_xyz, dtype=float)
    # gripper_link + R @ FINGER_MIDPOINT_IN_GRIPPER = finger_target
    grasp_pos = finger_target - R @ FINGER_MIDPOINT_IN_GRIPPER
    pregrasp_pos = grasp_pos - pregrasp_offset * approach

    grasp_pose = SE3Pose(position=grasp_pos, rotation=R)
    pregrasp_pose = SE3Pose(position=pregrasp_pos, rotation=R)
    return grasp_pose, pregrasp_pose, grasp_pos, pregrasp_pos


# ── Path segment bookkeeping ───────────────────────────────────────


@dataclass
class Segment:
    """One contiguous path chunk that will be played back.

    ``attach_body_id`` is the PyBullet body id of a movable object that
    should follow the gripper link during this segment's frames.  When
    it is ``None`` the playback loop doesn't touch any graspable.
    """

    path: np.ndarray  # (N, 24)
    attach_body_id: int | None
    attach_local_tf: np.ndarray | None  # (4, 4): mesh pose in gripper frame
    banner: str


def _assert_legs_constant(path: np.ndarray, label: str) -> None:
    """Verify that ankle and knee do not change across *path*.

    The legs are allowed to be in any static stance — HOME standing
    for most of the demo, or the squat stance during the under-table
    pick — but they must not move within a single subgroup planning
    segment.  This invariant catches bugs where a subgroup planner
    unexpectedly includes the leg joints.
    """
    assert np.allclose(
        path[:, 3:5], path[0, 3:5], atol=1e-6
    ), f"{label}: leg invariant broken — ankle/knee moved mid-segment"


def _assert_legs_frozen(path: np.ndarray, label: str) -> None:
    """Back-compat shim — same as :func:`_assert_legs_constant`."""
    _assert_legs_constant(path, label)


# ── High-level actions ─────────────────────────────────────────────


def _torso_arm_validator(current_full: np.ndarray, pointcloud: np.ndarray):
    """Build a validator closure that checks a 24-DOF config against a
    temporary ``autolife_torso_left_arm`` planner.  Used to filter out
    self/scene-colliding IK solutions before we hand them to the real
    planner as start/goal states.

    Also enforces joint bounds — the C++ ``validate`` only checks
    collision, but TRAC-IK and Gauss-Newton projection can both
    return solutions a hair outside the joint limits, and OMPL will
    silently discard them later.
    """
    val = create_planner(
        ARM_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=0.1),
        pointcloud=pointcloud,
        base_config=current_full,
    )
    lo = np.array(val._planner.lower_bounds())
    hi = np.array(val._planner.upper_bounds())
    margin = 1e-5

    def _check(full: np.ndarray) -> bool:
        reduced = full[TORSO_ARM_IDX]
        if np.any(reduced < lo + margin) or np.any(reduced > hi - margin):
            return False
        return val.validate(reduced)

    return _check


def _derive_pregrasp_from_grasp(
    grasp_full: np.ndarray,
    subgroup: str,
    active_idx: np.ndarray,
    gripper_target_pos: np.ndarray,
    base_config_override: np.ndarray | None = None,
    pin_legs: bool = False,
) -> np.ndarray:
    """Project ``grasp_full`` onto a "gripper-link at
    ``gripper_target_pos``" manifold using ``SymbolicContext.project``.

    This avoids TRAC-IK's random-branch issue: the Gauss-Newton
    projection moves the joints minimally from ``grasp_full`` to land
    the gripper at the desired position, keeping the arm on the same
    kinematic branch as the grasp config.  Much more reliable than
    calling TRAC-IK again with a slightly different target.

    For the 21-DOF body subgroup (``pin_legs=True``), we stack an
    ankle+knee pin onto the residual so the projection doesn't drift
    off the squat stance while satisfying the gripper constraint.
    """
    base = base_config_override if base_config_override is not None else grasp_full
    ctx_proj = SymbolicContext(subgroup, base_config=base)
    point_res: ca.SX = ctx_proj.link_translation(GRIPPER_LINK) - ca.DM(
        np.asarray(gripper_target_pos, dtype=float).tolist()
    )
    if pin_legs:
        residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
            ctx_proj.q[BODY_ANKLE_IDX] - ca.DM(float(base[3])),
            ctx_proj.q[BODY_KNEE_IDX] - ca.DM(float(base[4])),
            point_res,
        )
    else:
        residual = point_res
    raw = grasp_full[active_idx].copy()
    projected = ctx_proj.project(raw, residual)
    out = grasp_full.copy()
    out[active_idx] = projected
    return out


def pick(
    ik: IKBundle,
    current_full: np.ndarray,
    finger_target_xyz: np.ndarray,
    approach_world: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Plan current → pregrasp (free) → grasp (line) → pregrasp (line)."""
    grasp_pose, pregrasp_pose, grasp_pos, pregrasp_pos = build_grasp_pose(
        finger_target_xyz, approach_world
    )

    validator = _torso_arm_validator(current_full, pointcloud)
    # Solve grasp IK first, then derive pregrasp by projecting the
    # grasp config onto a gripper-at-pregrasp-pos manifold.  Retry
    # the whole pair until BOTH configs validate — the derived
    # pregrasp can still wobble into collision even when the grasp
    # branch is valid.
    original_seed = current_full.copy()
    grasp_full = None
    pregrasp_full = None
    for pair_attempt in range(80):
        jitter = np.random.uniform(-0.12, 0.12, size=original_seed.shape[0])
        jitter[:5] = 0.0
        seed = original_seed + jitter
        try:
            grasp_full = _solve_validated(
                ik,
                seed,
                grasp_pose,
                validator,
                f"{label} grasp (pair {pair_attempt})",
                max_attempts=15,
            )
        except RuntimeError:
            continue
        pregrasp_full = _derive_pregrasp_from_grasp(
            grasp_full,
            ARM_SUBGROUP,
            TORSO_ARM_IDX,
            pregrasp_pos,
        )
        if validator(pregrasp_full):
            break
    else:
        raise RuntimeError(
            f"{label}: no grasp+pregrasp pair was collision-free after 80 attempts"
        )
    assert grasp_full is not None and pregrasp_full is not None

    free_path = plan_arm_free(
        current_full, pregrasp_full, pointcloud, f"{label} free→pregrasp"
    )
    _assert_legs_frozen(free_path, f"{label} free")

    approach_path = plan_arm_line(
        pregrasp_full,
        grasp_full,
        pregrasp_pos,
        grasp_pos,
        pointcloud,
        f"{label} approach",
    )
    _assert_legs_frozen(approach_path, f"{label} approach")

    lift_path = plan_arm_line(
        grasp_full,
        pregrasp_full,
        grasp_pos,
        pregrasp_pos,
        pointcloud,
        f"{label} lift",
    )
    _assert_legs_frozen(lift_path, f"{label} lift")

    return [free_path, approach_path], [lift_path], pregrasp_full


def place(
    ik: IKBundle,
    current_full: np.ndarray,
    finger_target_xyz: np.ndarray,
    approach_world: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Plan current → pre-place (free) → place (line) → pre-place (line)."""
    place_pose, pre_place_pose, place_pos, pre_place_pos = build_grasp_pose(
        finger_target_xyz, approach_world
    )

    validator = _torso_arm_validator(current_full, pointcloud)
    original_seed = current_full.copy()
    place_full = None
    pre_place_full = None
    for pair_attempt in range(80):
        jitter = np.random.uniform(-0.12, 0.12, size=original_seed.shape[0])
        jitter[:5] = 0.0
        seed = original_seed + jitter
        try:
            place_full = _solve_validated(
                ik,
                seed,
                place_pose,
                validator,
                f"{label} place (pair {pair_attempt})",
                max_attempts=15,
            )
        except RuntimeError:
            continue
        pre_place_full = _derive_pregrasp_from_grasp(
            place_full,
            ARM_SUBGROUP,
            TORSO_ARM_IDX,
            pre_place_pos,
        )
        if validator(pre_place_full):
            break
    else:
        raise RuntimeError(
            f"{label}: no place+pre_place pair was collision-free after 80 attempts"
        )
    assert place_full is not None and pre_place_full is not None

    carry_free = plan_arm_free(
        current_full, pre_place_full, pointcloud, f"{label} free→preplace"
    )
    _assert_legs_frozen(carry_free, f"{label} carry")

    lower = plan_arm_line(
        pre_place_full,
        place_full,
        pre_place_pos,
        place_pos,
        pointcloud,
        f"{label} lower",
    )
    _assert_legs_frozen(lower, f"{label} lower")

    retreat = plan_arm_line(
        place_full,
        pre_place_full,
        place_pos,
        pre_place_pos,
        pointcloud,
        f"{label} retreat",
    )
    _assert_legs_frozen(retreat, f"{label} retreat")

    return [carry_free, lower], [retreat], pre_place_full


def navigate(
    current_full: np.ndarray,
    base_goal: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """BiT\\* base nav; returns the (N, 24) full-DOF path."""
    path = plan_base(current_full, base_goal, pointcloud, label)
    _assert_legs_constant(path, label)
    return path


def squat_or_stand(
    current_full: np.ndarray,
    target_stance: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """Plan the height chain from the current stance to *target_stance*.

    Legs DO move here (that's the whole point), so we skip the usual
    leg-invariance assertion.
    """
    return plan_height(current_full, target_stance, pointcloud, label)


# ── Stage-level orchestration helpers ──────────────────────────────


def run_pick(
    env: PyBulletEnv,
    ik: IKBundle,
    cloud: np.ndarray,
    current_full: np.ndarray,
    segments: list[Segment],
    *,
    label: str,
    finger_target: np.ndarray,
    approach_world: np.ndarray,
    object_id: int,
    object_initial_world_pos: np.ndarray,
    gripper_link_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan free→pregrasp + line approach + line lift, and capture the
    grasped object's rigid offset relative to the gripper link."""
    for _retry in range(10):
        try:
            pick_free, pick_lift, _ = pick(
                ik, current_full, finger_target, approach_world, cloud, label
            )
            break
        except RuntimeError as e:
            if _retry == 9:
                raise
            print(f"    [{label}] retry {_retry+1}: {e}")
    for p in pick_free:
        segments.append(
            Segment(
                path=p,
                attach_body_id=None,
                attach_local_tf=None,
                banner=f"{label}: approach",
            )
        )

    grasp_frame = pick_free[-1][-1]
    env.set_configuration(grasp_frame)
    env.sim.client.resetBasePositionAndOrientation(
        object_id, object_initial_world_pos.tolist(), [0.0, 0.0, 0.0, 1.0]
    )
    local_tf = capture_local_transform(env, gripper_link_idx, object_id)

    for p in pick_lift:
        segments.append(
            Segment(
                path=p,
                attach_body_id=object_id,
                attach_local_tf=local_tf,
                banner=f"{label}: lift",
            )
        )
    return pick_lift[-1][-1], local_tf


def run_place(
    ik: IKBundle,
    cloud: np.ndarray,
    current_full: np.ndarray,
    segments: list[Segment],
    *,
    label: str,
    finger_target: np.ndarray,
    approach_world: np.ndarray,
    object_id: int,
    local_tf: np.ndarray,
) -> np.ndarray:
    """Plan free→pre-place + line lower + line retreat, releasing the
    grasped object at the place frame."""
    for _retry in range(10):
        try:
            place_carry, place_retreat, _ = place(
                ik, current_full, finger_target, approach_world, cloud, label
            )
            break
        except RuntimeError as e:
            if _retry == 9:
                raise
            print(f"    [{label}] retry {_retry+1}: {e}")
    for p in place_carry:
        segments.append(
            Segment(
                path=p,
                attach_body_id=object_id,
                attach_local_tf=local_tf,
                banner=f"{label}: carry",
            )
        )
    for p in place_retreat:
        segments.append(
            Segment(
                path=p,
                attach_body_id=None,
                attach_local_tf=None,
                banner=f"{label}: retreat",
            )
        )
    return place_retreat[-1][-1]


def run_nav(
    current_full: np.ndarray,
    base_goal: np.ndarray,
    cloud: np.ndarray,
    segments: list[Segment],
    *,
    label: str,
    attach_body_id: int | None = None,
    attach_local_tf: np.ndarray | None = None,
) -> np.ndarray:
    """BiT\\* base nav with optional carried object."""
    nav_path = navigate(current_full, base_goal, cloud, label)
    segments.append(
        Segment(
            path=nav_path,
            attach_body_id=attach_body_id,
            attach_local_tf=attach_local_tf,
            banner=label,
        )
    )
    return nav_path[-1]


def run_squat(
    current_full: np.ndarray,
    target_stance: np.ndarray,
    cloud: np.ndarray,
    segments: list[Segment],
    *,
    label: str,
    attach_body_id: int | None = None,
    attach_local_tf: np.ndarray | None = None,
) -> np.ndarray:
    """Plan the height chain and append it as a segment.  Used for
    both squatting down and standing back up."""
    path = squat_or_stand(current_full, target_stance, cloud, label)
    segments.append(
        Segment(
            path=path,
            attach_body_id=attach_body_id,
            attach_local_tf=attach_local_tf,
            banner=label,
        )
    )
    return path[-1]


def run_upper_body_pick(
    env: PyBulletEnv,
    ik: IKBundle,
    cloud: np.ndarray,
    current_full: np.ndarray,
    segments: list[Segment],
    *,
    label: str,
    finger_target: np.ndarray,
    approach_world: np.ndarray,
    object_id: int,
    object_initial_world_pos: np.ndarray,
    gripper_link_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same shape as :func:`run_pick`, but plans the grasp on the
    ``autolife_body`` subgroup with a leg-pin CasADi constraint.  The
    robot must already be in its squat stance before this is called."""
    for _retry in range(10):
        try:
            pick_free, pick_lift, _ = pick_upper_body(
                ik, current_full, finger_target, approach_world, cloud, label
            )
            break
        except RuntimeError as e:
            if _retry == 9:
                raise
            print(f"    [{label}] retry {_retry+1}: {e}")
    for p in pick_free:
        segments.append(
            Segment(
                path=p,
                attach_body_id=None,
                attach_local_tf=None,
                banner=f"{label}: approach",
            )
        )

    grasp_frame = pick_free[-1][-1]
    env.set_configuration(grasp_frame)
    env.sim.client.resetBasePositionAndOrientation(
        object_id, object_initial_world_pos.tolist(), [0.0, 0.0, 0.0, 1.0]
    )
    local_tf = capture_local_transform(env, gripper_link_idx, object_id)

    for p in pick_lift:
        segments.append(
            Segment(
                path=p,
                attach_body_id=object_id,
                attach_local_tf=local_tf,
                banner=f"{label}: lift",
            )
        )
    return pick_lift[-1][-1], local_tf


# ── Playback ───────────────────────────────────────────────────────


def find_link_index(env: PyBulletEnv, link_name: str) -> int:
    client = env.sim.client
    for i in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    raise RuntimeError(f"link {link_name!r} not found on robot")


def capture_local_transform(
    env: PyBulletEnv, link_idx: int, body_id: int
) -> np.ndarray:
    """Return the 4×4 pose of *body_id* expressed in *link_idx*'s frame."""
    client = env.sim.client
    link_state = client.getLinkState(env.sim.skel_id, link_idx)
    link_pos = np.asarray(link_state[0])
    link_quat = np.asarray(link_state[1])  # xyzw
    link_R = np.asarray(client.getMatrixFromQuaternion(link_quat)).reshape(3, 3)

    obj_pos, obj_quat = client.getBasePositionAndOrientation(body_id)
    obj_pos = np.asarray(obj_pos)
    obj_R = np.asarray(client.getMatrixFromQuaternion(obj_quat)).reshape(3, 3)

    R_inv = link_R.T
    local = np.eye(4)
    local[:3, :3] = R_inv @ obj_R
    local[:3, 3] = R_inv @ (obj_pos - link_pos)
    return local


def apply_attachment(
    env: PyBulletEnv,
    link_idx: int,
    body_id: int,
    local_tf: np.ndarray,
) -> None:
    """Update the mesh pose so it stays rigidly fixed to the link."""
    client = env.sim.client
    link_state = client.getLinkState(env.sim.skel_id, link_idx)
    link_pos = np.asarray(link_state[0])
    link_quat = np.asarray(link_state[1])
    link_R = np.asarray(client.getMatrixFromQuaternion(link_quat)).reshape(3, 3)

    world_R = link_R @ local_tf[:3, :3]
    world_pos = link_R @ local_tf[:3, 3] + link_pos
    # Convert rotation matrix to quaternion (xyzw for PyBullet).
    m = world_R
    # Shepperd's method.
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = float(np.sqrt(t + 1.0) * 2.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0)
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0)
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0)
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    client.resetBasePositionAndOrientation(body_id, world_pos.tolist(), [x, y, z, w])


def play_segments(
    env: PyBulletEnv,
    segments: list[Segment],
    gripper_link_idx: int,
    fps: float = 60.0,
) -> None:
    """Interactive playback: SPACE play/pause, ←/→ step, close to exit."""
    client = env.sim.client
    dt = 1.0 / fps

    # Build a flat frame table so the user can scrub across segments.
    frames: list[tuple[int, int]] = []  # (segment_index, row_index)
    for si, seg in enumerate(segments):
        for ri in range(seg.path.shape[0]):
            frames.append((si, ri))

    idx = 0
    n = len(frames)
    playing = False
    space = ord(" ")
    left = pb.B3G_LEFT_ARROW
    right = pb.B3G_RIGHT_ARROW

    print("\nControls: SPACE play/pause   ←/→ step   close window to exit\n")
    for s in segments:
        print(f"  → {s.banner} ({s.path.shape[0]} wp)")

    last_banner = -1
    try:
        while client.isConnected():
            si, ri = frames[idx]
            seg = segments[si]
            env.set_configuration(seg.path[ri])
            if seg.attach_body_id is not None and seg.attach_local_tf is not None:
                apply_attachment(
                    env,
                    gripper_link_idx,
                    seg.attach_body_id,
                    seg.attach_local_tf,
                )
            if si != last_banner:
                print(f"[{si}] {seg.banner}")
                last_banner = si

            keys = client.getKeyboardEvents()
            if space in keys and keys[space] & pb.KEY_WAS_TRIGGERED:
                playing = not playing
            elif not playing and left in keys and keys[left] & pb.KEY_WAS_TRIGGERED:
                idx = (idx - 1) % n
            elif not playing and right in keys and keys[right] & pb.KEY_WAS_TRIGGERED:
                idx = (idx + 1) % n
            elif playing:
                idx = (idx + 1) % n

            time.sleep(dt)
    except pb.error:
        pass


# ── Main ───────────────────────────────────────────────────────────


def main(pcd_stride: int = 1, visualize: bool = True) -> None:
    """Build the scene, plan every phase, play everything back.

    ``pcd_stride`` downsamples the 151k-point collision cloud when
    planning is slow — pass e.g. ``--pcd_stride 2`` or ``4``.
    ``visualize=False`` runs headless (useful for smoke-testing).
    """
    env = PyBulletEnv(autolife_robot_config, visualize=visualize)
    print("── scene setup ──")
    load_room_meshes(env)
    cloud = load_room_pointcloud(stride=pcd_stride)
    print(f"  collision cloud: {len(cloud)} points (stride={pcd_stride})")
    env.add_pointcloud(cloud[::4], pointsize=2)

    if visualize:
        env.sim.client.resetDebugVisualizerCamera(
            cameraDistance=4.5,
            cameraYaw=-90.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[-0.5, -0.3, 0.7],
        )

    # Spawn the robot inside the room next to the dining table (east side).
    current_full = HOME_JOINTS.copy()
    current_full[:3] = BASE_NEAR_TABLE_EAST
    env.set_configuration(current_full)

    # Place every graspable at its initial world position.
    apple_id = place_graspable(env, "apple", APPLE_MESH_INIT)
    bottle_id = place_graspable(env, "bottle", BOTTLE_MESH_INIT)

    gripper_link = find_link_index(env, GRIPPER_LINK)
    ik = build_ik_bundle()
    segments: list[Segment] = []

    # Fix both Python and C random seeds so TRAC-IK's random restarts
    # land on the same kinematic branch every run.  Without this the
    # demo is ~50% reliable because TRAC-IK occasionally returns
    # self-colliding solutions near joint limits.
    import ctypes
    import random

    np.random.seed(42)
    random.seed(42)
    try:
        libc = ctypes.CDLL(None)
        libc.srand(42)
    except Exception:
        pass

    # ── Stage 1a: SQUAT down (height subgroup, 3 DOF) ───────────────
    print("\n── stage 1a: squat down (autolife_height subgroup) ──")
    current_full = run_squat(
        current_full,
        SQUAT_STANCE,
        cloud,
        segments,
        label="s1a squat (height subgroup)",
    )

    # ── Stage 1b: whole-upper-body pick (autolife_body + leg-pin) ──
    print(
        "\n── stage 1b: pick apple under table (autolife_body, leg-pin constraint) ──"
    )
    current_full, apple_local_tf = run_upper_body_pick(
        env,
        ik,
        cloud,
        current_full,
        segments,
        label="s1b pick apple (under table)",
        finger_target=APPLE_MESH_INIT + np.array([0, 0, APPLE_GRASP_Z]),
        approach_world=APPLE_APPROACH,
        object_id=apple_id,
        object_initial_world_pos=APPLE_MESH_INIT,
        gripper_link_idx=gripper_link,
    )

    # ── Stage 1c: STAND back up carrying the apple ──────────────────
    print("\n── stage 1c: stand up (height subgroup, apple attached) ──")
    current_full = run_squat(
        current_full,
        STANDING_STANCE,
        cloud,
        segments,
        label="s1c stand up (height subgroup)",
        attach_body_id=apple_id,
        attach_local_tf=apple_local_tf,
    )

    # ── Stage 2: place the apple on the dining-table top ────────────
    print("\n── stage 2: place apple on the dining table top ──")
    current_full = run_place(
        ik,
        cloud,
        current_full,
        segments,
        label="s2 place apple (table top)",
        finger_target=APPLE_MESH_PLACE + np.array([0, 0, APPLE_GRASP_Z]),
        approach_world=APPLE_APPROACH,
        object_id=apple_id,
        local_tf=apple_local_tf,
    )

    # ── Stage 3: navigate to the sofa ────────────────────────────────
    print("\n── stage 3: navigate east-of-table → sofa ──")
    current_full = run_nav(
        current_full,
        BASE_NEAR_SOFA,
        cloud,
        segments,
        label="s3 nav east-of-table → sofa",
    )

    # ── Stage 4: pick the bottle from the sofa ──────────────────────
    print("\n── stage 4: pick bottle from the sofa ──")
    current_full, bottle_local_tf = run_pick(
        env,
        ik,
        cloud,
        current_full,
        segments,
        label="s4 pick bottle (sofa)",
        finger_target=BOTTLE_MESH_INIT + np.array([0, 0, BOTTLE_GRASP_Z]),
        approach_world=BOTTLE_APPROACH,
        object_id=bottle_id,
        object_initial_world_pos=BOTTLE_MESH_INIT,
        gripper_link_idx=gripper_link,
    )

    # ── Stage 5: navigate to the coffee table, carrying the bottle ──
    print("\n── stage 5: navigate sofa → coffee table (carrying bottle) ──")
    current_full = run_nav(
        current_full,
        BASE_NEAR_COFFEE,
        cloud,
        segments,
        label="s5 nav sofa → coffee",
        attach_body_id=bottle_id,
        attach_local_tf=bottle_local_tf,
    )

    # ── Stage 6: place the bottle on the coffee table ───────────────
    print("\n── stage 6: place bottle on the coffee table ──")
    current_full = run_place(
        ik,
        cloud,
        current_full,
        segments,
        label="s6 place bottle (coffee table)",
        finger_target=BOTTLE_MESH_PLACE + np.array([0, 0, BOTTLE_GRASP_Z]),
        approach_world=BOTTLE_APPROACH,
        object_id=bottle_id,
        local_tf=bottle_local_tf,
    )

    # Reset everything back to the start of the demo so playback begins
    # from the spawned state, not the last planned configuration.
    env.set_configuration(segments[0].path[0])
    client = env.sim.client
    client.resetBasePositionAndOrientation(
        apple_id, APPLE_MESH_INIT.tolist(), [0.0, 0.0, 0.0, 1.0]
    )
    client.resetBasePositionAndOrientation(
        bottle_id, BOTTLE_MESH_INIT.tolist(), [0.0, 0.0, 0.0, 1.0]
    )

    total = sum(s.path.shape[0] for s in segments)
    print(f"\n── ready: {total} total frames across {len(segments)} segments ──")
    if not visualize:
        return
    play_segments(env, segments, gripper_link)


if __name__ == "__main__":
    Fire(main)
