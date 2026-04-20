from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

from autolife_planning.autolife import HOME_JOINTS, PLANNING_SUBGROUPS, autolife_robot_config
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

import time

SUPPORTED_GROUPS = {
    "autolife_left_arm",
    "autolife_right_arm",
    "autolife_dual_arm",
    "autolife_torso_left_arm",
    "autolife_torso_right_arm",
    "autolife_torso_dual_arm",
    "autolife_leg_torso_dual_arm",
}

FULL_DOF = len(autolife_robot_config.joint_names)


def _active_indices(group: str) -> np.ndarray:
    if group not in PLANNING_SUBGROUPS:
        raise ValueError(f"Unknown subgroup: {group}")
    subgroup_joints = PLANNING_SUBGROUPS[group]["joints"]
    full_joint_names = autolife_robot_config.joint_names
    return np.asarray([full_joint_names.index(j) for j in subgroup_joints], dtype=int)


def _validate_24d(name: str, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (FULL_DOF,):
        raise ValueError(f"{name} must be shape ({FULL_DOF},), got {q.shape}")
    return q


def _frozen_joints_same(group: str, start: np.ndarray, goal: np.ndarray, tol: float) -> bool:
    active = _active_indices(group)
    frozen = np.setdiff1d(np.arange(FULL_DOF), active)
    return np.allclose(start[frozen], goal[frozen], atol=tol, rtol=0.0)


@lru_cache(maxsize=1)
def _leg_torso_dual_arm_cost():
    import casadi as ca

    from autolife_planning.planning import Cost, SymbolicContext

    # Penalise the horizontal (XY) offset between Link_Waist_Pitch_to_Waist_Yaw
    # and Link_Ground_Vehicle — i.e. keep the waist directly above the base.
    # residual = [dx, dy]  →  cost = dx² + dy²
    ctx = SymbolicContext("autolife_leg_torso_dual_arm")
    waist_pos = ctx.link_translation("Link_Waist_Pitch_to_Waist_Yaw")
    ground_pos = ctx.link_translation("Link_Ground_Vehicle")
    residual = waist_pos[:2] - ground_pos[:2]
    return Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="leg_torso_dual_arm_balance_cost",
        weight=10.0,
    )


class AutolifePlanner:
    """Motion planner for the Autolife robot.

    Two MotionPlanners are created eagerly at construction and reused across
    all plan_to_joints calls — the C++ VAMP collision environment is never
    rebuilt.  Each call only calls set_subgroup() to switch active joints.

    Example::

        planner = AutolifePlanner()
        start = planner.default_start()
        goal  = start.copy()
        goal[7:14] = [0.3, -0.5, 0.3, 1.8, -0.2, -0.6, 0.5]
        path = planner.plan_to_joints("autolife_left_arm", start, goal)
    """

    def __init__(
        self,
        planner_name: str = "bitstar",
        time_limit: float = 2.0,
        tol: float = 1e-6,
    ) -> None:
        """
        Args:
            planner_name: OMPL planner name used for all calls.
            time_limit:   Default planning timeout in seconds (overridable per call).
            tol:          Equality tolerance used to detect frozen-joint mismatches.
        """
        self.planner_name = planner_name
        self.time_limit = time_limit
        self.tol = tol
        self.current_group = None
        self._planner = create_planner(
            "autolife_left_arm",
            config=PlannerConfig(planner_name=planner_name),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_start() -> np.ndarray:
        """Return the home full-body configuration (24-DOF)."""
        return HOME_JOINTS.copy()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def plan_to_joints(
        self,
        group: str,
        start: np.ndarray,
        goal: np.ndarray,
        planner_name: str | None = None,
        time_limit: float | None = None,
    ) -> np.ndarray | str | None:
        """Plan a collision-free path for *group* between two 24-DOF configs.

        Args:
            group:        One of SUPPORTED_GROUPS.
            start:        Full 24-DOF start vector.
            goal:         Full 24-DOF goal vector.
            planner_name: Ignored (planner algorithm is fixed at construction).
            time_limit:   Planning timeout in seconds; defaults to ``self.time_limit``.

        Returns:
            - ``"not same"`` if any frozen joint differs between start and goal.
            - ``np.ndarray`` of shape ``(N, 24)`` on success.
            - ``None`` when planning times out or fails.
        """
        if group not in SUPPORTED_GROUPS:
            raise ValueError(
                f"group must be one of {sorted(SUPPORTED_GROUPS)}, got {group!r}"
            )

        start = _validate_24d("start", start)
        goal = _validate_24d("goal", goal)

        if not _frozen_joints_same(group, start, goal, tol=self.tol):
            return "not same"

        time_limit = time_limit if time_limit is not None else self.time_limit

        self._planner.set_subgroup(group, base_config=start)

        if group == "autolife_leg_torso_dual_arm" and self.current_group != "autolife_leg_torso_dual_arm":
            # Cost dim must match the subgroup — set after set_subgroup.
            self._planner.set_costs([_leg_torso_dual_arm_cost()])
        else:
            self._planner.clear_costs()
        
        self.current_group = group

        start_sub = self._planner.extract_config(start)
        goal_sub = self._planner.extract_config(goal)

        result = self._planner.plan(start_sub, goal_sub, time_limit=time_limit)
        if not result.success or result.path is None:
            return None
        return self._planner.embed_path(result.path, base_config=start)

    @staticmethod
    def time_parameterize(
        path: np.ndarray,
        v_max: float = 0.5,
        a_max: float = 0.6,
        dt: float = 0.02,
        vel_scale: float = 1.0,
        a_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a joint-space waypoint path to a time-parameterized trajectory.

        Algorithm
        ---------
        1. Compute per-joint **total variation** along the whole path (not per
           segment) and derive the minimum total time *T* from a trapezoidal
           velocity profile on the joint with the largest travel.
        2. Scale the trapezoidal profile to arc-length space so the fastest joint
           reaches exactly *v_max* at the cruise phase.
        3. Evaluate the trapezoidal arc-length profile ``s(t)`` and invert it to
           assign timestamps to every waypoint.
        4. Fit a clamped cubic spline through the waypoints and resample at *dt*.

        Args:
            path:      ``(N, DOF)`` waypoints from the motion planner.
            v_max:     Maximum joint velocity (rad/s), uniform across all joints.
            a_max:     Maximum joint acceleration (rad/s²), uniform across all joints.
            dt:        Output sample period (seconds).
            vel_scale: Fractional scale applied to *v_max* (e.g. 0.5 → half speed).
            a_scale:   Fractional scale applied to *a_max* (e.g. 0.5 → half accel).

        Returns:
            times: ``(M,)`` timestamps of the output trajectory.
            traj:  ``(M, DOF)`` joint positions sampled at those timestamps.
        """
        v_max = v_max * vel_scale
        a_max = a_max * a_scale

        path = np.asarray(path, dtype=np.float64)
        N, DOF = path.shape
        if N == 1:
            return np.array([0.0]), path.copy()

        deltas = np.diff(path, axis=0)                          # (N-1, DOF)

        # ── Step 1: total variation per joint ────────────────────────────
        total_var = np.abs(deltas).sum(axis=0)                  # (DOF,)
        d_max = float(total_var.max())
        if d_max < 1e-12:
            return np.array([0.0, dt]), np.stack([path[0], path[-1]])

        # ── Step 2: total arc length & scaling to arc-length space ───────
        seg_lens = np.linalg.norm(deltas, axis=1)               # (N-1,)
        cum_lens  = np.concatenate([[0.0], np.cumsum(seg_lens)])
        L = float(cum_lens[-1])
        if L < 1e-12:
            return np.array([0.0, dt]), np.stack([path[0], path[-1]])

        # The joint with the largest total variation d_max must hit v_max
        # at the cruise phase.  Scale v_max / a_max to arc-length space:
        #   v_s = v_max * (L / d_max),   a_s = a_max * (L / d_max)
        scale = L / d_max
        v_s = v_max * scale
        a_s = a_max * scale

        # ── Step 3: total time T from trapezoidal profile in s-space ─────
        v_peak = np.sqrt(L * a_s)
        if v_peak <= v_s:                                        # triangular
            T = 2.0 * v_peak / a_s
        else:                                                    # trapezoidal
            T = L / v_s + v_s / a_s
        T = max(T, dt)

        # ── Step 4: s(t) — trapezoidal profile ───────────────────────────
        t_a      = min(v_s / a_s, T / 2.0)                      # accel/decel duration
        v_cruise = a_s * t_a                                     # actual cruise speed
        d_a      = 0.5 * a_s * t_a ** 2                         # distance during accel

        t_dense = np.arange(0.0, T + dt * 0.5, dt)
        s_dense = np.where(
            t_dense <= t_a,
            0.5 * a_s * t_dense ** 2,
            np.where(
                t_dense <= T - t_a,
                d_a + v_cruise * (t_dense - t_a),
                L - 0.5 * a_s * (T - t_dense) ** 2,
            ),
        )
        s_dense = np.clip(s_dense, 0.0, L)

        # ── Step 5: waypoint timestamps via s → t inversion ──────────────
        wp_times = np.interp(cum_lens, s_dense, t_dense)
        wp_times[0] = 0.0
        wp_times[-1] = T

        # ── Step 6: clamped cubic spline + resample ───────────────────────
        cs = CubicSpline(wp_times, path, bc_type="clamped")
        traj = cs(t_dense)
        return t_dense, traj
