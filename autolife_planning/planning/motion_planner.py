"""OMPL + VAMP motion planner.

Uses OMPL for planning algorithms and VAMP for SIMD-accelerated
collision checking.  The entire planning pipeline runs in C++ via
the ``_ompl_vamp`` extension — Python only crosses the boundary
once per ``plan()`` call.

Supports subgroup planning: each subgroup operates on a reduced
state space while frozen joints are expanded to the full 24-DOF
config before collision checks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from autolife_planning.types import PlannerConfig, PlanningResult, PlanningStatus


@runtime_checkable
class MotionPlannerBase(Protocol):
    """Protocol for motion planner backends."""

    @property
    def robot_name(self) -> str:
        ...

    @property
    def num_dof(self) -> int:
        ...

    def plan(self, start: np.ndarray, goal: np.ndarray) -> PlanningResult:
        ...

    def validate(self, configuration: np.ndarray) -> bool:
        ...


class MotionPlanner:
    """Motion planner using OMPL + VAMP C++ backend.

    All internals are private.  The public API accepts and returns
    only numpy arrays.

    For subgroup planners the helper methods :meth:`extract_config`,
    :meth:`embed_config`, and :meth:`embed_path` convert between the
    reduced DOF space used by the planner and the full 24-DOF body
    configuration.
    """

    def __init__(
        self,
        robot_name: str,
        config: PlannerConfig | None = None,
        pointcloud: np.ndarray | None = None,
    ) -> None:
        from autolife_planning._ompl_vamp import OmplVampPlanner
        from autolife_planning.config.robot_config import (
            PLANNING_SUBGROUPS,
            autolife_robot_config,
            subgroup_base_config,
        )

        if config is None:
            config = PlannerConfig()

        self._config = config
        self._robot_name = robot_name

        full_names = autolife_robot_config.joint_names
        sg = PLANNING_SUBGROUPS.get(robot_name)

        if sg is None:
            # Full-body planner
            self._planner = OmplVampPlanner()
            self._joint_names = list(full_names)
            self._subgroup_indices = None
        else:
            # Subgroup planner
            sg_joint_names = sg["joints"]
            active_indices = [full_names.index(j) for j in sg_joint_names]
            frozen = subgroup_base_config(robot_name).tolist()
            self._planner = OmplVampPlanner(active_indices, frozen)
            self._joint_names = list(sg_joint_names)
            self._subgroup_indices = np.array(active_indices)

        self._ndof = self._planner.dimension()
        self._coupled_joints = self._load_coupling()

        if pointcloud is not None:
            r_min, r_max = self._planner.min_max_radii()
            self._planner.add_pointcloud(
                np.asarray(pointcloud, dtype=np.float32).tolist(),
                r_min,
                r_max,
                config.point_radius,
            )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def num_dof(self) -> int:
        return self._ndof

    @property
    def joint_names(self) -> list[str]:
        """Joint names controlled by this planner, in DOF order."""
        return self._joint_names

    @property
    def is_subgroup(self) -> bool:
        """True if this planner controls a subset of the full body."""
        return self._subgroup_indices is not None

    @property
    def subgroup_indices(self) -> np.ndarray | None:
        """Indices of this planner's joints in the full 24-DOF config."""
        return self._subgroup_indices

    # ── Subgroup helpers ──────────────────────────────────────────────

    def _load_coupling(self) -> list[dict]:
        from autolife_planning.config.robot_config import PLANNING_SUBGROUPS

        sg = PLANNING_SUBGROUPS.get(self._robot_name)
        if sg is None:
            return []
        return sg.get("coupled_joints", [])

    def extract_config(self, full_config: np.ndarray) -> np.ndarray:
        """Extract this planner's joints from a full 24-DOF configuration."""
        full_config = np.asarray(full_config, dtype=np.float64)
        if self._subgroup_indices is None:
            return full_config.copy()
        return full_config[self._subgroup_indices].copy()

    def embed_config(
        self,
        config: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Embed a subgroup config into a full 24-DOF configuration."""
        config = np.asarray(config, dtype=np.float64)
        if self._subgroup_indices is None and not self._coupled_joints:
            return config.copy()
        from autolife_planning.config.robot_config import HOME_JOINTS

        if base_config is None:
            base_config = HOME_JOINTS
        full = np.array(base_config, dtype=np.float64)
        if self._subgroup_indices is not None:
            full[self._subgroup_indices] = config
        self._apply_coupling(full)
        return full

    def embed_path(
        self,
        path: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert a subgroup path ``(N, sub_dof)`` to ``(N, 24)``."""
        path = np.asarray(path, dtype=np.float64)
        if self._subgroup_indices is None and not self._coupled_joints:
            return path.copy()
        from autolife_planning.config.robot_config import HOME_JOINTS

        if base_config is None:
            base_config = HOME_JOINTS
        n = path.shape[0]
        full_path = np.tile(np.array(base_config, dtype=np.float64), (n, 1))
        if self._subgroup_indices is not None:
            full_path[:, self._subgroup_indices] = path
        for i in range(n):
            self._apply_coupling(full_path[i])
        return full_path

    def _apply_coupling(self, full_config: np.ndarray) -> None:
        """Set coupled slave joints from their master values, in-place."""
        if not self._coupled_joints:
            return
        from autolife_planning.config.robot_config import autolife_robot_config

        names = autolife_robot_config.joint_names
        for c in self._coupled_joints:
            master_idx = names.index(c["master"])
            slave_idx = names.index(c["slave"])
            full_config[slave_idx] = c["multiplier"] * full_config[master_idx] + c.get(
                "offset", 0.0
            )

    # ── Planning ──────────────────────────────────────────────────────

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        """Plan a collision-free path from start to goal."""
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if len(start) != self._ndof:
            raise ValueError(f"start has {len(start)} DOF, expected {self._ndof}")
        if len(goal) != self._ndof:
            raise ValueError(f"goal has {len(goal)} DOF, expected {self._ndof}")

        if not self._planner.validate(start.tolist()):
            return PlanningResult(
                status=PlanningStatus.INVALID_START,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )
        if not self._planner.validate(goal.tolist()):
            return PlanningResult(
                status=PlanningStatus.INVALID_GOAL,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )

        result = self._planner.plan(
            start.tolist(),
            goal.tolist(),
            self._config.planner_name,
            self._config.time_limit,
            self._config.simplify,
        )

        if not result.solved:
            return PlanningResult(
                status=PlanningStatus.FAILED,
                path=None,
                planning_time_ns=result.planning_time_ns,
                iterations=0,
                path_cost=float("inf"),
            )

        path_np = np.array(result.path, dtype=np.float64)

        return PlanningResult(
            status=PlanningStatus.SUCCESS,
            path=path_np,
            planning_time_ns=result.planning_time_ns,
            iterations=0,
            path_cost=result.path_cost,
        )

    def validate(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free."""
        configuration = np.asarray(configuration, dtype=np.float64)
        return self._planner.validate(configuration.tolist())

    def sample_valid(self) -> np.ndarray:
        """Sample a random collision-free configuration."""
        lo = np.array(self._planner.lower_bounds())
        hi = np.array(self._planner.upper_bounds())
        while True:
            config = np.random.uniform(lo, hi)
            if self._planner.validate(config.tolist()):
                return config


def available_robots() -> list[str]:
    """Return all available robot names for planning."""
    from autolife_planning.config.robot_config import PLANNING_SUBGROUPS

    return ["autolife"] + sorted(PLANNING_SUBGROUPS.keys())


def create_planner(
    robot_name: str = "autolife",
    config: PlannerConfig | None = None,
    pointcloud: np.ndarray | None = None,
) -> MotionPlanner:
    """Create a motion planner for any robot or subgroup.

    Args:
        robot_name: Robot or subgroup name. Use :func:`available_robots`
            to list all names.
        config: Planner configuration (uses defaults if None).
        pointcloud: ``(N, 3)`` obstacle point cloud (optional).

    Returns:
        A :class:`MotionPlanner` instance.
    """
    return MotionPlanner(robot_name, config, pointcloud)
