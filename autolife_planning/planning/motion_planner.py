"""VAMP-based motion planner.

Wraps the VAMP C++ library internally. No vamp types leak to the caller.
Mirrors the TracIKSolver pattern: configure -> create -> planner.plan() -> result.

Supports subgroup planning: each subgroup (e.g. ``autolife_left_high``)
operates on a subset of joints while the rest stay frozen.  Use
:meth:`extract_config` / :meth:`embed_config` / :meth:`embed_path` to
convert between the subgroup DOF space and the full 24-DOF body config.
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

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        ...

    def validate(self, configuration: np.ndarray) -> bool:
        ...


class MotionPlanner:
    """Motion planner wrapping the VAMP C++ library.

    All vamp internals are private. The public API accepts and returns
    only numpy arrays.

    For subgroup planners the helper methods :meth:`extract_config`,
    :meth:`embed_config`, and :meth:`embed_path` convert between the
    reduced DOF space used by the planner and the full 24-DOF body
    configuration used for visualization.
    """

    def __init__(
        self,
        robot_name: str,
        config: PlannerConfig | None = None,
        pointcloud: np.ndarray | None = None,
    ) -> None:
        if config is None:
            config = PlannerConfig()

        import vamp

        self._config = config
        self._robot_name = robot_name

        (
            self._vamp_module,
            self._planner_func,
            self._plan_settings,
            self._simp_settings,
        ) = vamp.configure_robot_and_planner_with_kwargs(
            robot_name, config.planner_name
        )

        self._env = vamp.Environment()
        self._sampler = self._vamp_module.halton()
        self._ndof: int = self._vamp_module.dimension()

        # Subgroup joint mapping
        self._joint_names: list[str] = list(self._vamp_module.joint_names())
        self._subgroup_indices: np.ndarray | None = self._compute_subgroup_indices()

        if pointcloud is not None:
            r_min, r_max = self._vamp_module.min_max_radii()
            self._env.add_pointcloud(
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
        """Indices of this planner's joints in the full 24-DOF config.

        ``None`` for the full-body ``autolife`` planner.
        """
        return self._subgroup_indices

    # ── Subgroup helpers ──────────────────────────────────────────────

    def _compute_subgroup_indices(self) -> np.ndarray | None:
        from autolife_planning.config.robot_config import autolife_robot_config

        full_names = autolife_robot_config.joint_names
        if len(self._joint_names) == len(full_names):
            return None
        try:
            return np.array([full_names.index(j) for j in self._joint_names])
        except ValueError:
            return None

    def extract_config(self, full_config: np.ndarray) -> np.ndarray:
        """Extract this planner's joints from a full 24-DOF configuration.

        For the full-body planner this is a copy of the input.
        """
        full_config = np.asarray(full_config, dtype=np.float64)
        if self._subgroup_indices is None:
            return full_config.copy()
        return full_config[self._subgroup_indices].copy()

    def embed_config(
        self,
        config: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Embed a subgroup config into a full 24-DOF configuration.

        Frozen joints are filled from *base_config* (defaults to
        ``HOME_JOINTS``).
        """
        config = np.asarray(config, dtype=np.float64)
        if self._subgroup_indices is None:
            return config.copy()
        from autolife_planning.config.robot_config import HOME_JOINTS

        if base_config is None:
            base_config = HOME_JOINTS
        full = np.array(base_config, dtype=np.float64)
        full[self._subgroup_indices] = config
        return full

    def embed_path(
        self,
        path: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert a subgroup path ``(N, sub_dof)`` to ``(N, 24)``.

        Frozen joints are filled from *base_config* (defaults to
        ``HOME_JOINTS``).
        """
        path = np.asarray(path, dtype=np.float64)
        if self._subgroup_indices is None:
            return path.copy()
        from autolife_planning.config.robot_config import HOME_JOINTS

        if base_config is None:
            base_config = HOME_JOINTS
        n = path.shape[0]
        full_path = np.tile(np.array(base_config, dtype=np.float64), (n, 1))
        full_path[:, self._subgroup_indices] = path
        return full_path

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        """Plan a collision-free path from start to goal.

        Input:
            start: Joint configuration array of length num_dof
            goal: Joint configuration array of length num_dof
        Output:
            PlanningResult with status, path as (N, ndof) numpy array
        """
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if len(start) != self._ndof:
            raise ValueError(f"start has {len(start)} DOF, expected {self._ndof}")
        if len(goal) != self._ndof:
            raise ValueError(f"goal has {len(goal)} DOF, expected {self._ndof}")

        # Validate start and goal
        if not self._vamp_module.validate(start, self._env):
            return PlanningResult(
                status=PlanningStatus.INVALID_START,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )
        if not self._vamp_module.validate(goal, self._env):
            return PlanningResult(
                status=PlanningStatus.INVALID_GOAL,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )

        result = self._planner_func(
            start, goal, self._env, self._plan_settings, self._sampler
        )

        if not result.solved:
            return PlanningResult(
                status=PlanningStatus.FAILED,
                path=None,
                planning_time_ns=result.nanoseconds,
                iterations=result.iterations,
                path_cost=float("inf"),
            )

        path = result.path

        if self._config.simplify:
            simplified = self._vamp_module.simplify(
                path, self._env, self._simp_settings, self._sampler
            )
            path = simplified.path

        if self._config.interpolate:
            path.interpolate_to_resolution(self._vamp_module.resolution())

        path_cost = float(path.cost())
        path_np = np.array(path.numpy())

        return PlanningResult(
            status=PlanningStatus.SUCCESS,
            path=path_np,
            planning_time_ns=result.nanoseconds,
            iterations=result.iterations,
            path_cost=path_cost,
        )

    def validate(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free.

        Input:
            configuration: Joint configuration array of length num_dof
        Output:
            True if collision-free
        """
        configuration = np.asarray(configuration, dtype=np.float64)
        return self._vamp_module.validate(configuration, self._env)

    def sample_valid(self) -> np.ndarray:
        """Sample a random collision-free configuration."""
        while True:
            config = self._sampler.next()
            if self._vamp_module.validate(config, self._env):
                return np.asarray(config, dtype=np.float64)


def available_robots() -> list[str]:
    """Return all available VAMP robot names.

    Includes the full-body ``autolife`` and all planning subgroups
    (e.g. ``autolife_left_high``, ``autolife_dual_mid``, …).
    """
    import vamp

    return [r for r in vamp.robots if r.startswith("autolife")]


def create_planner(
    robot_name: str = "autolife",
    config: PlannerConfig | None = None,
    pointcloud: np.ndarray | None = None,
) -> MotionPlanner:
    """Create a motion planner for any robot or subgroup.

    Args:
        robot_name: VAMP robot name.  Use :func:`available_robots` to
            list all names.  Common choices::

                "autolife"               # full body, 24 DOF
                "autolife_left_high"     # left arm, high stance, 7 DOF
                "autolife_dual_mid"      # both arms, mid stance, 14 DOF
                "autolife_torso_left_low"# waist+left arm, low stance, 9 DOF
                "autolife_body"          # body w/o base, 21 DOF

        config: Planner configuration (uses defaults if None).
        pointcloud: ``(N, 3)`` obstacle point cloud (optional).

    Returns:
        A :class:`MotionPlanner` instance.  For subgroup planners use
        :meth:`~MotionPlanner.extract_config` /
        :meth:`~MotionPlanner.embed_config` /
        :meth:`~MotionPlanner.embed_path` to convert between the
        reduced DOF space and full 24-DOF visualization configs.

    Examples::

        # Full body
        planner = create_planner("autolife")
        result = planner.plan(start_24dof, goal_24dof)

        # Subgroup (7-DOF left arm at high stance)
        planner = create_planner("autolife_left_high")
        start = planner.extract_config(HOME_JOINTS)   # → 7-DOF
        goal = planner.sample_valid()                  # → 7-DOF
        result = planner.plan(start, goal)             # path (N, 7)
        full_path = planner.embed_path(result.path)    # → (N, 24)
    """
    return MotionPlanner(robot_name, config, pointcloud)
