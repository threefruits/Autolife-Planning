"""Sanity test: the package imports and exposes its public surface.

Heavier behavioural tests live alongside this file and are added in a
follow-up commit; this one exists so CI has something to collect on the
very first run.
"""


def test_package_imports() -> None:
    import autolife_planning  # noqa: F401
    from autolife_planning import types
    from autolife_planning.planning import create_planner  # noqa: F401
    from autolife_planning.types import (  # noqa: F401
        IKResult,
        IKStatus,
        PlannerConfig,
        PlanningResult,
        PlanningStatus,
        SE3Pose,
    )

    assert hasattr(types, "PlannerConfig")
