"""Micro-benchmark for the OMPL+VAMP motion planner.

Measures two things on the same table-obstacle scene as
``examples/planning/motion.py``:

    1. Per-call collision check throughput  (``planner.validate``) —
       the raw VAMP SIMD pipeline.  Reported in nanoseconds / check
       and checks / second.
    2. End-to-end plan time for a handful of representative OMPL
       planners over the left arm (7 DOF), dual arm (14 DOF), and
       the full body (24 DOF).  Reported as median / p50-p95 wall
       clock over ``--repeats`` runs, excluding simplification and
       interpolation so the numbers reflect search only.

The script prints a compact markdown table intended to be lifted
verbatim into the docs landing page, plus a ``[Environment]`` block
naming the CPU and kernel the run was done on.

    pixi run -e dev python scripts/benchmarks/motion_planning.py
    pixi run -e dev python scripts/benchmarks/motion_planning.py --repeats 50
"""

from __future__ import annotations

import argparse
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh

import autolife_planning
from autolife_planning.autolife import HOME_JOINTS
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

# Feasibility planners — return on the first collision-free solution.
# These are what we quote in the docs landing pitch; ``plan_time_ns``
# reflects actual search wall-clock, not a time budget.
FEASIBILITY_PLANNERS = [
    "rrtc",
    "rrt",
    "kpiece",
    "bkpiece",
    "lbkpiece",
    "prm",
    "lazyprm",
    "est",
    "biest",
    "sbl",
    "stride",
]

# Asymptotically-optimal planners — run until the time budget, refining
# cost.  Their ``planning_time_ns`` is (by design) ~= the budget, so we
# report ``path_cost`` achieved at a fixed budget instead of "time to
# first solution", which OMPL doesn't expose cleanly.
AO_PLANNERS = ["rrtstar", "bitstar", "aitstar", "eitstar", "fmt"]

# Robot subgroups to benchmark, paired with a sensible time budget.
SUBGROUPS = [
    ("autolife_left_arm", 1.0),
    ("autolife_dual_arm", 1.0),
    ("autolife", 2.0),  # full body, 24 DOF
]

AO_BUDGET = 1.0  # seconds — cost-refinement budget for AO planners


def load_table(distance: float = 0.85, height: float = 0.35) -> np.ndarray:
    """Load the bundled table point cloud in front of the robot.

    Mirrors ``examples/planning/motion.py::load_table`` so the
    benchmark exercises exactly the scene the motion-planning demo
    uses.
    """
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


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "unknown"


def cpu_cores() -> str:
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        pmatch = re.search(r"^CPU\(s\):\s+(\d+)", out, re.M)
        smatch = re.search(r"Socket\(s\):\s+(\d+)", out)
        return f"{pmatch.group(1) if pmatch else '?'} logical CPUs" + (
            f", {smatch.group(1)} socket(s)" if smatch else ""
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def benchmark_validate(
    planner, config: np.ndarray, n_calls: int = 10_000
) -> tuple[float, float]:
    """Time ``planner.validate`` — per-call nanoseconds and ops / second.

    Uses the same config every call so we measure the raw collision
    pipeline, not Python-side randomness.
    """
    # Warm up once to let any lazy init settle.
    planner.validate(config)
    t0 = time.perf_counter_ns()
    for _ in range(n_calls):
        planner.validate(config)
    t1 = time.perf_counter_ns()
    ns_per = (t1 - t0) / n_calls
    ops = 1e9 / ns_per
    return ns_per, ops


def benchmark_plan(
    robot_name: str,
    planner_name: str,
    time_limit: float,
    cloud: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    """Return (wall-times in ms for successful runs, total attempts).

    Simplification and interpolation are disabled so the reported
    number is search time only.
    """
    planner = create_planner(
        robot_name,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            simplify=False,
            interpolate=False,
        ),
        base_config=HOME_JOINTS.copy(),
        pointcloud=cloud,
    )
    start = planner.extract_config(HOME_JOINTS)
    times_ms: list[float] = []
    for _ in range(repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(seed)
        goal = planner.sample_valid()
        result = planner.plan(start, goal)
        if result.success:
            times_ms.append(result.planning_time_ns / 1e6)
    return times_ms, repeats


def fmt_ms(samples: list[float], attempts: int) -> str:
    if not samples:
        return "    —"
    samples_sorted = sorted(samples)
    med = statistics.median(samples_sorted)
    p95 = samples_sorted[int(0.95 * (len(samples_sorted) - 1))]
    suffix = "" if len(samples) == attempts else f" ({len(samples)}/{attempts})"
    return f"{med:6.2f} / {p95:6.2f}{suffix}"


def benchmark_ao_cost(
    robot_name: str,
    planner_name: str,
    budget: float,
    cloud: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    """AO planners: report ``path_cost`` after ``budget`` seconds."""
    planner = create_planner(
        robot_name,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=budget,
            simplify=False,
            interpolate=False,
        ),
        base_config=HOME_JOINTS.copy(),
        pointcloud=cloud,
    )
    start = planner.extract_config(HOME_JOINTS)
    costs: list[float] = []
    for _ in range(repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(seed)
        goal = planner.sample_valid()
        result = planner.plan(start, goal)
        if result.success:
            costs.append(result.path_cost)
    return costs, repeats


def fmt_cost(samples: list[float], attempts: int) -> str:
    if not samples:
        return "    —"
    med = statistics.median(samples)
    suffix = "" if len(samples) == attempts else f" ({len(samples)}/{attempts})"
    return f"{med:6.2f}{suffix}"


class _silence_fds:
    """Context manager: redirect stdout+stderr fds to /dev/null.

    OMPL logs through its own C++ sinks which land on the process's
    fd 1 / 2 regardless of Python's ``sys.stdout`` redirection, so we
    dup the fds themselves.  Restored on exit.
    """

    def __enter__(self):
        import os

        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._saved = (os.dup(1), os.dup(2))
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *exc):
        import os

        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        os.close(self._saved[0])
        os.close(self._saved[1])
        os.close(self._devnull)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Plans per (planner, subgroup) cell (default: 20).",
    )
    ap.add_argument(
        "--validate-calls",
        type=int,
        default=10_000,
        help="Collision checks for the validate() microbenchmark.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    cloud = load_table()

    with _silence_fds():
        # --- Collision check microbenchmark (VAMP SIMD pipeline) ---
        warm = create_planner(
            "autolife_left_arm",
            config=PlannerConfig(planner_name="rrtc", time_limit=0.1),
            base_config=HOME_JOINTS.copy(),
            pointcloud=cloud,
        )
        warm_start = warm.extract_config(HOME_JOINTS)
        ns_per, ops = benchmark_validate(warm, warm_start, args.validate_calls)

        # --- Feasibility planners: time to first solution ---
        feas_rows: dict[str, dict[str, tuple[list[float], int]]] = {}
        for robot, budget in SUBGROUPS:
            feas_rows[robot] = {}
            for pl in FEASIBILITY_PLANNERS:
                samples, attempts = benchmark_plan(
                    robot, pl, budget, cloud, args.repeats, rng
                )
                feas_rows[robot][pl] = (samples, attempts)

        # --- AO planners: path cost after a fixed budget ---
        ao_rows: dict[str, dict[str, tuple[list[float], int]]] = {}
        for robot, _ in SUBGROUPS:
            ao_rows[robot] = {}
            for pl in AO_PLANNERS:
                samples, attempts = benchmark_ao_cost(
                    robot, pl, AO_BUDGET, cloud, args.repeats, rng
                )
                ao_rows[robot][pl] = (samples, attempts)

    # --- Report ---
    print()
    print("[Environment]")
    print(f"  CPU:    {cpu_model()} ({cpu_cores()})")
    print(f"  Kernel: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print()
    print("[Collision check]  planner.validate(config)")
    print(f"  {ns_per:.0f} ns / call   ~ {ops / 1e6:.2f} M checks / s")
    print()
    print(
        f"[Plan time — feasibility planners]  "
        f"median / p95 in ms, search-only, N = {args.repeats} per cell"
    )
    header = "  planner".ljust(14) + "".join(
        f"{name.split('_', 1)[-1][:14]:>18}" for name, _ in SUBGROUPS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pl in FEASIBILITY_PLANNERS:
        line = f"  {pl:<12}"
        for robot, _ in SUBGROUPS:
            samples, attempts = feas_rows[robot][pl]
            line += f"{fmt_ms(samples, attempts):>18}"
        print(line)
    print()
    print(
        f"[Path cost — AO planners]  "
        f"median cost after {AO_BUDGET:.1f}s budget, N = {args.repeats} per cell"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pl in AO_PLANNERS:
        line = f"  {pl:<12}"
        for robot, _ in SUBGROUPS:
            samples, attempts = ao_rows[robot][pl]
            line += f"{fmt_cost(samples, attempts):>18}"
        print(line)
    print()


if __name__ == "__main__":
    main()
