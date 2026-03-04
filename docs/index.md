# Autolife Planning

A planning library for the **Autolife robot**. It provides inverse kinematics (TRAC-IK), motion planning ([VAMP](https://github.com/KavrakiLab/vamp)), and collision-aware planning through a unified Python interface.

## Features

- **Inverse Kinematics** — TRAC-IK based solver with multiple solve strategies (speed, distance, manipulability)
- **Motion Planning** — VAMP-based planner with collision checking and path validation
- **Collision Geometry** — Spherized URDF representations for efficient collision detection
- **Rotation Utilities** — Conversions between quaternion, RPY, axis-angle, and rotation matrices

## Quick Install

First, install the required C/C++ system dependencies via conda:

```bash
conda create -n autolife -c conda-forge python=3.12 \
  boost cxx-compiler cmake ninja eigen orocos-kdl nlopt pybind11 pip
conda activate autolife
```

Then install the package:

```bash
pip install autolife-planning
```

See the [Getting Started](getting-started.md) guide for full details.
