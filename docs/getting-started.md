# Getting Started

## Prerequisites

- **Linux** (x86_64)
- **Python** 3.11–3.12
- [conda](https://docs.conda.io/) or [mamba](https://mamba.readthedocs.io/) package manager

The package depends on several C/C++ libraries that must be installed via conda-forge before `pip install`:

| Dependency | Purpose |
|------------|---------|
| `cmake`, `ninja` | Build system |
| `cxx-compiler` | C++ compiler toolchain |
| `boost` | C++ utility libraries |
| `eigen` | Linear algebra |
| `orocos-kdl` | Kinematics and dynamics |
| `nlopt` | Nonlinear optimization |
| `pybind11` | Python ↔ C++ bindings |

## Installation

### Using the environment file (recommended)

```bash
conda env create -f environment.yaml
conda activate autolife
```

This installs all system dependencies and `autolife-planning` in one step.

??? note "environment.yaml"
    ```yaml
    name: autolife
    channels:
      - conda-forge
    dependencies:
      - python>=3.11,<3.13
      - boost
      - cxx-compiler
      - cmake
      - ninja
      - eigen
      - orocos-kdl
      - nlopt
      - pybind11
      - pip
      - pip:
        - autolife-planning
    ```

### Manual installation

If you already have a conda environment with the prerequisites:

```bash
pip install autolife-planning
```

## Verify installation

```python
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose

solver = create_ik_solver("left_arm")
print(f"Chain: {solver.base_frame} → {solver.ee_frame} ({solver.num_joints} joints)")
```
