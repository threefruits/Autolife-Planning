#!/usr/bin/env python3
"""Generate subgroup URDFs for VAMP planning.

Takes the preprocessed simple URDF and produces subgroup variants by freezing
specified joints at given positions.  Also generates cricket JSON configs.

Subgroups:
  - Left/Right arm only (7 DOF) x high/low torso
  - Dual arm (14 DOF) x high/low torso
  - Torso + left/right arm (9 DOF) x high/low legs
  - Whole body without base (21 DOF)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Rotation helpers (pure-Python, no numpy) ─────────────────────────────────


def _rpy_to_matrix(r: float, p: float, y: float) -> list[list[float]]:
    """RPY (URDF fixed-axis XYZ) to 3x3 rotation matrix."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rz(theta: float) -> list[list[float]]:
    """Rotation matrix about Z axis."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _matmul3(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """3x3 matrix multiply."""
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


def _matrix_to_rpy(R: list[list[float]]) -> tuple[float, float, float]:
    """3x3 rotation matrix to RPY (URDF fixed-axis XYZ)."""
    if abs(R[2][0]) >= 1.0 - 1e-10:
        yaw = 0.0
        if R[2][0] < 0:
            pitch = math.pi / 2
            roll = math.atan2(R[0][1], R[0][2])
        else:
            pitch = -math.pi / 2
            roll = math.atan2(-R[0][1], -R[0][2])
    else:
        pitch = math.atan2(-R[2][0], math.sqrt(R[0][0] ** 2 + R[1][0] ** 2))
        roll = math.atan2(R[2][1], R[2][2])
        yaw = math.atan2(R[1][0], R[0][0])
    return roll, pitch, yaw


# ── Frozen joint value presets ───────────────────────────────────────────────

# Legs: high = standing tall, mid = medium crouch, low = deep crouch
LEGS_HIGH = {"Joint_Ankle": 0.0, "Joint_Knee": 0.0}
LEGS_MID = {"Joint_Ankle": 0.78, "Joint_Knee": 1.60}
LEGS_LOW = {"Joint_Ankle": 1.41, "Joint_Knee": 2.38}

# Waist: high = upright, mid/low = pitched forward
WAIST_HIGH = {"Joint_Waist_Pitch": 0.00, "Joint_Waist_Yaw": -0.14}
WAIST_MID = {"Joint_Waist_Pitch": 0.89, "Joint_Waist_Yaw": -0.14}
WAIST_LOW = {"Joint_Waist_Pitch": 0.95, "Joint_Waist_Yaw": -0.14}

# Neck: always frozen at zero
NECK_ZERO = {
    "Joint_Neck_Roll": 0.0,
    "Joint_Neck_Pitch": 0.0,
    "Joint_Neck_Yaw": 0.0,
}

# Arm home configs — symmetric between left and right.
# Symmetry: negate Shoulder_Inner, Shoulder_Outer, Elbow, Forearm; keep others.
LEFT_ARM_HOME = {
    "Joint_Left_Shoulder_Inner": 0.70,
    "Joint_Left_Shoulder_Outer": -0.14,
    "Joint_Left_UpperArm": -0.09,
    "Joint_Left_Elbow": 2.31,
    "Joint_Left_Forearm": 0.04,
    "Joint_Left_Wrist_Upper": -0.40,
    "Joint_Left_Wrist_Lower": 0.0,
}
RIGHT_ARM_HOME = {
    "Joint_Right_Shoulder_Inner": -0.70,
    "Joint_Right_Shoulder_Outer": 0.14,
    "Joint_Right_UpperArm": -0.09,
    "Joint_Right_Elbow": -2.31,
    "Joint_Right_Forearm": -0.04,
    "Joint_Right_Wrist_Upper": -0.40,
    "Joint_Right_Wrist_Lower": 0.0,
}

# ── Subgroup definitions ─────────────────────────────────────────────────────

SUBGROUPS = {
    # --- Single arm (7 DOF) ---
    "autolife_left_high": {
        "frozen": {**LEGS_HIGH, **WAIST_HIGH, **NECK_ZERO, **RIGHT_ARM_HOME},
        "end_effector": "Link_Left_Gripper_Right_Finger",
    },
    "autolife_left_mid": {
        "frozen": {**LEGS_MID, **WAIST_MID, **NECK_ZERO, **RIGHT_ARM_HOME},
        "end_effector": "Link_Left_Gripper_Right_Finger",
    },
    "autolife_left_low": {
        "frozen": {**LEGS_LOW, **WAIST_LOW, **NECK_ZERO, **RIGHT_ARM_HOME},
        "end_effector": "Link_Left_Gripper_Right_Finger",
    },
    "autolife_right_high": {
        "frozen": {**LEGS_HIGH, **WAIST_HIGH, **NECK_ZERO, **LEFT_ARM_HOME},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    "autolife_right_mid": {
        "frozen": {**LEGS_MID, **WAIST_MID, **NECK_ZERO, **LEFT_ARM_HOME},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    "autolife_right_low": {
        "frozen": {**LEGS_LOW, **WAIST_LOW, **NECK_ZERO, **LEFT_ARM_HOME},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    # --- Dual arm (14 DOF) ---
    "autolife_dual_high": {
        "frozen": {**LEGS_HIGH, **WAIST_HIGH, **NECK_ZERO},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    "autolife_dual_mid": {
        "frozen": {**LEGS_MID, **WAIST_MID, **NECK_ZERO},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    "autolife_dual_low": {
        "frozen": {**LEGS_LOW, **WAIST_LOW, **NECK_ZERO},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    # --- Torso + arm (9 DOF: 2 waist + 7 arm) ---
    "autolife_torso_left_high": {
        "frozen": {**LEGS_HIGH, **NECK_ZERO, **RIGHT_ARM_HOME},
        "end_effector": "Link_Left_Gripper_Right_Finger",
    },
    "autolife_torso_left_mid": {
        "frozen": {**LEGS_MID, **NECK_ZERO, **RIGHT_ARM_HOME},
        "end_effector": "Link_Left_Gripper_Right_Finger",
    },
    "autolife_torso_left_low": {
        "frozen": {**LEGS_LOW, **NECK_ZERO, **RIGHT_ARM_HOME},
        "end_effector": "Link_Left_Gripper_Right_Finger",
    },
    "autolife_torso_right_high": {
        "frozen": {**LEGS_HIGH, **NECK_ZERO, **LEFT_ARM_HOME},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    "autolife_torso_right_mid": {
        "frozen": {**LEGS_MID, **NECK_ZERO, **LEFT_ARM_HOME},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    "autolife_torso_right_low": {
        "frozen": {**LEGS_LOW, **NECK_ZERO, **LEFT_ARM_HOME},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
    # --- Whole body without base (21 DOF) ---
    "autolife_body": {
        "frozen": {},
        "end_effector": "Link_Right_Gripper_Right_Finger",
    },
}

# ── XML helpers (same as build_robot_description.py) ─────────────────────────


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Add pretty-print indentation."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    _indent(tree.getroot())
    tree.write(str(path), encoding="utf-8", xml_declaration=True)
    print(f"    wrote {path.name}")


# ── Core functions ───────────────────────────────────────────────────────────


def freeze_joint(joint: ET.Element, angle: float) -> None:
    """Convert a revolute joint to fixed, incorporating rotation at *angle*.

    For revolute joints the child-frame origin RPY is updated to include
    the additional rotation about the joint axis.  The XYZ is unchanged
    because the rotation is a pure rotation (no translation offset).
    """
    jtype = joint.get("type")
    if jtype == "fixed":
        return

    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

    axis_el = joint.find("axis")
    axis_str = axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1"
    ax, ay, az = (float(v) for v in axis_str.split())

    if jtype == "revolute":
        if abs(ax) > 0.01 or abs(ay) > 0.01:
            raise NotImplementedError(
                f"Non-Z-axis revolute joint not supported: axis={axis_str}"
            )
        rpy_str = origin.get("rpy", "0 0 0")
        r, p, y = (float(v) for v in rpy_str.split())
        R_orig = _rpy_to_matrix(r, p, y)
        R_joint = _rz(az * angle)
        R_new = _matmul3(R_orig, R_joint)
        nr, np_, ny = _matrix_to_rpy(R_new)
        origin.set("rpy", f"{nr:.10g} {np_:.10g} {ny:.10g}")

    elif jtype == "prismatic":
        xyz_str = origin.get("xyz", "0 0 0")
        x, y, z = (float(v) for v in xyz_str.split())
        x += ax * angle
        y += ay * angle
        z += az * angle
        origin.set("xyz", f"{x:.10g} {y:.10g} {z:.10g}")

    # Convert to fixed
    joint.set("type", "fixed")
    for tag in ("limit", "axis"):
        el = joint.find(tag)
        if el is not None:
            joint.remove(el)


def generate_subgroup_urdf(
    base_tree: ET.ElementTree, frozen_joints: dict[str, float]
) -> ET.ElementTree:
    """Create a subgroup URDF by freezing specified joints."""
    tree = copy.deepcopy(base_tree)
    root = tree.getroot()

    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if name in frozen_joints:
            freeze_joint(joint, frozen_joints[name])

    planning = [
        j.get("name")
        for j in root.findall("joint")
        if j.get("type") in ("revolute", "prismatic")
    ]
    print(f"    planning joints ({len(planning)} DOF): {planning}")
    return tree


def _snake_to_pascal(name: str) -> str:
    """autolife_left_high -> Autolife_Left_High

    Keeps underscores so that lower(name) in the FK template produces
    the original snake_case string, which VAMP's cmake expects as the
    nanobind submodule name.
    """
    return "_".join(w.capitalize() for w in name.split("_"))


def generate_cricket_config(name: str, end_effector: str, resolution: int = 64) -> dict:
    return {
        "name": _snake_to_pascal(name),
        "urdf": f"autolife/{name}_spherized.urdf",
        "srdf": "autolife/autolife.srdf",
        "end_effector": end_effector,
        "resolution": resolution,
        "template": "templates/fk_template.hh",
        "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template.hh"}],
        "output": f"{name}_fk.hh",
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing autolife_simple.urdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for subgroup URDFs (usually same as input-dir)",
    )
    parser.add_argument(
        "--cricket-dir",
        type=Path,
        required=True,
        help="Cricket resources directory for JSON configs",
    )
    args = parser.parse_args()

    input_urdf = args.input_dir / "autolife_simple.urdf"
    if not input_urdf.exists():
        raise FileNotFoundError(f"Base URDF not found: {input_urdf}")

    base_tree = ET.parse(str(input_urdf))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cricket_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:   {input_urdf}")
    print(f"Output:  {args.output_dir}")
    print(f"Cricket: {args.cricket_dir}")

    for name, config in SUBGROUPS.items():
        print(f"\n  [{name}]")
        tree = generate_subgroup_urdf(base_tree, config["frozen"])
        _write_xml(tree, args.output_dir / f"{name}.urdf")

        cricket = generate_cricket_config(name, config["end_effector"])
        cricket_path = args.cricket_dir / f"{name}.json"
        with open(cricket_path, "w") as f:
            json.dump(cricket, f, indent=4)
            f.write("\n")
        print(f"    wrote {cricket_path.name}")

    print(f"\nGenerated {len(SUBGROUPS)} subgroup URDFs and cricket configs.")
    print("\nNext steps:")
    print("  1. bash scripts/spherize_subgroups.sh")
    print("  2. bash scripts/generate_fk.sh")
    print("  3. Rebuild VAMP: cd third_party/vamp && pip install -e .")


if __name__ == "__main__":
    main()
