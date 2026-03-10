import os

import numpy as np

from autolife_planning.types.robot import CameraConfig, ChainConfig, RobotConfig

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCES_DIR = os.path.join(_PKG_ROOT, "resources", "robot", "autolife")

# Atomic joint groups — indices into the full 24-DOF configuration array.
# Order must match VAMP's URDF tree traversal: base, legs, waist, left arm, neck, right arm.
JOINT_GROUPS = {
    "base": slice(0, 3),  # Virtual_X, Virtual_Y, Virtual_Theta
    "legs": slice(3, 5),  # Ankle, Knee
    "waist": slice(5, 7),  # Waist_Pitch, Waist_Yaw
    "left_arm": slice(7, 14),  # Shoulder → Wrist (7 DOF)
    "neck": slice(14, 17),  # Roll, Pitch, Yaw
    "right_arm": slice(17, 24),  # Shoulder → Wrist (7 DOF)
}

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    "left_arm": ChainConfig(
        base_link="Link_Waist_Yaw_to_Shoulder_Inner",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "right_arm": ChainConfig(
        base_link="Link_Waist_Yaw_to_Shoulder_Inner",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_left": ChainConfig(
        base_link="Link_Ground_Vehicle",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=11,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_right": ChainConfig(
        base_link="Link_Ground_Vehicle",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=11,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_base_left": ChainConfig(
        base_link="Link_Zero_Point",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=14,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife_base.urdf"),
    ),
    "whole_body_base_right": ChainConfig(
        base_link="Link_Zero_Point",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=14,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife_base.urdf"),
    ),
}

VIZ_URDF_PATH = os.path.join(_RESOURCES_DIR, "autolife_viz.urdf")

autolife_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    joint_names=[
        # [0:3]   base (virtual planar joints)
        "Joint_Virtual_X",
        "Joint_Virtual_Y",
        "Joint_Virtual_Theta",
        # [3:5]   legs
        "Joint_Ankle",
        "Joint_Knee",
        # [5:7]   waist
        "Joint_Waist_Pitch",
        "Joint_Waist_Yaw",
        # [7:14]  left arm
        "Joint_Left_Shoulder_Inner",
        "Joint_Left_Shoulder_Outer",
        "Joint_Left_UpperArm",
        "Joint_Left_Elbow",
        "Joint_Left_Forearm",
        "Joint_Left_Wrist_Upper",
        "Joint_Left_Wrist_Lower",
        # [14:17] neck
        "Joint_Neck_Roll",
        "Joint_Neck_Pitch",
        "Joint_Neck_Yaw",
        # [17:24] right arm
        "Joint_Right_Shoulder_Inner",
        "Joint_Right_Shoulder_Outer",
        "Joint_Right_UpperArm",
        "Joint_Right_Elbow",
        "Joint_Right_Forearm",
        "Joint_Right_Wrist_Upper",
        "Joint_Right_Wrist_Lower",
    ],
    camera=CameraConfig(
        link_name="Link_Camera_Head_Forehead",
        width=640,
        height=480,
        fov=60.0,
        near=0.1,
        far=10.0,
    ),
)

# VAMP subgroup robot names for planning.
# Each maps to a separate VAMP module with the full robot body (all links and
# collision geometry) but only the listed joints movable.
_LEFT_ARM_JOINTS = [
    "Joint_Left_Shoulder_Inner",
    "Joint_Left_Shoulder_Outer",
    "Joint_Left_UpperArm",
    "Joint_Left_Elbow",
    "Joint_Left_Forearm",
    "Joint_Left_Wrist_Upper",
    "Joint_Left_Wrist_Lower",
]
_RIGHT_ARM_JOINTS = [
    "Joint_Right_Shoulder_Inner",
    "Joint_Right_Shoulder_Outer",
    "Joint_Right_UpperArm",
    "Joint_Right_Elbow",
    "Joint_Right_Forearm",
    "Joint_Right_Wrist_Upper",
    "Joint_Right_Wrist_Lower",
]
_WAIST_JOINTS = ["Joint_Waist_Pitch", "Joint_Waist_Yaw"]

PLANNING_SUBGROUPS = {
    # Single arm (7 DOF) -- high/mid/low torso
    "autolife_left_high": {"dof": 7, "joints": _LEFT_ARM_JOINTS},
    "autolife_left_mid": {"dof": 7, "joints": _LEFT_ARM_JOINTS},
    "autolife_left_low": {"dof": 7, "joints": _LEFT_ARM_JOINTS},
    "autolife_right_high": {"dof": 7, "joints": _RIGHT_ARM_JOINTS},
    "autolife_right_mid": {"dof": 7, "joints": _RIGHT_ARM_JOINTS},
    "autolife_right_low": {"dof": 7, "joints": _RIGHT_ARM_JOINTS},
    # Dual arm (14 DOF) -- high/mid/low torso
    "autolife_dual_high": {"dof": 14, "joints": _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS},
    "autolife_dual_mid": {"dof": 14, "joints": _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS},
    "autolife_dual_low": {"dof": 14, "joints": _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS},
    # Torso + arm (9 DOF: 2 waist + 7 arm) -- high/mid/low legs
    "autolife_torso_left_high": {"dof": 9, "joints": _WAIST_JOINTS + _LEFT_ARM_JOINTS},
    "autolife_torso_left_mid": {"dof": 9, "joints": _WAIST_JOINTS + _LEFT_ARM_JOINTS},
    "autolife_torso_left_low": {"dof": 9, "joints": _WAIST_JOINTS + _LEFT_ARM_JOINTS},
    "autolife_torso_right_high": {
        "dof": 9,
        "joints": _WAIST_JOINTS + _RIGHT_ARM_JOINTS,
    },
    "autolife_torso_right_mid": {"dof": 9, "joints": _WAIST_JOINTS + _RIGHT_ARM_JOINTS},
    "autolife_torso_right_low": {"dof": 9, "joints": _WAIST_JOINTS + _RIGHT_ARM_JOINTS},
    # Whole body without base (21 DOF)
    "autolife_body": {
        "dof": 21,
        "joints": [
            "Joint_Ankle",
            "Joint_Knee",
            *_WAIST_JOINTS,
            *_LEFT_ARM_JOINTS,
            "Joint_Neck_Roll",
            "Joint_Neck_Pitch",
            "Joint_Neck_Yaw",
            *_RIGHT_ARM_JOINTS,
        ],
    },
}

HOME_JOINTS = np.array(
    [
        # [0:3]   base
        0.0,
        0.0,
        0.0,
        # [3:5]   legs (high stance)
        0.0,
        0.0,
        # [5:7]   waist
        0.00,
        -0.14,
        # [7:14]  left arm
        0.70,
        -0.14,
        -0.09,
        2.31,
        0.04,
        -0.40,
        0.0,
        # [14:17] neck
        0.0,
        0.0,
        0.0,
        # [17:24] right arm (symmetric with left)
        -0.70,
        0.14,
        -0.09,
        -2.31,
        -0.04,
        -0.40,
        0.0,
    ]
)

# Frozen stance presets: legs + waist values for each height level.
# Must match the values in scripts/build_subgroup_descriptions.py.
_STANCE_PRESETS: dict[str, dict[str, float]] = {
    "high": {
        "Joint_Ankle": 0.0,
        "Joint_Knee": 0.0,
        "Joint_Waist_Pitch": 0.00,
        "Joint_Waist_Yaw": -0.14,
    },
    "mid": {
        "Joint_Ankle": 0.78,
        "Joint_Knee": 1.60,
        "Joint_Waist_Pitch": 0.89,
        "Joint_Waist_Yaw": -0.14,
    },
    "low": {
        "Joint_Ankle": 1.41,
        "Joint_Knee": 2.38,
        "Joint_Waist_Pitch": 0.95,
        "Joint_Waist_Yaw": -0.14,
    },
}


def subgroup_base_config(name: str) -> np.ndarray:
    """Return a 24-DOF base config with frozen joints at the correct stance values.

    For subgroups ending in ``_high``, ``_mid``, or ``_low`` the leg and waist
    joints are set to the corresponding frozen preset.  Movable joints will be
    overridden later by ``embed_config`` / ``embed_path``.
    """
    base = HOME_JOINTS.copy()
    joint_names = autolife_robot_config.joint_names
    for height in ("high", "mid", "low"):
        if name.endswith(f"_{height}"):
            for joint_name, value in _STANCE_PRESETS[height].items():
                base[joint_names.index(joint_name)] = value
            break
    return base
