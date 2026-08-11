"""The signing avatar: skeleton, poses and the wire format (plan §8.8).

    GlossSequence ─▶ (signsync.motion) ─▶ Animation ─▶ animation_to_dict ─▶ browser

Plan §8.8: a rig without expressive hands and face cannot render intelligible USL
however good the motion model is, so the default rig carries three joints per
finger on both hands and named facial channels for non-manual marking.
"""

from __future__ import annotations

from .export import animation_to_dict, export_animation, pose_to_dict, rig_to_dict
from .rig import (
    Animation,
    FaceChannel,
    Joint,
    Pose,
    Rig,
    default_rig,
    quat_between,
    quat_from_axis_angle,
    quat_identity,
    quat_multiply,
    quat_slerp,
    quat_to_matrix,
)

__all__ = [
    "Animation",
    "FaceChannel",
    "Joint",
    "Pose",
    "Rig",
    "animation_to_dict",
    "default_rig",
    "export_animation",
    "pose_to_dict",
    "quat_between",
    "quat_from_axis_angle",
    "quat_identity",
    "quat_multiply",
    "quat_slerp",
    "quat_to_matrix",
    "rig_to_dict",
]
