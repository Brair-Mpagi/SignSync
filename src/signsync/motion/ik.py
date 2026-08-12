"""Inverse kinematics for the arms (plan §8.7, stage 2).

Plan §8.7's second stage is "blend/interpolate between recorded clips using inverse
kinematics for smooth transitions". This is that IK: an analytic two-bone solver
placing the wrist at a target, which is what sign motion needs because a sign is
specified by *where the hands are*, not by shoulder and elbow angles.

Analytic rather than iterative (CCD/FABRIK): a two-bone chain has a closed-form
solution, so it is exact, deterministic and fast enough to run per frame on the CPU
budget plan §17 assumes. Iterative solvers also jitter between frames when the
target moves slightly, and jitter in an avatar's hands reads as a different sign.

The remaining degree of freedom — the elbow's swing around the shoulder-to-wrist
axis — is set from a pole hint rather than left arbitrary. Without it the elbow
drifts to wherever the maths lands, including through the torso, and elbow position
is visible information in signing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..avatar.rig import Pose, Rig, quat_between, quat_from_axis_angle, quat_multiply
from ..errors import SignSyncError

__all__ = ["IKResult", "solve_two_bone", "reach_wrist"]

_EPS = 1e-6


@dataclass(frozen=True)
class IKResult:
    """Solved rotations, and whether the target was actually reachable."""

    upper_rotation: np.ndarray
    lower_rotation: np.ndarray
    reached: bool
    """False when the target was out of range and the arm was extended towards it.

    Surfaced rather than silently clamped: a sign whose target the rig cannot reach
    is a rig or clip problem, and an avatar that quietly straightens its arm looks
    like it is signing something else.
    """

    error: float = 0.0


def solve_two_bone(
    origin: np.ndarray,
    target: np.ndarray,
    upper_length: float,
    lower_length: float,
    *,
    pole: np.ndarray | None = None,
    rest_direction: np.ndarray | None = None,
) -> IKResult:
    """Place the end of a two-bone chain at ``target``.

    Returns local rotations for the upper and lower bones, assuming both rest along
    ``rest_direction`` (the rig's bind direction, +y by default).
    """
    if upper_length <= 0 or lower_length <= 0:
        raise SignSyncError(
            f"bone lengths must be positive, got {upper_length} and {lower_length}"
        )

    origin = np.asarray(origin, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rest = np.asarray(rest_direction if rest_direction is not None else [0.0, 1.0, 0.0], np.float64)

    to_target = target - origin
    distance = float(np.linalg.norm(to_target))
    reach = upper_length + lower_length

    if distance < _EPS:
        identity = quat_from_axis_angle([1, 0, 0], 0.0)
        return IKResult(identity, identity, False, reach)

    reached = distance <= reach
    clamped = min(distance, reach * 0.999)
    # Also guard the degenerate near-fold case: a target closer than the difference
    # of the bone lengths cannot be reached by folding either.
    minimum = abs(upper_length - lower_length) * 1.001
    clamped = max(clamped, minimum)

    # Law of cosines for the interior angles.
    cos_elbow = (upper_length**2 + lower_length**2 - clamped**2) / (
        2 * upper_length * lower_length
    )
    elbow_angle = np.pi - np.arccos(np.clip(cos_elbow, -1.0, 1.0))

    cos_shoulder = (upper_length**2 + clamped**2 - lower_length**2) / (
        2 * upper_length * clamped
    )
    shoulder_offset = np.arccos(np.clip(cos_shoulder, -1.0, 1.0))

    direction = to_target / distance
    aim = quat_between(rest, direction)

    # Bend axis: perpendicular to the aim direction, biased towards the pole so the
    # elbow ends up where an elbow belongs.
    hint = np.asarray(pole, dtype=np.float64) if pole is not None else np.array([0.0, 0.0, -1.0])
    axis = np.cross(direction, hint)
    if np.linalg.norm(axis) < _EPS:
        axis = np.cross(direction, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < _EPS:
            axis = np.array([0.0, 0.0, 1.0])

    upper = quat_multiply(aim, quat_from_axis_angle(_local(aim, axis), -shoulder_offset))
    lower = quat_from_axis_angle(_local(aim, axis), elbow_angle)

    return IKResult(
        upper_rotation=upper,
        lower_rotation=lower,
        reached=reached,
        error=max(0.0, distance - reach),
    )


def _local(rotation: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Express a world-space axis in the frame the rotation produces."""
    from ..avatar.rig import quat_to_matrix

    matrix = quat_to_matrix(rotation)
    return matrix.T @ np.asarray(axis, dtype=np.float32)


def reach_wrist(
    rig: Rig,
    pose: Pose,
    side: str,
    target: np.ndarray,
    *,
    pole: np.ndarray | None = None,
) -> IKResult:
    """Rotate one arm in ``pose`` so its wrist lands on ``target`` (world space)."""
    if side not in ("left", "right"):
        raise SignSyncError(f"side must be 'left' or 'right', got {side!r}")

    positions = rig.forward_kinematics(pose)
    shoulder_index = rig.index(f"{side}_upper_arm")
    origin = positions[shoulder_index]

    upper_length = rig.joint(f"{side}_forearm").length
    lower_length = rig.joint(f"{side}_wrist").length

    # Elbows swing outward and back; the pole hint keeps them out of the torso.
    default_pole = np.array([1.0 if side == "left" else -1.0, 0.0, -1.0])
    result = solve_two_bone(
        origin,
        target,
        upper_length,
        lower_length,
        pole=pole if pole is not None else default_pole,
    )

    pose.set(rig, f"{side}_upper_arm", result.upper_rotation)
    pose.set(rig, f"{side}_forearm", result.lower_rotation)
    return result
