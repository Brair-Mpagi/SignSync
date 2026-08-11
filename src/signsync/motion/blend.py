"""Transition timing and easing (plan §8.7, stage 2).

Plan §8.7 warns that "naive keyframe-to-keyframe interpolation produces robotic
motion that Deaf users reject regardless of correctness". Two things fix most of
that, and both are here:

**Ease, don't ramp linearly.** Real limbs accelerate and decelerate. Linear
interpolation moves at constant speed and stops instantly, which is the single most
recognisable tell of machine-generated motion.

**Scale the transition to the distance travelled.** A fixed transition length makes
short movements look sluggish and long ones look teleported. Transition length here
is derived from how far the hands actually have to move, which is what makes the
result read as one continuous utterance rather than a slideshow.
"""

from __future__ import annotations

import numpy as np

from ..avatar.rig import Pose, Rig
from ..errors import SignSyncError

__all__ = ["ease_in_out", "ease_out", "transition_frames", "hand_travel"]


def ease_in_out(t: float) -> float:
    """Smoothstep easing on ``[0, 1]``."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    """Decelerating easing, for settling into a hold at the end of a sign."""
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - t) ** 3


def hand_travel(rig: Rig, start: Pose, end: Pose) -> float:
    """Distance both wrists move between two poses, in rig units."""
    a = rig.forward_kinematics(start)
    b = rig.forward_kinematics(end)
    total = 0.0
    for side in ("left", "right"):
        index = rig.index(f"{side}_wrist")
        total += float(np.linalg.norm(b[index] - a[index]))
    return total


def transition_frames(
    rig: Rig,
    start: Pose,
    end: Pose,
    *,
    fps: float = 30.0,
    minimum: int = 3,
    maximum: int = 12,
    speed: float = 2.4,
) -> int:
    """How many frames the hands need to travel between two poses.

    ``speed`` is in rig units per second — roughly shoulder-widths per second — and
    is a placeholder until real signing tempo is measured from the corpus. Plan §9.3
    records signing speed as a recruitment axis precisely because it varies a lot,
    so a single constant here is a known simplification, not a finding.
    """
    if minimum < 1 or maximum < minimum:
        raise SignSyncError(f"need 1 <= minimum ({minimum}) <= maximum ({maximum})")
    if fps <= 0 or speed <= 0:
        raise SignSyncError("fps and speed must be positive")

    distance = hand_travel(rig, start, end)
    frames = int(round(distance / speed * fps))
    return int(np.clip(frames, minimum, maximum))
