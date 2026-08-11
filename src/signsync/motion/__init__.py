"""Sign motion generation (plan §8.7).

    GlossSequence ─▶ ClipLibrary ─▶ MotionGenerator ─▶ Animation ─▶ (signsync.avatar)
                                          │
                                    IK + eased transitions + marker channels

Stages 1 and 2 of plan §8.7 are implemented: clip playback, and blended transitions
with inverse kinematics. Stage 3 — a learned motion model for novel sequences — is
not, and a gloss with no clip is reported rather than approximated.
"""

from __future__ import annotations

from .blend import ease_in_out, ease_out, hand_travel, transition_frames
from .generator import GeneratedMotion, MotionConfig, MotionGenerator
from .ik import IKResult, reach_wrist, solve_two_bone
from .library import ClipLibrary, ProceduralLibrary, RecordedLibrary, SignClip

__all__ = [
    "ClipLibrary",
    "GeneratedMotion",
    "IKResult",
    "MotionConfig",
    "MotionGenerator",
    "ProceduralLibrary",
    "RecordedLibrary",
    "SignClip",
    "ease_in_out",
    "ease_out",
    "hand_travel",
    "reach_wrist",
    "solve_two_bone",
    "transition_frames",
]
