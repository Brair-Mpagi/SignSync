"""Computer vision layer: camera in, normalised landmark features out (plan §8.1).

    frames --> Tracker --> LandmarkSequence --> normalise_sequence --> encode_sequence
                                                                          |
                                                                    (T, D) features

``capture`` is imported lazily via :func:`camera_stream` because it needs OpenCV;
everything else in this package runs on NumPy alone.
"""

from __future__ import annotations

from typing import Any

from .features import FeatureConfig, FeatureLayout, encode_sequence, motion_energy, resample
from .normalise import (
    BodyFrame,
    NormalisedSequence,
    StreamingNormaliser,
    estimate_body_frame,
    normalise_sequence,
)
from .schema import (
    FACE_GROUPS,
    FACE_INDICES,
    N_FACE,
    N_HAND,
    N_POSE,
    UPPER_BODY_POSE,
    Channel,
    FrameLandmarks,
    HandIndex,
    LandmarkSequence,
    PoseIndex,
)
from .tracker import MediaPipeHolisticTracker, ReplayTracker, Tracker, track_video

__all__ = [
    "BodyFrame",
    "Channel",
    "FACE_GROUPS",
    "FACE_INDICES",
    "FeatureConfig",
    "FeatureLayout",
    "FrameLandmarks",
    "HandIndex",
    "LandmarkSequence",
    "MediaPipeHolisticTracker",
    "N_FACE",
    "N_HAND",
    "N_POSE",
    "NormalisedSequence",
    "PoseIndex",
    "ReplayTracker",
    "StreamingNormaliser",
    "Tracker",
    "UPPER_BODY_POSE",
    "camera_stream",
    "encode_sequence",
    "estimate_body_frame",
    "motion_energy",
    "normalise_sequence",
    "resample",
    "track_video",
]


def camera_stream(**kwargs: Any) -> Any:
    """Open a :class:`~signsync.vision.capture.CameraStream` (needs the ``vision`` extra)."""
    from .capture import CameraSettings, CameraStream

    return CameraStream(CameraSettings(**kwargs))
