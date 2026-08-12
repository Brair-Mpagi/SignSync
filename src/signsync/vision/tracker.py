"""Landmark trackers.

The pipeline talks to a :class:`Tracker`, never to MediaPipe directly, for two
reasons. Plan §10 keeps the landmark tracker swappable ("MediaPipe *or equivalent*"),
and plan §17 requires the rest of the system to be runnable and testable on a
machine with no camera and no ``mediapipe`` install — which
:class:`ReplayTracker` provides.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..capabilities import require
from .schema import (
    FACE_INDICES,
    N_FACE,
    N_HAND,
    N_POSE,
    Channel,
    FrameLandmarks,
    LandmarkSequence,
)

__all__ = ["Tracker", "MediaPipeHolisticTracker", "ReplayTracker", "track_video"]


@runtime_checkable
class Tracker(Protocol):
    """Turns one video frame into one :class:`FrameLandmarks`."""

    def track(self, frame: np.ndarray, timestamp: float) -> FrameLandmarks:
        """Extract landmarks from a BGR image."""
        ...

    def close(self) -> None:
        """Release any underlying resources."""
        ...


class MediaPipeHolisticTracker:
    """MediaPipe Holistic hands + pose + face (plan §8.1).

    Only the face landmarks in :data:`~signsync.vision.schema.FACE_INDICES` are
    retained. The other ~430 mesh points describe facial *identity* rather than
    grammar: dropping them shrinks the feature vector and avoids storing more
    identifying detail about participants than the task needs (plan §16).
    """

    def __init__(
        self,
        *,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_face_landmarks: bool = False,
    ) -> None:
        mp = require("mediapipe", feature="live landmark tracking")
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._cv2 = require("opencv", feature="colour conversion for the tracker")

    def track(self, frame: np.ndarray, timestamp: float) -> FrameLandmarks:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)

        present = np.zeros(Channel.COUNT, dtype=bool)
        pose = _landmarks_to_array(getattr(results, "pose_landmarks", None), N_POSE)
        present[Channel.POSE] = pose is not None

        left = _landmarks_to_array(getattr(results, "left_hand_landmarks", None), N_HAND)
        present[Channel.LEFT_HAND] = left is not None

        right = _landmarks_to_array(getattr(results, "right_hand_landmarks", None), N_HAND)
        present[Channel.RIGHT_HAND] = right is not None

        face_full = _landmarks_to_array(getattr(results, "face_landmarks", None), None)
        face = None
        if face_full is not None and len(face_full) > max(FACE_INDICES):
            face = face_full[list(FACE_INDICES)]
        present[Channel.FACE] = face is not None

        return FrameLandmarks(
            pose=pose if pose is not None else np.zeros((N_POSE, 3), dtype=np.float32),
            left_hand=left if left is not None else np.zeros((N_HAND, 3), dtype=np.float32),
            right_hand=right if right is not None else np.zeros((N_HAND, 3), dtype=np.float32),
            face=face if face is not None else np.zeros((N_FACE, 3), dtype=np.float32),
            present=present,
            timestamp=timestamp,
        )

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> MediaPipeHolisticTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class ReplayTracker:
    """Replays a recorded :class:`LandmarkSequence`.

    Lets the whole downstream pipeline — recognition, translation, speech, avatar —
    be developed, demonstrated and tested with no camera, no ``mediapipe`` and no
    network. It ignores the image argument entirely and returns the next stored
    frame, so it is a drop-in :class:`Tracker`.
    """

    def __init__(self, sequence: LandmarkSequence, *, loop: bool = False) -> None:
        self.sequence = sequence
        self.loop = loop
        self._cursor = 0

    @classmethod
    def from_file(cls, path: str | Path, *, loop: bool = False) -> ReplayTracker:
        return cls(LandmarkSequence.load(path), loop=loop)

    @property
    def exhausted(self) -> bool:
        return not self.loop and self._cursor >= len(self.sequence)

    def track(
        self, frame: np.ndarray | None = None, timestamp: float | None = None
    ) -> FrameLandmarks:
        if len(self.sequence) == 0:
            raise StopIteration("replay sequence is empty")
        if self._cursor >= len(self.sequence):
            if not self.loop:
                raise StopIteration("replay sequence exhausted")
            self._cursor = 0
        result = self.sequence.frame(self._cursor)
        self._cursor += 1
        if timestamp is None:
            return result
        return FrameLandmarks(
            pose=result.pose,
            left_hand=result.left_hand,
            right_hand=result.right_hand,
            face=result.face,
            present=result.present,
            timestamp=timestamp,
        )

    def frames(self) -> Iterator[FrameLandmarks]:
        while not self.exhausted:
            try:
                yield self.track()
            except StopIteration:
                return

    def reset(self) -> None:
        self._cursor = 0

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def track_video(
    path: str | Path,
    tracker: Tracker | None = None,
    *,
    max_frames: int | None = None,
) -> LandmarkSequence:
    """Track a video file into a :class:`LandmarkSequence`.

    This is the offline half of data collection: recordings come back from district
    visits as video files and become landmark files here (plan §9.3).
    """
    cv2 = require("opencv", feature="reading video files")
    tracker = tracker or MediaPipeHolisticTracker()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frames: list[FrameLandmarks] = []
        index = 0
        while max_frames is None or index < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            frames.append(tracker.track(image, index / fps))
            index += 1
    finally:
        capture.release()

    return LandmarkSequence.from_frames(frames, fps=fps, source=str(path))


def _landmarks_to_array(landmark_list: Any, expected: int | None) -> np.ndarray | None:
    """Convert a MediaPipe landmark list to ``(N, 3)``, or ``None`` if absent."""
    if landmark_list is None:
        return None
    points = getattr(landmark_list, "landmark", None)
    if not points:
        return None
    array = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    if expected is not None and len(array) != expected:
        return None
    return array
