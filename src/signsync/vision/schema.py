"""Landmark schema — the contract between the camera and everything downstream.

Plan §8.1: frames become structured ``(x, y, z)`` landmark vectors rather than raw
pixels, because that keeps input dimensionality small enough to train on a modest
corpus, runs on a CPU, and generalises across skin tone, lighting and clothing.

Index conventions follow MediaPipe Holistic so a real tracker can populate these
structures without remapping:

* pose — 33 landmarks (BlazePose topology)
* hands — 21 landmarks each, wrist at index 0
* face — a curated subset of the 468-point mesh, see :data:`FACE_GROUPS`

Coordinates are image-normalised on capture (x, y in ``[0, 1]``, z roughly in the
same scale as x, negative towards the camera) and become signer-normalised in
:mod:`signsync.vision.normalise`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "N_POSE",
    "N_HAND",
    "N_FACE",
    "PoseIndex",
    "HandIndex",
    "FACE_GROUPS",
    "FACE_INDICES",
    "FACE_MIRROR_PAIRS",
    "FACE_MIRROR_PERM",
    "POSE_MIRROR_PAIRS",
    "UPPER_BODY_MIRROR_PERM",
    "mirror_permutation",
    "FACE_SUBSET_INDEX",
    "UPPER_BODY_POSE",
    "Channel",
    "FrameLandmarks",
    "LandmarkSequence",
]

N_POSE = 33
N_HAND = 21


class PoseIndex:
    """Named MediaPipe pose landmarks actually used by this project."""

    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24


#: Pose landmarks kept in the feature vector. Legs carry no linguistic information
#: and signers are frequently seated or framed at chest height, so including them
#: adds noise and missing values rather than signal.
UPPER_BODY_POSE: tuple[int, ...] = (
    PoseIndex.NOSE,
    PoseIndex.LEFT_EYE,
    PoseIndex.RIGHT_EYE,
    PoseIndex.LEFT_EAR,
    PoseIndex.RIGHT_EAR,
    PoseIndex.LEFT_SHOULDER,
    PoseIndex.RIGHT_SHOULDER,
    PoseIndex.LEFT_ELBOW,
    PoseIndex.RIGHT_ELBOW,
    PoseIndex.LEFT_WRIST,
    PoseIndex.RIGHT_WRIST,
    PoseIndex.LEFT_INDEX,
    PoseIndex.RIGHT_INDEX,
    PoseIndex.LEFT_THUMB,
    PoseIndex.RIGHT_THUMB,
    PoseIndex.LEFT_HIP,
    PoseIndex.RIGHT_HIP,
)


class HandIndex:
    """Named MediaPipe hand landmarks used for handshape descriptors."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


#: Face-mesh landmarks kept, grouped by the grammatical function they serve.
#:
#: Plan §8.1 says to keep only the landmarks relevant to non-manual markers. Brow
#: position marks questions and conditionals, mouth shape carries mouthing and
#: adverbial morphemes, and the head/eye points give the head pose that marks
#: negation and topic. The remaining ~430 mesh points describe face *identity*,
#: which is both useless here and a privacy liability we would rather not store.
FACE_GROUPS: dict[str, tuple[int, ...]] = {
    "left_brow": (70, 63, 105, 66, 107),
    "right_brow": (336, 296, 334, 293, 300),
    "left_eye": (33, 160, 158, 133, 153, 144),
    "right_eye": (362, 385, 387, 263, 373, 380),
    "outer_lips": (61, 39, 0, 269, 291, 405, 17, 181),
    "inner_lips": (78, 13, 308, 14),
    "nose": (168, 1),
    "jaw": (152,),
    "brow_centre": (10,),
    "cheeks": (234, 454),
}

#: Flat, ordered list of kept mesh indices. Order is stable and load-bearing: it
#: defines the layout of the face block in the feature vector.
FACE_INDICES: tuple[int, ...] = tuple(idx for group in FACE_GROUPS.values() for idx in group)

#: Reverse map from a full-mesh index to its position in the kept subset.
FACE_SUBSET_INDEX: dict[int, int] = {mesh: i for i, mesh in enumerate(FACE_INDICES)}

N_FACE = len(FACE_INDICES)

#: Mesh-index pairs that exchange identity when the signing space is mirrored.
#:
#: Reflecting a face is not just negating x. The left-brow landmarks have to *become*
#: the right-brow landmarks, or the mirrored face block is scrambled: point 0 would
#: hold the geometry of a point on the other side of the face, and every model reading
#: that block would see noise. This matters for every left-handed signer, whose clips
#: are mirrored into canonical form by :mod:`signsync.vision.normalise`.
FACE_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    # brows, in matching order
    (70, 336),
    (63, 296),
    (105, 334),
    (66, 293),
    (107, 300),
    # eyes: outer/inner corners and upper/lower lids
    (33, 263),
    (160, 387),
    (158, 385),
    (133, 362),
    (153, 380),
    (144, 373),
    # lips
    (61, 291),
    (39, 269),
    (181, 405),
    (78, 308),
    # cheeks
    (234, 454),
)


#: Pose landmarks that exchange identity when the signing space is mirrored.
POSE_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (PoseIndex.LEFT_EYE, PoseIndex.RIGHT_EYE),
    (PoseIndex.LEFT_EAR, PoseIndex.RIGHT_EAR),
    (PoseIndex.LEFT_SHOULDER, PoseIndex.RIGHT_SHOULDER),
    (PoseIndex.LEFT_ELBOW, PoseIndex.RIGHT_ELBOW),
    (PoseIndex.LEFT_WRIST, PoseIndex.RIGHT_WRIST),
    (PoseIndex.LEFT_INDEX, PoseIndex.RIGHT_INDEX),
    (PoseIndex.LEFT_THUMB, PoseIndex.RIGHT_THUMB),
    (PoseIndex.LEFT_HIP, PoseIndex.RIGHT_HIP),
)


def mirror_permutation(
    order: Sequence[int], pairs: Iterable[tuple[int, int]]
) -> tuple[int, ...]:
    """Permutation that relabels ``order`` under a left-right mirror.

    Mirroring a body is not just negating x: the left shoulder has to *become* the
    right shoulder, and the left brow the right brow. Without the relabelling, a
    mirrored clip's skeleton is internally inconsistent — arms crossed over,
    landmarks holding the opposite side's geometry — and every downstream model
    reads it as an anatomically impossible pose rather than as the same sign.

    Landmarks with no partner in ``order`` (midline points, or a partner that was
    filtered out) map to themselves.
    """
    position = {landmark: i for i, landmark in enumerate(order)}
    permutation = list(range(len(order)))
    for left, right in pairs:
        if left in position and right in position:
            i, j = position[left], position[right]
            permutation[i], permutation[j] = j, i
    return tuple(permutation)


#: ``face[..., FACE_MIRROR_PERM, :]`` reorders a face block into its mirrored
#: identity. Midline points (nose bridge, nose tip, chin, forehead, lip centres) map
#: to themselves.
FACE_MIRROR_PERM: tuple[int, ...] = mirror_permutation(FACE_INDICES, FACE_MIRROR_PAIRS)

#: The same, for the upper-body pose block.
UPPER_BODY_MIRROR_PERM: tuple[int, ...] = mirror_permutation(UPPER_BODY_POSE, POSE_MIRROR_PAIRS)


class Channel:
    """Ordering of the presence mask columns."""

    POSE = 0
    LEFT_HAND = 1
    RIGHT_HAND = 2
    FACE = 3
    COUNT = 4

    NAMES = ("pose", "left_hand", "right_hand", "face")


def _zeros(n: int) -> np.ndarray:
    return np.zeros((n, 3), dtype=np.float32)


@dataclass(frozen=True)
class FrameLandmarks:
    """One tracked frame.

    Any channel may be absent — a hand leaves the frame, the face is turned away,
    tracking drops out. Absent channels are zero-filled *and* flagged, never
    silently interpolated: a model that cannot tell "hand at the origin" from "hand
    not detected" will learn the tracker's failure modes as if they were signs.
    """

    pose: np.ndarray = field(default_factory=lambda: _zeros(N_POSE))
    left_hand: np.ndarray = field(default_factory=lambda: _zeros(N_HAND))
    right_hand: np.ndarray = field(default_factory=lambda: _zeros(N_HAND))
    face: np.ndarray = field(default_factory=lambda: _zeros(N_FACE))
    present: np.ndarray = field(default_factory=lambda: np.zeros(Channel.COUNT, dtype=bool))
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        for name, expected in (
            ("pose", N_POSE),
            ("left_hand", N_HAND),
            ("right_hand", N_HAND),
            ("face", N_FACE),
        ):
            arr = getattr(self, name)
            if arr.shape != (expected, 3):
                raise ValueError(f"{name} must have shape ({expected}, 3), got {arr.shape}")
        if self.present.shape != (Channel.COUNT,):
            raise ValueError(
                f"present must have shape ({Channel.COUNT},), got {self.present.shape}"
            )

    @property
    def has_any_hand(self) -> bool:
        return bool(self.present[Channel.LEFT_HAND] or self.present[Channel.RIGHT_HAND])


@dataclass
class LandmarkSequence:
    """A time series of tracked frames, stored channel-wise for vectorised maths.

    Shapes: ``pose (T, 33, 3)``, ``left_hand``/``right_hand (T, 21, 3)``,
    ``face (T, N_FACE, 3)``, ``present (T, 4)``, ``timestamps (T,)``.
    """

    pose: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    face: np.ndarray
    present: np.ndarray
    timestamps: np.ndarray
    fps: float = 30.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = len(self.pose)
        for name, expected in (
            ("pose", (t, N_POSE, 3)),
            ("left_hand", (t, N_HAND, 3)),
            ("right_hand", (t, N_HAND, 3)),
            ("face", (t, N_FACE, 3)),
            ("present", (t, Channel.COUNT)),
            ("timestamps", (t,)),
        ):
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    def __len__(self) -> int:
        return len(self.pose)

    @property
    def duration(self) -> float:
        """Seconds spanned by the sequence."""
        return len(self) / self.fps

    @classmethod
    def empty(cls, n_frames: int = 0, fps: float = 30.0) -> LandmarkSequence:
        return cls(
            pose=np.zeros((n_frames, N_POSE, 3), dtype=np.float32),
            left_hand=np.zeros((n_frames, N_HAND, 3), dtype=np.float32),
            right_hand=np.zeros((n_frames, N_HAND, 3), dtype=np.float32),
            face=np.zeros((n_frames, N_FACE, 3), dtype=np.float32),
            present=np.zeros((n_frames, Channel.COUNT), dtype=bool),
            timestamps=np.arange(n_frames, dtype=np.float32) / fps,
            fps=fps,
        )

    @classmethod
    def from_frames(
        cls, frames: list[FrameLandmarks], fps: float = 30.0, **meta: Any
    ) -> LandmarkSequence:
        if not frames:
            return cls.empty(0, fps)
        stack = lambda name: np.stack([getattr(f, name) for f in frames]).astype(  # noqa: E731
            np.float32
        )
        return cls(
            pose=stack("pose"),
            left_hand=stack("left_hand"),
            right_hand=stack("right_hand"),
            face=stack("face"),
            present=np.stack([f.present for f in frames]),
            timestamps=np.array([f.timestamp for f in frames], dtype=np.float32),
            fps=fps,
            meta=dict(meta),
        )

    def frame(self, i: int) -> FrameLandmarks:
        return FrameLandmarks(
            pose=self.pose[i],
            left_hand=self.left_hand[i],
            right_hand=self.right_hand[i],
            face=self.face[i],
            present=self.present[i],
            timestamp=float(self.timestamps[i]),
        )

    def slice(self, start: int, stop: int) -> LandmarkSequence:
        """Frame-range view, used by continuous-signing segmentation (plan §8.3)."""
        start = max(0, start)
        stop = min(len(self), stop)
        if stop < start:
            raise ValueError(f"stop {stop} precedes start {start}")
        return LandmarkSequence(
            pose=self.pose[start:stop],
            left_hand=self.left_hand[start:stop],
            right_hand=self.right_hand[start:stop],
            face=self.face[start:stop],
            present=self.present[start:stop],
            timestamps=self.timestamps[start:stop],
            fps=self.fps,
            meta=dict(self.meta),
        )

    def coverage(self) -> dict[str, float]:
        """Fraction of frames in which each channel was tracked.

        Used as a quality gate at annotation time: a clip where the dominant hand
        is tracked in 40% of frames is a recording problem, not training data.
        """
        if len(self) == 0:
            return dict.fromkeys(Channel.NAMES, 0.0)
        return {
            name: float(self.present[:, i].mean()) for i, name in enumerate(Channel.NAMES)
        }

    def save(self, path: str | Path) -> Path:
        """Write to a compressed ``.npz``.

        Landmark files are derived from identifiable people; see
        ``docs/data-protection.md`` before deciding where this path points.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pose=self.pose,
            left_hand=self.left_hand,
            right_hand=self.right_hand,
            face=self.face,
            present=self.present,
            timestamps=self.timestamps,
            fps=np.float32(self.fps),
            meta=np.array(_encode_meta(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LandmarkSequence:
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                pose=data["pose"],
                left_hand=data["left_hand"],
                right_hand=data["right_hand"],
                face=data["face"],
                present=data["present"],
                timestamps=data["timestamps"],
                fps=float(data["fps"]),
                meta=_decode_meta(str(data["meta"])),
            )


def _encode_meta(meta: dict[str, Any]) -> str:
    import json

    return json.dumps(meta, sort_keys=True)


def _decode_meta(raw: str) -> dict[str, Any]:
    import json

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
