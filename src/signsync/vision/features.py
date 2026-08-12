"""Feature encoding — normalised landmarks to the matrix a temporal model sees.

Produces a ``(T, D)`` float32 array with a documented block layout
(:class:`FeatureLayout`). The layout is exposed rather than hidden because per-sign
confusion analysis (plan §15) needs to attribute errors to a channel: "this model
confuses these two signs because it is ignoring the face" is only answerable if the
face block's column range is known.

Blocks, in order:

===============  ================================================================
``body``         upper-body skeleton in body-frame units
``dominant_local`` wrist-centred handshape of the signing hand
``weak_local``     wrist-centred handshape of the non-dominant hand
``dominant_wrist`` where the dominant hand is, in the body frame
``weak_wrist``     where the non-dominant hand is
``dominant_shape`` handshape descriptors (extension, spread, curl)
``weak_shape``     handshape descriptors
``face``         head-rotation-free facial geometry
``head_pose``    yaw/pitch/roll — non-manual grammar
``present``      per-channel tracking flags
===============  ================================================================

Derivatives (velocity, optionally acceleration) are appended for every block except
``present``: movement is a phonemic parameter of a sign, and a model given only
static positions has to rediscover differencing from limited data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .normalise import NormalisedSequence
from .schema import HandIndex

__all__ = [
    "FeatureConfig",
    "FeatureLayout",
    "encode_sequence",
    "handshape_descriptor",
    "resample",
    "motion_energy",
]

_EPS = 1e-8

#: Finger chains as (mcp, pip, tip) for the curl angles.
_FINGER_CHAINS: tuple[tuple[int, int, int], ...] = (
    (HandIndex.THUMB_MCP, HandIndex.THUMB_IP, HandIndex.THUMB_TIP),
    (HandIndex.INDEX_MCP, HandIndex.INDEX_PIP, HandIndex.INDEX_TIP),
    (HandIndex.MIDDLE_MCP, HandIndex.MIDDLE_PIP, HandIndex.MIDDLE_TIP),
    (HandIndex.RING_MCP, HandIndex.RING_PIP, HandIndex.RING_TIP),
    (HandIndex.PINKY_MCP, HandIndex.PINKY_PIP, HandIndex.PINKY_TIP),
)

#: Width of :func:`handshape_descriptor`: 5 extensions + 4 spreads + 5 curls.
HANDSHAPE_DIM = 14


@dataclass(frozen=True)
class FeatureConfig:
    """What goes into the feature vector.

    ``include_face`` defaults to True. Turning it off is supported for ablation
    studies and for clips where the face was not tracked, but it should never be
    the default for a released model: non-manual markers carry negation, question
    and conditional marking, so a face-free model cannot represent those
    distinctions at all (plan §8.7).
    """

    include_face: bool = True
    include_velocity: bool = True
    include_acceleration: bool = False
    include_handshape: bool = True


@dataclass
class FeatureLayout:
    """Column ranges of each block in the encoded matrix."""

    blocks: dict[str, slice] = field(default_factory=dict)
    dim: int = 0

    def add(self, name: str, width: int) -> None:
        self.blocks[name] = slice(self.dim, self.dim + width)
        self.dim += width

    def select(self, features: np.ndarray, name: str) -> np.ndarray:
        """Extract one block from an encoded matrix."""
        return features[..., self.blocks[name]]

    def names(self) -> list[str]:
        return list(self.blocks)


def handshape_descriptor(local_hand: np.ndarray) -> np.ndarray:
    """Rotation-free summary of a handshape: extension, spread and curl.

    Complements the raw landmark block rather than replacing it. Raw coordinates
    carry orientation (phonemic, so it must be kept), but a small model on a small
    corpus struggles to derive "index extended, others closed" from 63 coordinates.
    These 14 numbers state it directly.
    """
    hand = np.asarray(local_hand, dtype=np.float32)
    tips = hand[list(HandIndex.FINGER_TIPS)]

    extension = np.linalg.norm(tips, axis=-1)

    spread = np.linalg.norm(tips[1:] - tips[:-1], axis=-1)

    curls = np.empty(len(_FINGER_CHAINS), dtype=np.float32)
    for i, (mcp, pip, tip) in enumerate(_FINGER_CHAINS):
        proximal = hand[pip] - hand[mcp]
        distal = hand[tip] - hand[pip]
        denom = np.linalg.norm(proximal) * np.linalg.norm(distal)
        if denom < _EPS:
            curls[i] = 0.0
        else:
            cosine = float(np.clip(np.dot(proximal, distal) / denom, -1.0, 1.0))
            curls[i] = np.arccos(cosine)

    return np.concatenate([extension, spread, curls]).astype(np.float32)


def _derivative(values: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference derivative with edge replication.

    Central rather than forward differences so a movement's velocity peak stays
    aligned with the frame it happens on; a half-frame shift matters when signs are
    only 10–20 frames long.
    """
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, 1.0 / fps, axis=0).astype(np.float32)


def encode_sequence(
    sequence: NormalisedSequence, config: FeatureConfig | None = None
) -> tuple[np.ndarray, FeatureLayout]:
    """Encode a normalised clip as ``(T, D)`` features plus its layout."""
    cfg = config or FeatureConfig()
    n = len(sequence)
    layout = FeatureLayout()
    blocks: list[np.ndarray] = []

    def add(name: str, arr: np.ndarray) -> None:
        flat = arr.reshape(n, -1).astype(np.float32)
        layout.add(name, flat.shape[1])
        blocks.append(flat)

    add("body", sequence.body)
    add("dominant_local", sequence.dominant_local)
    add("weak_local", sequence.weak_local)
    add("dominant_wrist", sequence.dominant_wrist)
    add("weak_wrist", sequence.weak_wrist)

    if cfg.include_handshape:
        for name, hand in (
            ("dominant_shape", sequence.dominant_local),
            ("weak_shape", sequence.weak_local),
        ):
            desc = np.stack([handshape_descriptor(hand[t]) for t in range(n)]) if n else np.zeros(
                (0, HANDSHAPE_DIM), dtype=np.float32
            )
            add(name, desc)

    if cfg.include_face:
        add("face", sequence.face)
        add("head_pose", sequence.head_pose)

    static = np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0), dtype=np.float32)

    derived: list[np.ndarray] = []
    if cfg.include_velocity:
        velocity = _derivative(static, sequence.fps)
        layout.add("velocity", velocity.shape[1])
        derived.append(velocity)
        if cfg.include_acceleration:
            acceleration = _derivative(velocity, sequence.fps)
            layout.add("acceleration", acceleration.shape[1])
            derived.append(acceleration)

    # Presence flags go last and are never differentiated: the derivative of a
    # tracking dropout is a meaningless spike.
    presence = sequence.present.astype(np.float32)
    layout.add("present", presence.shape[1])

    features = np.concatenate([static, *derived, presence], axis=1)
    return np.ascontiguousarray(features, dtype=np.float32), layout


def resample(features: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly resample a ``(T, D)`` sequence to a fixed frame count.

    Classifiers over fixed-size inputs need this, and it doubles as the temporal
    augmentation primitive (plan §9.4). Interpolation rather than padding, so that
    signing speed — which varies hugely between signers — does not become a feature
    the model can latch onto.
    """
    features = np.asarray(features, dtype=np.float32)
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    t = len(features)
    if t == n_frames:
        return features.copy()
    if t == 0:
        return np.zeros((n_frames, features.shape[1]), dtype=np.float32)
    if t == 1:
        return np.repeat(features, n_frames, axis=0)

    source = np.linspace(0.0, 1.0, t)
    target = np.linspace(0.0, 1.0, n_frames)
    out = np.empty((n_frames, features.shape[1]), dtype=np.float32)
    for d in range(features.shape[1]):
        out[:, d] = np.interp(target, source, features[:, d])
    return out


def motion_energy(sequence: NormalisedSequence, *, smooth: int = 3) -> np.ndarray:
    """Per-frame hand motion magnitude, used to find sign boundaries.

    Only the hands contribute. Body sway and head movement continue through the
    pauses between signs, so including them would smear exactly the minima that
    :mod:`signsync.recognition.segmentation` looks for.
    """
    n = len(sequence)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    hands = np.concatenate(
        [
            sequence.dominant_wrist,
            sequence.weak_wrist,
            sequence.dominant_local.reshape(n, -1),
            sequence.weak_local.reshape(n, -1),
        ],
        axis=1,
    )
    velocity = _derivative(hands, sequence.fps)
    energy = np.linalg.norm(velocity, axis=1).astype(np.float32)

    if smooth > 1 and n >= smooth:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        energy = np.convolve(energy, kernel, mode="same").astype(np.float32)
    return energy
