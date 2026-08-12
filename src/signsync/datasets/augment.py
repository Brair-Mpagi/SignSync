"""Augmentation, applied conservatively (plan §9.4).

Augmentation for sign language is not the same problem as augmentation for object
recognition, because the transformations that leave a photo's label unchanged do
not leave a sign's meaning unchanged. Direction, orientation and rhythm are
phonemic. So:

* **Bounds are tight and enforced.** Rotation is limited to a few degrees, temporal
  rescaling to the range real signers actually vary over. Values outside the bounds
  raise rather than clamp, because a silently clamped augmentation is a config bug
  that never surfaces.
* **Mirroring is not in the default policy.** Mirroring swaps the dominant hand,
  which is a legitimate way to model left-handed signers, but it also reverses
  every directional sign — "I-give-you" becomes "you-give-me". It is available only
  through an explicit opt-in with that trade-off named in the signature.
* **Augmentation runs on raw image-space landmarks**, before normalisation, so it
  simulates recording variation (camera angle, distance, framing) rather than
  perturbing an already-canonicalised representation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..errors import SignSyncError
from ..vision.schema import Channel, LandmarkSequence

__all__ = ["AugmentationPolicy", "augment", "mirror_handedness", "temporal_rescale"]

_MAX_ROTATION_DEG = 12.0
_MIN_SPEED, _MAX_SPEED = 0.6, 1.6


@dataclass(frozen=True)
class AugmentationPolicy:
    """Bounds for one augmentation pass. Zero disables an axis."""

    translation: float = 0.04
    """Fraction of the frame the whole signer may be shifted by."""

    scale: float = 0.12
    """Relative zoom, standing in for camera distance."""

    rotation_deg: float = 6.0
    """In-plane rotation, standing in for a tilted camera."""

    speed: float = 0.15
    """Relative temporal rescaling, standing in for signing tempo."""

    landmark_noise: float = 0.003
    """Per-landmark jitter, standing in for tracker imprecision."""

    frame_dropout: float = 0.02
    """Probability of marking a frame's hands untracked, standing in for occlusion."""

    def __post_init__(self) -> None:
        if self.rotation_deg > _MAX_ROTATION_DEG:
            raise SignSyncError(
                f"rotation_deg={self.rotation_deg} exceeds {_MAX_ROTATION_DEG}°; large rotations "
                "change the spatial relationships that carry meaning in signing (plan §9.4)"
            )
        for name in ("translation", "scale", "speed", "landmark_noise", "frame_dropout"):
            value = getattr(self, name)
            if value < 0:
                raise SignSyncError(f"{name} must be non-negative, got {value}")
        if not 1 - self.speed >= _MIN_SPEED - 1e-9 or not 1 + self.speed <= _MAX_SPEED + 1e-9:
            raise SignSyncError(
                f"speed={self.speed} would rescale outside [{_MIN_SPEED}, {_MAX_SPEED}]×; "
                "extreme tempo changes destroy the rhythm distinctions between signs"
            )
        if not 0.0 <= self.frame_dropout <= 0.5:
            raise SignSyncError(f"frame_dropout must be in [0, 0.5], got {self.frame_dropout}")

    @classmethod
    def none(cls) -> AugmentationPolicy:
        """A no-op policy, for evaluation runs."""
        return cls(
            translation=0.0,
            scale=0.0,
            rotation_deg=0.0,
            speed=0.0,
            landmark_noise=0.0,
            frame_dropout=0.0,
        )


def temporal_rescale(sequence: LandmarkSequence, factor: float) -> LandmarkSequence:
    """Resample a clip to ``factor`` times its length (``>1`` = slower signing)."""
    if not _MIN_SPEED <= factor <= _MAX_SPEED:
        raise SignSyncError(
            f"temporal factor {factor} outside [{_MIN_SPEED}, {_MAX_SPEED}]; see plan §9.4"
        )
    n = len(sequence)
    target = max(2, int(round(n * factor)))
    if n < 2 or target == n:
        return sequence

    source_grid = np.linspace(0.0, 1.0, n)
    target_grid = np.linspace(0.0, 1.0, target)

    def interp(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(n, -1)
        out = np.empty((target, flat.shape[1]), dtype=np.float32)
        for d in range(flat.shape[1]):
            out[:, d] = np.interp(target_grid, source_grid, flat[:, d])
        return out.reshape((target, *values.shape[1:])).astype(np.float32)

    # Presence is boolean: nearest-neighbour, since an interpolated half-tracked
    # frame is not a state the tracker can ever produce.
    indices = np.clip(np.round(target_grid * (n - 1)).astype(int), 0, n - 1)

    return LandmarkSequence(
        pose=interp(sequence.pose),
        left_hand=interp(sequence.left_hand),
        right_hand=interp(sequence.right_hand),
        face=interp(sequence.face),
        present=sequence.present[indices],
        timestamps=(np.arange(target, dtype=np.float32) / sequence.fps),
        fps=sequence.fps,
        meta={**sequence.meta, "augment_speed": factor},
    )


def mirror_handedness(
    sequence: LandmarkSequence, *, i_accept_direction_reversal: bool
) -> LandmarkSequence:
    """Mirror a clip left-to-right, swapping the hands.

    Only valid for signs whose meaning does not depend on direction. Mirroring
    reverses spatial agreement — a verb directed from signer to addressee becomes
    one directed from addressee to signer, which is a different sentence. The
    keyword argument exists so that using this is a deliberate, greppable decision
    rather than a checkbox in a config file.
    """
    if not i_accept_direction_reversal:
        raise SignSyncError(
            "mirroring reverses directional signs and swaps the dominant hand; pass "
            "i_accept_direction_reversal=True only for a vocabulary verified to be "
            "direction-independent (plan §9.4)"
        )

    def flip(points: np.ndarray) -> np.ndarray:
        out = points.copy()
        out[..., 0] = 1.0 - out[..., 0]
        return out.astype(np.float32)

    present = sequence.present.copy()
    present[:, [Channel.LEFT_HAND, Channel.RIGHT_HAND]] = present[
        :, [Channel.RIGHT_HAND, Channel.LEFT_HAND]
    ]
    return LandmarkSequence(
        pose=flip(sequence.pose),
        left_hand=flip(sequence.right_hand),
        right_hand=flip(sequence.left_hand),
        face=flip(sequence.face),
        present=present,
        timestamps=sequence.timestamps.copy(),
        fps=sequence.fps,
        meta={**sequence.meta, "augment_mirrored": True},
    )


def augment(
    sequence: LandmarkSequence,
    policy: AugmentationPolicy | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> LandmarkSequence:
    """Apply one random augmentation pass within the policy's bounds."""
    policy = policy or AugmentationPolicy()
    rng = rng or np.random.default_rng()
    result = sequence

    if policy.speed > 0:
        factor = float(rng.uniform(1 - policy.speed, 1 + policy.speed))
        result = temporal_rescale(result, factor)

    n = len(result)
    if n == 0:
        return result

    centre = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    scale = 1.0 + float(rng.uniform(-policy.scale, policy.scale)) if policy.scale else 1.0
    angle = np.deg2rad(float(rng.uniform(-policy.rotation_deg, policy.rotation_deg)))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    shift = (
        np.array(
            [*rng.uniform(-policy.translation, policy.translation, size=2), 0.0],
            dtype=np.float32,
        )
        if policy.translation
        else np.zeros(3, dtype=np.float32)
    )

    def transform(points: np.ndarray) -> np.ndarray:
        moved = (points - centre) * scale
        moved = moved @ rotation.T
        moved = moved + centre + shift
        if policy.landmark_noise:
            moved = moved + rng.normal(0.0, policy.landmark_noise, moved.shape)
        return moved.astype(np.float32)

    present = result.present.copy()
    if policy.frame_dropout:
        drops = rng.random(n) < policy.frame_dropout
        present[drops, Channel.LEFT_HAND] = False
        present[drops, Channel.RIGHT_HAND] = False

    return replace(
        result,
        pose=transform(result.pose),
        left_hand=transform(result.left_hand),
        right_hand=transform(result.right_hand),
        face=transform(result.face),
        present=present,
        meta={**result.meta, "augmented": True},
    )
