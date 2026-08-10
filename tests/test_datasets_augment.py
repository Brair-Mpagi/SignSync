from __future__ import annotations

import numpy as np
import pytest

from signsync.datasets.augment import (
    AugmentationPolicy,
    augment,
    mirror_handedness,
    temporal_rescale,
)
from signsync.errors import SignSyncError
from signsync.vision.features import encode_sequence, resample
from signsync.vision.normalise import normalise_sequence
from signsync.vision.schema import Channel


def encoded(sequence, n=32) -> np.ndarray:
    features, _ = encode_sequence(normalise_sequence(sequence))
    return resample(features, n)


def test_augmentation_preserves_the_sign_after_normalisation(clip):
    """Augmentation simulates recording variation; normalisation should undo it."""
    rng = np.random.default_rng(0)
    augmented = augment(clip, AugmentationPolicy(landmark_noise=0.0, frame_dropout=0.0), rng=rng)

    baseline = encoded(clip)
    varied = encoded(augmented)
    from signsync.vision.synthetic import synthetic_sign

    different_sign = encoded(synthetic_sign("WATER", "signer-a"))

    assert np.linalg.norm(baseline - varied) < np.linalg.norm(baseline - different_sign)


def test_augmentation_actually_changes_the_raw_landmarks(clip):
    augmented = augment(clip, rng=np.random.default_rng(1))
    assert not np.allclose(augmented.pose[: len(clip)], clip.pose[: len(augmented)])
    assert augmented.meta["augmented"] is True


def test_no_op_policy_leaves_geometry_alone(clip):
    unchanged = augment(clip, AugmentationPolicy.none(), rng=np.random.default_rng(2))
    np.testing.assert_allclose(unchanged.pose, clip.pose, atol=1e-6)


def test_rotation_bound_is_enforced_not_clamped():
    """A silently clamped augmentation is a config bug that never surfaces."""
    with pytest.raises(SignSyncError, match="change the spatial relationships"):
        AugmentationPolicy(rotation_deg=45.0)


def test_extreme_tempo_is_rejected():
    with pytest.raises(SignSyncError, match="rhythm"):
        AugmentationPolicy(speed=0.9)


def test_negative_bounds_are_rejected():
    with pytest.raises(SignSyncError, match="non-negative"):
        AugmentationPolicy(translation=-0.1)


def test_temporal_rescale_changes_length_but_not_frame_rate(clip):
    slower = temporal_rescale(clip, 1.4)
    assert len(slower) > len(clip)
    assert slower.fps == clip.fps
    assert slower.meta["augment_speed"] == pytest.approx(1.4)


def test_temporal_rescale_refuses_meaning_changing_factors(clip):
    with pytest.raises(SignSyncError, match="plan §9.4"):
        temporal_rescale(clip, 3.0)


def test_temporal_rescale_keeps_presence_boolean(clip):
    rescaled = temporal_rescale(clip, 1.3)
    assert rescaled.present.dtype == bool


def test_mirroring_requires_an_explicit_acknowledgement(clip):
    with pytest.raises(SignSyncError, match="direction-independent"):
        mirror_handedness(clip, i_accept_direction_reversal=False)


def test_mirroring_swaps_the_hands(clip):
    mirrored = mirror_handedness(clip, i_accept_direction_reversal=True)
    np.testing.assert_allclose(mirrored.left_hand[..., 1], clip.right_hand[..., 1])
    np.testing.assert_array_equal(
        mirrored.present[:, Channel.LEFT_HAND], clip.present[:, Channel.RIGHT_HAND]
    )
    assert mirrored.meta["augment_mirrored"] is True


def test_frame_dropout_marks_hands_untracked(clip):
    heavy = augment(clip, AugmentationPolicy(frame_dropout=0.5), rng=np.random.default_rng(3))
    assert heavy.present[:, Channel.RIGHT_HAND].mean() < clip.present[:, Channel.RIGHT_HAND].mean()


def test_augmentation_is_reproducible_for_a_seed(clip):
    a = augment(clip, rng=np.random.default_rng(11))
    b = augment(clip, rng=np.random.default_rng(11))
    np.testing.assert_allclose(a.pose, b.pose)
