from __future__ import annotations

import numpy as np
import pytest

from signsync.vision.normalise import (
    StreamingNormaliser,
    estimate_body_frame,
    head_pose_from_pose,
    normalise_sequence,
)
from signsync.vision.schema import N_POSE, Channel, LandmarkSequence, PoseIndex
from signsync.vision.synthetic import SignerStyle, synthetic_sign


def _shift_and_scale(seq: LandmarkSequence, *, offset: float, scale: float) -> LandmarkSequence:
    """Simulate the same signing recorded from a different position and distance."""
    moved = LandmarkSequence(
        pose=seq.pose * scale + offset,
        left_hand=seq.left_hand * scale + offset,
        right_hand=seq.right_hand * scale + offset,
        face=seq.face * scale + offset,
        present=seq.present.copy(),
        timestamps=seq.timestamps.copy(),
        fps=seq.fps,
    )
    return moved


def test_normalisation_is_invariant_to_camera_position_and_distance(clip):
    """Plan §8.1: normalise for signer position and scale."""
    a = normalise_sequence(clip)
    b = normalise_sequence(_shift_and_scale(clip, offset=0.13, scale=1.7))

    np.testing.assert_allclose(a.body, b.body, atol=1e-4)
    np.testing.assert_allclose(a.right_wrist, b.right_wrist, atol=1e-4)


def test_normalisation_preserves_the_difference_between_two_signs(signer):
    """Invariance must not go so far as to erase the signal."""
    hello = normalise_sequence(synthetic_sign("HELLO", signer))
    water = normalise_sequence(synthetic_sign("WATER", signer))
    n = min(len(hello), len(water))
    assert np.abs(hello.right_wrist[:n] - water.right_wrist[:n]).max() > 0.1


def test_normalisation_reduces_between_signer_distance(signer, other_signer):
    """The point of normalisation: same sign, different signers, closer together."""
    raw_a = synthetic_sign("HOSPITAL", signer)
    raw_b = synthetic_sign("HOSPITAL", other_signer)
    n = min(len(raw_a), len(raw_b))

    raw_gap = np.abs(raw_a.pose[:n] - raw_b.pose[:n]).mean()
    norm_a = normalise_sequence(raw_a)
    norm_b = normalise_sequence(raw_b)
    norm_gap = np.abs(norm_a.body[:n] - norm_b.body[:n]).mean() * float(
        np.median([signer.shoulder_width, other_signer.shoulder_width])
    )
    assert norm_gap < raw_gap


def test_body_frame_is_none_when_shoulders_are_degenerate():
    pose = np.zeros((N_POSE, 3), dtype=np.float32)
    assert estimate_body_frame(pose) is None


def test_degenerate_shoulders_do_not_explode_the_sequence(clip):
    broken = LandmarkSequence(
        pose=np.zeros_like(clip.pose),
        left_hand=clip.left_hand.copy(),
        right_hand=clip.right_hand.copy(),
        face=clip.face.copy(),
        present=clip.present.copy(),
        timestamps=clip.timestamps.copy(),
        fps=clip.fps,
    )
    result = normalise_sequence(broken)
    assert np.isfinite(result.body).all()
    assert np.isfinite(result.right_wrist).all()


def test_absent_hand_normalises_to_zero_not_to_noise(clip):
    clip.present[:, Channel.LEFT_HAND] = False
    result = normalise_sequence(clip)
    np.testing.assert_array_equal(result.left_local, np.zeros_like(result.left_local))
    np.testing.assert_array_equal(result.left_wrist, np.zeros_like(result.left_wrist))


def test_head_pose_detects_a_tilt():
    pose = np.zeros((N_POSE, 3), dtype=np.float32)
    pose[PoseIndex.LEFT_EAR] = [0.6, 0.5, 0.0]
    pose[PoseIndex.RIGHT_EAR] = [0.4, 0.5, 0.0]
    pose[PoseIndex.LEFT_EYE] = [0.55, 0.50, 0.0]
    pose[PoseIndex.RIGHT_EYE] = [0.45, 0.55, 0.0]
    pose[PoseIndex.NOSE] = [0.5, 0.56, 0.0]

    upright = np.zeros((N_POSE, 3), dtype=np.float32)
    upright[PoseIndex.LEFT_EAR] = [0.6, 0.5, 0.0]
    upright[PoseIndex.RIGHT_EAR] = [0.4, 0.5, 0.0]
    upright[PoseIndex.LEFT_EYE] = [0.55, 0.52, 0.0]
    upright[PoseIndex.RIGHT_EYE] = [0.45, 0.52, 0.0]
    upright[PoseIndex.NOSE] = [0.5, 0.56, 0.0]

    assert abs(head_pose_from_pose(pose)[2]) > abs(head_pose_from_pose(upright)[2])


def test_head_pose_is_zero_without_a_face():
    assert head_pose_from_pose(np.zeros((N_POSE, 3), dtype=np.float32)).tolist() == [0, 0, 0]


def test_per_frame_reference_cancels_body_shift_and_sequence_reference_keeps_it():
    """Body shifting is grammar, so the default reference mode must preserve it."""
    style = SignerStyle.derived("signer-shift")
    seq = synthetic_sign("NAME", style)
    drift = np.linspace(0, 0.08, len(seq), dtype=np.float32)[:, None, None]
    shifted = LandmarkSequence(
        pose=seq.pose + drift,
        left_hand=seq.left_hand + drift,
        right_hand=seq.right_hand + drift,
        face=seq.face + drift,
        present=seq.present.copy(),
        timestamps=seq.timestamps.copy(),
        fps=seq.fps,
    )

    per_frame = normalise_sequence(shifted, reference="per_frame")
    sequence = normalise_sequence(shifted, reference="sequence")

    body_drift = lambda r: float(np.abs(r.body[-1, 0] - r.body[0, 0]).max())  # noqa: E731
    assert body_drift(sequence) > body_drift(per_frame)


def test_streaming_normaliser_converges_to_the_offline_reference(clip):
    streaming = StreamingNormaliser(momentum=0.5, fps=clip.fps)
    for i in range(len(clip)):
        streaming(clip.frame(i))

    reference = streaming.reference
    assert reference is not None
    # The synthetic signer's shoulders are a fixed width apart in image units.
    style = SignerStyle.derived("signer-a")
    assert reference.scale == pytest.approx(style.shoulder_width, abs=0.02)


def test_streaming_normaliser_starts_without_a_reference():
    assert StreamingNormaliser().reference is None


def test_streaming_normaliser_rejects_bad_momentum():
    with pytest.raises(ValueError):
        StreamingNormaliser(momentum=1.0)


def test_empty_sequence_normalises_to_empty():
    result = normalise_sequence(LandmarkSequence.empty(0))
    assert len(result) == 0
