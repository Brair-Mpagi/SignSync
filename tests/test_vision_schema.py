from __future__ import annotations

import numpy as np
import pytest

from signsync.vision.schema import (
    FACE_GROUPS,
    FACE_INDICES,
    FACE_SUBSET_INDEX,
    N_FACE,
    N_HAND,
    N_POSE,
    Channel,
    FrameLandmarks,
    LandmarkSequence,
)


def test_face_subset_is_small_and_consistent():
    """Plan §8.1: keep only the mesh points that carry non-manual grammar."""
    assert N_FACE == len(FACE_INDICES)
    assert len(set(FACE_INDICES)) == N_FACE, "duplicate face landmark index"
    assert N_FACE < 60, "face subset has grown into identity territory"
    assert max(FACE_INDICES) < 468
    assert all(FACE_SUBSET_INDEX[m] == i for i, m in enumerate(FACE_INDICES))


def test_face_groups_cover_brows_mouth_and_head():
    for required in ("left_brow", "right_brow", "outer_lips", "inner_lips"):
        assert FACE_GROUPS[required], f"{required} must not be empty"


def test_frame_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="left_hand"):
        FrameLandmarks(left_hand=np.zeros((5, 3), dtype=np.float32))


def test_sequence_roundtrip(tmp_path, clip):
    path = clip.save(tmp_path / "clip.npz")
    restored = LandmarkSequence.load(path)

    assert len(restored) == len(clip)
    assert restored.fps == clip.fps
    assert restored.meta["gloss"] == "HELLO"
    np.testing.assert_allclose(restored.pose, clip.pose)
    np.testing.assert_array_equal(restored.present, clip.present)


def test_empty_sequence_has_consistent_shapes():
    seq = LandmarkSequence.empty(0)
    assert len(seq) == 0
    assert seq.pose.shape == (0, N_POSE, 3)
    assert seq.left_hand.shape == (0, N_HAND, 3)
    assert seq.coverage() == dict.fromkeys(Channel.NAMES, 0.0)


def test_slice_keeps_schema(clip):
    part = clip.slice(2, 7)
    assert len(part) == 5
    assert part.fps == clip.fps
    np.testing.assert_allclose(part.pose[0], clip.pose[2])


def test_slice_clamps_out_of_range(clip):
    assert len(clip.slice(-5, 10_000)) == len(clip)
    with pytest.raises(ValueError):
        clip.slice(5, 2)


def test_coverage_reports_tracking_dropouts(clip):
    coverage = clip.coverage()
    assert coverage["pose"] == 1.0
    assert 0.0 <= coverage["left_hand"] <= 1.0
