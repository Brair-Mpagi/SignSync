from __future__ import annotations

import numpy as np
import pytest

from signsync.vision.features import (
    HANDSHAPE_DIM,
    FeatureConfig,
    encode_sequence,
    handshape_descriptor,
    motion_energy,
    resample,
)
from signsync.vision.normalise import normalise_sequence
from signsync.vision.schema import N_HAND
from signsync.vision.synthetic import synthetic_sentence, synthetic_sign


def test_encoding_is_finite_and_documented(clip):
    features, layout = encode_sequence(normalise_sequence(clip))

    assert features.shape == (len(clip), layout.dim)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    for block in ("body", "dominant_local", "weak_local", "face", "head_pose", "present", "velocity"):
        assert block in layout.blocks
    assert sum(s.stop - s.start for s in layout.blocks.values()) == layout.dim


def test_layout_can_isolate_a_channel_for_error_analysis(clip):
    """Plan §15 wants per-sign confusion attributable to a channel."""
    features, layout = encode_sequence(normalise_sequence(clip))
    face = layout.select(features, "face")
    assert face.shape[0] == len(clip)
    assert face.shape[1] == layout.blocks["face"].stop - layout.blocks["face"].start


def test_face_can_be_excluded_but_is_on_by_default(clip):
    normalised = normalise_sequence(clip)
    assert FeatureConfig().include_face is True

    _, without = encode_sequence(normalised, FeatureConfig(include_face=False))
    _, with_face = encode_sequence(normalised, FeatureConfig(include_face=True))
    assert without.dim < with_face.dim
    assert "face" not in without.blocks


def test_acceleration_is_opt_in(clip):
    normalised = normalise_sequence(clip)
    _, base = encode_sequence(normalised, FeatureConfig(include_acceleration=False))
    _, accelerated = encode_sequence(normalised, FeatureConfig(include_acceleration=True))
    assert "acceleration" not in base.blocks
    assert accelerated.dim > base.dim


def test_presence_flags_are_never_differentiated(clip):
    """The derivative of a tracking dropout is a meaningless spike."""
    _, layout = encode_sequence(normalise_sequence(clip))
    assert layout.blocks["present"].start > layout.blocks["velocity"].start


def test_handshape_descriptor_separates_open_from_closed():
    open_hand = np.zeros((N_HAND, 3), dtype=np.float32)
    for i in range(1, N_HAND):
        open_hand[i] = [0.0, -0.05 * i, 0.0]
    closed_hand = np.zeros((N_HAND, 3), dtype=np.float32)

    open_desc = handshape_descriptor(open_hand)
    closed_desc = handshape_descriptor(closed_hand)

    assert open_desc.shape == (HANDSHAPE_DIM,)
    assert open_desc[:5].sum() > closed_desc[:5].sum()


def test_resample_preserves_endpoints_and_changes_length():
    values = np.linspace(0, 1, 17, dtype=np.float32)[:, None]
    out = resample(values, 40)
    assert out.shape == (40, 1)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(1.0)


def test_resample_handles_degenerate_lengths():
    single = np.ones((1, 3), dtype=np.float32)
    assert resample(single, 5).shape == (5, 3)
    assert resample(np.zeros((0, 3), dtype=np.float32), 4).shape == (4, 3)
    with pytest.raises(ValueError):
        resample(single, 0)


def test_motion_energy_dips_between_signs():
    """Plan §8.3: sign boundaries show up as troughs in hand motion."""
    sentence = synthetic_sentence(["HELLO", "HOSPITAL"], "signer-a", pause_frames=8)
    energy = motion_energy(normalise_sequence(sentence))

    assert len(energy) == len(sentence)
    boundaries = sentence.meta["boundaries"]
    gap_start, gap_end = boundaries[0][1], boundaries[1][0]
    during_gap = energy[gap_start:gap_end].mean()
    during_signs = np.concatenate([energy[: boundaries[0][1]], energy[boundaries[1][0] :]]).mean()
    assert during_gap < during_signs


def test_motion_energy_of_empty_sequence():
    from signsync.vision.schema import LandmarkSequence

    assert len(motion_energy(normalise_sequence(LandmarkSequence.empty(0)))) == 0


def test_same_gloss_encodes_more_similarly_than_different_glosses(signer):
    """The synthetic generator has to make recognition possible, or it is useless."""

    def encode(gloss: str, who) -> np.ndarray:
        features, _ = encode_sequence(normalise_sequence(synthetic_sign(gloss, who)))
        return resample(features, 32).ravel()

    from signsync.vision.synthetic import SignerStyle

    a = encode("HOSPITAL", signer)
    same_sign_other_signer = encode("HOSPITAL", SignerStyle.derived("signer-c"))
    different_sign = encode("WATER", signer)

    assert np.linalg.norm(a - same_sign_other_signer) < np.linalg.norm(a - different_sign)
