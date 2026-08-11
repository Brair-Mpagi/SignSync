"""Handedness canonicalisation (plan §8.1, §9.3).

A left-handed signer produces the mirror image of the same sign, not a different
sign. If normalisation does not undo that, every model has to learn each sign
twice, and any model that summarises a class with one template cannot represent it
at all — which is how a system ends up failing completely on roughly a tenth of the
signing population.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from signsync.vision.features import encode_sequence, resample
from signsync.vision.normalise import detect_dominant_hand, normalise_sequence
from signsync.vision.schema import (
    FACE_INDICES,
    FACE_MIRROR_PERM,
    N_FACE,
    UPPER_BODY_MIRROR_PERM,
    UPPER_BODY_POSE,
    PoseIndex,
    mirror_permutation,
)
from signsync.vision.synthetic import SignerStyle, synthetic_sign

GLOSS = "HOSPITAL"


def encode(style: SignerStyle, dominant: str) -> np.ndarray:
    normalised = normalise_sequence(synthetic_sign(GLOSS, style), dominant=dominant)
    features, _ = encode_sequence(normalised)
    return resample(features, 32)


@pytest.fixture
def pair():
    right = replace(SignerStyle.derived("signer-a"), noise=0.0, dropout=0.0, left_handed=False)
    return right, replace(right, left_handed=True)


# --------------------------------------------------------------------------- permutations


def test_mirror_permutations_are_involutions():
    for permutation in (FACE_MIRROR_PERM, UPPER_BODY_MIRROR_PERM):
        indices = np.array(permutation)
        assert sorted(indices) == list(range(len(indices))), "not a permutation"
        assert (indices[indices] == np.arange(len(indices))).all(), "mirroring twice must undo"


def test_face_permutation_swaps_the_brows():
    left_brow = FACE_INDICES.index(70)
    right_brow = FACE_INDICES.index(336)
    assert FACE_MIRROR_PERM[left_brow] == right_brow
    assert FACE_MIRROR_PERM[right_brow] == left_brow


def test_midline_landmarks_map_to_themselves():
    chin = FACE_INDICES.index(152)
    assert FACE_MIRROR_PERM[chin] == chin
    nose = UPPER_BODY_POSE.index(PoseIndex.NOSE)
    assert UPPER_BODY_MIRROR_PERM[nose] == nose


def test_pose_permutation_swaps_the_shoulders():
    left = UPPER_BODY_POSE.index(PoseIndex.LEFT_SHOULDER)
    right = UPPER_BODY_POSE.index(PoseIndex.RIGHT_SHOULDER)
    assert UPPER_BODY_MIRROR_PERM[left] == right


def test_permutation_tolerates_a_partner_that_was_filtered_out():
    permutation = mirror_permutation((PoseIndex.LEFT_SHOULDER,), ((PoseIndex.LEFT_SHOULDER, PoseIndex.RIGHT_SHOULDER),))
    assert permutation == (0,), "an unpaired landmark must map to itself"


# --------------------------------------------------------------------------- canonicalisation


def test_canonicalisation_makes_a_left_handed_signer_match_a_right_handed_one(pair):
    """The whole point: the same sign, from either hand, encodes the same way."""
    right, left = pair
    canonical = np.linalg.norm(encode(right, "right") - encode(left, "left"))
    uncorrected = np.linalg.norm(encode(right, "right") - encode(left, "right"))

    assert canonical < uncorrected / 2, (
        f"canonicalisation did not recover the sign (canonical={canonical:.2f}, "
        f"uncorrected={uncorrected:.2f})"
    )


def test_canonicalisation_preserves_the_distinction_between_signs(pair):
    """Invariance must not collapse different signs onto each other."""
    right, left = pair
    same_sign = np.linalg.norm(encode(right, "right") - encode(left, "left"))

    other, _ = encode_sequence(normalise_sequence(synthetic_sign("WATER", right), dominant="right"))
    different_sign = np.linalg.norm(encode(right, "right") - resample(other, 32))

    assert same_sign < different_sign


def test_mirroring_is_recorded_on_the_result(pair):
    right, left = pair
    assert normalise_sequence(synthetic_sign(GLOSS, left), dominant="left").mirrored is True
    assert normalise_sequence(synthetic_sign(GLOSS, right), dominant="right").mirrored is False


def test_canonicalisation_can_be_switched_off_for_handedness_research(pair):
    _, left = pair
    result = normalise_sequence(
        synthetic_sign(GLOSS, left), dominant="left", canonicalise_handedness=False
    )
    assert result.mirrored is False
    assert result.dominant == "left"


def test_detection_finds_the_moving_hand_in_a_one_handed_sign():
    """Detection is reliable when one hand is idle; two-handed signs are the hard case."""
    one_handed = next(
        g
        for g in ("HELLO", "HELP", "WATER", "NAME", "SCHOOL", "DOCTOR")
        if not synthetic_sign(g, "signer-a").present[:, 1].all()
    )
    right = replace(SignerStyle.derived("signer-a"), left_handed=False)
    assert detect_dominant_hand(synthetic_sign(one_handed, right)) == "right"
    assert detect_dominant_hand(synthetic_sign(one_handed, replace(right, left_handed=True))) == "left"


def test_detection_defaults_to_right_without_evidence():
    from signsync.vision.schema import LandmarkSequence

    assert detect_dominant_hand(LandmarkSequence.empty(0)) == "right"


def test_normalise_rejects_an_unknown_handedness(pair):
    right, _ = pair
    with pytest.raises(ValueError, match="dominant must be"):
        normalise_sequence(synthetic_sign(GLOSS, right), dominant="either")


def test_face_block_is_not_scrambled_by_mirroring(pair):
    """The bug this guards: a mirrored face whose brow points hold the other brow."""
    right, left = pair
    a = normalise_sequence(synthetic_sign(GLOSS, right), dominant="right")
    b = normalise_sequence(synthetic_sign(GLOSS, left), dominant="left")
    assert a.face.shape[1] == N_FACE
    np.testing.assert_allclose(a.face, b.face, atol=1e-4)
