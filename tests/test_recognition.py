from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from signsync.datasets.corpus import CorpusLoader
from signsync.datasets.synthetic import SyntheticCorpusSpec, build_synthetic_corpus
from signsync.errors import SignSyncError
from signsync.recognition.base import UNKNOWN_GLOSS, RecogniserConfig, Vocabulary, top_k
from signsync.recognition.prototype import PrototypeRecogniser
from signsync.recognition.train import (
    TrainingRun,
    confusion_pairs,
    evaluate,
    train_from_corpus,
)
from signsync.vision.features import encode_sequence
from signsync.vision.normalise import normalise_sequence
from signsync.vision.synthetic import SignerStyle, synthetic_sign

TODAY = date(2026, 8, 10)
VOCAB = ("HELLO", "HELP", "HOSPITAL", "WATER", "NAME", "SCHOOL")


def features_for(gloss: str, signer: str) -> np.ndarray:
    """Encode a clip the way the training pipeline does: handedness from the signer."""
    style = SignerStyle.derived(signer)
    dominant = "left" if style.left_handed else "right"
    normalised = normalise_sequence(synthetic_sign(gloss, style), dominant=dominant)
    encoded, _ = encode_sequence(normalised)
    return encoded


# --------------------------------------------------------------------------- vocabulary


def test_vocabulary_roundtrip(tmp_path):
    vocab = Vocabulary(VOCAB)
    assert vocab.decode(vocab.encode(["HELP", "WATER"])) == ["HELP", "WATER"]
    assert Vocabulary.load(vocab.save(tmp_path / "v.json")).glosses == VOCAB


def test_vocabulary_rejects_duplicates_and_emptiness():
    with pytest.raises(SignSyncError, match="duplicate"):
        Vocabulary(("A", "A"))
    with pytest.raises(SignSyncError, match="empty"):
        Vocabulary(())


def test_vocabulary_errors_are_specific():
    vocab = Vocabulary(VOCAB)
    with pytest.raises(SignSyncError, match="not in the vocabulary"):
        vocab.index("MISSING")
    with pytest.raises(SignSyncError, match="outside vocabulary"):
        vocab.gloss(99)


def test_top_k_is_ordered():
    probabilities = np.array([0.1, 0.6, 0.3])
    assert top_k(probabilities, Vocabulary(("A", "B", "C")), 2) == (("B", 0.6), ("C", 0.3))


# --------------------------------------------------------------------------- prototype


@pytest.fixture(scope="module")
def fitted():
    train_signers = ["signer-a", "signer-b", "signer-c", "signer-d"]
    sequences, labels = [], []
    for signer in train_signers:
        for gloss in VOCAB:
            sequences.append(features_for(gloss, signer))
            labels.append(gloss)
    return PrototypeRecogniser(RecogniserConfig(min_confidence=0.3)).fit(sequences, labels)


def test_prototype_generalises_to_an_unseen_signer(fitted):
    """The only accuracy number that means anything (plan §8.3, §14)."""
    correct = sum(fitted.predict(features_for(g, "signer-held-out")).gloss == g for g in VOCAB)
    assert correct >= len(VOCAB) - 1, "prototype failed on a signer it had not seen"


def test_probabilities_are_a_distribution(fitted):
    probabilities = fitted.predict_proba(features_for("HELP", "signer-a"))
    assert probabilities.shape == (len(VOCAB),)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)
    assert (probabilities >= 0).all()


def test_confidence_is_not_saturated(fitted):
    """A recogniser that reports 100% on everything has no usable confidence gate."""
    confidences = [fitted.predict(features_for(g, "signer-a")).confidence for g in VOCAB]
    assert max(confidences) < 0.9999


def test_low_confidence_yields_unknown_not_a_guess(fitted):
    """Plan §16.3: be transparent about limits rather than confidently wrong."""
    strict = PrototypeRecogniser(RecogniserConfig(min_confidence=0.999))
    strict.fit([features_for(g, "signer-a") for g in VOCAB], list(VOCAB))
    prediction = strict.predict(features_for("HELP", "signer-z"))
    assert prediction.gloss == UNKNOWN_GLOSS
    assert prediction.is_unknown
    assert prediction.alternatives, "alternatives must survive an abstention"


def test_alternatives_are_carried_for_downstream_disambiguation(fitted):
    prediction = fitted.predict(features_for("WATER", "signer-a"))
    assert len(prediction.alternatives) == 3
    assert prediction.alternatives[0][1] >= prediction.alternatives[1][1]


def test_unfitted_recogniser_says_so():
    with pytest.raises(SignSyncError, match="not been fitted"):
        PrototypeRecogniser().predict(np.zeros((10, 5), dtype=np.float32))
    with pytest.raises(SignSyncError, match="not been fitted"):
        _ = PrototypeRecogniser().vocabulary


def test_fit_rejects_a_gloss_with_no_examples():
    with pytest.raises(SignSyncError, match="no training examples"):
        PrototypeRecogniser().fit(
            [features_for("HELP", "signer-a")], ["HELP"], Vocabulary(("HELP", "GHOST"))
        )


def test_fit_rejects_mismatched_inputs():
    with pytest.raises(SignSyncError, match="labels"):
        PrototypeRecogniser().fit([features_for("HELP", "signer-a")], ["HELP", "EXTRA"])
    with pytest.raises(SignSyncError, match="empty training set"):
        PrototypeRecogniser().fit([], [])


def test_prototype_roundtrips_through_disk(fitted, tmp_path):
    path = fitted.save(tmp_path / "model.npz")
    restored = PrototypeRecogniser.load(path)

    sample = features_for("HOSPITAL", "signer-new")
    np.testing.assert_allclose(
        restored.predict_proba(sample), fitted.predict_proba(sample), atol=1e-5
    )
    assert restored.vocabulary.glosses == fitted.vocabulary.glosses


def test_saving_an_unfitted_model_is_refused(tmp_path):
    with pytest.raises(SignSyncError, match="unfitted"):
        PrototypeRecogniser().save(tmp_path / "x.npz")


def test_variable_length_clips_are_accepted(fitted):
    short = features_for("HELP", "signer-a")[:5]
    assert fitted.predict(short).gloss in {*VOCAB, UNKNOWN_GLOSS}


def test_wrong_shape_is_rejected(fitted):
    with pytest.raises(SignSyncError, match=r"\(T, D\)"):
        fitted.predict(np.zeros((10,), dtype=np.float32))


# --------------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def corpus_loader(tmp_path_factory):
    root = tmp_path_factory.mktemp("train-corpus")
    corpus, registry = build_synthetic_corpus(
        root,
        SyntheticCorpusSpec(n_signers=8, vocabulary=VOCAB, repeats_per_gloss=2, sentences=()),
        today=TODAY,
    )
    return CorpusLoader(corpus, registry, on=TODAY)


def test_training_from_a_corpus_reports_held_out_accuracy(corpus_loader):
    result = train_from_corpus(
        corpus_loader, run=TrainingRun(backend="prototype", augmentations=1, seed=1)
    )

    assert result.split is not None
    assert set(result.split.signers["train"]) & set(result.split.signers["test"]) == set()
    assert result.test_accuracy is not None
    assert result.train_accuracy > 0.9
    assert result.test_accuracy > 0.8, "prototype should generalise on unseen signers"
    assert "test acc" in result.summary()


def test_confidences_are_calibrated_on_held_out_signers(corpus_loader):
    """An uncalibrated gate abstains on everything or on nothing (plan §16.3)."""
    from signsync.recognition.dataset import feature_sets_for_split

    result = train_from_corpus(corpus_loader, run=TrainingRun(augmentations=0, seed=1))
    assert result.report["calibrated"] is True
    assert 0.05 < result.report["temperature"] < 20.0, "temperature ran to a grid bound"

    sets = feature_sets_for_split(corpus_loader, result.split, augmentations=0)
    confidences = [result.recogniser.predict(s).confidence for s in sets["test"].sequences]
    assert min(confidences) > 0.4, "correct predictions must clear the abstention gate"
    assert min(confidences) < 0.99, "confidence must still vary, or it says nothing"


def test_calibration_needs_a_fitted_model_and_data(fitted):
    with pytest.raises(SignSyncError, match="not been fitted"):
        PrototypeRecogniser().calibrate([], [])
    with pytest.raises(SignSyncError, match="empty set"):
        fitted.calibrate([], [])


def test_temperature_fitting_does_not_run_away_on_a_perfect_set():
    """Hard targets would drive T to zero whenever validation has no errors."""
    from signsync.recognition.base import fit_temperature

    scores = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    labels = np.array([0, 1, 2])
    temperature = fit_temperature(scores, labels)
    assert temperature > 0.05, "temperature collapsed to the grid floor"

    scaled = np.exp(scores[0] / temperature)
    assert (scaled / scaled.sum()).max() < 0.999


def test_temperature_fitting_validates_its_inputs():
    from signsync.recognition.base import fit_temperature

    with pytest.raises(SignSyncError, match="n_classes"):
        fit_temperature(np.zeros((3, 2)), np.zeros(5, dtype=np.int64))
    with pytest.raises(SignSyncError, match="smoothing"):
        fit_temperature(np.zeros((2, 2)), np.zeros(2, dtype=np.int64), smoothing=1.5)
    assert fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=np.int64)) == 1.0


def test_training_never_sees_withdrawn_signers(corpus_loader):
    result = train_from_corpus(corpus_loader, run=TrainingRun(augmentations=0))
    used = {s for side in result.report["signers"].values() for s in side}
    assert "signer-04" not in used  # withdrawn
    assert "signer-05" not in used  # retention lapsed


def test_augmentation_applies_to_training_only(corpus_loader):
    from signsync.datasets.splits import signer_independent_split
    from signsync.recognition.dataset import feature_sets_for_split

    split = signer_independent_split(
        corpus_loader.corpus, records=corpus_loader.permitted_clips(), seed=0
    )
    sets = feature_sets_for_split(corpus_loader, split, augmentations=3)

    assert len(sets["train"]) == 4 * len(split.train)
    assert len(sets["test"]) == len(split.test)


def test_vocabulary_comes_from_the_training_side(corpus_loader):
    result = train_from_corpus(corpus_loader, run=TrainingRun(augmentations=0))
    assert set(result.vocabulary.glosses) <= set(VOCAB)


def test_evaluate_rejects_an_empty_set(corpus_loader):
    from signsync.recognition.dataset import FeatureSet

    empty = FeatureSet([], [], [], [], feature_dim=0)
    with pytest.raises(SignSyncError, match="empty feature set"):
        evaluate(PrototypeRecogniser(), empty)


def test_confusion_pairs_names_specific_sign_pairs(fitted):
    """Plan §15 wants systematically confused pairs, not just an accuracy number."""
    from signsync.recognition.dataset import FeatureSet

    sequences = [features_for(g, "signer-x") for g in VOCAB]
    feature_set = FeatureSet(
        sequences=sequences,
        labels=list(VOCAB),
        clip_ids=list(VOCAB),
        signers=["signer-x"] * len(VOCAB),
        feature_dim=sequences[0].shape[1],
    )
    pairs = confusion_pairs(fitted, feature_set)
    assert all(len(p) == 3 and p[0] != p[1] for p in pairs)
