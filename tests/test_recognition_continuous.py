from __future__ import annotations

import numpy as np
import pytest

from signsync.errors import SignSyncError
from signsync.recognition.base import RecogniserConfig
from signsync.recognition.infer import StreamingConfig, StreamingRecogniser
from signsync.recognition.prototype import PrototypeRecogniser
from signsync.recognition.segmentation import (
    ContinuousRecogniser,
    SegmentationConfig,
    segment_motion,
)
from signsync.vision.features import encode_sequence
from signsync.vision.normalise import normalise_sequence
from signsync.vision.schema import LandmarkSequence
from signsync.vision.synthetic import synthetic_sentence, synthetic_sign

GLOSSES = ("HELLO", "HOSPITAL", "WATER", "HELP")


@pytest.fixture(scope="module")
def recogniser():
    sequences, labels = [], []
    for signer in ("signer-a", "signer-b", "signer-c"):
        for gloss in GLOSSES:
            encoded, _ = encode_sequence(normalise_sequence(synthetic_sign(gloss, signer)))
            sequences.append(encoded)
            labels.append(gloss)
    return PrototypeRecogniser(RecogniserConfig(min_confidence=0.3)).fit(sequences, labels)


# --------------------------------------------------------------------------- segmentation


def test_segmenter_finds_roughly_one_span_per_sign():
    sentence = synthetic_sentence(["HELLO", "HOSPITAL", "WATER"], "signer-a", pause_frames=10)
    segments = segment_motion(normalise_sequence(sentence))
    assert 2 <= len(segments) <= 5, f"expected ~3 spans, got {len(segments)}"


def test_segments_are_ordered_and_disjoint():
    sentence = synthetic_sentence(["HELLO", "HELP", "WATER"], "signer-b", pause_frames=8)
    segments = segment_motion(normalise_sequence(sentence))
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert earlier.end_frame <= later.start_frame


def test_segment_reports_frames_and_seconds():
    sentence = synthetic_sentence(["HELLO", "WATER"], "signer-a")
    segment = segment_motion(normalise_sequence(sentence))[0]
    assert segment.duration == pytest.approx(segment.n_frames / sentence.fps)
    assert segment.start == pytest.approx(segment.start_frame / sentence.fps)


def test_duration_prior_splits_an_over_long_span():
    """A span this long is usually two signs joined by a hold."""
    sentence = synthetic_sentence(["HELLO", "HOSPITAL", "WATER"], "signer-a", pause_frames=0)
    normalised = normalise_sequence(sentence)

    permissive = segment_motion(normalised, SegmentationConfig(max_duration=30.0))
    strict = segment_motion(normalised, SegmentationConfig(max_duration=0.4))
    assert len(strict) > len(permissive)


def test_short_bursts_are_treated_as_transitions():
    sentence = synthetic_sentence(["HELLO", "WATER"], "signer-a", pause_frames=6)
    normalised = normalise_sequence(sentence)
    config = SegmentationConfig(min_duration=5.0, max_duration=10.0)
    assert len(segment_motion(normalised, config)) == 0


def test_empty_sequence_segments_to_nothing():
    assert segment_motion(normalise_sequence(LandmarkSequence.empty(0))) == []


def test_segmentation_config_validates_its_prior():
    with pytest.raises(SignSyncError, match="max_duration"):
        SegmentationConfig(min_duration=2.0, max_duration=1.0)
    with pytest.raises(SignSyncError, match="energy_quantile"):
        SegmentationConfig(energy_quantile=1.5)


# --------------------------------------------------------------------------- continuous


def test_continuous_recogniser_returns_timed_glosses(recogniser):
    sentence = synthetic_sentence(["HELLO", "HOSPITAL"], "signer-d", pause_frames=10)
    normalised = normalise_sequence(sentence)
    features, _ = encode_sequence(normalised)

    predictions = ContinuousRecogniser(recogniser).recognise(normalised, features)

    assert predictions
    assert all(p.end > p.start for p in predictions)
    assert all(p.gloss in {*GLOSSES, "<unknown>"} for p in predictions)
    for earlier, later in zip(predictions, predictions[1:], strict=False):
        assert earlier.start <= later.start


def test_continuous_recogniser_can_drop_abstentions(recogniser):
    sentence = synthetic_sentence(["HELLO", "WATER"], "signer-e", pause_frames=10)
    normalised = normalise_sequence(sentence)
    features, _ = encode_sequence(normalised)

    strict = PrototypeRecogniser(RecogniserConfig(min_confidence=0.999))
    strict.fit(
        [
            encode_sequence(normalise_sequence(synthetic_sign(g, "signer-a")))[0]
            for g in GLOSSES
        ],
        list(GLOSSES),
    )
    assert ContinuousRecogniser(strict, drop_unknown=True).recognise(normalised, features) == []


def test_continuous_recogniser_rejects_mismatched_features(recogniser):
    sentence = synthetic_sentence(["HELLO"], "signer-a")
    normalised = normalise_sequence(sentence)
    with pytest.raises(SignSyncError, match="frames"):
        ContinuousRecogniser(recogniser).recognise(normalised, np.zeros((3, 10), dtype=np.float32))


# --------------------------------------------------------------------------- streaming


def _stream(recogniser, sequence, config=None):
    streaming = StreamingRecogniser(recogniser, config or StreamingConfig(fps=sequence.fps))
    emitted = []
    # Lead-in of still frames so the detector can learn the rest energy level.
    for _ in range(20):
        streaming.push(sequence.frame(0))
    for i in range(len(sequence)):
        prediction = streaming.push(sequence.frame(i))
        if prediction is not None:
            emitted.append(prediction)
    final = streaming.flush()
    if final is not None:
        emitted.append(final)
    return emitted


def test_streaming_emits_nothing_while_calibrating(recogniser):
    clip = synthetic_sign("HELLO", "signer-a")
    streaming = StreamingRecogniser(recogniser, StreamingConfig(fps=clip.fps))
    assert streaming.rest_level is None
    for _ in range(5):
        assert streaming.push(clip.frame(0)) is None
    assert streaming.rest_level is None, "calibration should not finish this early"


def test_streaming_detects_a_sign_from_a_frame_stream(recogniser):
    clip = synthetic_sign("HOSPITAL", "signer-f")
    emitted = _stream(recogniser, clip)
    assert emitted, "no sign detected in a stream containing one"
    assert emitted[0].end > emitted[0].start


def test_streaming_reports_signing_state(recogniser):
    clip = synthetic_sign("WATER", "signer-a")
    streaming = StreamingRecogniser(recogniser, StreamingConfig(fps=clip.fps))
    for _ in range(20):
        streaming.push(clip.frame(0))
    assert not streaming.is_signing
    for i in range(len(clip)):
        streaming.push(clip.frame(i))
    assert streaming.is_signing or streaming.rest_level is not None


def test_streaming_debounces_a_repeated_gloss(recogniser):
    clip = synthetic_sign("HELP", "signer-a")
    doubled = synthetic_sentence(["HELP", "HELP"], "signer-a", pause_frames=4)
    emitted = _stream(recogniser, doubled, StreamingConfig(fps=clip.fps, debounce=10.0))
    glosses = [p.gloss for p in emitted]
    assert len(glosses) == len(set(glosses)), f"debounce failed: {glosses}"


def test_streaming_reset_clears_state(recogniser):
    clip = synthetic_sign("HELLO", "signer-a")
    streaming = StreamingRecogniser(recogniser, StreamingConfig(fps=clip.fps))
    for i in range(len(clip)):
        streaming.push(clip.frame(i))
    streaming.reset()
    assert streaming.rest_level is None
    assert not streaming.is_signing


def test_streaming_config_rejects_an_oscillating_detector():
    with pytest.raises(SignSyncError, match="oscillate"):
        StreamingConfig(onset_ratio=1.0, offset_ratio=2.0)
    with pytest.raises(SignSyncError, match="min_frames"):
        StreamingConfig(min_frames=1)
    with pytest.raises(SignSyncError, match="max_frames"):
        StreamingConfig(min_frames=10, max_frames=5)
