from __future__ import annotations

import pytest

from signsync.errors import SignSyncError
from signsync.motion.library import RecordedLibrary
from signsync.pipeline import SignSyncPipeline
from signsync.recognition.base import RecogniserConfig, SignPrediction
from signsync.recognition.prototype import PrototypeRecogniser
from signsync.speech.base import AudioClip
from signsync.speech.stt import ScriptedSTT
from signsync.vision.features import encode_sequence
from signsync.vision.normalise import normalise_sequence
from signsync.vision.synthetic import synthetic_sentence, synthetic_sign

GLOSSES = ("HELLO", "HELP", "HOSPITAL", "WATER", "NEED")


@pytest.fixture(scope="module")
def recogniser():
    sequences, labels = [], []
    for signer in ("signer-a", "signer-b", "signer-c"):
        for gloss in GLOSSES:
            features, _ = encode_sequence(normalise_sequence(synthetic_sign(gloss, signer)))
            sequences.append(features)
            labels.append(gloss)
    return PrototypeRecogniser(RecogniserConfig(min_confidence=0.3)).fit(sequences, labels)


@pytest.fixture
def pipeline():
    return SignSyncPipeline()


def test_pipeline_starts_without_any_optional_component(pipeline):
    """Plan §17: a missing component disables a feature, it does not stop the system."""
    capabilities = pipeline.capabilities()
    assert capabilities["avatar"] is True
    assert capabilities["recognition"] is False
    assert capabilities["speech_output"] is False


def test_mode_a_from_glosses(pipeline):
    result = pipeline.sign_to_speech(["ME", "NEED", "HELP"])
    assert result.text == "I need help."
    assert result.glosses == ("ME", "NEED", "HELP")
    assert result.speech.text == "I need help."


def test_mode_a_from_landmarks(recogniser):
    pipeline = SignSyncPipeline(recogniser=recogniser)
    sequence = synthetic_sentence(["HOSPITAL", "WATER"], "signer-e", pause_frames=10)
    result = pipeline.sign_to_speech(sequence)

    assert result.glosses, "nothing was recognised from the landmark sequence"
    assert result.text


def test_mode_a_without_a_recogniser_says_how_to_get_one(pipeline):
    sequence = synthetic_sign("HELLO", "signer-a")
    with pytest.raises(SignSyncError, match="signsync train"):
        pipeline.sign_to_speech(sequence)


def test_low_confidence_asks_for_a_repeat_rather_than_guessing(pipeline):
    """Plan §16.3: the signer can repeat; the hearing user cannot detect an error."""
    result = pipeline.sign_to_speech(
        [SignPrediction("ME", 0.99), SignPrediction("NEED", 0.2)]
    )
    assert result.needs_repeat
    assert "low_confidence" in [w.code for w in result.warnings]


def test_confident_recognised_signs_do_not_ask_for_a_repeat(pipeline):
    result = pipeline.sign_to_speech(
        [SignPrediction("ME", 0.95), SignPrediction("NEED", 0.92), SignPrediction("WATER", 0.9)]
    )
    assert not result.needs_repeat


def test_mode_b_from_text(pipeline):
    result = pipeline.speech_to_sign("Where is the hospital?")
    assert result.glosses == ("HOSPITAL", "WHERE")
    assert len(result.animation) > 0
    assert result.transcript.text == "Where is the hospital?"


def test_mode_b_from_audio_uses_the_configured_recogniser():
    pipeline = SignSyncPipeline(stt=ScriptedSTT("i need water"))
    result = pipeline.speech_to_sign(AudioClip.silence(1.0))
    assert result.transcript.text == "i need water"
    assert result.glosses == ("ME", "NEED", "WATER")


def test_mode_b_reports_signs_it_does_not_have():
    pipeline = SignSyncPipeline(library=RecordedLibrary())
    result = pipeline.speech_to_sign("I need help.")
    assert result.motion.missing == ("ME", "NEED", "HELP")
    assert "missing_signs" in [w.code for w in result.warnings]


def test_mode_b_labels_generated_motion(pipeline):
    result = pipeline.speech_to_sign("I need help.")
    assert "generated_motion" in [w.code for w in result.warnings]


def test_untranslatable_words_reach_the_warnings(pipeline):
    result = pipeline.speech_to_sign("You must go to the hospital.")
    codes = [w.code for w in result.warnings]
    assert "untranslated_words" in codes
    assert any("must" in w.message for w in result.warnings)


def test_mode_c_routes_by_input_direction(pipeline):
    signed = pipeline.conversation_turn(signing=["ME", "NEED", "HELP"])
    spoken = pipeline.conversation_turn(speech="Where is the hospital?")
    assert signed["direction"] == "sign_to_speech"
    assert spoken["direction"] == "speech_to_sign"


def test_mode_c_refuses_to_arbitrate_simultaneous_input(pipeline):
    """Who has the floor is an interface decision, not one to make silently."""
    with pytest.raises(SignSyncError, match="floor"):
        pipeline.conversation_turn(signing=["HELLO"], speech="hello")
    with pytest.raises(SignSyncError, match="needs either"):
        pipeline.conversation_turn()


def test_latency_is_measured_across_the_round_trip(pipeline):
    pipeline.sign_to_speech(["ME", "NEED", "HELP"])
    pipeline.speech_to_sign("I need help.")
    report = pipeline.latency_report()

    stages = {s["stage"] for s in report["stages"]}
    assert {"translation", "motion"} <= stages
    assert report["total_p95_ms"] > 0
    assert isinstance(report["meets_o11"], bool)
    assert report["bottleneck"] in stages


def test_deployment_warnings_are_repeated_on_every_result(pipeline):
    """A warnings field that is always present is harder for a client to forget."""
    a = pipeline.sign_to_speech(["HELLO"])
    b = pipeline.speech_to_sign("hello")
    for result in (a, b):
        assert "unvalidated_lexicon" in [w.code for w in result.warnings]
