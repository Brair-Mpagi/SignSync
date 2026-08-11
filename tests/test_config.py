"""Environment configuration.

A container's only channel for configuration is the environment, so these tests
guard the path that a compose file or `docker run -e` actually exercises. The
failure they exist to prevent is the quiet one: a deployment that sets
``SIGNSYNC_MODEL``, starts successfully, and recognises nothing.
"""

from __future__ import annotations

import pytest

from signsync.config import PREFIX, Settings, load_recogniser, pipeline_from_env, settings_from_env
from signsync.errors import SignSyncError
from signsync.recognition.base import RecogniserConfig
from signsync.recognition.prototype import PrototypeRecogniser
from signsync.vision.features import encode_sequence
from signsync.vision.normalise import normalise_sequence
from signsync.vision.synthetic import synthetic_sign


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from an unconfigured environment."""
    for name in (
        "MODEL",
        "LEXICON",
        "CLIPS",
        "VOICE",
        "MIN_CONFIDENCE",
        "REQUIRE_MODEL",
    ):
        monkeypatch.delenv(PREFIX + name, raising=False)


@pytest.fixture
def model_path(tmp_path):
    sequences, labels = [], []
    for signer in ("signer-a", "signer-b"):
        for gloss in ("HELLO", "HELP"):
            features, _ = encode_sequence(normalise_sequence(synthetic_sign(gloss, signer)))
            sequences.append(features)
            labels.append(gloss)
    model = PrototypeRecogniser(RecogniserConfig()).fit(sequences, labels)
    return model.save(tmp_path / "recogniser.npz")


def test_empty_environment_yields_defaults():
    settings = settings_from_env()
    assert settings.model is None
    assert settings.min_confidence == 0.6
    assert settings.require_model is False


def test_model_path_is_read_from_the_environment(monkeypatch, model_path):
    monkeypatch.setenv(PREFIX + "MODEL", str(model_path))
    assert settings_from_env().model == model_path


def test_a_missing_model_path_fails_at_startup(monkeypatch, tmp_path):
    """Not at request time: a service that silently lacks recognition looks broken."""
    monkeypatch.setenv(PREFIX + "MODEL", str(tmp_path / "nope.npz"))
    with pytest.raises(SignSyncError, match="does not exist"):
        settings_from_env()


def test_bad_numbers_and_booleans_are_rejected(monkeypatch):
    monkeypatch.setenv(PREFIX + "MIN_CONFIDENCE", "very high")
    with pytest.raises(SignSyncError, match="must be a number"):
        settings_from_env()

    monkeypatch.delenv(PREFIX + "MIN_CONFIDENCE")
    monkeypatch.setenv(PREFIX + "REQUIRE_MODEL", "sometimes")
    with pytest.raises(SignSyncError, match="must be a boolean"):
        settings_from_env()


@pytest.mark.parametrize(
    ("value", "expected"), [("1", True), ("true", True), ("on", True), ("0", False), ("no", False)]
)
def test_boolean_spellings(monkeypatch, value, expected):
    monkeypatch.setenv(PREFIX + "REQUIRE_MODEL", value)
    assert settings_from_env().require_model is expected


def test_pipeline_from_env_loads_the_recogniser(monkeypatch, model_path):
    monkeypatch.setenv(PREFIX + "MODEL", str(model_path))
    pipeline = pipeline_from_env()

    assert pipeline.capabilities()["recognition"] is True
    assert pipeline.sign_to_speech(["ME", "NEED", "HELP"]).text == "I need help."


def test_pipeline_without_a_model_still_serves_the_avatar():
    """Plan §18.3 Mode B is a legitimate deployment on its own."""
    pipeline = pipeline_from_env()
    assert pipeline.capabilities()["recognition"] is False
    assert len(pipeline.speech_to_sign("I need help.").animation) > 0


def test_require_model_refuses_to_start_without_one(monkeypatch):
    monkeypatch.setenv(PREFIX + "REQUIRE_MODEL", "1")
    with pytest.raises(SignSyncError, match="REQUIRE_MODEL"):
        pipeline_from_env()


def test_confidence_threshold_reaches_the_pipeline(monkeypatch):
    monkeypatch.setenv(PREFIX + "MIN_CONFIDENCE", "0.9")
    assert pipeline_from_env().low_confidence_threshold == pytest.approx(0.9)


def test_unknown_model_format_is_named(tmp_path):
    weird = tmp_path / "model.bin"
    weird.write_bytes(b"")
    with pytest.raises(SignSyncError, match="unrecognised model format"):
        load_recogniser(weird)


def test_settings_describe_is_legible_at_startup():
    lines = Settings().describe()
    assert any("sign recognition disabled" in line for line in lines)
    assert any("bundled placeholder" in line for line in lines)
