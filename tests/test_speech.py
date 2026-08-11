from __future__ import annotations

import numpy as np
import pytest

from signsync.errors import SignSyncError
from signsync.speech import (
    AudioClip,
    NullSTT,
    ScriptedSTT,
    SpeechToText,
    TextOnlyTTS,
    TextToSpeech,
    best_available_stt,
    best_available_tts,
    estimate_speech_duration,
    read_wav,
    write_wav,
)


def tone(seconds: float = 1.0, rate: int = 16_000) -> AudioClip:
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False, dtype=np.float32)
    return AudioClip((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)


# --------------------------------------------------------------------------- audio


def test_audio_clip_reports_duration_and_silence():
    assert tone(2.0).duration == pytest.approx(2.0)
    assert not tone().is_silent
    assert AudioClip.silence(0.5).is_silent
    assert AudioClip(np.zeros(0, dtype=np.float32)).is_silent


def test_audio_clip_rejects_stereo_and_bad_rates():
    with pytest.raises(SignSyncError, match="mono"):
        AudioClip(np.zeros((10, 2), dtype=np.float32))
    with pytest.raises(SignSyncError, match="sample_rate"):
        AudioClip(np.zeros(10, dtype=np.float32), 0)


def test_wav_roundtrip(tmp_path):
    original = tone(0.25)
    restored = read_wav(write_wav(original, tmp_path / "a.wav"))
    assert restored.sample_rate == original.sample_rate
    np.testing.assert_allclose(restored.samples, original.samples, atol=1e-3)


# --------------------------------------------------------------------------- STT


def test_adapters_satisfy_the_protocols():
    assert isinstance(NullSTT(), SpeechToText)
    assert isinstance(ScriptedSTT(["hi"]), SpeechToText)
    assert isinstance(TextOnlyTTS(), TextToSpeech)


def test_scripted_stt_returns_its_lines_in_order():
    stt = ScriptedSTT(["where is the hospital", "thank you"])
    assert stt.transcribe(tone()).text == "where is the hospital"
    assert stt.transcribe(tone()).text == "thank you"
    assert stt.exhausted
    assert stt.transcribe(tone()).is_empty


def test_scripted_stt_can_loop_and_reset():
    stt = ScriptedSTT("hello", loop=True)
    assert [stt.transcribe(tone()).text for _ in range(3)] == ["hello"] * 3
    stt.reset()
    assert stt.transcribe(tone()).text == "hello"


def test_scripted_stt_accepts_a_bare_string():
    assert ScriptedSTT("one line").transcribe(tone()).text == "one line"


def test_null_stt_is_empty_and_says_it_is_not_confident():
    """A caller must not be able to mistake it for a working recogniser."""
    transcript = NullSTT().transcribe(tone())
    assert transcript.is_empty
    assert transcript.confidence == 0.0
    assert transcript.engine == "null"


def test_best_available_stt_never_raises():
    """An absent optional dependency degrades speech, it does not stop the system."""
    assert isinstance(best_available_stt(), SpeechToText)


# --------------------------------------------------------------------------- TTS


def test_text_only_tts_is_explicit_about_producing_no_sound():
    result = TextOnlyTTS().synthesise("I need help.")
    assert result.text == "I need help."
    assert result.is_audible is False
    assert result.audio is None
    assert "no speech engine" in result.detail
    assert result.duration > 0, "the client still needs to pace the conversation"


def test_text_only_tts_handles_empty_text():
    result = TextOnlyTTS().synthesise("")
    assert result.duration == 0.0
    assert not result.is_audible


def test_best_available_tts_never_raises():
    assert isinstance(best_available_tts(), TextToSpeech)


def test_duration_estimate_scales_with_length_and_has_a_floor():
    short = estimate_speech_duration("Yes.")
    long = estimate_speech_duration(" ".join(["word"] * 60))
    assert short >= 0.4, "a very short utterance still takes a moment to say"
    assert long > short
    assert estimate_speech_duration("") == 0.0


def test_duration_estimate_rejects_a_nonsense_rate():
    with pytest.raises(SignSyncError, match="words_per_minute"):
        estimate_speech_duration("hello", words_per_minute=0)


def test_missing_speech_extra_is_reported_with_the_install_hint():
    from signsync.capabilities import available
    from signsync.errors import MissingDependencyError
    from signsync.speech.stt import WhisperSTT

    if available("whisper"):
        pytest.skip("whisper installed; nothing to assert about its absence")
    with pytest.raises(MissingDependencyError, match="speech"):
        WhisperSTT()
