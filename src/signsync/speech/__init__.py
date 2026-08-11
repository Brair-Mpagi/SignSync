"""Speech in and out, behind swappable adapters (plan §8.5).

    microphone ─▶ SpeechToText ─▶ Transcript ─▶ (translation)
    (translation) ─▶ TextToSpeech ─▶ SpeechResult ─▶ speaker

Both factories degrade rather than raise when no engine is installed, and the
results say so — ``Transcript.is_empty`` and ``SpeechResult.is_audible`` — because
a caller that assumes success leaves one participant in a conversation waiting.
"""

from __future__ import annotations

from .base import (
    AudioClip,
    SpeechResult,
    SpeechSegment,
    SpeechToText,
    TextToSpeech,
    Transcript,
    estimate_speech_duration,
)
from .stt import NullSTT, ScriptedSTT, WhisperSTT, best_available_stt
from .tts import CommandTTS, PiperTTS, TextOnlyTTS, best_available_tts, read_wav, write_wav

__all__ = [
    "AudioClip",
    "CommandTTS",
    "NullSTT",
    "PiperTTS",
    "ScriptedSTT",
    "SpeechResult",
    "SpeechSegment",
    "SpeechToText",
    "TextOnlyTTS",
    "TextToSpeech",
    "Transcript",
    "WhisperSTT",
    "best_available_stt",
    "best_available_tts",
    "estimate_speech_duration",
    "read_wav",
    "write_wav",
]
