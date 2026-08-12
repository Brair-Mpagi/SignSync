"""Speech contracts (plan §8.5, §10).

Plan §8.5 requires the speech layer to be "modularised so different models/engines
can be swapped without redesigning the pipeline", and plan §10 lists the engines as
examples rather than commitments. So the pipeline depends on
:class:`SpeechToText` and :class:`TextToSpeech` only.

The awkward case this design has to handle honestly: a deployment with no speech
engine installed at all. Plan §17 expects the system to run in places where
installing a 1.5 GB model is not realistic, so "no audio" must be a supported
outcome rather than a crash — but it must also be *visible*, because a caller that
believes it spoke when it did not will leave a hearing user waiting in silence.
Hence :attr:`SpeechResult.is_audible`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import SignSyncError

__all__ = [
    "AudioClip",
    "SpeechSegment",
    "Transcript",
    "SpeechResult",
    "SpeechToText",
    "TextToSpeech",
    "estimate_speech_duration",
]

#: Words per minute for estimating how long speech *would* take.
#:
#: Used for pacing the avatar and the UI when no audio is produced. 150 wpm is
#: unhurried conversational English; a clinic conversation through a translation
#: system is not the place to optimise for speed.
WORDS_PER_MINUTE = 150.0


@dataclass(frozen=True)
class AudioClip:
    """Mono float32 audio in ``[-1, 1]``."""

    samples: np.ndarray
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise SignSyncError(f"audio must be mono (1-D), got shape {self.samples.shape}")
        if self.sample_rate <= 0:
            raise SignSyncError(f"sample_rate must be positive, got {self.sample_rate}")

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def is_silent(self) -> bool:
        """Whether the clip carries no usable signal.

        Checked before transcription so an unplugged microphone produces "I heard
        nothing" rather than a hallucinated sentence — speech models are notorious
        for inventing text from silence.
        """
        return len(self.samples) == 0 or float(np.abs(self.samples).max()) < 1e-4

    @classmethod
    def silence(cls, seconds: float, sample_rate: int = 16_000) -> AudioClip:
        return cls(np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate)


@dataclass(frozen=True)
class SpeechSegment:
    """A timed span of recognised speech."""

    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass(frozen=True)
class Transcript:
    """Recognised speech (plan §8.5)."""

    text: str
    confidence: float = 1.0
    language: str = "en"
    segments: tuple[SpeechSegment, ...] = ()
    engine: str = "unknown"
    duration: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class SpeechResult:
    """Synthesised speech, or an honest report that none was produced."""

    text: str
    audio: AudioClip | None = None
    engine: str = "unknown"
    estimated_duration: float = 0.0
    detail: str = ""
    """Why there is no audio, when there is none. Shown to the operator, not the user."""

    @property
    def is_audible(self) -> bool:
        """Whether this actually produced sound.

        Callers must check. A UI that assumes success will show "speaking…" to a
        Deaf signer while the hearing person hears nothing and the conversation
        stalls with neither party knowing why.
        """
        return self.audio is not None and len(self.audio) > 0

    @property
    def duration(self) -> float:
        if self.audio is not None and len(self.audio) > 0:
            return self.audio.duration
        return self.estimated_duration


@runtime_checkable
class SpeechToText(Protocol):
    """English speech in, text out."""

    @property
    def name(self) -> str: ...

    def transcribe(self, audio: AudioClip) -> Transcript: ...


@runtime_checkable
class TextToSpeech(Protocol):
    """Text in, speech out — or a documented absence of it."""

    @property
    def name(self) -> str: ...

    def synthesise(self, text: str) -> SpeechResult: ...


def estimate_speech_duration(text: str, words_per_minute: float = WORDS_PER_MINUTE) -> float:
    """How long this text would take to say.

    Needed even when nothing is spoken: the conversation UI paces turn-taking by it,
    and objective O6 budgets under a second of added latency, which is only
    measurable against an expected duration.
    """
    words = len(text.split())
    if words == 0:
        return 0.0
    if words_per_minute <= 0:
        raise SignSyncError(f"words_per_minute must be positive, got {words_per_minute}")
    # A floor, because very short utterances ("Yes.") still take a moment to say
    # and a near-zero estimate makes the UI flash.
    return max(0.4, words / words_per_minute * 60.0)
