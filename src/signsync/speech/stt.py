"""Speech-to-text adapters (plan §8.5).

Plan §8.5 requires tolerance of conversational speech, background noise, and the
range of English accents encountered in Uganda. None of that is achieved by an
adapter — it is achieved by choosing and evaluating a model, which this module is
deliberately structured to make swappable, and which
:mod:`signsync.evaluation` measures.

``docs/limitations.md`` records that no accent-specific evaluation set exists yet.
That is a data gap, not a code gap, and it should not be hidden behind a
confident-looking default.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..capabilities import require
from ..errors import SignSyncError
from .base import AudioClip, SpeechSegment, Transcript

__all__ = ["WhisperSTT", "ScriptedSTT", "NullSTT", "best_available_stt"]


class WhisperSTT:
    """Whisper-class recognition through ``faster-whisper`` (plan §10).

    ``model_size`` matters more here than usual: plan §17 targets clinic laptops
    without GPUs, where the large models are unusable in real time. Start at
    ``"small"`` and measure against objective O11 before going bigger.
    """

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ) -> None:
        module = require("whisper", feature="English speech recognition")
        self.model_size = model_size
        self.language = language
        self._model = module.WhisperModel(model_size, device=device, compute_type=compute_type)

    @property
    def name(self) -> str:
        return f"faster-whisper:{self.model_size}"

    def transcribe(self, audio: AudioClip) -> Transcript:
        if audio.is_silent:
            # Speech models hallucinate fluent sentences from silence. Returning
            # empty is not a missed opportunity, it is the correct answer.
            return Transcript(text="", confidence=0.0, engine=self.name, duration=audio.duration)

        segments, info = self._model.transcribe(
            audio.samples, language=self.language, vad_filter=True
        )
        collected = [
            SpeechSegment(
                text=segment.text.strip(),
                start=float(segment.start),
                end=float(segment.end),
                confidence=_probability(segment),
            )
            for segment in segments
        ]
        text = " ".join(s.text for s in collected).strip()
        confidence = min((s.confidence for s in collected), default=0.0)
        return Transcript(
            text=text,
            confidence=confidence,
            language=getattr(info, "language", self.language),
            segments=tuple(collected),
            engine=self.name,
            duration=audio.duration,
        )


def _probability(segment: object) -> float:
    """Segment confidence, from whichever field the backend exposes."""
    import math

    logprob = getattr(segment, "avg_logprob", None)
    if logprob is None:
        return 1.0
    return float(min(1.0, math.exp(float(logprob))))


@dataclass
class ScriptedSTT:
    """Returns pre-set transcripts, ignoring the audio.

    The offline path for demos, CI and interface development: the rest of the
    pipeline can be exercised with no microphone, no model download and no network
    (plan §17). It is also how the conversation flow gets tested deterministically,
    since a real recogniser makes every test a coin flip on its own accuracy.
    """

    lines: list[str] = field(default_factory=list)
    confidence: float = 1.0
    loop: bool = False
    _cursor: int = 0

    def __init__(
        self, lines: Iterable[str] | str = (), *, confidence: float = 1.0, loop: bool = False
    ) -> None:
        self.lines = [lines] if isinstance(lines, str) else list(lines)
        self.confidence = confidence
        self.loop = loop
        self._cursor = 0

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def exhausted(self) -> bool:
        return not self.loop and self._cursor >= len(self.lines)

    def transcribe(self, audio: AudioClip) -> Transcript:
        if not self.lines:
            return Transcript(text="", confidence=0.0, engine=self.name)
        if self._cursor >= len(self.lines):
            if not self.loop:
                return Transcript(text="", confidence=0.0, engine=self.name)
            self._cursor = 0

        text = self.lines[self._cursor]
        self._cursor += 1
        duration = audio.duration if len(audio) else 0.0
        return Transcript(
            text=text,
            confidence=self.confidence,
            segments=(SpeechSegment(text, 0.0, duration, self.confidence),),
            engine=self.name,
            duration=duration,
        )

    def reset(self) -> None:
        self._cursor = 0


class NullSTT:
    """Recognises nothing, and says so.

    Used when no engine is installed. It exists so the pipeline can be assembled
    and inspected without speech input; a caller that mistakes it for a working
    recogniser will see empty transcripts with zero confidence rather than
    plausible invented text.
    """

    @property
    def name(self) -> str:
        return "null"

    def transcribe(self, audio: AudioClip) -> Transcript:
        return Transcript(
            text="",
            confidence=0.0,
            engine=self.name,
            duration=audio.duration if len(audio) else 0.0,
        )


def best_available_stt(**kwargs: object) -> object:
    """The best speech recogniser this machine can actually run.

    Falls back to :class:`NullSTT` rather than raising: an absent optional
    dependency should degrade the speech feature, not prevent the sign-recognition
    half of the system from starting (plan §17).
    """
    from ..capabilities import available

    if available("whisper"):
        try:
            return WhisperSTT(**kwargs)  # type: ignore[arg-type]
        except (SignSyncError, OSError, RuntimeError):
            # A present-but-unusable install (no model files, unsupported CPU) is
            # exactly the situation the fallback exists for.
            return NullSTT()
    return NullSTT()
