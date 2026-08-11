"""Text-to-speech adapters (plan §8.5, objective O6).

Three backends, in preference order:

1. :class:`PiperTTS` — local neural TTS, no network, the intended production path
   for plan §17's local-first deployment.
2. :class:`CommandTTS` — an OS speech command (``espeak-ng``, macOS ``say``) if one
   is installed. Not pleasant, but audible, which beats silent.
3. :class:`TextOnlyTTS` — produces no audio and reports that clearly.

The third is not a placeholder to be quietly tolerated. Mode A of plan §18.3 ends
in speech: if nothing is spoken, the hearing participant gets nothing, and the
client has to show the text instead. That decision belongs to the client, which is
why :attr:`SpeechResult.is_audible` exists and why this module never fabricates a
silent clip to make the return type look successful.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from ..capabilities import require
from ..errors import SignSyncError
from .base import AudioClip, SpeechResult, estimate_speech_duration

__all__ = ["PiperTTS", "CommandTTS", "TextOnlyTTS", "best_available_tts", "write_wav"]

#: OS speech commands to look for, as (executable, argv builder).
_COMMANDS: tuple[tuple[str, str], ...] = (
    ("espeak-ng", "espeak-ng"),
    ("espeak", "espeak"),
    ("say", "say"),
)


class PiperTTS:
    """Local neural TTS via Piper (plan §10, §17)."""

    def __init__(self, voice: str | Path, *, sample_rate: int = 22_050) -> None:
        self._piper = require("piper", feature="local text-to-speech")
        self.voice = str(voice)
        self.sample_rate = sample_rate
        self._voice = self._piper.PiperVoice.load(self.voice)

    @property
    def name(self) -> str:
        return f"piper:{Path(self.voice).stem}"

    def synthesise(self, text: str) -> SpeechResult:
        if not text.strip():
            return SpeechResult(text="", engine=self.name, detail="empty text")

        chunks = [np.frombuffer(chunk, dtype=np.int16) for chunk in self._voice.synthesize_stream_raw(text)]
        if not chunks:
            return SpeechResult(
                text=text,
                engine=self.name,
                estimated_duration=estimate_speech_duration(text),
                detail="synthesiser returned no audio",
            )
        samples = (np.concatenate(chunks).astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        return SpeechResult(
            text=text, audio=AudioClip(samples, self.sample_rate), engine=self.name
        )


class CommandTTS:
    """An OS speech command, used when no neural voice is installed.

    Writes to a temporary WAV rather than playing directly, so the audio can be
    streamed to a browser client — the deployment target in plan §17 is often a
    phone on the other side of the room, not the machine running inference.
    """

    def __init__(self, executable: str | None = None, *, words_per_minute: int = 150) -> None:
        self.executable = executable or _find_command()
        if self.executable is None:
            raise SignSyncError(
                "no OS speech command found (looked for espeak-ng, espeak, say). "
                'Install one, or use the "speech" extra for a local neural voice.'
            )
        self.words_per_minute = words_per_minute

    @property
    def name(self) -> str:
        return f"command:{Path(self.executable).name}"

    def synthesise(self, text: str) -> SpeechResult:
        if not text.strip():
            return SpeechResult(text="", engine=self.name, detail="empty text")

        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "speech.wav"
            command = _build_command(self.executable, text, target, self.words_per_minute)
            try:
                subprocess.run(command, check=True, capture_output=True, timeout=30)
                clip = read_wav(target)
            except (subprocess.SubprocessError, OSError) as exc:
                return SpeechResult(
                    text=text,
                    engine=self.name,
                    estimated_duration=estimate_speech_duration(text),
                    detail=f"speech command failed: {exc}",
                )
        return SpeechResult(text=text, audio=clip, engine=self.name)


class TextOnlyTTS:
    """Produces no audio, and is explicit about it.

    The last resort. It returns the text and an estimated duration so the client can
    still pace a conversation, with ``is_audible`` False so the client knows to show
    the text rather than wait for sound that is not coming.
    """

    @property
    def name(self) -> str:
        return "text-only"

    def synthesise(self, text: str) -> SpeechResult:
        return SpeechResult(
            text=text,
            audio=None,
            engine=self.name,
            estimated_duration=estimate_speech_duration(text),
            detail=(
                "no speech engine installed; install the 'speech' extra or an OS speech "
                "command (espeak-ng). The client should display this text instead."
            ),
        )


def best_available_tts(voice: str | Path | None = None) -> object:
    """The best voice this machine can actually produce."""
    from ..capabilities import available

    if voice is not None and available("piper"):
        try:
            return PiperTTS(voice)
        except (SignSyncError, OSError, RuntimeError):
            pass
    if _find_command() is not None:
        try:
            return CommandTTS()
        except SignSyncError:
            pass
    return TextOnlyTTS()


def _find_command() -> str | None:
    for executable, _ in _COMMANDS:
        found = shutil.which(executable)
        if found:
            return found
    return None


def _build_command(executable: str, text: str, target: Path, wpm: int) -> list[str]:
    name = Path(executable).name
    if name == "say":  # macOS
        return [executable, "-o", str(target), "--data-format=LEI16@22050", text]
    return [executable, "-w", str(target), "-s", str(wpm), text]


def read_wav(path: str | Path) -> AudioClip:
    """Read a mono 16-bit WAV into an :class:`AudioClip`."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()

    if width != 2:
        raise SignSyncError(f"{path}: expected 16-bit audio, got {width * 8}-bit")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return AudioClip(samples.astype(np.float32), rate)


def write_wav(clip: AudioClip, path: str | Path) -> Path:
    """Write an :class:`AudioClip` as a 16-bit mono WAV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(clip.samples, -1.0, 1.0)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(clip.sample_rate)
        handle.writeframes((samples * 32767.0).astype(np.int16).tobytes())
    return target
