"""Deployment configuration from the environment.

A container cannot be given a Python object, so the settings a deployment needs to
vary — which model to load, which voice, how strict the confidence gate is — have to
arrive as environment variables. This module is the single place that reads them, so
the mapping from ``SIGNSYNC_*`` to pipeline behaviour is greppable rather than
scattered through the API.

The rule this follows: **a misconfigured deployment must fail loudly at startup, not
quietly at request time.** A ``SIGNSYNC_MODEL`` pointing at a file that does not
exist is a deployment mistake, and starting anyway would leave a service that
accepts sign input and recognises nothing — which looks, from the clinic, exactly
like a system that does not work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import SignSyncError

__all__ = ["Settings", "settings_from_env", "pipeline_from_env"]

PREFIX = "SIGNSYNC_"


@dataclass(frozen=True)
class Settings:
    """Everything a deployment can set without changing code."""

    model: Path | None = None
    """Trained recogniser (``.npz`` prototype, or ``.pt`` torch checkpoint)."""

    lexicon: Path | None = None
    """Override the bundled placeholder lexicon with a reviewed one."""

    clips: Path | None = None
    """Recorded motion library. Without it the avatar uses generated motion, which
    the API labels as generated (plan §8.7)."""

    voice: Path | None = None
    """Piper voice model for local speech output."""

    min_confidence: float = 0.6
    """Below this, the pipeline warns and asks the signer to repeat (plan §16.3)."""

    require_model: bool = False
    """Refuse to start without a recogniser.

    Off by default so the avatar-only deployments in plan §18.3 Mode B work, on for
    sites that need Mode A — where starting without recognition is a silent
    half-broken service rather than a reduced one."""

    def describe(self) -> list[str]:
        """Human-readable summary, printed at startup."""
        lines = [
            f"model     : {self.model or '(none — sign recognition disabled)'}",
            f"lexicon   : {self.lexicon or '(bundled placeholder)'}",
            f"clips     : {self.clips or '(none — generated motion)'}",
            f"voice     : {self.voice or '(none — best available)'}",
            f"confidence: warn below {self.min_confidence:.0%}",
        ]
        return lines


def _path(name: str) -> Path | None:
    raw = os.environ.get(PREFIX + name, "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        raise SignSyncError(
            f"{PREFIX}{name} points at {path}, which does not exist. "
            "Fix the path or unset the variable; starting without it would leave a "
            "service that silently lacks the feature."
        )
    return path


def _float(name: str, default: float) -> float:
    raw = os.environ.get(PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SignSyncError(f"{PREFIX}{name} must be a number, got {raw!r}") from None


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(PREFIX + name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise SignSyncError(f"{PREFIX}{name} must be a boolean, got {raw!r}")


def settings_from_env() -> Settings:
    """Read ``SIGNSYNC_*`` from the environment, failing loudly on bad values."""
    return Settings(
        model=_path("MODEL"),
        lexicon=_path("LEXICON"),
        clips=_path("CLIPS"),
        voice=_path("VOICE"),
        min_confidence=_float("MIN_CONFIDENCE", 0.6),
        require_model=_bool("REQUIRE_MODEL", False),
    )


def load_recogniser(path: Path):  # type: ignore[no-untyped-def]
    """Load a recogniser, choosing the backend from the file extension."""
    if path.suffix == ".npz":
        from .recognition.prototype import PrototypeRecogniser

        return PrototypeRecogniser.load(path)
    if path.suffix in (".pt", ".pth"):
        from .recognition.torch_runtime import TorchRecogniser

        return TorchRecogniser.load(path)
    raise SignSyncError(
        f"unrecognised model format {path.suffix!r} for {path}; expected .npz (prototype) "
        "or .pt/.pth (torch)"
    )


def pipeline_from_env(settings: Settings | None = None):  # type: ignore[no-untyped-def]
    """Build a :class:`~signsync.pipeline.SignSyncPipeline` from the environment."""
    from .motion.library import RecordedLibrary
    from .pipeline import SignSyncPipeline
    from .speech.stt import best_available_stt
    from .speech.tts import best_available_tts
    from .translation.lexicon import Lexicon

    settings = settings or settings_from_env()

    recogniser = load_recogniser(settings.model) if settings.model else None
    if recogniser is None and settings.require_model:
        raise SignSyncError(
            f"{PREFIX}REQUIRE_MODEL is set but no {PREFIX}MODEL was given. Train one with "
            "`signsync train`, or unset REQUIRE_MODEL for an avatar-only deployment."
        )

    return SignSyncPipeline(
        recogniser=recogniser,
        lexicon=Lexicon.load(settings.lexicon) if settings.lexicon else None,
        library=RecordedLibrary.load(settings.clips) if settings.clips else None,
        stt=best_available_stt(),  # type: ignore[arg-type]
        tts=best_available_tts(settings.voice),  # type: ignore[arg-type]
        low_confidence_threshold=settings.min_confidence,
    )
