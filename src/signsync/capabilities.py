"""Runtime capability probing for optional dependencies.

Plan §17 requires the system to run local-first on modest hardware with unreliable
connectivity. That means no module may hard-import a heavy optional dependency at
package import time: a clinic laptop without ``torch`` must still be able to run the
recognition demo, and a machine without a camera must still be able to serve the
avatar.

Every optional backend therefore goes through :func:`require`, which raises a
:class:`~signsync.errors.MissingDependencyError` naming the extra to install, and
:func:`report`, which powers ``signsync doctor``.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

from .errors import MissingDependencyError

__all__ = ["Capability", "CAPABILITIES", "available", "require", "report", "format_report"]


@dataclass(frozen=True)
class Capability:
    """An optional dependency and the honest consequence of it being absent."""

    name: str
    """Short identifier, e.g. ``"torch"``."""

    package: str
    """Importable module name to probe."""

    extra: str
    """Name of the pyproject optional-dependency group that installs it."""

    enables: str
    """What becomes possible when it is present."""

    fallback: str
    """What happens when it is absent. Never "nothing works" — there is always a path."""


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="mediapipe",
        package="mediapipe",
        extra="vision",
        enables="live hand/pose/face landmark tracking from a camera",
        fallback="replay tracking from recorded landmark files; synthetic sequences in tests",
    ),
    Capability(
        name="opencv",
        package="cv2",
        extra="vision",
        enables="camera capture, video decoding, frame preprocessing",
        fallback="landmark files are read directly; no video I/O",
    ),
    Capability(
        name="torch",
        package="torch",
        extra="models",
        enables="training and running the LSTM/TCN/Transformer recognisers",
        fallback="the NumPy prototype recogniser (nearest class mean over DTW-aligned features)",
    ),
    Capability(
        name="onnxruntime",
        package="onnxruntime",
        extra="runtime",
        enables="quantised CPU inference for deployment",
        fallback="native PyTorch or NumPy inference, slower but functionally identical",
    ),
    Capability(
        name="whisper",
        package="faster_whisper",
        extra="speech",
        enables="English speech recognition from a microphone",
        fallback="typed text input, and scripted transcripts in tests",
    ),
    Capability(
        name="piper",
        package="piper",
        extra="speech",
        enables="local neural text-to-speech",
        fallback="an OS speech command if present, otherwise text-only output",
    ),
    Capability(
        name="fastapi",
        package="fastapi",
        extra="api",
        enables="the HTTP + WebSocket backend and the browser client",
        fallback="the command-line demos, which exercise the same pipeline",
    ),
)

_BY_NAME = {c.name: c for c in CAPABILITIES}


@lru_cache(maxsize=None)
def available(name: str) -> bool:
    """Return whether the capability's package can be imported.

    Uses :func:`importlib.util.find_spec` rather than a real import so probing stays
    cheap and side-effect free — importing ``torch`` costs seconds we do not want to
    pay just to render a capability table.
    """
    cap = _lookup(name)
    try:
        return importlib.util.find_spec(cap.package) is not None
    except (ImportError, ValueError):
        # A broken or partially installed distribution: treat as unavailable rather
        # than crashing the caller, which is usually `signsync doctor` diagnosing
        # exactly that situation.
        return False


def require(name: str, *, feature: str | None = None) -> ModuleType:
    """Import and return the capability's module, or explain how to install it.

    ``feature`` names the caller in the error message, so the user is told which of
    their actions failed rather than just which package is missing.
    """
    cap = _lookup(name)
    try:
        return importlib.import_module(cap.package)
    except ImportError as exc:
        raise MissingDependencyError(
            feature or cap.enables, extra=cap.extra, package=cap.package
        ) from exc


def report() -> list[tuple[Capability, bool]]:
    """Capability status for the whole package, in declaration order."""
    return [(cap, available(cap.name)) for cap in CAPABILITIES]


def format_report(*, colour: bool = False) -> str:
    """Render :func:`report` as the human-readable ``signsync doctor`` output."""
    ok_mark, missing_mark = ("\033[32m✓\033[0m", "\033[33m○\033[0m") if colour else ("[ok]", "[--]")
    width = max(len(cap.name) for cap in CAPABILITIES)

    lines = ["SignSync capability report", ""]
    missing: list[Capability] = []
    for cap, ready in report():
        mark = ok_mark if ready else missing_mark
        lines.append(f"  {mark} {cap.name:<{width}}  {cap.enables}")
        if not ready:
            missing.append(cap)
            lines.append(f"       {'':<{width}}  without it: {cap.fallback}")

    if missing:
        extras = sorted({cap.extra for cap in missing})
        lines += [
            "",
            f"{len(missing)} optional capabilit{'y is' if len(missing) == 1 else 'ies are'} "
            "unavailable. The pipeline still runs; see the fallbacks above.",
            "  To enable them:  pip install -e \"." + f'[{",".join(extras)}]"',
        ]
    else:
        lines += ["", "All optional capabilities available."]
    return "\n".join(lines)


def _lookup(name: str) -> Capability:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown capability {name!r}; known: {sorted(_BY_NAME)}") from None
