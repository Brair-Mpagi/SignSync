"""Request and response shapes for the API.

Serialisation lives here rather than in the route handlers so that the WebSocket
and HTTP paths return the *same* structures — a client that learns the shape from
one should not have to relearn it for the other.

Every response includes ``warnings``. That is a deliberate schema decision: plan
§16.3 requires the product to be transparent about its limits, and a warnings field
that is always present is much harder for a client author to forget than one that
appears only when something is wrong.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..avatar.export import animation_to_dict
from ..pipeline import PipelineWarning, SignToSpeechResult, SpeechToSignResult

__all__ = [
    "SignToEnglishRequest",
    "EnglishToSignRequest",
    "SpeechToSignRequest",
    "serialise_warnings",
    "serialise_sign_to_speech",
    "serialise_speech_to_sign",
    "serialise_animation",
]


class SignToEnglishRequest(BaseModel):
    """Mode A, non-streaming."""

    glosses: list[str] = Field(default_factory=list, description="Recognised gloss sequence")
    markers: list[str] = Field(
        default_factory=list,
        description="Non-manual markers observed, e.g. head_shake, brow_raise",
    )
    speak: bool = Field(default=True, description="Also synthesise speech")


class EnglishToSignRequest(BaseModel):
    """Mode B, from typed text."""

    text: str = Field(default="", description="English sentence to render in USL")
    include_rig: bool = Field(
        default=False,
        description="Bundle the skeleton with the animation. Clients normally fetch "
        "/api/rig once and leave this off.",
    )


class SpeechToSignRequest(BaseModel):
    """Mode B, from speech or its transcript."""

    text: str = Field(default="", description="Transcript, when the client did its own STT")
    include_rig: bool = False


def serialise_warnings(warnings: tuple[PipelineWarning, ...]) -> list[dict[str, str]]:
    return [{"code": w.code, "message": w.message} for w in warnings]


def serialise_sign_to_speech(
    result: SignToSpeechResult, *, include_audio: bool = False
) -> dict[str, Any]:
    """Mode A response.

    ``audible`` is separate from the presence of an ``audio`` field on purpose: a
    client must be able to tell "speech was produced" from "speech was requested",
    or it will wait for sound that is never coming.
    """
    payload: dict[str, Any] = {
        "text": result.text,
        "glosses": list(result.glosses),
        "confidence": round(float(result.confidence), 4),
        "needsRepeat": result.needs_repeat,
        "frame": result.translation.frame.describe(),
        "unresolved": list(result.translation.unresolved),
        "predictions": [
            {
                "gloss": p.gloss,
                "confidence": round(float(p.confidence), 4),
                "start": round(float(p.start), 3),
                "end": round(float(p.end), 3),
            }
            for p in result.predictions
        ],
        "speech": {
            "engine": result.speech.engine,
            "audible": result.speech.is_audible,
            "duration": round(float(result.speech.duration), 3),
            "detail": result.speech.detail,
        },
        "warnings": serialise_warnings(result.warnings),
    }
    if include_audio and result.speech.is_audible and result.speech.audio is not None:
        import base64

        import numpy as np

        samples = np.clip(result.speech.audio.samples, -1.0, 1.0)
        pcm = (samples * 32767.0).astype("<i2").tobytes()
        payload["speech"]["pcm16"] = base64.b64encode(pcm).decode("ascii")
        payload["speech"]["sampleRate"] = result.speech.audio.sample_rate
    return payload


def serialise_speech_to_sign(
    result: SpeechToSignResult, *, include_rig: bool = False
) -> dict[str, Any]:
    """Mode B response."""
    rig = None
    if include_rig:
        from ..avatar.rig import default_rig

        rig = default_rig()

    return {
        "transcript": result.transcript.text,
        "transcriptConfidence": round(float(result.transcript.confidence), 4),
        "glosses": list(result.glosses),
        "notation": result.sequence.notation(),
        "frame": result.sequence.frame.describe(),
        "markers": [
            {
                "marker": m.marker.value,
                "start": round(m.start, 3),
                "end": round(m.end, 3),
                "intensity": round(m.intensity, 3),
                "scope": list(m.scopes_glosses),
            }
            for m in result.sequence.markers
        ],
        "animation": animation_to_dict(result.animation, rig),
        "missing": list(result.motion.missing),
        "generated": list(result.motion.procedural),
        "complete": result.motion.is_complete,
        "warnings": serialise_warnings(result.warnings),
    }


def serialise_animation(animation: Any, rig: Any = None) -> dict[str, Any]:
    return animation_to_dict(animation, rig)
