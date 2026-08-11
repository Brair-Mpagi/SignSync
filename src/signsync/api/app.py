"""FastAPI backend (plan §10, §18.3).

Serves the three modes of plan §18.3 over HTTP and WebSocket, plus the browser
client.

Design notes worth stating:

* **Every response carries its warnings.** Confidence, an unvalidated lexicon,
  generated rather than recorded motion, absent audio — the client cannot make an
  honest interface without them (plan §16.3), so they are part of the response
  schema rather than a log line.
* **Landmarks are sent, not video.** The WebSocket protocol takes landmark frames
  from the client's own tracker, which keeps bandwidth at a few KB/s instead of a
  video stream, and means the participant's face never leaves their device unless
  they choose to record (plan §16, §17).
* **Responses are gzipped.** Animation payloads are long runs of similar floats and
  compress by roughly an order of magnitude, which matters on the connectivity plan
  §17 anticipates.
* **The API never fabricates capability.** ``/health`` reports what this deployment
  can actually do, so the client hides what is unavailable rather than offering a
  button that silently does nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..capabilities import require
from ..errors import ConsentError, SignSyncError
from ..pipeline import SignSyncPipeline

# Imported at module scope, not inside create_app, for two reasons: `require`
# still produces the friendly "install the api extra" message, and FastAPI
# resolves route annotations against module globals — a request model imported
# inside the factory is invisible to it, and every body parameter silently
# becomes a query parameter.
fastapi = require("fastapi", feature="the HTTP and WebSocket backend")

from fastapi import WebSocket  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from .schemas import (  # noqa: E402
    EnglishToSignRequest,
    SignToEnglishRequest,
    SpeechToSignRequest,
    serialise_sign_to_speech,
    serialise_speech_to_sign,
    serialise_warnings,
)

__all__ = ["create_app"]

STATIC_DIR = "static"


def create_app(pipeline: SignSyncPipeline | None = None) -> Any:
    """Build the FastAPI application.

    Takes the pipeline as an argument so tests and deployments can inject a
    configured one — a module-level singleton would make it impossible to test the
    "no recogniser loaded" and "no speech engine" paths, which are the ones most
    likely to be hit in the field.
    """
    pipeline = pipeline or SignSyncPipeline()
    app = fastapi.FastAPI(
        title="SignSync",
        version="0.1.0",
        description=(
            "Bidirectional Ugandan Sign Language <-> English translation. "
            "Output is provisional and is not a substitute for a qualified interpreter."
        ),
    )
    app.state.pipeline = pipeline

    # Animation frames are long runs of similar floats; gzip is worth an order of
    # magnitude on them and costs nothing to enable.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(SignSyncError)
    async def _handle_known_errors(request: Any, exc: SignSyncError) -> Any:
        # Errors raised on purpose carry actionable messages; consent failures are
        # 403 rather than 500 because they are a policy outcome, not a fault.
        status = 403 if isinstance(exc, ConsentError) else 400
        return JSONResponse(status_code=status, content={"error": str(exc)})

    # ------------------------------------------------------------------ status

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "capabilities": pipeline.capabilities(),
            "warnings": serialise_warnings(pipeline.deployment_warnings()),
            "disclaimer": (
                "An assistive tool, not a certified interpreter. Do not rely on it in "
                "medical, legal or safety-critical situations."
            ),
        }

    @app.get("/api/rig")
    async def rig() -> dict[str, Any]:
        from ..avatar.export import rig_to_dict

        return rig_to_dict(pipeline.rig)

    @app.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        return pipeline.latency_report()

    @app.get("/api/lexicon")
    async def lexicon() -> dict[str, Any]:
        return {
            "validated": pipeline.lexicon.is_validated,
            "reviewed_by": pipeline.lexicon.reviewed_by,
            "warning": pipeline.lexicon.warning,
            "size": len(pipeline.lexicon),
            "glosses": pipeline.lexicon.glosses(),
        }

    # ------------------------------------------------------------------ mode A

    @app.post("/api/sign-to-english")
    async def sign_to_english(request: SignToEnglishRequest) -> dict[str, Any]:
        from ..datasets.schema import MarkerType

        markers = tuple(MarkerType(m) for m in request.markers)
        result = pipeline.sign_to_speech(
            list(request.glosses), markers=markers, speak=request.speak
        )
        return serialise_sign_to_speech(result, include_audio=request.speak)

    # ------------------------------------------------------------------ mode B

    @app.post("/api/english-to-sign")
    async def english_to_sign(request: EnglishToSignRequest) -> dict[str, Any]:
        result = pipeline.speech_to_sign(request.text)
        return serialise_speech_to_sign(result, include_rig=request.include_rig)

    @app.post("/api/speech-to-sign")
    async def speech_to_sign(request: SpeechToSignRequest) -> dict[str, Any]:
        if not pipeline.capabilities()["speech_input"] and not request.text:
            raise SignSyncError(
                "no speech recogniser is available in this deployment; send `text` instead"
            )
        result = pipeline.speech_to_sign(request.text or "")
        return serialise_speech_to_sign(result, include_rig=request.include_rig)

    # ------------------------------------------------------------------ mode C

    @app.websocket("/ws/sign")
    async def sign_stream(websocket: WebSocket) -> None:
        """Streaming Mode A: landmark frames in, translations out.

        The client tracks landmarks locally and sends those rather than video, so
        the bandwidth is a few KB/s and the raw camera feed never leaves the
        device.
        """
        from ..recognition.infer import StreamingConfig, StreamingRecogniser
        from ..vision.schema import Channel, FrameLandmarks

        await websocket.accept()
        if pipeline.recogniser is None:
            await websocket.send_json(
                {"type": "error", "error": "no recogniser loaded in this deployment"}
            )
            await websocket.close()
            return

        streaming = StreamingRecogniser(
            pipeline.recogniser, StreamingConfig(fps=30.0), feature_config=pipeline.feature_config
        )
        pending: list[str] = []

        try:
            while True:
                message = await websocket.receive_json()
                kind = message.get("type", "frame")

                if kind == "reset":
                    streaming.reset()
                    pending.clear()
                    await websocket.send_json({"type": "reset"})
                    continue

                if kind == "end":
                    prediction = streaming.flush()
                    if prediction is not None:
                        pending.append(prediction.gloss)
                        await websocket.send_json(_prediction_message(prediction))
                    result = pipeline.sign_to_speech(pending, speak=False)
                    await websocket.send_json(
                        {"type": "translation", **serialise_sign_to_speech(result)}
                    )
                    pending.clear()
                    streaming.reset()
                    continue

                frame = _frame_from_message(message, FrameLandmarks, Channel)
                prediction = streaming.push(frame)
                if prediction is not None:
                    pending.append(prediction.gloss)
                    await websocket.send_json(_prediction_message(prediction))
        except Exception as exc:  # noqa: BLE001 - a dropped socket must not kill the server
            await _close_quietly(websocket, exc)

    @app.websocket("/ws/speak")
    async def speak_stream(websocket: WebSocket) -> None:
        """Streaming Mode B: English text in, avatar animation out."""
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                text = str(message.get("text", "")).strip()
                if not text:
                    await websocket.send_json({"type": "error", "error": "empty text"})
                    continue
                result = pipeline.speech_to_sign(text)
                await websocket.send_json(
                    {"type": "animation", **serialise_speech_to_sign(result, include_rig=False)}
                )
        except Exception as exc:  # noqa: BLE001
            await _close_quietly(websocket, exc)

    # ------------------------------------------------------------------ client

    static_path = Path(__file__).resolve().parent / STATIC_DIR
    if static_path.is_dir():
        app.mount("/static", StaticFiles(directory=static_path), name="static")

        @app.get("/")
        async def index() -> Any:
            return FileResponse(static_path / "index.html")

    return app


def _prediction_message(prediction: Any) -> dict[str, Any]:
    return {
        "type": "sign",
        "gloss": prediction.gloss,
        "confidence": round(float(prediction.confidence), 4),
        "start": round(float(prediction.start), 3),
        "end": round(float(prediction.end), 3),
        "alternatives": [
            {"gloss": g, "confidence": round(float(c), 4)} for g, c in prediction.alternatives
        ],
    }


def _frame_from_message(message: dict[str, Any], frame_cls: Any, channel_cls: Any) -> Any:
    """Build a FrameLandmarks from a client message.

    Missing channels arrive as ``null`` and become zero-filled *and flagged*, which
    is the same contract the tracker uses: downstream must be able to tell "hand at
    the origin" from "hand not detected".
    """
    import numpy as np

    from ..vision.schema import N_FACE, N_HAND, N_POSE

    def block(name: str, expected: int) -> tuple[np.ndarray, bool]:
        raw = message.get(name)
        if not raw:
            return np.zeros((expected, 3), dtype=np.float32), False
        array = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
        if len(array) != expected:
            return np.zeros((expected, 3), dtype=np.float32), False
        return array, True

    pose, has_pose = block("pose", N_POSE)
    left, has_left = block("leftHand", N_HAND)
    right, has_right = block("rightHand", N_HAND)
    face, has_face = block("face", N_FACE)

    present = np.zeros(channel_cls.COUNT, dtype=bool)
    present[channel_cls.POSE] = has_pose
    present[channel_cls.LEFT_HAND] = has_left
    present[channel_cls.RIGHT_HAND] = has_right
    present[channel_cls.FACE] = has_face

    return frame_cls(
        pose=pose,
        left_hand=left,
        right_hand=right,
        face=face,
        present=present,
        timestamp=float(message.get("timestamp", 0.0)),
    )


async def _close_quietly(websocket: WebSocket, exc: Exception) -> None:
    """Report an error to the client if the socket is still open, then close."""
    try:
        await websocket.send_json({"type": "error", "error": str(exc)})
        await websocket.close()
    except Exception:  # noqa: BLE001 - the socket is already gone
        pass
