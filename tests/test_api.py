from __future__ import annotations

import types

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from signsync.api import create_app  # noqa: E402
from signsync.motion.library import RecordedLibrary  # noqa: E402
from signsync.pipeline import SignSyncPipeline  # noqa: E402
from signsync.recognition.base import RecogniserConfig  # noqa: E402
from signsync.recognition.prototype import PrototypeRecogniser  # noqa: E402
from signsync.speech.stt import ScriptedSTT  # noqa: E402
from signsync.vision.features import encode_sequence  # noqa: E402
from signsync.vision.normalise import normalise_sequence  # noqa: E402
from signsync.vision.synthetic import synthetic_sign  # noqa: E402

GLOSSES = ("HELLO", "HELP", "HOSPITAL", "WATER")


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def recogniser():
    sequences, labels = [], []
    for signer in ("signer-a", "signer-b", "signer-c"):
        for gloss in GLOSSES:
            features, _ = encode_sequence(normalise_sequence(synthetic_sign(gloss, signer)))
            sequences.append(features)
            labels.append(gloss)
    return PrototypeRecogniser(RecogniserConfig(min_confidence=0.3)).fit(sequences, labels)


# --------------------------------------------------------------------------- asgi target


def test_uvicorn_target_resolves_to_the_application_not_a_module():
    """`uvicorn signsync.api:app` is what the container runs.

    A submodule named `app` would shadow this attribute — module __getattr__ only
    runs when normal lookup fails — and uvicorn would receive the module, starting
    cleanly and then failing on every request with "'module' object is not
    callable". That is why the routes live in server.py.
    """
    import signsync.api as api

    resolved = api.app
    assert isinstance(resolved, FastAPI), f"expected a FastAPI app, got {type(resolved)}"
    assert resolved is api.app, "the app should be built once, not per attribute access"


def test_importing_the_package_does_not_shadow_the_app():
    import signsync.api as api

    assert not isinstance(getattr(api, "app", None), types.ModuleType)


# --------------------------------------------------------------------------- status


def test_health_reports_real_capabilities(client):
    """Plan §16.3: the client hides what is unavailable rather than faking it."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["capabilities"]) == {
        "recognition",
        "speech_input",
        "speech_output",
        "avatar",
        "validated_lexicon",
    }
    assert body["capabilities"]["recognition"] is False  # none loaded in this app
    assert "not a certified interpreter" in body["disclaimer"]


def test_health_warns_about_the_unvalidated_lexicon(client):
    codes = [w["code"] for w in client.get("/health").json()["warnings"]]
    assert "unvalidated_lexicon" in codes


def test_capabilities_reflect_an_equipped_deployment(recogniser):
    equipped = TestClient(
        create_app(SignSyncPipeline(recogniser=recogniser, stt=ScriptedSTT("hello")))
    )
    capabilities = equipped.get("/health").json()["capabilities"]
    assert capabilities["recognition"] is True
    assert capabilities["speech_input"] is True


def test_rig_endpoint_describes_the_skeleton(client):
    rig = client.get("/api/rig").json()
    assert rig["joints"][0]["parent"] is None
    assert any(j["name"] == "right_index_3" for j in rig["joints"])
    assert "brow_furrow" in rig["faceChannels"]


def test_lexicon_endpoint_admits_it_is_provisional(client):
    body = client.get("/api/lexicon").json()
    assert body["validated"] is False
    assert body["reviewed_by"] is None
    assert body["size"] > 0


def test_metrics_report_against_the_latency_objective(client):
    client.post("/api/english-to-sign", json={"text": "I need help."})
    body = client.get("/api/metrics").json()
    assert "total_p95_ms" in body
    assert isinstance(body["meets_o11"], bool)


# --------------------------------------------------------------------------- mode A


def test_sign_to_english_translates_and_reports_confidence(client):
    body = client.post("/api/sign-to-english", json={"glosses": ["ME", "NEED", "HELP"]}).json()
    assert body["text"] == "I need help."
    assert body["glosses"] == ["ME", "NEED", "HELP"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert "warnings" in body


def test_markers_change_the_translation(client):
    plain = client.post("/api/sign-to-english", json={"glosses": ["YOU", "UNDERSTAND"]}).json()
    marked = client.post(
        "/api/sign-to-english",
        json={"glosses": ["YOU", "UNDERSTAND"], "markers": ["head_shake"]},
    ).json()
    assert "not" not in plain["text"].lower()
    assert "not" in marked["text"].lower()


def test_response_says_whether_speech_was_actually_produced(client):
    """A client that cannot tell will wait for audio that never arrives."""
    speech = client.post("/api/sign-to-english", json={"glosses": ["HELLO"]}).json()["speech"]
    assert speech["audible"] is False
    assert speech["detail"]
    assert speech["duration"] > 0, "the client still needs to pace the exchange"


def test_unresolved_glosses_are_reported(client):
    body = client.post("/api/sign-to-english", json={"glosses": ["ME", "BLORP"]}).json()
    assert "BLORP" in body["unresolved"]
    assert body["needsRepeat"] is True


def test_empty_glosses_are_accepted(client):
    body = client.post("/api/sign-to-english", json={"glosses": []}).json()
    assert body["text"] == ""


# --------------------------------------------------------------------------- mode B


def test_english_to_sign_returns_glosses_markers_and_motion(client):
    body = client.post("/api/english-to-sign", json={"text": "Where is the hospital?"}).json()

    assert body["glosses"] == ["HOSPITAL", "WHERE"]
    assert body["markers"][0]["marker"] == "brow_furrow"
    assert body["markers"][0]["scope"] == ["HOSPITAL", "WHERE"]
    assert len(body["animation"]["frames"]) > 0
    assert body["animation"]["segments"][0]["gloss"] == "HOSPITAL"


def test_generated_motion_is_labelled_as_generated(client):
    body = client.post("/api/english-to-sign", json={"text": "I need help."}).json()
    assert body["generated"], "procedural motion must be declared, not passed off as recorded"
    assert "generated_motion" in [w["code"] for w in body["warnings"]]


def test_missing_signs_are_reported_rather_than_approximated(client):
    empty = TestClient(create_app(SignSyncPipeline(library=RecordedLibrary())))
    body = empty.post("/api/english-to-sign", json={"text": "I need help."}).json()

    assert body["missing"] == ["ME", "NEED", "HELP"]
    assert body["complete"] is False
    assert len(body["animation"]["frames"]) == 0


def test_rig_is_only_bundled_when_asked(client):
    without = client.post("/api/english-to-sign", json={"text": "hello"}).json()
    with_rig = client.post(
        "/api/english-to-sign", json={"text": "hello", "include_rig": True}
    ).json()
    assert "rig" not in without["animation"]
    assert "rig" in with_rig["animation"]


def test_speech_to_sign_without_an_engine_explains_itself(client):
    response = client.post("/api/speech-to-sign", json={"text": ""})
    assert response.status_code == 400
    assert "text" in response.json()["error"]


def test_speech_to_sign_accepts_a_client_side_transcript(client):
    body = client.post("/api/speech-to-sign", json={"text": "I need water"}).json()
    assert body["glosses"] == ["ME", "NEED", "WATER"]


# --------------------------------------------------------------------------- websockets


def test_speak_socket_streams_animations(client):
    with client.websocket_connect("/ws/speak") as socket:
        socket.send_json({"text": "I need help."})
        message = socket.receive_json()

    assert message["type"] == "animation"
    assert message["glosses"] == ["ME", "NEED", "HELP"]
    assert len(message["animation"]["frames"]) > 0


def test_speak_socket_reports_empty_input_without_closing(client):
    with client.websocket_connect("/ws/speak") as socket:
        socket.send_json({"text": "   "})
        assert socket.receive_json()["type"] == "error"
        socket.send_json({"text": "hello"})
        assert socket.receive_json()["type"] == "animation"


def test_sign_socket_refuses_politely_without_a_recogniser(client):
    with client.websocket_connect("/ws/sign") as socket:
        message = socket.receive_json()
    assert message["type"] == "error"
    assert "recogniser" in message["error"]


def test_sign_socket_recognises_streamed_landmark_frames(recogniser):
    """Landmarks, not video: a few KB/s, and the camera feed stays on the device."""
    app = TestClient(create_app(SignSyncPipeline(recogniser=recogniser)))
    clip = synthetic_sign("HOSPITAL", "signer-d")

    with app.websocket_connect("/ws/sign") as socket:
        for _ in range(20):  # let the detector learn the rest level
            socket.send_json(_frame_message(clip, 0))
        for index in range(len(clip)):
            socket.send_json(_frame_message(clip, index))
        socket.send_json({"type": "end"})

        messages = []
        while True:
            message = socket.receive_json()
            messages.append(message)
            if message["type"] in ("translation", "error"):
                break

    assert messages[-1]["type"] == "translation"
    assert any(m["type"] == "sign" for m in messages), "no sign was detected in the stream"


def test_sign_socket_handles_reset(recogniser):
    app = TestClient(create_app(SignSyncPipeline(recogniser=recogniser)))
    with app.websocket_connect("/ws/sign") as socket:
        socket.send_json({"type": "reset"})
        assert socket.receive_json()["type"] == "reset"


def _frame_message(clip, index: int) -> dict:
    frame = clip.frame(index)
    return {
        "type": "frame",
        "pose": frame.pose.tolist(),
        "leftHand": frame.left_hand.tolist(),
        "rightHand": frame.right_hand.tolist(),
        "face": frame.face.tolist(),
        "timestamp": float(frame.timestamp),
    }
