"""SignSync — bidirectional Ugandan Sign Language <-> spoken English translation.

The package mirrors the component breakdown in ``docs/plan.md``:

``vision``
    Camera capture and landmark extraction (plan §8.1).
``recognition``
    Temporal sign models, isolated and continuous (plan §8.2, §8.3).
``translation``
    Semantic intermediate representation and both translation directions
    (plan §7.2, §8.4, §8.6).
``speech``
    Swappable speech-to-text and text-to-speech adapters (plan §8.5).
``motion`` / ``avatar``
    Sign motion generation and the rig that performs it (plan §8.7, §8.8).
``datasets``
    Corpus schema, consent enforcement, signer-independent splits (plan §9, §16).
``evaluation``
    Automatic metrics and human-evaluation tooling (plan §15).
``api``
    FastAPI backend tying the pipeline together in real time (plan §10).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
