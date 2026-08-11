"""FastAPI backend (plan §10).

    create_app(pipeline) -> FastAPI

Imported lazily by the CLI so the package still works without the ``api`` extra
(plan §17). Use ``signsync serve`` to run it, or point uvicorn at
``signsync.api:app`` for a default pipeline.
"""

from __future__ import annotations

from typing import Any

from .app import create_app

__all__ = ["create_app", "app"]


def __getattr__(name: str) -> Any:
    """Provide ``signsync.api:app`` for ``uvicorn`` without building it on import."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
