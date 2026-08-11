"""FastAPI backend (plan §10).

    create_app(pipeline) -> FastAPI

Imported lazily by the CLI so the package still works without the ``api`` extra
(plan §17). Use ``signsync serve`` to run it, or point uvicorn at
``signsync.api:app`` — the container does the latter.

The routes live in ``server.py`` and not ``app.py`` on purpose. A submodule named
``app`` becomes a real attribute of this package as soon as it is imported, which
shadows the ``app`` below: module ``__getattr__`` only runs when normal lookup
*fails*, so ``uvicorn signsync.api:app`` would receive the module rather than the
application and fail with "'module' object is not callable" — in the container, at
request time, having started cleanly.
"""

from __future__ import annotations

from typing import Any

from .server import create_app

__all__ = ["create_app", "app"]

_app: Any = None


def __getattr__(name: str) -> Any:
    """Provide ``signsync.api:app`` for ``uvicorn`` without building it on import.

    Configured from ``SIGNSYNC_*`` environment variables, because that is the only
    channel a container has. Without this, a compose file could set
    ``SIGNSYNC_MODEL`` and get a service that quietly recognises nothing.

    Cached, because ASGI servers may resolve the attribute more than once and
    rebuilding the pipeline would reload the model each time.
    """
    global _app
    if name == "app":
        if _app is None:
            from ..config import pipeline_from_env

            _app = create_app(pipeline_from_env())
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
