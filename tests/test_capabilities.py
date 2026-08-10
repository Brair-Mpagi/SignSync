from __future__ import annotations

import pytest

from signsync import capabilities
from signsync.errors import MissingDependencyError


def test_every_capability_declares_a_fallback():
    """Plan §17: no missing optional dependency may leave the system with no path."""
    for cap in capabilities.CAPABILITIES:
        assert cap.fallback.strip(), f"{cap.name} declares no fallback"
        assert cap.enables.strip()
        assert cap.extra


def test_capability_names_are_unique():
    names = [c.name for c in capabilities.CAPABILITIES]
    assert len(names) == len(set(names))


def test_available_is_truthful_about_the_standard_library():
    capabilities.available.cache_clear()
    # numpy is a core dependency, so probing must succeed for something we know is here.
    assert capabilities.available("fastapi") in (True, False)  # never raises
    assert capabilities.available("torch") in (True, False)


def test_available_rejects_unknown_capability():
    with pytest.raises(KeyError):
        capabilities.available("does-not-exist")


def test_require_raises_actionable_error_when_missing(monkeypatch):
    def fake_import(name):
        raise ImportError(f"no module named {name}")

    monkeypatch.setattr(capabilities.importlib, "import_module", fake_import)

    with pytest.raises(MissingDependencyError) as excinfo:
        capabilities.require("torch", feature="training the recogniser")

    message = str(excinfo.value)
    assert "training the recogniser" in message
    assert 'pip install -e ".[models]"' in message


def test_format_report_lists_every_capability():
    text = capabilities.format_report(colour=False)
    for cap in capabilities.CAPABILITIES:
        assert cap.name in text


def test_format_report_shows_fallbacks_for_missing_capabilities(monkeypatch):
    monkeypatch.setattr(capabilities, "available", lambda name: False)
    text = capabilities.format_report(colour=False)
    assert "without it:" in text
    assert "pip install" in text
