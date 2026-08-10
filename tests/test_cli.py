from __future__ import annotations

import pytest

from signsync import __version__, cli
from signsync.errors import SignSyncError


def test_doctor_reports_capabilities(capsys):
    assert cli.main(["doctor", "--no-colour"]) == 0
    out = capsys.readouterr().out
    assert "capability report" in out
    assert "mediapipe" in out


def test_no_command_prints_help_and_fails(capsys):
    assert cli.main([]) == 1
    assert "usage: signsync" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_deliberate_errors_are_reported_without_a_traceback(capsys, monkeypatch):
    def boom(args):
        raise SignSyncError("consent record expired")

    monkeypatch.setattr(cli, "_cmd_doctor", boom)
    assert cli.main(["doctor"]) == 2
    assert "error: consent record expired" in capsys.readouterr().err
