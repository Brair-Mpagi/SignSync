"""The browser client's maths must agree with the server's.

The client re-implements forward kinematics in JavaScript, because the wire format
is joint offsets plus quaternions rather than baked positions (plan §8.8, §17). A
divergence between the two implementations renders a subtly wrong avatar — an arm
bent the wrong way is a different sign — and no Python test would notice, because
Python is not the code that draws it.

So this test runs the client's own function under Node and compares it to the
server's, and skips when Node is unavailable rather than pretending to have checked.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from signsync.avatar import animation_to_dict, default_rig
from signsync.motion import MotionGenerator, ProceduralLibrary

STATIC = Path(__file__).resolve().parents[1] / "src" / "signsync" / "api" / "static"


def test_static_client_is_dependency_free():
    """Plan §17: a clinic laptop that has never been online must still render."""
    for name in ("index.html", "app.js", "avatar.js", "style.css"):
        assert (STATIC / name).is_file(), f"{name} missing from the shipped client"

    # Look for remote references, not for the word "CDN" — the sources explain in
    # prose why they do not use one.
    for name in ("index.html", "app.js", "avatar.js", "style.css"):
        source = (STATIC / name).read_text(encoding="utf-8")
        for scheme in ("http://", "https://", "//unpkg", "//cdn"):
            assert scheme not in source, f"{name} references a remote host ({scheme})"


def test_client_shows_the_disclaimer_as_markup():
    """A warning the user can dismiss is a warning that is not there (plan §16.3)."""
    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="disclaimer"' in markup
    assert 'id="warnings"' in markup
    assert 'id="confidence-fill"' in markup


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_forward_kinematics_matches_python(tmp_path):
    rig = default_rig()
    animation = MotionGenerator(ProceduralLibrary(rig=rig), rig).generate(
        ["HOSPITAL", "WHERE"]
    ).animation
    payload = animation_to_dict(animation, rig)

    index = len(animation) // 2
    expected = rig.forward_kinematics(animation.poses[index])

    fixture = tmp_path / "fk.json"
    fixture.write_text(
        json.dumps({"rig": payload["rig"], "frame": payload["frames"][index]}), encoding="utf-8"
    )

    script = f"""
    import {{ forwardKinematics }} from {str(STATIC / "avatar.js")!r};
    import fs from 'fs';
    const data = JSON.parse(fs.readFileSync({str(fixture)!r}, 'utf8'));
    console.log(JSON.stringify(forwardKinematics(data.rig, data.frame)));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    got = np.array(json.loads(result.stdout), dtype=np.float64)

    assert got.shape == expected.shape
    # Tolerance covers the 4-decimal rounding the wire format applies.
    np.testing.assert_allclose(got, expected, atol=1e-3)
