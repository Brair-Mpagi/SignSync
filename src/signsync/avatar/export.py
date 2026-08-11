"""Animation export for the browser renderer (plan §8.8, §10, §17).

Plan §8.8 renders through Three.js/WebGL "for browser accessibility and low
deployment friction", and plan §17 wants it running on common laptops and phones
with no dedicated GPU. So the wire format is joint hierarchy plus per-frame
quaternions — a few hundred floats per frame — rather than baked vertex data. The
client rigs a mesh to the same skeleton once and animates it locally, which keeps
the payload small enough for the connectivity plan §17 expects.

Rotations stay quaternions on the wire. Converting to Euler angles for
readability would reintroduce gimbal lock at the wrist, where rotation is phonemic.

On size: a 44-joint rig at 30 fps is about 5,300 floats per second of animation,
which is roughly 70 KB/s of raw JSON. That is deliberately left to HTTP compression
rather than hand-rolled encoding — long runs of similar floats gzip by an order of
magnitude, and the API enables compression for exactly this response. Inventing a
bespoke binary format here would save less than gzip does and would cost every
future client an implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .rig import Animation, FaceChannel, Pose, Rig

__all__ = ["animation_to_dict", "rig_to_dict", "export_animation", "pose_to_dict"]

FORMAT_VERSION = 1


def rig_to_dict(rig: Rig) -> dict[str, Any]:
    """Skeleton definition: sent once, then referenced by every animation."""
    return {
        "version": FORMAT_VERSION,
        "joints": [
            {"name": j.name, "parent": j.parent, "offset": list(j.offset)} for j in rig.joints
        ],
        "faceChannels": list(FaceChannel.ALL),
    }


def pose_to_dict(pose: Pose, *, precision: int = 4) -> dict[str, Any]:
    """One frame. Rounded, because four decimals is well below visible precision
    and halves the payload."""
    return {
        "rotations": np.round(pose.rotations, precision).tolist(),
        "root": np.round(pose.root, precision).tolist(),
        "face": {k: round(float(v), precision) for k, v in pose.face.items() if abs(v) > 1e-4},
    }


def animation_to_dict(
    animation: Animation, rig: Rig | None = None, *, precision: int = 4
) -> dict[str, Any]:
    """Serialise an animation, optionally bundling the skeleton with it."""
    payload: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "fps": animation.fps,
        "duration": animation.duration,
        "glosses": list(animation.glosses),
        "segments": [
            {"start": round(s, 4), "end": round(e, 4), "gloss": g} for s, e, g in animation.segments
        ],
        "frames": [pose_to_dict(p, precision=precision) for p in animation.poses],
    }
    if rig is not None:
        payload["rig"] = rig_to_dict(rig)
    if animation.notes:
        payload["notes"] = animation.notes
    return payload


def export_animation(
    animation: Animation, path: str | Path, rig: Rig | None = None, *, indent: int | None = None
) -> Path:
    """Write an animation to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(animation_to_dict(animation, rig), indent=indent) + "\n", encoding="utf-8"
    )
    return target
