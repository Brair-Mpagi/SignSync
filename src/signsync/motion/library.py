"""The sign motion library (plan §8.7, stage 1).

Plan §8.7 stages sign generation deliberately:

1. play back recorded motion-capture or video-derived clips;
2. blend between them with IK for smooth transitions;
3. (post-MVP research) learn a motion model for novel sequences.

This module is stage 1's storage, plus the honest handling of stage 1's central
problem: **what happens when a gloss has no recorded clip.** Plan §14 lists robotic,
unintelligible avatar motion as a high-impact risk, and the way a system gets there
is by inventing plausible-looking motion for signs it does not have. So:

* A recorded clip is used when there is one.
* Otherwise the word is **fingerspelled** if it can be, which is what a human signer
  does with an unknown word.
* Otherwise the library returns nothing and says so, and the client shows the gloss
  as text. It never approximates.

:class:`ProceduralLibrary` is the exception, and it is clearly labelled: it produces
motion from the shared synthetic signature so demos, tests and the browser client
have something to render before any recording session has happened. Its output is
not USL and must never be shown to a user as if it were.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..avatar.rig import FaceChannel, Pose, Rig, default_rig, quat_from_axis_angle
from ..vision.synthetic import sign_signature
from .ik import reach_wrist

__all__ = [
    "SignClip",
    "ClipLibrary",
    "RecordedLibrary",
    "ProceduralLibrary",
    "FINGERSPELL_ALPHABET",
]

#: Letters the fingerspelling fallback can produce.
FINGERSPELL_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class SignClip:
    """Motion for one gloss."""

    gloss: str
    poses: list[Pose]
    fps: float = 30.0
    source: str = "unknown"
    """Where this motion came from: a signer id, ``"procedural"``, ``"fingerspelled"``.

    Kept because a Deaf evaluator reviewing avatar output (plan §14) needs to know
    which clips are real recordings and which are generated, and because a corpus
    withdrawal has to be able to find the clips derived from that signer."""

    two_handed: bool = False
    hold_frames: int = 2
    """Frames held at the end. Signs do not stop dead; a hold is what separates one
    sign from the next in continuous signing."""

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def duration(self) -> float:
        return len(self.poses) / self.fps

    @property
    def is_recorded(self) -> bool:
        return self.source not in ("procedural", "fingerspelled", "unknown")


@runtime_checkable
class ClipLibrary(Protocol):
    """Looks up motion for a gloss."""

    def get(self, gloss: str) -> SignClip | None: ...

    def __contains__(self, gloss: object) -> bool: ...


@dataclass
class RecordedLibrary:
    """Clips derived from recorded signers, loaded from disk."""

    clips: dict[str, SignClip] = field(default_factory=dict)
    rig: Rig = field(default_factory=default_rig)

    def __contains__(self, gloss: object) -> bool:
        return str(gloss).upper() in self.clips

    def __len__(self) -> int:
        return len(self.clips)

    def get(self, gloss: str) -> SignClip | None:
        return self.clips.get(gloss.upper())

    def add(self, clip: SignClip) -> None:
        self.clips[clip.gloss.upper()] = clip

    def coverage(self, glosses: list[str]) -> tuple[list[str], list[str]]:
        """Split a vocabulary into ``(have, missing)``.

        The number to report before a demo: an avatar that can sign 12 of the 50
        signs in a domain is not ready for that domain, and this is how that is
        known in advance rather than discovered in front of users.
        """
        have = [g for g in glosses if g.upper() in self.clips]
        missing = [g for g in glosses if g.upper() not in self.clips]
        return have, missing

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "clips": [
                {
                    "gloss": clip.gloss,
                    "fps": clip.fps,
                    "source": clip.source,
                    "two_handed": clip.two_handed,
                    "hold_frames": clip.hold_frames,
                    "frames": [
                        {
                            "rotations": np.round(p.rotations, 5).tolist(),
                            "root": np.round(p.root, 5).tolist(),
                            "face": {k: round(v, 4) for k, v in p.face.items()},
                        }
                        for p in clip.poses
                    ],
                }
                for clip in self.clips.values()
            ],
        }
        target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, rig: Rig | None = None) -> RecordedLibrary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        library = cls(rig=rig or default_rig())
        for entry in data.get("clips", []):
            poses = [
                Pose(
                    rotations=np.asarray(frame["rotations"], dtype=np.float32),
                    root=np.asarray(frame.get("root", [0, 0, 0]), dtype=np.float32),
                    face=dict(frame.get("face", {})),
                )
                for frame in entry["frames"]
            ]
            library.add(
                SignClip(
                    gloss=entry["gloss"],
                    poses=poses,
                    fps=float(entry.get("fps", 30.0)),
                    source=entry.get("source", "unknown"),
                    two_handed=bool(entry.get("two_handed", False)),
                    hold_frames=int(entry.get("hold_frames", 2)),
                )
            )
        return library


@dataclass
class ProceduralLibrary:
    """Generates motion from the shared synthetic signature. **Not USL.**

    Exists so the pipeline is demonstrable end to end before any signer has been
    recorded (plan §17 — the system must be runnable offline), and so the browser
    client has something to render in CI. Every clip it produces is tagged
    ``source="procedural"``, and :attr:`SignClip.is_recorded` is False, so a client
    can label it as generated rather than passing it off as a real sign.
    """

    rig: Rig = field(default_factory=default_rig)
    fps: float = 30.0

    def __contains__(self, gloss: object) -> bool:
        return bool(str(gloss).strip())

    def get(self, gloss: str) -> SignClip | None:
        gloss = gloss.strip().upper()
        if not gloss:
            return None

        signature = sign_signature(gloss)
        poses: list[Pose] = []
        for frame in range(signature.n_frames):
            pose = self.rig.rest_pose()
            _apply_handshape(self.rig, pose, "right", signature.curls, signature.spread)

            target = _wrist_target(signature.wrist_path[frame], mirror=False)
            reach_wrist(self.rig, pose, "right", target)

            if signature.two_handed:
                _apply_handshape(self.rig, pose, "left", signature.curls, signature.spread)
                reach_wrist(
                    self.rig, pose, "left", _wrist_target(signature.wrist_path[frame], True)
                )
            else:
                _rest_arm(self.rig, pose, "left")

            pose.face[FaceChannel.BROW_RAISE] = max(0.0, signature.brow)
            pose.face[FaceChannel.BROW_FURROW] = max(0.0, -signature.brow)
            pose.face[FaceChannel.HEAD_TILT] = signature.head_tilt
            poses.append(pose)

        return SignClip(
            gloss=gloss,
            poses=poses,
            fps=self.fps,
            source="procedural",
            two_handed=signature.two_handed,
        )


def _wrist_target(path_point: np.ndarray, mirror: bool) -> np.ndarray:
    """Map a body-frame trajectory point onto the rig's world space.

    The vision pipeline's body frame has its origin between the shoulders with +y
    downward; the rig places the chest a fixed distance below the root. Keeping the
    two in the same units is what lets a recognised sign and a generated sign be
    compared at all.
    """
    x, y, z = float(path_point[0]), float(path_point[1]), float(path_point[2])
    if mirror:
        x = -x
    chest_height = 0.25  # rig units below the root, matching the default skeleton
    return np.array([x * 0.9, chest_height + y * 0.7, 0.35 + z], dtype=np.float32)


def _apply_handshape(rig: Rig, pose: Pose, side: str, curls: np.ndarray, spread: float) -> None:
    """Curl the fingers of one hand.

    Applied per joint rather than as a named handshape because the library has no
    handshape inventory yet; a real one would come from the HamNoSys annotation
    plan §9.3 specifies.
    """
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    for i, finger in enumerate(fingers):
        curl = float(curls[i % len(curls)])
        for segment in (1, 2, 3):
            joint = f"{side}_{finger}_{segment}"
            angle = curl * (0.5 if segment == 1 else 0.85)
            axis = np.array([1.0, 0.0, 0.0])
            rotation = quat_from_axis_angle(axis, angle)
            if segment == 1:
                # Spread happens at the knuckle only.
                sideways = quat_from_axis_angle(
                    np.array([0.0, 0.0, 1.0]),
                    (i - 2) * 0.12 * spread * (1 if side == "left" else -1),
                )
                from ..avatar.rig import quat_multiply

                rotation = quat_multiply(sideways, rotation)
            pose.set(rig, joint, rotation)


def _rest_arm(rig: Rig, pose: Pose, side: str) -> None:
    """Leave the non-dominant arm hanging, rather than frozen mid-air."""
    positions = rig.forward_kinematics(pose)
    shoulder = positions[rig.index(f"{side}_upper_arm")]
    target = shoulder + np.array([0.05 if side == "left" else -0.05, 1.05, 0.05], dtype=np.float32)
    reach_wrist(rig, pose, side, target)
