"""Synthetic landmark generation for tests, demos and CI.

Not a substitute for data. Real signs are collected from real signers under consent
(plan §9), and nothing here tells you anything about USL. What this module provides
is a **deterministic stand-in with the right structure**, so that:

* the pipeline can be developed and demonstrated before the corpus exists;
* CI can exercise recognition end to end without shipping video of a real person;
* the ``data/samples/`` fixtures are derived from no identifiable individual and are
  therefore safe to commit (see ``docs/data-protection.md``).

Each gloss gets a stable, distinct motion signature derived from its name, and each
signer gets a stable body geometry, position, speed and noise level. That separation
is what makes signer-independent evaluation meaningful on synthetic data too: a
model that has memorised signer geometry instead of sign motion will fail on a
held-out synthetic signer exactly as it would on a held-out real one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .schema import (
    N_FACE,
    N_HAND,
    N_POSE,
    Channel,
    FrameLandmarks,
    HandIndex,
    LandmarkSequence,
    PoseIndex,
)

__all__ = ["SignerStyle", "synthetic_sign", "synthetic_sentence", "stable_seed"]

_PATHS = ("line", "arc", "circle", "repeat", "tap")


def stable_seed(*parts: str) -> int:
    """Seed derived from strings, stable across processes and Python versions.

    ``hash()`` is randomised per process, which would make "deterministic" fixtures
    change between runs and turn any recognition test into a coin flip.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class SignerStyle:
    """Per-signer variation: body geometry, framing, tempo, tracking quality.

    Mirrors the diversity axes plan §9.3 requires from real recruitment — body
    size, camera distance, signing speed — so held-out-signer evaluation on
    synthetic data probes the same failure mode.
    """

    signer_id: str
    shoulder_width: float = 0.22
    centre: tuple[float, float] = (0.5, 0.45)
    speed: float = 1.0
    noise: float = 0.004
    dropout: float = 0.02
    """Probability of a per-frame tracking dropout on the non-dominant hand."""
    left_handed: bool = False

    @classmethod
    def derived(cls, signer_id: str) -> SignerStyle:
        """A plausible, stable style for a signer identifier."""
        rng = np.random.default_rng(stable_seed("signer", signer_id))
        return cls(
            signer_id=signer_id,
            shoulder_width=float(rng.uniform(0.16, 0.30)),
            centre=(float(rng.uniform(0.42, 0.58)), float(rng.uniform(0.38, 0.52))),
            speed=float(rng.uniform(0.7, 1.4)),
            noise=float(rng.uniform(0.002, 0.008)),
            dropout=float(rng.uniform(0.0, 0.06)),
            left_handed=bool(rng.random() < 0.12),
        )


def _sign_parameters(gloss: str) -> dict[str, object]:
    """Stable motion signature for a gloss."""
    rng = np.random.default_rng(stable_seed("gloss", gloss.upper()))
    return {
        "start": rng.uniform([-1.1, -0.9], [1.1, 0.9]),
        "end": rng.uniform([-1.1, -0.9], [1.1, 0.9]),
        "path": _PATHS[int(rng.integers(len(_PATHS)))],
        "amplitude": float(rng.uniform(0.15, 0.55)),
        "curls": rng.uniform(0.0, 1.6, size=5),
        "spread": float(rng.uniform(0.4, 1.0)),
        "two_handed": bool(rng.random() < 0.45),
        "brow": float(rng.uniform(-1.0, 1.0)),
        "head_tilt": float(rng.uniform(-0.25, 0.25)),
        "frames": int(rng.integers(18, 40)),
        "repeats": int(rng.integers(2, 4)),
    }


def _trajectory(params: dict[str, object], n: int) -> np.ndarray:
    """Wrist path in body-frame units, shape ``(n, 3)``."""
    start = np.asarray(params["start"], dtype=np.float64)
    end = np.asarray(params["end"], dtype=np.float64)
    amp = float(params["amplitude"])  # type: ignore[arg-type]
    path = str(params["path"])
    t = np.linspace(0.0, 1.0, n)

    if path == "repeat":
        cycles = int(params["repeats"])  # type: ignore[arg-type]
        phase = 0.5 - 0.5 * np.cos(2 * np.pi * cycles * t)
        base = start[None, :] + phase[:, None] * (end - start)[None, :]
        depth = np.zeros(n)
    elif path == "circle":
        angle = 2 * np.pi * t
        centre = (start + end) / 2
        base = centre[None, :] + amp * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        depth = 0.3 * amp * np.sin(angle)
    elif path == "arc":
        base = start[None, :] + t[:, None] * (end - start)[None, :]
        base[:, 1] -= amp * np.sin(np.pi * t)
        depth = np.zeros(n)
    elif path == "tap":
        phase = np.abs(np.sin(np.pi * t * int(params["repeats"])))  # type: ignore[arg-type]
        base = start[None, :] + (0.25 * phase)[:, None] * (end - start)[None, :]
        depth = -0.2 * phase
    else:  # line, with ease-in-ease-out so velocity is not a step function
        eased = 0.5 - 0.5 * np.cos(np.pi * t)
        base = start[None, :] + eased[:, None] * (end - start)[None, :]
        depth = np.zeros(n)

    return np.concatenate([base, depth[:, None]], axis=1)


def _hand_points(curls: np.ndarray, spread: float, phase: float) -> np.ndarray:
    """Wrist-relative hand landmarks for a handshape, in body-frame units."""
    hand = np.zeros((N_HAND, 3), dtype=np.float64)
    finger_chains = (
        (HandIndex.THUMB_CMC, HandIndex.THUMB_MCP, HandIndex.THUMB_IP, HandIndex.THUMB_TIP),
        (HandIndex.INDEX_MCP, HandIndex.INDEX_PIP, HandIndex.INDEX_DIP, HandIndex.INDEX_TIP),
        (HandIndex.MIDDLE_MCP, HandIndex.MIDDLE_PIP, HandIndex.MIDDLE_DIP, HandIndex.MIDDLE_TIP),
        (HandIndex.RING_MCP, HandIndex.RING_PIP, HandIndex.RING_DIP, HandIndex.RING_TIP),
        (HandIndex.PINKY_MCP, HandIndex.PINKY_PIP, HandIndex.PINKY_DIP, HandIndex.PINKY_TIP),
    )
    base_angles = np.linspace(-0.6, 0.6, len(finger_chains)) * spread
    lengths = np.array([0.055, 0.075, 0.080, 0.072, 0.058])

    for f, chain in enumerate(finger_chains):
        direction = np.array([np.sin(base_angles[f]), -np.cos(base_angles[f]), 0.0])
        point = np.zeros(3)
        curl = float(curls[f]) * (0.9 + 0.1 * np.sin(2 * np.pi * phase))
        for j, idx in enumerate(chain):
            angle = curl * (j + 1) / len(chain)
            segment = lengths[f] / len(chain) * 2.4
            step = direction * np.cos(angle) + np.array([0.0, 0.0, 1.0]) * np.sin(angle) * 0.3
            step[1] += np.sin(angle) * 0.35
            point = point + step * segment
            hand[idx] = point
    return hand


def synthetic_sign(
    gloss: str,
    signer: SignerStyle | str = "signer-a",
    *,
    fps: float = 30.0,
    seed: int | None = None,
    n_frames: int | None = None,
) -> LandmarkSequence:
    """Generate one isolated sign clip in image coordinates.

    Output is *un-normalised*, i.e. what a tracker would emit, so callers exercise
    :func:`~signsync.vision.normalise.normalise_sequence` on the way in.
    """
    style = SignerStyle.derived(signer) if isinstance(signer, str) else signer
    params = _sign_parameters(gloss)
    rng = np.random.default_rng(
        stable_seed("clip", gloss, style.signer_id) if seed is None else seed
    )

    n = n_frames or max(8, int(round(int(params["frames"]) / style.speed)))  # type: ignore[arg-type]
    scale = style.shoulder_width
    cx, cy = style.centre

    traj = _trajectory(params, n)
    if style.left_handed:
        traj[:, 0] *= -1.0

    frames: list[FrameLandmarks] = []
    for t in range(n):
        phase = t / max(n - 1, 1)
        pose = np.zeros((N_POSE, 3), dtype=np.float64)

        # Torso: shoulders on the x axis, hips below, head above.
        pose[PoseIndex.LEFT_SHOULDER] = [cx + scale / 2, cy, 0.0]
        pose[PoseIndex.RIGHT_SHOULDER] = [cx - scale / 2, cy, 0.0]
        pose[PoseIndex.LEFT_HIP] = [cx + scale * 0.4, cy + scale * 1.6, 0.0]
        pose[PoseIndex.RIGHT_HIP] = [cx - scale * 0.4, cy + scale * 1.6, 0.0]

        head_tilt = float(params["head_tilt"]) * np.sin(np.pi * phase)  # type: ignore[arg-type]
        head_y = cy - scale * 0.75
        pose[PoseIndex.NOSE] = [cx + head_tilt * scale * 0.2, head_y, -0.02]
        pose[PoseIndex.LEFT_EYE] = [cx + scale * 0.12, head_y - scale * 0.12, 0.0]
        pose[PoseIndex.RIGHT_EYE] = [cx - scale * 0.12, head_y - scale * 0.12, 0.0]
        pose[PoseIndex.LEFT_EAR] = [cx + scale * 0.30, head_y - scale * 0.08, 0.02]
        pose[PoseIndex.RIGHT_EAR] = [cx - scale * 0.30, head_y - scale * 0.08, 0.02]

        dominant_wrist = np.array(
            [cx + traj[t, 0] * scale * 0.9, cy + traj[t, 1] * scale * 0.9, traj[t, 2] * scale]
        )
        other_wrist = dominant_wrist.copy()
        other_wrist[0] = 2 * cx - other_wrist[0]
        if not params["two_handed"]:
            other_wrist = np.array([cx - scale * 0.6, cy + scale * 1.2, 0.0])

        dom_left = style.left_handed
        left_wrist = dominant_wrist if dom_left else other_wrist
        right_wrist = other_wrist if dom_left else dominant_wrist
        pose[PoseIndex.LEFT_WRIST] = left_wrist
        pose[PoseIndex.RIGHT_WRIST] = right_wrist
        # Elbows hang below the shoulder-to-wrist midpoint, as a real arm does.
        for shoulder, wrist, elbow in (
            (PoseIndex.LEFT_SHOULDER, left_wrist, PoseIndex.LEFT_ELBOW),
            (PoseIndex.RIGHT_SHOULDER, right_wrist, PoseIndex.RIGHT_ELBOW),
        ):
            mid = (pose[shoulder] + wrist) / 2
            mid[1] += scale * 0.35
            pose[elbow] = mid
        pose[PoseIndex.LEFT_INDEX] = left_wrist + [0.01, -0.03, 0.0]
        pose[PoseIndex.RIGHT_INDEX] = right_wrist + [-0.01, -0.03, 0.0]
        pose[PoseIndex.LEFT_THUMB] = left_wrist + [0.02, -0.01, 0.0]
        pose[PoseIndex.RIGHT_THUMB] = right_wrist + [-0.02, -0.01, 0.0]

        curls = np.asarray(params["curls"], dtype=np.float64)
        shape = _hand_points(curls, float(params["spread"]), phase) * scale  # type: ignore[arg-type]
        dominant_hand = shape + dominant_wrist
        support_hand = shape * [-1, 1, 1] + other_wrist

        left_hand = dominant_hand if dom_left else support_hand
        right_hand = support_hand if dom_left else dominant_hand

        face = _face_points(
            centre=(pose[PoseIndex.NOSE][0], head_y),
            scale=scale,
            brow=float(params["brow"]),  # type: ignore[arg-type]
            mouth=0.3 + 0.2 * np.sin(2 * np.pi * phase),
        )

        present = np.ones(Channel.COUNT, dtype=bool)
        support_present = bool(params["two_handed"]) and rng.random() >= style.dropout
        if dom_left:
            present[Channel.RIGHT_HAND] = support_present
        else:
            present[Channel.LEFT_HAND] = support_present

        noise = style.noise
        frames.append(
            FrameLandmarks(
                pose=(pose + rng.normal(0, noise, pose.shape)).astype(np.float32),
                left_hand=(left_hand + rng.normal(0, noise, left_hand.shape)).astype(np.float32),
                right_hand=(right_hand + rng.normal(0, noise, right_hand.shape)).astype(np.float32),
                face=(face + rng.normal(0, noise * 0.5, face.shape)).astype(np.float32),
                present=present,
                timestamp=t / fps,
            )
        )

    return LandmarkSequence.from_frames(
        frames, fps=fps, gloss=gloss.upper(), signer_id=style.signer_id, synthetic=True
    )


def _face_points(centre: tuple[float, float], scale: float, brow: float, mouth: float) -> np.ndarray:
    """Face-mesh subset laid out around a centre, with brow height and mouth aperture."""
    cx, cy = centre
    face = np.zeros((N_FACE, 3), dtype=np.float64)
    # Landmarks are laid out on an ellipse; the specific positions do not matter,
    # only that brow height and mouth aperture vary consistently with the markers,
    # since that is what the feature encoder measures.
    for i in range(N_FACE):
        angle = 2 * np.pi * i / N_FACE
        face[i] = [
            cx + 0.28 * scale * np.cos(angle),
            cy + 0.34 * scale * np.sin(angle),
            0.0,
        ]
    # Ordering follows FACE_GROUPS: brows first (10 points), then eyes (12), lips (12).
    face[0:10, 1] -= brow * scale * 0.10
    face[22:34, 1] += mouth * scale * 0.06
    return face


def synthetic_sentence(
    glosses: list[str],
    signer: SignerStyle | str = "signer-a",
    *,
    fps: float = 30.0,
    pause_frames: int = 4,
) -> LandmarkSequence:
    """Concatenate signs into a continuous clip with short transitions.

    Used to exercise the continuous-signing segmenter (plan §8.3). The transitions
    are linear interpolations between the last and first frames of adjacent signs —
    a crude stand-in for co-articulation, which in real signing reshapes the signs
    themselves rather than just bridging them.
    """
    style = SignerStyle.derived(signer) if isinstance(signer, str) else signer
    frames: list[FrameLandmarks] = []
    boundaries: list[tuple[int, int, str]] = []

    for gloss in glosses:
        clip = synthetic_sign(gloss, style, fps=fps)
        if frames and pause_frames > 0:
            frames.extend(_interpolate(frames[-1], clip.frame(0), pause_frames))
        start = len(frames)
        frames.extend(clip.frame(i) for i in range(len(clip)))
        boundaries.append((start, len(frames), gloss.upper()))

    sequence = LandmarkSequence.from_frames(
        frames,
        fps=fps,
        signer_id=style.signer_id,
        synthetic=True,
        glosses=[g.upper() for g in glosses],
        boundaries=[[s, e, g] for s, e, g in boundaries],
    )
    # Restamp: concatenated clips each start their timestamps at zero.
    sequence.timestamps = (np.arange(len(sequence), dtype=np.float32) / fps).astype(np.float32)
    return sequence


def _interpolate(a: FrameLandmarks, b: FrameLandmarks, n: int) -> list[FrameLandmarks]:
    out: list[FrameLandmarks] = []
    for i in range(1, n + 1):
        w = i / (n + 1)
        out.append(
            FrameLandmarks(
                pose=((1 - w) * a.pose + w * b.pose).astype(np.float32),
                left_hand=((1 - w) * a.left_hand + w * b.left_hand).astype(np.float32),
                right_hand=((1 - w) * a.right_hand + w * b.right_hand).astype(np.float32),
                face=((1 - w) * a.face + w * b.face).astype(np.float32),
                present=a.present & b.present,
                timestamp=a.timestamp,
            )
        )
    return out
