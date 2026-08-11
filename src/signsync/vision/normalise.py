"""Signer normalisation — remove the camera, keep the language.

Plan §8.1 requires landmarks "normalised for signer position/scale so the model
generalises across body sizes and camera distances", and plan §14 (Risk 2) makes
generalisation to unseen signers a top risk. This module is the main defence.

Three separate normalisations, because they must preserve different things:

**Body frame.** Origin at the mid-shoulder point, scale by shoulder width, roll
corrected so the shoulder axis is horizontal. This removes where the signer sits
and how far away the camera is.

**Hands.** Translated to the wrist and scaled by hand size, but *not* rotation
normalised. Palm orientation is phonemic in sign languages — a rotation-invariant
hand encoding throws away the difference between distinct signs. The wrist's
position in the body frame is kept separately, because *where* a sign is made
carries meaning.

Hands are also emitted as **dominant** and **non-dominant** rather than right and
left, and a left-dominant signer's space is mirrored back to canonical form. This
is how sign languages describe signs in the first place: a left-handed signer
produces the mirror image of the same sign, not a different sign. Without this,
every classifier has to learn each sign twice, and any model that summarises a
class — including the prototype recogniser — is left trying to represent a
bimodal class with one template. Note this is the opposite situation from
:func:`~signsync.datasets.augment.mirror_handedness`: mirroring a left-dominant
signer *recovers* the canonical form, while mirroring a right-dominant one
invents a clip that reverses every directional sign.

**Face.** Centred on the nose bridge and scaled by inter-ocular distance, with head
rotation factored out into an explicit ``head_pose`` channel. Head tilt and shake
are grammatical (negation, topic marking), so they are promoted to their own
signal rather than left entangled with mouth shape.

Reference estimation
--------------------
The body frame can be estimated per frame, once per sequence, or with a running
average. The default is ``"sequence"``: per-frame estimation would silently cancel
out body shifts, and body shifting is grammar (role shift, spatial referencing),
not noise. Streaming callers use :class:`StreamingNormaliser`, which tracks a
running estimate because it has no future frames to average over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .schema import (
    FACE_MIRROR_PERM,
    FACE_SUBSET_INDEX,
    POSE_MIRROR_PAIRS,
    Channel,
    FrameLandmarks,
    HandIndex,
    LandmarkSequence,
    PoseIndex,
    mirror_permutation,
)

__all__ = [
    "BodyFrame",
    "NormalisedSequence",
    "normalise_sequence",
    "StreamingNormaliser",
    "estimate_body_frame",
    "head_pose_from_pose",
]

ReferenceMode = Literal["sequence", "per_frame", "running"]

_EPS = 1e-6
_MIN_SHOULDER_WIDTH = 1e-3

# Subset positions of the face landmarks used for centring and scaling.
_FACE_NOSE_BRIDGE = FACE_SUBSET_INDEX[168]
_FACE_LEFT_EYE_OUTER = FACE_SUBSET_INDEX[33]
_FACE_RIGHT_EYE_OUTER = FACE_SUBSET_INDEX[263]


@dataclass(frozen=True)
class BodyFrame:
    """Rigid transform from image coordinates into the signer's own frame."""

    origin: np.ndarray  # (3,) mid-shoulder point
    scale: float  # shoulder width in image units
    rotation: np.ndarray  # (3, 3) roll correction about the view axis

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map ``(..., 3)`` image points into the body frame."""
        centred = points - self.origin
        return (centred @ self.rotation.T) / self.scale

    @classmethod
    def identity(cls) -> BodyFrame:
        return cls(
            origin=np.zeros(3, dtype=np.float32),
            scale=1.0,
            rotation=np.eye(3, dtype=np.float32),
        )


def estimate_body_frame(pose: np.ndarray) -> BodyFrame | None:
    """Body frame from one pose frame, or ``None`` if the shoulders are unusable.

    Returns ``None`` rather than a degenerate transform when the shoulders coincide
    (tracking failure, or the signer turned fully side-on): dividing by a
    near-zero shoulder width would blow the landmarks up to values that look like
    violent motion to a temporal model.
    """
    left = pose[PoseIndex.LEFT_SHOULDER].astype(np.float64)
    right = pose[PoseIndex.RIGHT_SHOULDER].astype(np.float64)
    axis = left - right
    width = float(np.linalg.norm(axis[:2]))
    if width < _MIN_SHOULDER_WIDTH or not np.isfinite(width):
        return None

    # Roll: rotate about the view axis so the shoulder line lies along +x.
    angle = float(np.arctan2(axis[1], axis[0]))
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return BodyFrame(
        origin=((left + right) / 2.0).astype(np.float32),
        scale=width,
        rotation=rotation,
    )


def head_pose_from_pose(pose: np.ndarray) -> np.ndarray:
    """Coarse ``(yaw, pitch, roll)`` in radians from pose landmarks.

    Estimated from the ears, eyes and nose rather than the face mesh: those points
    survive partial face occlusion, and this signal only needs to be good enough to
    separate a head shake from a head tilt from a nod, which is what non-manual
    negation and topic marking require.
    """
    nose = pose[PoseIndex.NOSE].astype(np.float64)
    left_ear = pose[PoseIndex.LEFT_EAR].astype(np.float64)
    right_ear = pose[PoseIndex.RIGHT_EAR].astype(np.float64)
    left_eye = pose[PoseIndex.LEFT_EYE].astype(np.float64)
    right_eye = pose[PoseIndex.RIGHT_EYE].astype(np.float64)

    ear_span = float(np.linalg.norm(left_ear[:2] - right_ear[:2]))
    if ear_span < _EPS:
        return np.zeros(3, dtype=np.float32)

    # Yaw: the nose drifts towards the ear the head turns away from.
    ear_mid = (left_ear + right_ear) / 2.0
    yaw = float(np.clip((nose[0] - ear_mid[0]) / (ear_span / 2.0), -1.0, 1.0)) * (np.pi / 2)

    # Pitch: vertical offset of the nose from the eye line, in ear-span units.
    eye_mid = (left_eye + right_eye) / 2.0
    pitch = float(np.clip((nose[1] - eye_mid[1]) / (ear_span / 2.0) - 0.35, -1.0, 1.0)) * (
        np.pi / 3
    )

    # Roll: tilt of the eye line.
    roll = float(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0] + _EPS))
    return np.array([yaw, pitch, roll], dtype=np.float32)


@dataclass
class NormalisedSequence:
    """Signer-normalised landmarks, split into linguistically distinct channels.

    ``body`` keeps the upper-body skeleton in body-frame units (1.0 = shoulder
    width). ``dominant_local``/``weak_local`` are wrist-centred handshapes, and
    ``dominant_wrist``/``weak_wrist`` the hands' *locations* in the body frame —
    kept separate because location and handshape are independent parameters of a
    sign. ``face`` is head-rotation-free facial geometry and ``head_pose`` the
    rotation that was removed.

    Channels are dominant/non-dominant, not right/left: see the module docstring.
    ``dominant`` records which physical hand was detected as dominant, and
    ``mirrored`` whether the signing space was flipped to reach canonical form.
    """

    body: np.ndarray  # (T, n_pose_kept, 3)
    dominant_local: np.ndarray  # (T, 21, 3)
    weak_local: np.ndarray  # (T, 21, 3)
    dominant_wrist: np.ndarray  # (T, 3)
    weak_wrist: np.ndarray  # (T, 3)
    face: np.ndarray  # (T, N_FACE, 3)
    head_pose: np.ndarray  # (T, 3)
    present: np.ndarray  # (T, 4) bool — pose, weak, dominant, face
    fps: float
    dominant: str = "right"
    mirrored: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.body)

    @property
    def duration(self) -> float:
        return len(self) / self.fps


def _robust_reference(frames: list[BodyFrame | None]) -> BodyFrame:
    """Median body frame over the frames where the pose was tracked.

    A median rather than a mean: a handful of frames with a mis-detected shoulder
    would drag a mean far enough to distort the whole clip.
    """
    valid = [f for f in frames if f is not None]
    if not valid:
        return BodyFrame.identity()
    origin = np.median(np.stack([f.origin for f in valid]), axis=0).astype(np.float32)
    scale = float(np.median([f.scale for f in valid]))
    angles = np.array([np.arctan2(f.rotation[1, 0], f.rotation[0, 0]) for f in valid])
    # Circular median via the mean resultant vector — angles wrap, so a plain
    # median across ±pi would land in the wrong place.
    angle = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return BodyFrame(origin=origin, scale=max(scale, _MIN_SHOULDER_WIDTH), rotation=rotation)


def _normalise_hand(hand: np.ndarray, present: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(wrist_position, wrist_centred_handshape)`` for one frame."""
    wrist = hand[HandIndex.WRIST].astype(np.float32)
    if not present:
        return wrist, np.zeros_like(hand, dtype=np.float32)
    local = hand.astype(np.float32) - wrist
    span = float(np.linalg.norm(local[HandIndex.MIDDLE_MCP]))
    if span < _EPS:
        # Degenerate detection: keep the raw offsets rather than amplifying noise.
        return wrist, local
    return wrist, local / span


def _normalise_face(face: np.ndarray, present: bool, head_pose: np.ndarray) -> np.ndarray:
    """Centre on the nose bridge, scale by inter-ocular distance, undo head roll."""
    if not present:
        return np.zeros_like(face, dtype=np.float32)
    centred = face.astype(np.float32) - face[_FACE_NOSE_BRIDGE]
    span = float(
        np.linalg.norm(face[_FACE_LEFT_EYE_OUTER][:2] - face[_FACE_RIGHT_EYE_OUTER][:2])
    )
    if span > _EPS:
        centred = centred / span
    roll = float(head_pose[2])
    cos_r, sin_r = np.cos(-roll), np.sin(-roll)
    rot = np.array([[cos_r, -sin_r, 0.0], [sin_r, cos_r, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return centred @ rot.T


def detect_dominant_hand(sequence: LandmarkSequence) -> str:
    """Which hand is doing the signing: ``"right"`` or ``"left"``.

    Decided by how much each hand is tracked and how far it travels. The dominant
    hand is present more often and moves more: the non-dominant hand is idle in
    one-handed signs and, in two-handed signs, either mirrors the dominant hand or
    stays still as a base. Ties resolve to right-handed, which is the majority case
    and the safer default when the evidence is genuinely absent.

    **Prefer the signer's recorded handedness where it exists.** Handedness is a
    property of the signer, not of a clip, and this function cannot see that: a
    symmetric two-handed sign gives it no evidence either way, so it falls back to
    right-handed even for a left-handed signer. Detecting per clip therefore makes
    a signer's own clips *disagree with each other*, which is worse than a
    consistently wrong guess — the channels swap from clip to clip and every model
    sees noise. :class:`~signsync.datasets.schema.SignerProfile` records handedness
    for exactly this reason; :func:`signsync.recognition.dataset.encode_clip` uses
    it. Use this function only for a stream from an unknown signer, and preferably
    over several clips rather than one.
    """
    n = len(sequence)
    if n == 0:
        return "right"

    scores = []
    for hand, channel in (
        (sequence.right_hand, Channel.RIGHT_HAND),
        (sequence.left_hand, Channel.LEFT_HAND),
    ):
        present = sequence.present[:, channel]
        coverage = float(present.mean())
        if n > 1 and present.any():
            wrists = hand[:, HandIndex.WRIST, :]
            travel = float(np.linalg.norm(np.diff(wrists, axis=0), axis=1)[present[1:]].sum())
        else:
            travel = 0.0
        scores.append(coverage * (1.0 + travel))

    return "left" if scores[1] > scores[0] * 1.05 else "right"


def _mirror_x(points: np.ndarray) -> np.ndarray:
    """Reflect body-frame points across the signer's midline."""
    out = np.asarray(points, dtype=np.float32).copy()
    out[..., 0] *= -1.0
    return out


def _mirror_face(face: np.ndarray) -> np.ndarray:
    """Reflect a face block, exchanging left/right paired landmarks.

    Negating x alone is not a reflection of a face: it would leave the left-brow
    landmarks holding right-brow geometry, so the mirrored face block would be
    systematically scrambled for every left-handed signer, and the non-manual
    channel — which is where negation and question marking live — would carry noise
    instead of grammar.
    """
    return _mirror_x(face)[..., list(FACE_MIRROR_PERM), :]


def normalise_sequence(
    sequence: LandmarkSequence,
    *,
    reference: ReferenceMode = "sequence",
    pose_indices: tuple[int, ...] | None = None,
    dominant: str = "auto",
    canonicalise_handedness: bool = True,
) -> NormalisedSequence:
    """Normalise a whole clip.

    ``reference="sequence"`` (default) estimates one body frame for the clip, so
    body shifts stay visible in the output. ``"per_frame"`` re-estimates every
    frame, which stabilises a moving camera at the cost of erasing those shifts —
    use it only when the camera itself is handheld.

    ``dominant`` is ``"auto"`` (detect), ``"right"`` or ``"left"``; pass the signer's
    recorded handedness when the corpus knows it. ``canonicalise_handedness=False``
    keeps a left-dominant signer's raw geometry, which is what you want when
    *studying* handedness rather than recognising signs.
    """
    from .schema import UPPER_BODY_POSE

    kept = UPPER_BODY_POSE if pose_indices is None else pose_indices
    n = len(sequence)
    if n == 0:
        return NormalisedSequence(
            body=np.zeros((0, len(kept), 3), dtype=np.float32),
            dominant_local=np.zeros((0, 21, 3), dtype=np.float32),
            weak_local=np.zeros((0, 21, 3), dtype=np.float32),
            dominant_wrist=np.zeros((0, 3), dtype=np.float32),
            weak_wrist=np.zeros((0, 3), dtype=np.float32),
            face=np.zeros((0, sequence.face.shape[1], 3), dtype=np.float32),
            head_pose=np.zeros((0, 3), dtype=np.float32),
            present=np.zeros((0, Channel.COUNT), dtype=bool),
            fps=sequence.fps,
            meta=dict(sequence.meta),
        )

    if dominant == "auto":
        dominant = detect_dominant_hand(sequence)
    elif dominant not in ("left", "right"):
        raise ValueError(f"dominant must be 'auto', 'left' or 'right', got {dominant!r}")

    per_frame = [
        estimate_body_frame(sequence.pose[t]) if sequence.present[t, Channel.POSE] else None
        for t in range(n)
    ]
    fallback = _robust_reference(per_frame)

    body = np.zeros((n, len(kept), 3), dtype=np.float32)
    left_local = np.zeros((n, sequence.left_hand.shape[1], 3), dtype=np.float32)
    right_local = np.zeros_like(left_local)
    left_wrist = np.zeros((n, 3), dtype=np.float32)
    right_wrist = np.zeros((n, 3), dtype=np.float32)
    face = np.zeros_like(sequence.face, dtype=np.float32)
    head = np.zeros((n, 3), dtype=np.float32)

    for t in range(n):
        if reference == "per_frame":
            frame_ref = per_frame[t] or fallback
        else:
            frame_ref = fallback

        body[t] = frame_ref.apply(sequence.pose[t][list(kept)])

        if sequence.present[t, Channel.POSE]:
            head[t] = head_pose_from_pose(sequence.pose[t])

        for hand_arr, present_idx, wrist_out, local_out in (
            (sequence.left_hand[t], Channel.LEFT_HAND, left_wrist, left_local),
            (sequence.right_hand[t], Channel.RIGHT_HAND, right_wrist, right_local),
        ):
            present = bool(sequence.present[t, present_idx])
            wrist, local = _normalise_hand(hand_arr, present)
            wrist_out[t] = frame_ref.apply(wrist) if present else 0.0
            local_out[t] = local

        face[t] = _normalise_face(
            sequence.face[t], bool(sequence.present[t, Channel.FACE]), head[t]
        )

    present = sequence.present.copy()
    mirror = canonicalise_handedness and dominant == "left"

    if dominant == "left":
        dominant_local, weak_local = left_local, right_local
        dominant_wrist, weak_wrist = left_wrist, right_wrist
        present = present[:, [Channel.POSE, Channel.RIGHT_HAND, Channel.LEFT_HAND, Channel.FACE]]
    else:
        dominant_local, weak_local = right_local, left_local
        dominant_wrist, weak_wrist = right_wrist, left_wrist

    if mirror:
        # Reflect the signing space so a left-dominant signer's clip becomes the
        # canonical form of the same sign, rather than a second variant every model
        # would otherwise have to learn separately.
        body = _mirror_x(body)[..., list(mirror_permutation(kept, POSE_MIRROR_PAIRS)), :]
        dominant_local = _mirror_x(dominant_local)
        weak_local = _mirror_x(weak_local)
        dominant_wrist = _mirror_x(dominant_wrist)
        weak_wrist = _mirror_x(weak_wrist)
        face = _mirror_face(face)
        head = head * np.array([-1.0, 1.0, -1.0], dtype=np.float32)  # yaw and roll flip

    return NormalisedSequence(
        body=body,
        dominant_local=dominant_local,
        weak_local=weak_local,
        dominant_wrist=dominant_wrist,
        weak_wrist=weak_wrist,
        face=face,
        head_pose=head,
        present=present,
        fps=sequence.fps,
        dominant=dominant,
        mirrored=mirror,
        meta=dict(sequence.meta),
    )


class StreamingNormaliser:
    """Frame-at-a-time normalisation for the live pipeline.

    Keeps an exponentially weighted body-frame estimate, because a streaming caller
    has no future frames to take a median over. ``momentum`` trades stability
    against responsiveness: high values ignore a signer who genuinely moves, low
    values let tracking jitter wobble the whole coordinate system.
    """

    def __init__(
        self, *, momentum: float = 0.95, fps: float = 30.0, dominant: str = "right"
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if dominant not in ("left", "right"):
            raise ValueError(
                f"dominant must be 'left' or 'right', got {dominant!r}; the streaming path "
                "cannot detect handedness from a single frame, so it must be told"
            )
        self.dominant = dominant
        self.momentum = momentum
        self.fps = fps
        self._reference: BodyFrame | None = None

    @property
    def reference(self) -> BodyFrame | None:
        """Current body-frame estimate, or ``None`` before the first tracked pose."""
        return self._reference

    def reset(self) -> None:
        self._reference = None

    def update_reference(self, frame: FrameLandmarks) -> BodyFrame:
        observed = (
            estimate_body_frame(frame.pose) if frame.present[Channel.POSE] else None
        )
        if observed is None:
            return self._reference or BodyFrame.identity()
        if self._reference is None:
            self._reference = observed
            return observed

        m = self.momentum
        prev = self._reference
        angle_prev = np.arctan2(prev.rotation[1, 0], prev.rotation[0, 0])
        angle_new = np.arctan2(observed.rotation[1, 0], observed.rotation[0, 0])
        # Blend on the shortest arc so a wrap across ±pi does not spin the frame.
        delta = (angle_new - angle_prev + np.pi) % (2 * np.pi) - np.pi
        angle = angle_prev + (1 - m) * delta
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        self._reference = BodyFrame(
            origin=(m * prev.origin + (1 - m) * observed.origin).astype(np.float32),
            scale=max(m * prev.scale + (1 - m) * observed.scale, _MIN_SHOULDER_WIDTH),
            rotation=np.array(
                [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),
        )
        return self._reference

    def __call__(self, frame: FrameLandmarks) -> NormalisedSequence:
        """Normalise a single frame, returned as a length-1 sequence."""
        self.update_reference(frame)
        single = LandmarkSequence.from_frames([frame], fps=self.fps)
        result = normalise_sequence(single, reference="sequence", dominant=self.dominant)
        # Re-apply using the running reference rather than this frame's own estimate.
        ref = self._reference or BodyFrame.identity()
        from .schema import UPPER_BODY_POSE

        sign = -1.0 if result.mirrored else 1.0
        body = ref.apply(frame.pose[list(UPPER_BODY_POSE)])
        body[..., 0] *= sign
        result.body[0] = body

        hands = {
            "left": (frame.left_hand, Channel.LEFT_HAND),
            "right": (frame.right_hand, Channel.RIGHT_HAND),
        }
        weak_side = "right" if self.dominant == "left" else "left"
        for side, target in ((self.dominant, result.dominant_wrist), (weak_side, result.weak_wrist)):
            hand, channel = hands[side]
            if frame.present[channel]:
                wrist = ref.apply(hand[HandIndex.WRIST])
                wrist[0] *= sign
                target[0] = wrist
        return result
