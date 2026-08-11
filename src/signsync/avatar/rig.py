"""Avatar skeleton (plan §8.8).

Plan §8.8 is blunt about the requirement: "a rig without expressive hands and face
cannot render intelligible USL regardless of how good the motion model is". So the
rig carries full finger articulation (three joints per finger, both hands) and a set
of named facial channels for non-manual marking, not just an upper body.

Rotations are quaternions, and blending uses slerp. Euler angles would be simpler to
write and would gimbal-lock the wrist — which is exactly the joint whose rotation is
phonemic in sign languages, so a wrist that flips at certain orientations does not
produce awkward motion, it produces a different sign.

Coordinate convention, shared with :mod:`signsync.vision.normalise` so recognition
and generation can be compared directly:

* +x to the signer's left (as the viewer sees it, mirrored)
* +y downward, matching image coordinates
* +z towards the viewer
* one unit = one shoulder width
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import SignSyncError

__all__ = [
    "Joint",
    "Rig",
    "Pose",
    "Animation",
    "FaceChannel",
    "default_rig",
    "quat_identity",
    "quat_from_axis_angle",
    "quat_multiply",
    "quat_slerp",
    "quat_to_matrix",
    "quat_between",
]

_EPS = 1e-8


class FaceChannel:
    """Named facial channels driven by non-manual markers (plan §8.7).

    Deliberately a small, grammatical set rather than a full blendshape rig: these
    are the channels the marker vocabulary in
    :class:`~signsync.datasets.schema.MarkerType` actually needs, and each one maps
    to a grammatical function rather than to an emotion.
    """

    BROW_RAISE = "brow_raise"
    BROW_FURROW = "brow_furrow"
    MOUTH_OPEN = "mouth_open"
    MOUTH_WIDE = "mouth_wide"
    CHEEKS_PUFF = "cheeks_puff"
    SQUINT = "squint"
    HEAD_SHAKE = "head_shake"
    HEAD_NOD = "head_nod"
    HEAD_TILT = "head_tilt"
    EYE_GAZE_X = "eye_gaze_x"
    EYE_GAZE_Y = "eye_gaze_y"

    ALL = (
        BROW_RAISE,
        BROW_FURROW,
        MOUTH_OPEN,
        MOUTH_WIDE,
        CHEEKS_PUFF,
        SQUINT,
        HEAD_SHAKE,
        HEAD_NOD,
        HEAD_TILT,
        EYE_GAZE_X,
        EYE_GAZE_Y,
    )


# --------------------------------------------------------------------------- quaternions
# Stored as (w, x, y, z).


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_normalise(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        return quat_identity()
    return (np.asarray(q, dtype=np.float32) / norm).astype(np.float32)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return quat_identity()
    axis = axis / norm
    half = angle / 2.0
    return np.array(
        [np.cos(half), *(axis * np.sin(half))], dtype=np.float32
    )


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation, taking the shorter arc.

    Linear interpolation of quaternion components would move the joint at an uneven
    angular rate — fast in the middle of a transition, slow at the ends — which is
    precisely the "robotic" quality plan §8.7 warns Deaf evaluators reject.
    """
    a = quat_normalise(a).astype(np.float64)
    b = quat_normalise(b).astype(np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        # q and -q are the same rotation; flip so we take the short way round.
        b, dot = -b, -dot
    if dot > 0.9995:
        return quat_normalise(a + t * (b - a))

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    scale_a = np.sin((1.0 - t) * theta) / sin_theta
    scale_b = np.sin(t * theta) / sin_theta
    return quat_normalise(scale_a * a + scale_b * b)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalise(q).astype(np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest rotation taking ``source`` onto ``target``."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return quat_identity()
    a, b = a / na, b / nb
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - _EPS:
        return quat_identity()
    if dot < -1.0 + _EPS:
        # Opposite vectors: any perpendicular axis is a valid 180° rotation.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < _EPS:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return quat_from_axis_angle(axis, np.pi)
    axis = np.cross(a, b)
    return quat_from_axis_angle(axis, np.arccos(dot))


# --------------------------------------------------------------------------- skeleton


@dataclass(frozen=True)
class Joint:
    """One bone, positioned relative to its parent in the bind pose."""

    name: str
    parent: str | None
    offset: tuple[float, float, float]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.offset))


def _hand_joints(side: str, sign: float) -> list[Joint]:
    """Finger chain for one hand.

    Three joints per finger. Fewer cannot express the handshapes that distinguish
    signs; the thumb in particular needs its own chain, since thumb position alone
    separates whole handshape classes.
    """
    wrist = f"{side}_wrist"
    joints: list[Joint] = []
    fingers = (
        ("thumb", 0.22 * sign, 0.02, 0.06, 0.075),
        ("index", 0.11 * sign, -0.10, 0.09, 0.085),
        ("middle", 0.02 * sign, -0.11, 0.10, 0.090),
        ("ring", -0.07 * sign, -0.10, 0.09, 0.080),
        ("pinky", -0.15 * sign, -0.08, 0.07, 0.065),
    )
    for name, x, y, proximal, distal in fingers:
        base = f"{side}_{name}_1"
        joints.append(Joint(base, wrist, (x, y, 0.02)))
        joints.append(Joint(f"{side}_{name}_2", base, (0.0, -proximal, 0.0)))
        joints.append(Joint(f"{side}_{name}_3", f"{side}_{name}_2", (0.0, -distal, 0.0)))
    return joints


def _build_default_joints() -> list[Joint]:
    joints = [
        Joint("root", None, (0.0, 0.0, 0.0)),
        Joint("hips", "root", (0.0, 1.55, 0.0)),
        Joint("spine", "hips", (0.0, -0.75, 0.0)),
        Joint("chest", "spine", (0.0, -0.55, 0.0)),
        Joint("neck", "chest", (0.0, -0.25, 0.0)),
        Joint("head", "neck", (0.0, -0.35, 0.0)),
    ]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        joints += [
            Joint(f"{side}_shoulder", "chest", (0.18 * sign, -0.05, 0.0)),
            Joint(f"{side}_upper_arm", f"{side}_shoulder", (0.32 * sign, 0.05, 0.0)),
            Joint(f"{side}_forearm", f"{side}_upper_arm", (0.0, 0.62, 0.0)),
            Joint(f"{side}_wrist", f"{side}_forearm", (0.0, 0.55, 0.0)),
        ]
        joints += _hand_joints(side, sign)
    return joints


@dataclass
class Rig:
    """Joint hierarchy with forward kinematics."""

    joints: tuple[Joint, ...]

    def __post_init__(self) -> None:
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise SignSyncError("duplicate joint names in rig")
        self._index = {name: i for i, name in enumerate(names)}
        for joint in self.joints:
            if joint.parent is not None and joint.parent not in self._index:
                raise SignSyncError(f"joint {joint.name!r} has unknown parent {joint.parent!r}")
            if joint.parent is not None and self._index[joint.parent] >= self._index[joint.name]:
                raise SignSyncError(
                    f"joint {joint.name!r} appears before its parent {joint.parent!r}; "
                    "forward kinematics needs parents first"
                )

    def __len__(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    def index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise SignSyncError(f"no joint named {name!r} in rig") from None

    def joint(self, name: str) -> Joint:
        return self.joints[self.index(name)]

    def chain(self, name: str) -> list[str]:
        """Joint names from the root down to ``name``."""
        result: list[str] = []
        current: str | None = name
        while current is not None:
            result.append(current)
            current = self.joint(current).parent
        return list(reversed(result))

    def forward_kinematics(self, pose: Pose) -> np.ndarray:
        """World-space joint positions, shape ``(n_joints, 3)``."""
        positions = np.zeros((len(self), 3), dtype=np.float32)
        rotations = np.zeros((len(self), 3, 3), dtype=np.float32)

        for i, joint in enumerate(self.joints):
            local = quat_to_matrix(pose.rotations[i])
            if joint.parent is None:
                rotations[i] = local
                positions[i] = pose.root + np.asarray(joint.offset, dtype=np.float32)
            else:
                p = self.index(joint.parent)
                rotations[i] = rotations[p] @ local
                positions[i] = positions[p] + rotations[p] @ np.asarray(
                    joint.offset, dtype=np.float32
                )
        return positions

    def rest_pose(self) -> Pose:
        return Pose(
            rotations=np.tile(quat_identity(), (len(self), 1)),
            root=np.zeros(3, dtype=np.float32),
        )


@dataclass
class Pose:
    """Local joint rotations plus a root translation, and facial channel weights."""

    rotations: np.ndarray  # (n_joints, 4) quaternions
    root: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    face: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rotations = np.asarray(self.rotations, dtype=np.float32)
        if self.rotations.ndim != 2 or self.rotations.shape[1] != 4:
            raise SignSyncError(
                f"rotations must be (n_joints, 4) quaternions, got {self.rotations.shape}"
            )
        self.root = np.asarray(self.root, dtype=np.float32).reshape(3)

    def copy(self) -> Pose:
        return Pose(self.rotations.copy(), self.root.copy(), dict(self.face))

    def set(self, rig: Rig, joint: str, rotation: np.ndarray) -> Pose:
        self.rotations[rig.index(joint)] = quat_normalise(rotation)
        return self

    def blend(self, other: Pose, t: float) -> Pose:
        """Interpolate towards ``other``, slerping every joint."""
        t = float(np.clip(t, 0.0, 1.0))
        rotations = np.stack(
            [quat_slerp(a, b, t) for a, b in zip(self.rotations, other.rotations, strict=True)]
        )
        face = {
            key: (1 - t) * self.face.get(key, 0.0) + t * other.face.get(key, 0.0)
            for key in set(self.face) | set(other.face)
        }
        return Pose(rotations, (1 - t) * self.root + t * other.root, face)


@dataclass
class Animation:
    """A timed sequence of poses, with the glosses and markers that produced it."""

    poses: list[Pose]
    fps: float = 30.0
    glosses: tuple[str, ...] = ()
    segments: tuple[tuple[float, float, str], ...] = ()
    """``(start, end, gloss)`` spans, so a client can highlight the sign in progress."""

    notes: str = ""

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def duration(self) -> float:
        return len(self.poses) / self.fps if self.fps else 0.0

    def gloss_at(self, time: float) -> str | None:
        for start, end, gloss in self.segments:
            if start <= time < end:
                return gloss
        return None

    def concat(self, other: Animation) -> Animation:
        if abs(self.fps - other.fps) > 1e-6:
            raise SignSyncError(f"cannot concatenate {self.fps} fps with {other.fps} fps")
        offset = self.duration
        return Animation(
            poses=self.poses + other.poses,
            fps=self.fps,
            glosses=self.glosses + other.glosses,
            segments=self.segments
            + tuple((s + offset, e + offset, g) for s, e, g in other.segments),
        )


def default_rig() -> Rig:
    """The standard SignSync skeleton: upper body, both hands, head."""
    return Rig(tuple(_build_default_joints()))
