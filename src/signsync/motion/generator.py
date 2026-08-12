"""Gloss sequence to continuous avatar motion (plan §8.7).

    GlossSequence ──▶ clips ──▶ co-articulated transitions ──▶ Animation
                                        │
                              non-manual marker channels

Two things this module is careful about, both from plan §8.7 and §14:

**Transitions are not concatenation.** Real signing has no gaps between signs; the
hands travel from the end of one to the start of the next (movement epenthesis)
while the handshape is already changing. Playing clips back to back produces the
stop-start motion Deaf evaluators reject as robotic. Transitions here are slerped
with an ease curve whose length scales with how far the hands have to travel.

**Markers are applied over their spans, not per sign.** A brow raise covering a
whole clause is a yes/no question; over one sign it marks a topic. The generator
takes the spans from :class:`~signsync.datasets.schema.NonManualMarker` and ramps the
facial channels in and out, because an instantaneous switch reads as a twitch.

A gloss with no clip is reported in :attr:`GeneratedMotion.missing` and skipped —
never approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..avatar.rig import Animation, FaceChannel, Pose, Rig, default_rig
from ..datasets.schema import MarkerType, NonManualMarker
from ..errors import SignSyncError
from ..translation.english_to_sign import GlossSequence
from .blend import ease_in_out, transition_frames
from .library import ClipLibrary, ProceduralLibrary, SignClip

__all__ = ["MotionConfig", "GeneratedMotion", "MotionGenerator"]

#: Facial channel driven by each grammatical marker.
_MARKER_CHANNELS: dict[MarkerType, str] = {
    MarkerType.BROW_RAISE: FaceChannel.BROW_RAISE,
    MarkerType.BROW_FURROW: FaceChannel.BROW_FURROW,
    MarkerType.HEAD_SHAKE: FaceChannel.HEAD_SHAKE,
    MarkerType.HEAD_NOD: FaceChannel.HEAD_NOD,
    MarkerType.HEAD_TILT: FaceChannel.HEAD_TILT,
    MarkerType.PUFFED_CHEEKS: FaceChannel.CHEEKS_PUFF,
    MarkerType.SQUINT: FaceChannel.SQUINT,
    MarkerType.MOUTH_MORPHEME: FaceChannel.MOUTH_WIDE,
    MarkerType.MOUTHING: FaceChannel.MOUTH_OPEN,
}


@dataclass(frozen=True)
class MotionConfig:
    """Timing of transitions and marker ramps."""

    fps: float = 30.0
    min_transition_frames: int = 3
    max_transition_frames: int = 12
    marker_ramp: float = 0.12
    """Seconds a facial channel takes to reach full value. An instantaneous switch
    reads as a twitch rather than as grammar."""

    lead_in_frames: int = 4
    """Frames easing out of rest at the start, so the avatar does not snap into the
    first sign."""

    tail_frames: int = 6

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise SignSyncError(f"fps must be positive, got {self.fps}")
        if self.min_transition_frames > self.max_transition_frames:
            raise SignSyncError("min_transition_frames exceeds max_transition_frames")


@dataclass
class GeneratedMotion:
    """An animation plus an honest account of what could not be signed."""

    animation: Animation
    missing: tuple[str, ...] = ()
    """Glosses with no clip. The client must show these as text."""

    procedural: tuple[str, ...] = ()
    """Glosses rendered from generated rather than recorded motion.

    Surfaced so the client can label them and so a Deaf evaluator reviewing avatar
    quality (plan §14) knows which motion is real."""

    @property
    def is_complete(self) -> bool:
        return not self.missing

    @property
    def is_fully_recorded(self) -> bool:
        return not self.missing and not self.procedural


class MotionGenerator:
    """Turns a gloss sequence into avatar motion."""

    def __init__(
        self,
        library: ClipLibrary | None = None,
        rig: Rig | None = None,
        config: MotionConfig | None = None,
    ) -> None:
        self.rig = rig or default_rig()
        # `library or ...` would be wrong: an empty RecordedLibrary is falsy, and
        # substituting the procedural library for it would silently generate motion
        # for signs nobody has recorded — the exact failure this class reports.
        self.library = ProceduralLibrary(rig=self.rig) if library is None else library
        self.config = config or MotionConfig()

    def generate(self, sequence: GlossSequence | list[str]) -> GeneratedMotion:
        """Generate motion for a gloss sequence, applying its non-manual markers."""
        if isinstance(sequence, GlossSequence):
            glosses = list(sequence.glosses)
            markers = list(sequence.markers)
        else:
            glosses = [g.upper() for g in sequence]
            markers = []

        clips: list[SignClip] = []
        missing: list[str] = []
        procedural: list[str] = []

        for gloss in glosses:
            clip = self.library.get(gloss)
            if clip is None:
                missing.append(gloss)
                continue
            clips.append(clip)
            if not clip.is_recorded:
                procedural.append(gloss)

        animation = self._assemble(clips)
        if markers:
            _apply_markers(animation, markers, self.config)

        return GeneratedMotion(
            animation=animation,
            missing=tuple(missing),
            procedural=tuple(dict.fromkeys(procedural)),
        )

    def _transition_length(self, start: Pose, end: Pose) -> int:
        return transition_frames(
            self.rig,
            start,
            end,
            fps=self.config.fps,
            minimum=self.config.min_transition_frames,
            maximum=self.config.max_transition_frames,
        )

    def _assemble(self, clips: list[SignClip]) -> Animation:
        """Concatenate clips with co-articulated transitions between them."""
        rest = self.rig.rest_pose()
        if not clips:
            return Animation(poses=[], fps=self.config.fps)

        poses: list[Pose] = []
        segments: list[tuple[float, float, str]] = []

        # Ease out of rest rather than snapping into the first handshape. The lead-in
        # scales with distance for the same reason transitions do: the hands start
        # at rest and the first sign may be anywhere, so a fixed frame count makes
        # the avatar lunge into the opening sign.
        first = clips[0].poses[0]
        lead_in = max(
            self.config.lead_in_frames,
            self._transition_length(rest, first),
        )
        for i in range(lead_in):
            poses.append(rest.blend(first, ease_in_out((i + 1) / (lead_in + 1))))

        for index, clip in enumerate(clips):
            if index > 0:
                previous = poses[-1]
                n = self._transition_length(previous, clip.poses[0])
                for i in range(n):
                    poses.append(previous.blend(clip.poses[0], ease_in_out((i + 1) / (n + 1))))

            start = len(poses) / self.config.fps
            poses.extend(pose.copy() for pose in clip.poses)
            for _ in range(clip.hold_frames):
                poses.append(clip.poses[-1].copy())
            segments.append((start, len(poses) / self.config.fps, clip.gloss))

        last = poses[-1]
        tail = max(self.config.tail_frames, self._transition_length(last, rest))
        for i in range(tail):
            poses.append(last.blend(rest, ease_in_out((i + 1) / (tail + 1))))

        return Animation(
            poses=poses,
            fps=self.config.fps,
            glosses=tuple(c.gloss for c in clips),
            segments=tuple(segments),
        )


def _apply_markers(
    animation: Animation, markers: list[NonManualMarker], config: MotionConfig
) -> None:
    """Ramp facial channels over each marker's span.

    The span is the grammar. A marker applied to the whole animation regardless of
    its declared scope would turn every topic-marked noun into a question.
    """
    if not animation.poses:
        return

    fps = animation.fps
    ramp = max(1, int(round(config.marker_ramp * fps)))

    for marker in markers:
        channel = _MARKER_CHANNELS.get(marker.marker)
        if channel is None:
            continue

        # Prefer the gloss scope when present: gloss spans survive the transition
        # frames inserted between signs, whereas the annotation's seconds refer to
        # the sequence before those frames existed.
        start_time, end_time = _resolve_span(animation, marker)
        start = int(round(start_time * fps))
        end = min(len(animation.poses), int(round(end_time * fps)))
        if end <= start:
            continue

        for index in range(start, end):
            distance_in = index - start
            distance_out = end - 1 - index
            weight = min(1.0, (min(distance_in, distance_out) + 1) / ramp)
            pose = animation.poses[index]
            pose.face[channel] = max(
                pose.face.get(channel, 0.0), float(marker.intensity) * weight
            )


def _resolve_span(animation: Animation, marker: NonManualMarker) -> tuple[float, float]:
    """Marker span in animation time."""
    scoped = set(marker.scopes_glosses)
    if scoped and animation.segments:
        spans = [(s, e) for s, e, gloss in animation.segments if gloss in scoped]
        if spans:
            return min(s for s, _ in spans), max(e for _, e in spans)
    return marker.start, marker.end if marker.end > marker.start else animation.duration
