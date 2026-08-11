"""Continuous signing: finding where one sign ends and the next begins (plan §8.3).

Plan §8.3 calls this the hardest recognition milestone, and plan §14 warns against
treating it as a quick extension of isolated recognition. What is implemented here
is an honest baseline, not a solution:

Signs are separated by *movement epenthesis* — the transitional motion that carries
the hands from the end of one sign to the start of the next. Transitions are
typically faster and less structured than the signs around them, and the hands
briefly slow at the hold points at each sign's edges. So boundaries are looked for
at troughs in hand motion energy, constrained by a duration prior so that a
momentary hold inside a long sign does not split it in two.

What this does **not** handle, and what a learned segmenter would be needed for:

* **Co-articulation.** Adjacent signs reshape each other; the boundary is often not
  a clean trough at all.
* **Repetition.** A sign with internal repetition looks like several short signs.
* **Fingerspelling.** Rapid letter sequences have no energy troughs between letters.

Those limits are listed in ``docs/limitations.md`` rather than hidden, because a
segmenter that silently splits a repeated sign into three produces a fluent,
confident and wrong translation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import SignSyncError
from ..vision.features import motion_energy
from ..vision.normalise import NormalisedSequence
from .base import SignPrediction, Recogniser

__all__ = ["Segment", "SegmentationConfig", "segment_motion", "ContinuousRecogniser"]


@dataclass(frozen=True)
class Segment:
    """A candidate sign span, in frames and seconds."""

    start_frame: int
    end_frame: int
    fps: float
    mean_energy: float = 0.0

    @property
    def start(self) -> float:
        return self.start_frame / self.fps

    @property
    def end(self) -> float:
        return self.end_frame / self.fps

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


@dataclass(frozen=True)
class SegmentationConfig:
    """Duration prior and sensitivity for boundary detection."""

    min_duration: float = 0.20
    """Shorter spans are treated as transitional movement, not signs."""

    max_duration: float = 2.50
    """Longer spans are split at their weakest internal trough — a span this long is
    usually two signs joined by a hold rather than one very slow sign."""

    energy_quantile: float = 0.35
    """Frames below this quantile of motion energy count as candidate boundaries.
    A quantile rather than an absolute threshold, because energy scale varies with
    signing style and normalisation."""

    smooth_frames: int = 5
    merge_gap: float = 0.10
    """Active spans closer together than this are one sign interrupted by a hold."""

    def __post_init__(self) -> None:
        if self.min_duration <= 0 or self.max_duration <= self.min_duration:
            raise SignSyncError(
                f"need 0 < min_duration ({self.min_duration}) < max_duration ({self.max_duration})"
            )
        if not 0.0 < self.energy_quantile < 1.0:
            raise SignSyncError(f"energy_quantile must be in (0, 1), got {self.energy_quantile}")


def segment_motion(
    sequence: NormalisedSequence, config: SegmentationConfig | None = None
) -> list[Segment]:
    """Split a continuous clip into candidate sign spans."""
    config = config or SegmentationConfig()
    fps = sequence.fps
    n = len(sequence)
    if n == 0:
        return []

    energy = motion_energy(sequence, smooth=config.smooth_frames)
    threshold = float(np.quantile(energy, config.energy_quantile))
    active = energy > threshold

    spans = _runs(active)
    spans = _merge_close(spans, int(round(config.merge_gap * fps)))
    spans = [s for s in spans if (s[1] - s[0]) / fps >= config.min_duration]

    max_frames = int(round(config.max_duration * fps))
    split_spans: list[tuple[int, int]] = []
    for start, end in spans:
        split_spans.extend(_split_long(start, end, energy, max_frames, int(round(config.min_duration * fps))))

    return [
        Segment(
            start_frame=start,
            end_frame=end,
            fps=fps,
            mean_energy=float(energy[start:end].mean()) if end > start else 0.0,
        )
        for start, end in split_spans
    ]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as half-open ``[start, end)`` index pairs."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _merge_close(spans: list[tuple[int, int]], gap_frames: int) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap_frames:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def _split_long(
    start: int, end: int, energy: np.ndarray, max_frames: int, min_frames: int
) -> list[tuple[int, int]]:
    """Recursively split an over-long span at its weakest interior trough."""
    if end - start <= max_frames:
        return [(start, end)]

    interior_start, interior_end = start + min_frames, end - min_frames
    if interior_end <= interior_start:
        return [(start, end)]

    cut = interior_start + int(np.argmin(energy[interior_start:interior_end]))
    return _split_long(start, cut, energy, max_frames, min_frames) + _split_long(
        cut, end, energy, max_frames, min_frames
    )


class ContinuousRecogniser:
    """Segment, then recognise each span with an isolated recogniser.

    A two-stage design rather than an end-to-end sequence model, matching plan
    §8.2's staged progression: it reuses the isolated recogniser that Phase 2
    produces, so continuous recognition can be evaluated before the Phase 3 sequence
    model exists. Its ceiling is the segmenter's — a boundary error is unrecoverable
    downstream, since the recogniser only ever sees the span it is given.
    """

    def __init__(
        self,
        recogniser: Recogniser,
        config: SegmentationConfig | None = None,
        *,
        drop_unknown: bool = False,
    ) -> None:
        self.recogniser = recogniser
        self.config = config or SegmentationConfig()
        self.drop_unknown = drop_unknown

    def recognise(
        self, sequence: NormalisedSequence, features: np.ndarray
    ) -> list[SignPrediction]:
        """Recognise a gloss sequence from a continuous clip.

        ``features`` must be the encoding of ``sequence`` — they are passed
        separately so the caller can reuse an encoding it already computed rather
        than paying for it twice on the live path.
        """
        if len(features) != len(sequence):
            raise SignSyncError(
                f"features have {len(features)} frames but the sequence has {len(sequence)}"
            )

        predictions: list[SignPrediction] = []
        for segment in segment_motion(sequence, self.config):
            span = features[segment.start_frame : segment.end_frame]
            if len(span) < 2:
                continue
            prediction = self.recogniser.predict(span)
            if prediction.is_unknown and self.drop_unknown:
                continue
            predictions.append(
                SignPrediction(
                    gloss=prediction.gloss,
                    confidence=prediction.confidence,
                    start=segment.start,
                    end=segment.end,
                    alternatives=prediction.alternatives,
                )
            )
        return predictions
