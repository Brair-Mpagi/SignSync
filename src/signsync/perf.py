"""Latency and throughput measurement.

Plan §14 lists "latency too high for real-time use" as a live risk whose mitigation
ends with *"measure FPS/latency continuously, don't assume"*. Objectives O1
(≥25 FPS tracking on a mid-range laptop CPU) and O11 (<2 s conversational
round-trip) are numeric, so the system carries its own measurement rather than
relying on someone remembering to profile it.

Every stage of the live pipeline reports through a :class:`LatencyTracker`; the API
exposes the aggregate so a deployment can be checked against the objectives on the
hardware it actually runs on, not on a developer laptop.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

__all__ = ["FpsMeter", "LatencyTracker", "StageTiming", "LatencyReport"]


class FpsMeter:
    """Frame rate over a sliding window.

    A window rather than a running total since start: a pipeline that averaged
    30 FPS overall while stalling to 5 FPS whenever both hands are visible fails
    the objective in exactly the situation that matters, and a cumulative mean
    hides that.
    """

    def __init__(self, window: int = 60) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> None:
        self._times.append(time.perf_counter() if now is None else now)

    @property
    def fps(self) -> float:
        """Frames per second over the window, or 0.0 before two ticks."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span

    def meets(self, target_fps: float) -> bool:
        return self.fps >= target_fps

    def reset(self) -> None:
        self._times.clear()


@dataclass(frozen=True)
class StageTiming:
    """Latency summary for one pipeline stage, in milliseconds."""

    stage: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


@dataclass(frozen=True)
class LatencyReport:
    """Per-stage timings plus the end-to-end total."""

    stages: tuple[StageTiming, ...]
    total_p95_ms: float

    def meets(self, budget_ms: float) -> bool:
        """Whether the 95th-percentile round trip fits the budget.

        p95 and not the mean: turn-taking breaks down on the slow exchanges, and a
        mean that passes while one exchange in twenty takes four seconds is not a
        system people can hold a conversation through.
        """
        return self.total_p95_ms <= budget_ms

    def slowest(self) -> StageTiming | None:
        return max(self.stages, key=lambda s: s.p95_ms, default=None)


class LatencyTracker:
    """Records per-stage durations for the live pipeline."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._samples: dict[str, deque[float]] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time a block and file it under ``stage``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - start) * 1000.0)

    def record(self, stage: str, milliseconds: float) -> None:
        self._samples.setdefault(stage, deque(maxlen=self._window)).append(milliseconds)

    def stage(self, name: str) -> StageTiming | None:
        samples = self._samples.get(name)
        if not samples:
            return None
        ordered = sorted(samples)
        return StageTiming(
            stage=name,
            count=len(ordered),
            mean_ms=sum(ordered) / len(ordered),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            max_ms=ordered[-1],
        )

    def report(self) -> LatencyReport:
        stages = tuple(s for s in (self.stage(name) for name in self._samples) if s is not None)
        # Stages run in sequence, so the round trip is bounded by the sum of the
        # per-stage p95s. That is pessimistic — the slow frames are not always the
        # same frames — which is the right direction to be wrong in for a budget.
        total = sum(s.p95_ms for s in stages)
        return LatencyReport(stages=stages, total_p95_ms=total)

    def reset(self) -> None:
        self._samples.clear()


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]
