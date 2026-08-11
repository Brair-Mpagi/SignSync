"""Live recognition from a frame stream (plan §8.2, objective O11).

Offline recognition gets a clip whose boundaries someone has already decided. Live
recognition does not: frames arrive one at a time, the signer starts and stops when
they choose, and the system has to decide *when* a sign happened as well as *what*
it was.

The approach here is onset/offset detection on hand motion energy, with a rest
threshold learned from the stream's own quiet frames:

    rest ──(energy rises)──▶ signing ──(energy falls, held)──▶ emit ──▶ rest

Two behaviours worth stating, both aimed at not being confidently wrong in front of
a user (plan §16.3):

* A span is only emitted after the motion has stayed low for ``hold_frames``. Signs
  contain internal holds, and emitting at the first quiet frame chops them in half.
* Consecutive identical predictions inside the debounce window are suppressed, so a
  signer pausing mid-sign does not produce "HELP HELP HELP".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..errors import SignSyncError
from ..vision.features import FeatureConfig, encode_sequence
from ..vision.normalise import StreamingNormaliser
from ..vision.schema import FrameLandmarks, LandmarkSequence
from .base import Recogniser, SignPrediction

__all__ = ["StreamingConfig", "StreamingRecogniser"]


@dataclass(frozen=True)
class StreamingConfig:
    """Onset/offset sensitivity and buffering limits."""

    fps: float = 30.0
    min_frames: int = 8
    """Shorter bursts are transitional movement, not signs."""

    max_frames: int = 90
    """Hard cap on a buffered span, so a signer who never pauses still gets output
    rather than an unboundedly growing buffer."""

    onset_ratio: float = 2.0
    """Energy must exceed this multiple of the observed rest level to start a sign."""

    offset_ratio: float = 1.2
    hold_frames: int = 6
    """Quiet frames required before a span is closed. Signs contain internal holds."""

    debounce: float = 0.6
    """Seconds within which a repeat of the same gloss is suppressed."""

    calibration_frames: int = 15
    """Frames used to learn the rest energy level before recognition starts."""

    def __post_init__(self) -> None:
        if self.min_frames < 2:
            raise SignSyncError(f"min_frames must be at least 2, got {self.min_frames}")
        if self.max_frames <= self.min_frames:
            raise SignSyncError("max_frames must exceed min_frames")
        if self.offset_ratio > self.onset_ratio:
            raise SignSyncError(
                "offset_ratio must not exceed onset_ratio, or the detector will "
                "oscillate between states on a steady signal"
            )


class StreamingRecogniser:
    """Frame-at-a-time wrapper around an isolated recogniser."""

    def __init__(
        self,
        recogniser: Recogniser,
        config: StreamingConfig | None = None,
        *,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.recogniser = recogniser
        self.config = config or StreamingConfig()
        self.feature_config = feature_config
        self.normaliser = StreamingNormaliser(fps=self.config.fps)

        self._buffer: deque[FrameLandmarks] = deque(maxlen=self.config.max_frames)
        self._rest_samples: list[float] = []
        self._rest_level: float | None = None
        self._previous_wrists: np.ndarray | None = None
        self._active = False
        self._quiet = 0
        self._frame_index = 0
        self._last: tuple[str, float] | None = None

    @property
    def is_signing(self) -> bool:
        """Whether a sign is currently in progress."""
        return self._active

    @property
    def rest_level(self) -> float | None:
        """Learned rest energy, or ``None`` while still calibrating."""
        return self._rest_level

    def reset(self) -> None:
        self._buffer.clear()
        self._rest_samples.clear()
        self._rest_level = None
        self._previous_wrists = None
        self._active = False
        self._quiet = 0
        self._last = None
        self.normaliser.reset()

    def push(self, frame: FrameLandmarks) -> SignPrediction | None:
        """Feed one tracked frame; returns a prediction when a sign completes."""
        self._frame_index += 1
        energy = self._frame_energy(frame)

        if self._rest_level is None:
            self._rest_samples.append(energy)
            if len(self._rest_samples) >= self.config.calibration_frames:
                # Median, not mean: if the signer starts moving during calibration,
                # a mean would set the rest level so high that no onset ever fires.
                self._rest_level = max(float(np.median(self._rest_samples)), 1e-4)
            return None

        onset = self._rest_level * self.config.onset_ratio
        offset = self._rest_level * self.config.offset_ratio

        if not self._active:
            if energy > onset:
                self._active = True
                self._quiet = 0
                self._buffer.clear()
                self._buffer.append(frame)
            else:
                # Keep adapting to the room while at rest, so a signer settling into
                # a new position does not leave the threshold stuck at an old value.
                self._rest_level = 0.95 * self._rest_level + 0.05 * energy
            return None

        self._buffer.append(frame)
        self._quiet = self._quiet + 1 if energy < offset else 0

        if self._quiet >= self.config.hold_frames or len(self._buffer) >= self.config.max_frames:
            return self._emit()
        return None

    def flush(self) -> SignPrediction | None:
        """Close any in-progress span, e.g. when the stream ends."""
        return self._emit() if self._active else None

    def _emit(self) -> SignPrediction | None:
        frames = list(self._buffer)
        self._active = False
        self._quiet = 0
        self._buffer.clear()

        if len(frames) < self.config.min_frames:
            return None

        sequence = LandmarkSequence.from_frames(frames, fps=self.config.fps)
        from ..vision.normalise import normalise_sequence

        features, _ = encode_sequence(normalise_sequence(sequence), self.feature_config)
        prediction = self.recogniser.predict(features)

        end = self._frame_index / self.config.fps
        start = end - len(frames) / self.config.fps
        prediction = SignPrediction(
            gloss=prediction.gloss,
            confidence=prediction.confidence,
            start=start,
            end=end,
            alternatives=prediction.alternatives,
        )

        if self._last is not None:
            last_gloss, last_end = self._last
            if last_gloss == prediction.gloss and start - last_end < self.config.debounce:
                return None
        self._last = (prediction.gloss, end)
        return prediction

    def _frame_energy(self, frame: FrameLandmarks) -> float:
        """Wrist movement since the previous frame, in normalised units."""
        normalised = self.normaliser(frame)
        wrists = np.concatenate([normalised.dominant_wrist[0], normalised.weak_wrist[0]])
        if self._previous_wrists is None:
            self._previous_wrists = wrists
            return 0.0
        energy = float(np.linalg.norm(wrists - self._previous_wrists))
        self._previous_wrists = wrists
        return energy
