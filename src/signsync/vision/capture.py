"""Camera capture (plan §8.1).

Thin wrapper over OpenCV that yields ``(frame, timestamp)`` pairs and measures its
own throughput, so objective O1 (≥25 FPS on a mid-range laptop CPU, no GPU) can be
checked on the deployment hardware instead of assumed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np

from ..capabilities import require
from ..perf import FpsMeter

__all__ = ["CameraSettings", "CameraStream"]


@dataclass(frozen=True)
class CameraSettings:
    """Requested capture settings. A camera may not honour all of them."""

    device: int | str = 0
    width: int = 640
    height: int = 480
    fps: float = 30.0
    mirror: bool = True
    """Flip horizontally for a front-facing camera.

    This is a *display* convenience so signers see themselves as in a mirror. It is
    applied before tracking, which is fine because the tracker labels hands from
    body context rather than image side — but note it must never be used as data
    augmentation: mirroring a sign swaps the dominant hand and can change the sign
    (plan §9.4).
    """


class CameraStream:
    """Iterable camera source with an FPS meter attached."""

    def __init__(self, settings: CameraSettings | None = None) -> None:
        self.settings = settings or CameraSettings()
        self._cv2 = require("opencv", feature="camera capture")
        self._capture: Any | None = None
        self.fps_meter = FpsMeter()

    def open(self) -> CameraStream:
        cv2 = self._cv2
        capture = cv2.VideoCapture(self.settings.device)
        if not capture.isOpened():
            raise OSError(f"could not open camera device {self.settings.device!r}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
        self._capture = capture
        return self

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        start = time.perf_counter()
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            if self.settings.mirror:
                frame = self._cv2.flip(frame, 1)
            self.fps_meter.tick()
            yield frame, time.perf_counter() - start

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> CameraStream:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
