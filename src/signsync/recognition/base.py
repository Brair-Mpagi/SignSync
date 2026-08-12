"""Recognition contracts: vocabulary, predictions, and the recogniser interface.

Plan §8.2 stages the models deliberately (LSTM → TCN → transformer → multimodal),
so the pipeline must not be written against any one of them. Everything downstream
depends on :class:`Recogniser` and :class:`SignPrediction` only.

One decision encoded here rather than left to callers: a recogniser that is not
confident returns :data:`UNKNOWN_GLOSS`, not its best guess. Plan §16.3 requires the
system to be transparent about its limits in-product, and a confidently-presented
wrong sign in a clinic is worse than an honest "I did not catch that" — the user can
repeat a sign, but cannot detect a fluent mistranslation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import SignSyncError

__all__ = [
    "UNKNOWN_GLOSS",
    "Vocabulary",
    "SignPrediction",
    "Recogniser",
    "RecogniserConfig",
    "fit_temperature",
    "top_k",
]

#: Emitted when no gloss clears the confidence threshold.
UNKNOWN_GLOSS = "<unknown>"


@dataclass(frozen=True)
class Vocabulary:
    """Ordered gloss list with index lookup."""

    glosses: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.glosses:
            raise SignSyncError("vocabulary cannot be empty")
        if len(set(self.glosses)) != len(self.glosses):
            duplicates = sorted({g for g in self.glosses if self.glosses.count(g) > 1})
            raise SignSyncError(f"duplicate glosses in vocabulary: {duplicates}")

    def __len__(self) -> int:
        return len(self.glosses)

    def __iter__(self):
        return iter(self.glosses)

    def __contains__(self, gloss: object) -> bool:
        return gloss in self.glosses

    def index(self, gloss: str) -> int:
        try:
            return self.glosses.index(gloss)
        except ValueError:
            raise SignSyncError(f"gloss {gloss!r} is not in the vocabulary") from None

    def gloss(self, index: int) -> str:
        if not 0 <= index < len(self.glosses):
            raise SignSyncError(f"index {index} outside vocabulary of {len(self.glosses)}")
        return self.glosses[index]

    def encode(self, glosses: list[str]) -> np.ndarray:
        return np.array([self.index(g) for g in glosses], dtype=np.int64)

    def decode(self, indices: np.ndarray) -> list[str]:
        return [self.gloss(int(i)) for i in indices]

    @classmethod
    def from_corpus(cls, corpus) -> Vocabulary:
        return cls(tuple(corpus.vocabulary()))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self.glosses), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        return cls(tuple(json.loads(Path(path).read_text(encoding="utf-8"))))


@dataclass(frozen=True)
class SignPrediction:
    """One recognised sign, with the evidence for it.

    ``alternatives`` is carried through the pipeline because a downstream translator
    can often resolve a near-tie from context that the recogniser has no access to —
    "PAIN" and "PAINT" look similar and are not equally likely after "ME SICK".
    """

    gloss: str
    confidence: float
    start: float = 0.0
    end: float = 0.0
    alternatives: tuple[tuple[str, float], ...] = ()

    @property
    def is_unknown(self) -> bool:
        return self.gloss == UNKNOWN_GLOSS

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def __str__(self) -> str:
        return f"{self.gloss} ({self.confidence:.0%})"


@runtime_checkable
class Recogniser(Protocol):
    """Maps a feature sequence to a sign."""

    @property
    def vocabulary(self) -> Vocabulary: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for one ``(T, D)`` clip."""
        ...

    def predict(self, features: np.ndarray) -> SignPrediction:
        """Best gloss for one ``(T, D)`` clip, or ``UNKNOWN_GLOSS`` if unsure."""
        ...


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    smoothing: float = 0.05,
) -> float:
    """Temperature that makes ``softmax(scores / T)`` honest about being wrong.

    ``scores`` are ``(n, n_classes)`` with higher meaning more likely — logits for a
    neural model, negative distances for the prototype recogniser.

    Why this exists: a model's raw confidences are calibrated on the data it was fit
    to, where it is nearly always right, so it reports high confidence on held-out
    signers too — right up until it is wrong about them. The confidence number is
    not decorative here. Plan §16.3 puts it in front of users as the signal for when
    to distrust a translation, and :data:`UNKNOWN_GLOSS` abstention is driven by it,
    so an uncalibrated model either abstains on everything or on nothing.

    Fit this on **held-out signers** (the validation split), never on training data —
    calibrating on training data reproduces exactly the overconfidence it is meant
    to correct.

    ``smoothing`` keeps the fit well-posed. Against hard targets, a calibration set
    the model gets entirely right has its likelihood maximised as the temperature
    goes to zero: the fit runs away to "always 100% confident", which is the failure
    this function exists to prevent. Smoothed targets put the optimum at a finite
    temperature whose top-1 confidence lands near ``1 - smoothing``, leaving the
    abstention threshold something to work with. Small validation sets are the norm
    at plan §9.2's early corpus sizes, so the degenerate case is the common one, not
    a corner.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or len(scores) != len(labels):
        raise SignSyncError(
            f"expected (n, n_classes) scores matching {len(labels)} labels, got {scores.shape}"
        )
    if not 0.0 <= smoothing < 1.0:
        raise SignSyncError(f"smoothing must be in [0, 1), got {smoothing}")
    if len(scores) == 0:
        return 1.0

    candidates = grid if grid is not None else np.geomspace(0.05, 20.0, 80)
    rows = np.arange(len(labels))
    n_classes = scores.shape[1]

    best_temperature, best_loss = 1.0, np.inf
    for temperature in candidates:
        scaled = scores / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probabilities = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        loss = -(
            (1.0 - smoothing) * log_probabilities[rows, labels]
            + (smoothing / n_classes) * log_probabilities.sum(axis=1)
        ).mean()
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), float(loss)
    return best_temperature


def top_k(
    probabilities: np.ndarray, vocabulary: Vocabulary, k: int = 3
) -> tuple[tuple[str, float], ...]:
    """Top ``k`` ``(gloss, probability)`` pairs, highest first."""
    if probabilities.ndim != 1:
        raise SignSyncError(f"expected a 1-D probability vector, got shape {probabilities.shape}")
    order = np.argsort(probabilities)[::-1][:k]
    return tuple((vocabulary.gloss(int(i)), float(probabilities[i])) for i in order)


@dataclass
class RecogniserConfig:
    """Settings shared by every recogniser implementation."""

    n_frames: int = 32
    """Frames each clip is resampled to. Signing speed varies enough between
    signers that padding to a fixed length would make tempo a feature."""

    min_confidence: float = 0.45
    """Below this, emit ``UNKNOWN_GLOSS`` rather than a guess."""

    top_k: int = 3
    feature_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_frames < 2:
            raise SignSyncError(f"n_frames must be at least 2, got {self.n_frames}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise SignSyncError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
