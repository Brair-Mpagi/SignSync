"""NumPy prototype recogniser — the no-dependency fallback.

Plan §17 requires the system to run local-first on modest hardware, and plan §8.2
starts the model progression at a *baseline* rather than a transformer. This is the
step before that baseline: a nearest-class-mean classifier over standardised,
fixed-length feature sequences, in pure NumPy.

Its job is not accuracy. Its job is to make the whole pipeline — camera to
recognition to translation to speech to avatar — runnable, testable and
demonstrable on a laptop with no ``torch``, no GPU and no network, so that the
integration work of plan Phase 9 is not blocked on the modelling work of Phase 2.
Any accuracy it reports on real data should be read as a floor, not a result
(docs/limitations.md).

Why nearest class mean and not k-NN: prediction cost is O(classes) rather than
O(training clips), which keeps the live path bounded as the corpus grows from 50 to
500+ signs (plan §9.2), and the class means double as an interpretable per-sign
template.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..errors import SignSyncError
from ..vision.features import resample
from .base import (
    UNKNOWN_GLOSS,
    RecogniserConfig,
    SignPrediction,
    Vocabulary,
    fit_temperature,
    top_k,
)

__all__ = ["PrototypeRecogniser"]

_EPS = 1e-8


class PrototypeRecogniser:
    """Nearest class mean over standardised, length-normalised feature sequences."""

    def __init__(self, config: RecogniserConfig | None = None) -> None:
        self.config = config or RecogniserConfig()
        self._vocabulary: Vocabulary | None = None
        self._prototypes: np.ndarray | None = None  # (n_classes, n_frames * D)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._temperature = 1.0
        self.calibrated = False

    @property
    def vocabulary(self) -> Vocabulary:
        if self._vocabulary is None:
            raise SignSyncError("recogniser has not been fitted")
        return self._vocabulary

    @property
    def is_fitted(self) -> bool:
        return self._prototypes is not None

    def fit(
        self,
        sequences: list[np.ndarray],
        labels: list[str],
        vocabulary: Vocabulary | None = None,
    ) -> PrototypeRecogniser:
        """Fit class templates from ``(T, D)`` feature sequences and gloss labels."""
        if len(sequences) != len(labels):
            raise SignSyncError(f"{len(sequences)} sequences but {len(labels)} labels")
        if not sequences:
            raise SignSyncError("cannot fit on an empty training set")

        vocab = vocabulary or Vocabulary(tuple(sorted(set(labels))))
        matrix = np.stack([self._flatten(s) for s in sequences])

        # Standardise per feature. Sign features live on wildly different scales —
        # normalised landmark coordinates around 1.0, joint angles in radians,
        # velocities an order of magnitude smaller — and a Euclidean distance over
        # unstandardised features is dominated by whichever block happens to be
        # largest rather than by which one distinguishes the signs.
        self._mean = matrix.mean(axis=0)
        self._scale = matrix.std(axis=0)
        self._scale[self._scale < _EPS] = 1.0
        standardised = (matrix - self._mean) / self._scale

        prototypes = np.zeros((len(vocab), standardised.shape[1]), dtype=np.float32)
        counts = np.zeros(len(vocab), dtype=np.int64)
        for row, label in zip(standardised, labels, strict=True):
            index = vocab.index(label)
            prototypes[index] += row
            counts[index] += 1

        missing = [vocab.gloss(i) for i, c in enumerate(counts) if c == 0]
        if missing:
            raise SignSyncError(
                f"no training examples for gloss(es) {missing}; either drop them from the "
                "vocabulary or collect examples (an unseen class can never be predicted)"
            )
        prototypes /= counts[:, None]
        self._prototypes = prototypes
        self._vocabulary = vocab

        # Provisional temperature from the training set's own distance spread, so
        # the model is usable before calibration. Without any scaling the distances
        # are large enough that softmax saturates and every prediction reads as
        # 100% confident. Call :meth:`calibrate` with held-out signers to replace
        # this with an honest value — the training set cannot provide one.
        distances = self._distances(standardised)
        spread = float(np.median(np.ptp(distances, axis=1)))
        self._temperature = max(spread / 4.0, _EPS)
        self.calibrated = False
        return self

    def calibrate(self, sequences: list[np.ndarray], labels: list[str]) -> float:
        """Rescale confidences using held-out signers. Returns the new temperature.

        See :func:`~signsync.recognition.base.fit_temperature`. Pass the validation
        split — calibrating on training clips reproduces the overconfidence this is
        meant to remove, and the abstention threshold then never fires on the
        signers it exists to protect.
        """
        if self._prototypes is None or self._mean is None or self._scale is None:
            raise SignSyncError("recogniser has not been fitted")
        if not sequences:
            raise SignSyncError("cannot calibrate on an empty set")

        matrix = np.stack([self._flatten(s) for s in sequences])
        standardised = (matrix - self._mean) / self._scale
        scores = -self._distances(standardised)
        indices = np.array([self.vocabulary.index(label) for label in labels], dtype=np.int64)

        self._temperature = max(fit_temperature(scores, indices), _EPS)
        self.calibrated = True
        return self._temperature

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._prototypes is None or self._mean is None or self._scale is None:
            raise SignSyncError("recogniser has not been fitted")
        standardised = (self._flatten(features)[None, :] - self._mean) / self._scale
        distances = self._distances(standardised)[0]
        logits = -distances / self._temperature
        logits -= logits.max()
        exponentiated = np.exp(logits)
        return (exponentiated / exponentiated.sum()).astype(np.float32)

    def predict(self, features: np.ndarray) -> SignPrediction:
        probabilities = self.predict_proba(features)
        alternatives = top_k(probabilities, self.vocabulary, self.config.top_k)
        gloss, confidence = alternatives[0]
        if confidence < self.config.min_confidence:
            return SignPrediction(
                gloss=UNKNOWN_GLOSS, confidence=confidence, alternatives=alternatives
            )
        return SignPrediction(gloss=gloss, confidence=confidence, alternatives=alternatives)

    def _flatten(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise SignSyncError(f"expected (T, D) features, got shape {features.shape}")
        return resample(features, self.config.n_frames).ravel()

    def _distances(self, standardised: np.ndarray) -> np.ndarray:
        assert self._prototypes is not None
        # ||a - b||^2 expanded, so the whole batch is one matrix product.
        a2 = (standardised**2).sum(axis=1)[:, None]
        b2 = (self._prototypes**2).sum(axis=1)[None, :]
        cross = standardised @ self._prototypes.T
        return np.sqrt(np.maximum(a2 + b2 - 2 * cross, 0.0))

    def save(self, path: str | Path) -> Path:
        if self._prototypes is None or self._mean is None or self._scale is None:
            raise SignSyncError("cannot save an unfitted recogniser")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            prototypes=self._prototypes,
            mean=self._mean,
            scale=self._scale,
            temperature=np.float32(self._temperature),
            calibrated=np.bool_(self.calibrated),
            glosses=np.array(list(self.vocabulary.glosses)),
            config=np.array(
                json.dumps(
                    {
                        "n_frames": self.config.n_frames,
                        "min_confidence": self.config.min_confidence,
                        "top_k": self.config.top_k,
                        "feature_config": self.config.feature_config,
                    }
                )
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> PrototypeRecogniser:
        with np.load(Path(path), allow_pickle=False) as data:
            config = RecogniserConfig(**json.loads(str(data["config"])))
            model = cls(config)
            model._prototypes = data["prototypes"]
            model._mean = data["mean"]
            model._scale = data["scale"]
            model._temperature = float(data["temperature"])
            model.calibrated = bool(data["calibrated"]) if "calibrated" in data else False
            model._vocabulary = Vocabulary(tuple(str(g) for g in data["glosses"]))
        return model
