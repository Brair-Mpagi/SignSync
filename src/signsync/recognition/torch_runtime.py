"""Training loop and inference wrapper for the PyTorch models.

Separated from :mod:`signsync.recognition.torch_models` so architectures stay
readable on their own, and imported lazily by :mod:`signsync.recognition.train` so
the package keeps working without the ``models`` extra.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..errors import SignSyncError
from ..vision.features import resample
from .base import UNKNOWN_GLOSS, RecogniserConfig, SignPrediction, Vocabulary, top_k
from .dataset import FeatureSet
from .torch_models import ModelConfig, build_model, torch

__all__ = ["TorchTrainingConfig", "TorchRecogniser", "fit_torch_model"]

nn = torch.nn


@dataclass(frozen=True)
class TorchTrainingConfig:
    """Optimisation settings."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    """Epochs without validation improvement before stopping. Early stopping is on
    by default because signer-independent validation is the only signal that
    separates learning the signs from learning the signers."""

    label_smoothing: float = 0.05
    seed: int = 0
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


class TorchRecogniser:
    """Inference wrapper implementing the :class:`~signsync.recognition.base.Recogniser` protocol."""

    def __init__(
        self,
        model,  # nn.Module
        vocabulary: Vocabulary,
        config: RecogniserConfig,
        *,
        model_name: str,
        model_config: ModelConfig,
        normalisation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.model = model.eval()
        self._vocabulary = vocabulary
        self.config = config
        self.model_name = model_name
        self.model_config = model_config
        self.normalisation = normalisation

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _prepare(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise SignSyncError(f"expected (T, D) features, got shape {features.shape}")
        prepared = resample(features, self.config.n_frames)
        if self.normalisation is not None:
            mean, scale = self.normalisation
            prepared = (prepared - mean) / scale
        return prepared

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(self._prepare(features)[None, ...])
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=-1)[0]
        return probabilities.cpu().numpy().astype(np.float32)

    def predict(self, features: np.ndarray) -> SignPrediction:
        probabilities = self.predict_proba(features)
        alternatives = top_k(probabilities, self.vocabulary, self.config.top_k)
        gloss, confidence = alternatives[0]
        if confidence < self.config.min_confidence:
            return SignPrediction(UNKNOWN_GLOSS, confidence, alternatives=alternatives)
        return SignPrediction(gloss, confidence, alternatives=alternatives)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "vocabulary": list(self.vocabulary.glosses),
            "model_name": self.model_name,
            "model_config": asdict(self.model_config),
            "recogniser_config": {
                "n_frames": self.config.n_frames,
                "min_confidence": self.config.min_confidence,
                "top_k": self.config.top_k,
                "feature_config": self.config.feature_config,
            },
            "normalisation": None
            if self.normalisation is None
            else [self.normalisation[0], self.normalisation[1]],
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, **build_kwargs) -> TorchRecogniser:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        model_config = ModelConfig(**payload["model_config"])
        model = build_model(payload["model_name"], model_config, **build_kwargs)
        model.load_state_dict(payload["state_dict"])
        normalisation = payload.get("normalisation")
        return cls(
            model,
            Vocabulary(tuple(payload["vocabulary"])),
            RecogniserConfig(**payload["recogniser_config"]),
            model_name=payload["model_name"],
            model_config=model_config,
            normalisation=None
            if normalisation is None
            else (np.asarray(normalisation[0]), np.asarray(normalisation[1])),
        )


def _stack(feature_set: FeatureSet, vocabulary: Vocabulary, n_frames: int):
    x = np.stack([resample(s, n_frames) for s in feature_set.sequences]).astype(np.float32)
    y = np.array([vocabulary.index(label) for label in feature_set.labels], dtype=np.int64)
    return x, y


def fit_torch_model(
    train_set: FeatureSet,
    val_set: FeatureSet | None,
    *,
    model_name: str = "lstm",
    recogniser_config: RecogniserConfig | None = None,
    model_config: ModelConfig | None = None,
    training: TorchTrainingConfig | None = None,
    streams: dict[str, slice] | None = None,
    vocabulary: Vocabulary | None = None,
) -> tuple[TorchRecogniser, dict]:
    """Train one model and return it with its training history."""
    recogniser_config = recogniser_config or RecogniserConfig()
    training = training or TorchTrainingConfig()
    vocab = vocabulary or train_set.vocabulary or Vocabulary(tuple(sorted(set(train_set.labels))))

    torch.manual_seed(training.seed)
    np.random.seed(training.seed)

    x_train, y_train = _stack(train_set, vocab, recogniser_config.n_frames)

    # Standardise using training statistics only. Computing them over the whole
    # corpus would leak held-out signers' feature distributions into training, which
    # is precisely the signer information the split exists to withhold.
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    scale = x_train.std(axis=(0, 1), keepdims=True)
    scale[scale < 1e-8] = 1.0
    x_train = (x_train - mean) / scale

    config = model_config or ModelConfig(
        input_dim=x_train.shape[-1], n_classes=len(vocab)
    )
    device = torch.device(training.resolve_device())
    model = build_model(model_name, config, streams=streams).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=training.label_smoothing)

    train_x = torch.from_numpy(x_train).to(device)
    train_y = torch.from_numpy(y_train).to(device)

    val_tensors = None
    if val_set is not None and len(val_set) > 0:
        x_val, y_val = _stack(val_set, vocab, recogniser_config.n_frames)
        x_val = (x_val - mean) / scale
        val_tensors = (torch.from_numpy(x_val).to(device), torch.from_numpy(y_val).to(device))

    history: dict[str, list[float]] = {"loss": [], "val_accuracy": []}
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score, stale = -1.0, 0

    for _ in range(training.epochs):
        model.train()
        permutation = torch.randperm(len(train_x), device=device)
        epoch_loss = 0.0
        for start in range(0, len(permutation), training.batch_size):
            batch = permutation[start : start + training.batch_size]
            optimiser.zero_grad()
            loss = criterion(model(train_x[batch]), train_y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += float(loss) * len(batch)
        history["loss"].append(epoch_loss / len(train_x))

        if val_tensors is None:
            continue
        model.eval()
        with torch.no_grad():
            predictions = model(val_tensors[0]).argmax(dim=-1)
            accuracy = float((predictions == val_tensors[1]).float().mean())
        history["val_accuracy"].append(accuracy)

        if accuracy > best_score:
            best_score, stale = accuracy, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= training.patience:
                break

    model.load_state_dict(best_state)
    recogniser = TorchRecogniser(
        model.cpu(),
        vocab,
        recogniser_config,
        model_name=model_name,
        model_config=config,
        normalisation=(mean[0], scale[0]),
    )
    return recogniser, {
        "history": history,
        "best_val_accuracy": best_score if val_tensors is not None else None,
        "epochs_run": len(history["loss"]),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def training_report(report: dict) -> str:
    return json.dumps(report, indent=2, default=float)
