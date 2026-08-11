"""Backend-agnostic training entry point.

Owns the parts of training that must not depend on which model is being trained:
the split is validated on every run, the vocabulary comes from the *training* side
only, and evaluation happens on held-out signers.

``backend="prototype"`` uses the NumPy fallback and needs nothing beyond the core
dependency; any other backend name is a PyTorch model from
:mod:`signsync.recognition.torch_models` and requires the ``models`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datasets.augment import AugmentationPolicy
from ..datasets.corpus import CorpusLoader
from ..datasets.splits import Split, signer_independent_split, validate_split
from ..errors import SignSyncError
from ..vision.features import FeatureConfig
from .base import Recogniser, RecogniserConfig, Vocabulary
from .dataset import FeatureSet, feature_sets_for_split
from .prototype import PrototypeRecogniser

__all__ = ["TrainingRun", "TrainingResult", "train", "evaluate", "train_from_corpus"]


@dataclass
class TrainingRun:
    """Everything that determines a training run, recorded so it can be repeated."""

    backend: str = "prototype"
    recogniser: RecogniserConfig = field(default_factory=RecogniserConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    augmentations: int = 2
    augmentation_policy: AugmentationPolicy = field(default_factory=AugmentationPolicy)
    seed: int = 0
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """A fitted recogniser plus how it did on held-out signers."""

    recogniser: Recogniser
    vocabulary: Vocabulary
    split: Split | None
    train_accuracy: float
    val_accuracy: float | None
    test_accuracy: float | None
    report: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"backend      : {self.report.get('backend', 'unknown')}",
            f"vocabulary   : {len(self.vocabulary)} glosses",
            f"train acc    : {self.train_accuracy:.1%}",
        ]
        if self.val_accuracy is not None:
            lines.append(f"val acc      : {self.val_accuracy:.1%}  (held-out signers)")
        if self.test_accuracy is not None:
            lines.append(f"test acc     : {self.test_accuracy:.1%}  (held-out signers)")
        if self.split is not None:
            lines.append(f"split        : {self.split.summary()}")
        return "\n".join(lines)


def evaluate(recogniser: Recogniser, feature_set: FeatureSet) -> float:
    """Top-1 accuracy, counting a low-confidence ``<unknown>`` as wrong.

    Counting abstentions as errors is the honest accounting: a system that abstains
    on everything is not a correct system, it is an unusable one. Precision on the
    non-abstained subset is a separate number, reported by
    :mod:`signsync.evaluation`.
    """
    if len(feature_set) == 0:
        raise SignSyncError("cannot evaluate on an empty feature set")
    correct = sum(
        recogniser.predict(sequence).gloss == label
        for sequence, label in zip(feature_set.sequences, feature_set.labels, strict=True)
    )
    return correct / len(feature_set)


def train(
    feature_sets: dict[str, FeatureSet],
    run: TrainingRun | None = None,
) -> TrainingResult:
    """Fit a recogniser on prepared feature sets."""
    run = run or TrainingRun()
    if "train" not in feature_sets:
        raise SignSyncError("feature_sets must contain a 'train' entry")

    train_set = feature_sets["train"]
    val_set = feature_sets.get("val")
    test_set = feature_sets.get("test")

    # The vocabulary is the training side's. A gloss the model was never trained on
    # cannot be predicted, so including it would silently add an always-wrong class
    # and quietly depress every reported metric.
    vocabulary = Vocabulary(tuple(sorted(set(train_set.labels))))

    report: dict[str, Any] = {"backend": run.backend, "train_size": len(train_set)}

    if run.backend == "prototype":
        prototype = PrototypeRecogniser(run.recogniser).fit(
            train_set.sequences, train_set.labels, vocabulary
        )
        if val_set is not None and len(val_set) > 0:
            # Calibrate confidences on held-out signers. Without this the model
            # ranks signs correctly but reports confidences fitted to signers it has
            # already seen, so the abstention threshold either rejects almost every
            # correct prediction on a new signer or accepts almost every wrong one —
            # and the confidence shown to the user means nothing either way.
            known = [
                (sequence, label)
                for sequence, label in zip(val_set.sequences, val_set.labels, strict=True)
                if label in vocabulary
            ]
            if known:
                report["temperature"] = prototype.calibrate(
                    [s for s, _ in known], [label for _, label in known]
                )
        report["calibrated"] = prototype.calibrated
        recogniser: Recogniser = prototype
    else:
        from .torch_models import ModelConfig
        from .torch_runtime import TorchTrainingConfig, fit_torch_model

        model_kwargs = dict(run.model_kwargs)
        training_config = model_kwargs.pop("training", None) or TorchTrainingConfig(seed=run.seed)
        model_config = model_kwargs.pop("model_config", None) or ModelConfig(
            input_dim=train_set.feature_dim, n_classes=len(vocabulary)
        )
        recogniser, torch_report = fit_torch_model(
            train_set,
            val_set,
            model_name=run.backend,
            recogniser_config=run.recogniser,
            model_config=model_config,
            training=training_config,
            vocabulary=vocabulary,
            **model_kwargs,
        )
        report.update(torch_report)

    unseen = {
        name: sorted(set(feature_sets[name].labels) - set(vocabulary.glosses))
        for name in ("val", "test")
        if name in feature_sets
    }
    report["glosses_unseen_in_training"] = {k: v for k, v in unseen.items() if v}

    return TrainingResult(
        recogniser=recogniser,
        vocabulary=vocabulary,
        split=None,
        train_accuracy=evaluate(recogniser, train_set),
        val_accuracy=evaluate(recogniser, val_set) if val_set else None,
        test_accuracy=evaluate(recogniser, test_set) if test_set else None,
        report=report,
    )


def train_from_corpus(
    loader: CorpusLoader,
    *,
    run: TrainingRun | None = None,
    split: Split | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    save_to: str | Path | None = None,
) -> TrainingResult:
    """Split a corpus by signer, encode it, train, and report on held-out signers.

    The split is validated here even when supplied by the caller: a split file
    edited months later must fail loudly rather than quietly inflate an accuracy
    number (plan §14, Risk 2).
    """
    run = run or TrainingRun()
    records = loader.permitted_clips()
    isolated = [r for r in records if not r.is_continuous]
    if not isolated:
        raise SignSyncError("no consented isolated clips available to train on")

    if split is None:
        split = signer_independent_split(
            loader.corpus, ratios=ratios, seed=run.seed, records=isolated
        )
    else:
        validate_split(loader.corpus, split, records=isolated)

    feature_sets = feature_sets_for_split(
        loader,
        split,
        feature_config=run.feature_config,
        augmentations=run.augmentations,
        policy=run.augmentation_policy,
        seed=run.seed,
    )
    result = train(feature_sets, run)
    result.split = split
    result.report["signers"] = {name: list(ids) for name, ids in split.signers.items()}

    if save_to is not None:
        _save(result, save_to)
    return result


def _save(result: TrainingResult, path: str | Path) -> Path:
    target = Path(path)
    saver = getattr(result.recogniser, "save", None)
    if saver is None:
        raise SignSyncError(f"{type(result.recogniser).__name__} cannot be saved")
    return saver(target)


def confusion_pairs(
    recogniser: Recogniser, feature_set: FeatureSet, *, limit: int = 10
) -> list[tuple[str, str, int]]:
    """Most frequent ``(true, predicted, count)`` confusions.

    Plan §15 asks for a per-sign confusion matrix specifically to catch
    systematically confused sign *pairs* — near-minimal pairs that differ only in
    handshape or movement are where a landmark model fails, and an aggregate
    accuracy number cannot show them.
    """
    counts: dict[tuple[str, str], int] = {}
    for sequence, label in zip(feature_set.sequences, feature_set.labels, strict=True):
        predicted = recogniser.predict(sequence).gloss
        if predicted != label:
            key = (label, predicted)
            counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return [(true, predicted, count) for (true, predicted), count in ordered]
