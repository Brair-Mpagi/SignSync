"""Corpus to training arrays.

Sits between :mod:`signsync.datasets` (which owns consent and splitting) and the
models (which own architecture). It never bypasses the loader, so consent gating
applies to every array a model ever sees.

Augmentation is applied to the training side only, and the caller cannot switch that
off — augmenting evaluation data inflates the score by testing on distortions of the
same clips, which is the kind of quiet self-deception plan §15 is written to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..datasets.augment import AugmentationPolicy, augment
from ..datasets.corpus import ClipData, CorpusLoader
from ..datasets.schema import ClipRecord
from ..datasets.splits import Split
from ..errors import SignSyncError
from ..vision.features import FeatureConfig, encode_sequence
from ..vision.normalise import normalise_sequence
from .base import Vocabulary

__all__ = ["FeatureSet", "build_feature_set", "encode_clip"]


@dataclass
class FeatureSet:
    """Feature sequences with their labels and provenance.

    ``signers`` is kept alongside ``labels`` so that any downstream grouping —
    cross-validation, per-signer error analysis — can stay signer-aware without
    going back to the manifest.
    """

    sequences: list[np.ndarray]
    labels: list[str]
    clip_ids: list[str]
    signers: list[str]
    feature_dim: int
    vocabulary: Vocabulary | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sequences)

    def __post_init__(self) -> None:
        lengths = {len(self.sequences), len(self.labels), len(self.clip_ids), len(self.signers)}
        if len(lengths) > 1:
            raise SignSyncError(f"FeatureSet fields have mismatched lengths: {sorted(lengths)}")

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def signer_ids(self) -> list[str]:
        return sorted(set(self.signers))


def encode_clip(
    data: ClipData,
    feature_config: FeatureConfig | None = None,
    *,
    dominant: str = "auto",
) -> np.ndarray:
    """Landmarks to ``(T, D)`` features for one clip.

    ``dominant`` should come from the signer's profile rather than be detected —
    see :func:`~signsync.vision.normalise.detect_dominant_hand` for why per-clip
    detection makes one signer's clips disagree with each other.
    """
    normalised = normalise_sequence(data.sequence, dominant=dominant)
    features, _ = encode_sequence(normalised, feature_config)
    return features


def build_feature_set(
    loader: CorpusLoader,
    records: list[ClipRecord],
    *,
    feature_config: FeatureConfig | None = None,
    augmentations: int = 0,
    policy: AugmentationPolicy | None = None,
    seed: int = 0,
    isolated_only: bool = True,
) -> FeatureSet:
    """Encode a set of clips, optionally with augmented copies.

    ``augmentations`` copies are added *per clip* and only make sense for a training
    split; callers building an evaluation set must leave it at zero. The clip ids of
    augmented copies are suffixed so per-clip error analysis can tell them apart from
    the originals.
    """
    if augmentations < 0:
        raise SignSyncError(f"augmentations must be non-negative, got {augmentations}")

    chosen = [r for r in records if not (isolated_only and r.is_continuous)]
    if not chosen:
        raise SignSyncError(
            "no clips to encode"
            + (" (all candidates are continuous; pass isolated_only=False)" if records else "")
        )

    rng = np.random.default_rng(seed)
    policy = policy or AugmentationPolicy()

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    clip_ids: list[str] = []
    signers: list[str] = []
    skipped: list[tuple[str, str]] = []

    for data in loader.load_all(chosen):
        record = data.record
        label = record.glosses[0] if isolated_only else "|".join(record.glosses)
        # Handedness comes from the signer's profile, not from the clip. A
        # left-handed signer mirrors every sign, and canonicalising per clip would
        # let one signer's clips disagree with each other.
        profile = loader.corpus.signers.get(record.signer_id)
        dominant = profile.handedness.value if profile is not None else "auto"

        sequences.append(encode_clip(data, feature_config, dominant=dominant))
        labels.append(label)
        clip_ids.append(record.clip_id)
        signers.append(record.signer_id)

        for i in range(augmentations):
            varied = augment(data.sequence, policy, rng=rng)
            sequences.append(
                encode_clip(
                    ClipData(record=record, sequence=varied), feature_config, dominant=dominant
                )
            )
            labels.append(label)
            clip_ids.append(f"{record.clip_id}#aug{i}")
            signers.append(record.signer_id)

    skipped = list(getattr(loader, "rejected", []))
    if not sequences:
        raise SignSyncError(
            "every candidate clip was rejected by consent or quality gates: "
            + "; ".join(f"{cid}: {reason}" for cid, reason in skipped[:5])
        )

    return FeatureSet(
        sequences=sequences,
        labels=labels,
        clip_ids=clip_ids,
        signers=signers,
        feature_dim=sequences[0].shape[1],
        vocabulary=Vocabulary(tuple(sorted(set(labels)))),
        meta={"augmentations": augmentations, "skipped": skipped},
    )


def feature_sets_for_split(
    loader: CorpusLoader,
    split: Split,
    *,
    feature_config: FeatureConfig | None = None,
    augmentations: int = 2,
    policy: AugmentationPolicy | None = None,
    seed: int = 0,
) -> dict[str, FeatureSet]:
    """Build train/val/test feature sets, augmenting the training side only."""
    by_id = {c.clip_id: c for c in loader.corpus.clips}
    result: dict[str, FeatureSet] = {}
    for name in ("train", "val", "test"):
        clip_ids = split[name]
        if not clip_ids:
            continue
        result[name] = build_feature_set(
            loader,
            [by_id[cid] for cid in clip_ids],
            feature_config=feature_config,
            augmentations=augmentations if name == "train" else 0,
            policy=policy,
            seed=seed,
        )
    return result
