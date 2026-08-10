"""Train/validation/test splitting, signer-independent by construction.

Plan §14 lists "poor generalisation to unseen signers" as a high-likelihood,
high-impact risk, and plan §8.3 requires splitting "by signer identity, never by
clip". The failure this prevents is silent: a clip-level split lets the same signer
appear on both sides, the model learns that signer's geometry and tempo, accuracy
looks excellent, and the system then fails on the first person it meets in a clinic.

So the split functions here refuse to produce an overlapping split, and
:func:`validate_split` is called by the training pipeline rather than left to
reviewer discipline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..errors import SplitError
from .schema import ClipRecord, Corpus

__all__ = ["Split", "signer_independent_split", "signer_kfold", "validate_split", "SPLIT_NAMES"]

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """Clip identifiers per side, plus the signers each side owns."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    signers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    seed: int = 0

    def __getitem__(self, name: str) -> tuple[str, ...]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)  # type: ignore[no-any-return]

    def sizes(self) -> dict[str, int]:
        return {name: len(self[name]) for name in SPLIT_NAMES}

    def summary(self) -> str:
        parts = [
            f"{name}: {len(self[name])} clips / {len(self.signers.get(name, ()))} signers"
            for name in SPLIT_NAMES
        ]
        return " | ".join(parts)


def signer_independent_split(
    corpus: Corpus,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    records: list[ClipRecord] | None = None,
    require_full_vocabulary: bool = True,
) -> Split:
    """Split a corpus so that no signer appears on more than one side.

    Signers are assigned whole, largest first, each to whichever side is furthest
    below its target share. Greedy rather than random assignment because signer clip
    counts are usually very uneven — one enthusiastic participant can contribute a
    third of the corpus, and a random assignment would then miss the requested
    ratios by a wide margin.

    ``require_full_vocabulary`` fails the split when a gloss exists only outside the
    training set. That situation is not necessarily wrong (few-shot evaluation is a
    legitimate design), but it is almost always an accident, and it makes evaluation
    numbers uninterpretable.
    """
    clips = list(records if records is not None else corpus.clips)
    if not clips:
        raise SplitError("cannot split an empty corpus")
    if len(ratios) != 3 or any(r < 0 for r in ratios):
        raise SplitError(f"ratios must be three non-negative numbers, got {ratios}")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise SplitError("ratios must sum to more than zero")
    normalised = tuple(r / total_ratio for r in ratios)

    by_signer: dict[str, list[ClipRecord]] = {}
    for clip in clips:
        by_signer.setdefault(clip.signer_id, []).append(clip)

    wanted_sides = [name for name, ratio in zip(SPLIT_NAMES, normalised, strict=True) if ratio > 0]
    if len(by_signer) < len(wanted_sides):
        raise SplitError(
            f"{len(by_signer)} signer(s) cannot fill {len(wanted_sides)} non-empty splits "
            f"({', '.join(wanted_sides)}). Recruit more signers (plan §9.3), or evaluate on a "
            "corpus from a separate collection round — do not fall back to a clip-level split."
        )

    order = sorted(by_signer, key=lambda s: (-len(by_signer[s]), s))
    rng = random.Random(seed)
    # Shuffle within equal clip counts so the seed has an effect without letting a
    # large signer land in a small split.
    rng.shuffle(order)
    order.sort(key=lambda s: -len(by_signer[s]))

    n_total = len(clips)
    targets = dict(zip(SPLIT_NAMES, (r * n_total for r in normalised), strict=True))
    assigned: dict[str, list[ClipRecord]] = {name: [] for name in SPLIT_NAMES}
    signers: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    def place(signer: str, side: str) -> None:
        assigned[side].extend(by_signer[signer])
        signers[side].append(signer)

    # Seed every requested side with one signer before distributing the rest.
    # Purely greedy assignment leaves the smallest side empty whenever the largest
    # side's deficit stays bigger than the small side's target — which happens for
    # any corpus with roughly as many signers as splits, i.e. exactly the corpus
    # sizes plan §9.2 expects early on. The smallest sides claim the smallest
    # signers, so seeding costs the ratios as little as possible.
    remaining = list(order)
    for side in sorted(wanted_sides, key=lambda name: targets[name]):
        if remaining:
            place(remaining.pop(), side)

    for signer in remaining:
        side = min(
            wanted_sides,
            key=lambda name: (len(assigned[name]) - targets[name], name),
        )
        place(signer, side)

    split = Split(
        train=tuple(c.clip_id for c in assigned["train"]),
        val=tuple(c.clip_id for c in assigned["val"]),
        test=tuple(c.clip_id for c in assigned["test"]),
        signers={name: tuple(sorted(signers[name])) for name in SPLIT_NAMES},
        seed=seed,
    )
    validate_split(corpus, split, require_full_vocabulary=require_full_vocabulary, records=clips)
    return split


def signer_kfold(
    corpus: Corpus, *, k: int = 5, seed: int = 0, records: list[ClipRecord] | None = None
) -> list[Split]:
    """Leave-signers-out cross-validation folds.

    With the corpus sizes plan §9.2 anticipates for V1/V2, a single held-out signer
    set is small enough that one unusual signer dominates the reported accuracy.
    Cross-validation over signer groups gives a spread instead of a single number
    that is mostly luck.
    """
    clips = list(records if records is not None else corpus.clips)
    signers = sorted({c.signer_id for c in clips})
    if k < 2:
        raise SplitError(f"k must be at least 2, got {k}")
    if len(signers) < k:
        raise SplitError(f"{len(signers)} signers cannot make {k} signer-disjoint folds")

    rng = random.Random(seed)
    shuffled = signers[:]
    rng.shuffle(shuffled)
    buckets: list[list[str]] = [shuffled[i::k] for i in range(k)]

    folds: list[Split] = []
    for i, held_out in enumerate(buckets):
        test_signers = set(held_out)
        train_clips = [c for c in clips if c.signer_id not in test_signers]
        test_clips = [c for c in clips if c.signer_id in test_signers]
        fold = Split(
            train=tuple(c.clip_id for c in train_clips),
            val=(),
            test=tuple(c.clip_id for c in test_clips),
            signers={
                "train": tuple(sorted({c.signer_id for c in train_clips})),
                "val": (),
                "test": tuple(sorted(test_signers)),
            },
            seed=seed + i,
        )
        validate_split(corpus, fold, require_full_vocabulary=False, records=clips)
        folds.append(fold)
    return folds


def validate_split(
    corpus: Corpus,
    split: Split,
    *,
    require_full_vocabulary: bool = True,
    records: list[ClipRecord] | None = None,
) -> None:
    """Raise :class:`SplitError` if a split would produce misleading numbers.

    Called by the training pipeline on every run, including splits loaded from disk:
    a split file edited by hand months later must fail here rather than quietly
    inflate a reported accuracy.
    """
    clips = {c.clip_id: c for c in (records if records is not None else corpus.clips)}

    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for clip_id in split[name]:
            if clip_id not in clips:
                raise SplitError(f"split references unknown clip {clip_id!r}")
            if clip_id in seen:
                raise SplitError(f"clip {clip_id!r} appears in both {seen[clip_id]} and {name}")
            seen[clip_id] = name

    signers_by_side = {
        name: {clips[cid].signer_id for cid in split[name]} for name in SPLIT_NAMES
    }
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1 :]:
            overlap = signers_by_side[left] & signers_by_side[right]
            if overlap:
                raise SplitError(
                    f"signer(s) {sorted(overlap)} appear in both {left} and {right}. "
                    "A signer-dependent split reports memorisation as accuracy "
                    "(plan §8.3, §14)."
                )

    if require_full_vocabulary and split.train:
        train_vocab = {g for cid in split.train for g in clips[cid].glosses}
        for name in ("val", "test"):
            missing = {g for cid in split[name] for g in clips[cid].glosses} - train_vocab
            if missing:
                raise SplitError(
                    f"gloss(es) {sorted(missing)} appear in {name} but never in train. "
                    "Pass require_full_vocabulary=False if this few-shot split is deliberate."
                )
