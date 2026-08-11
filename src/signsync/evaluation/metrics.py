"""Automatic metrics (plan §15).

Plan §15 opens by saying accuracy alone is not sufficient evidence of success, and
lists a different metric family for each stage. All of them are here — and all of
them are, by the plan's own framing, the *easy* half. The hard half is
:mod:`signsync.evaluation.human`, and :mod:`signsync.evaluation.report` refuses to
call a system evaluated without it.

Two choices worth defending:

* **Per-sign confusion pairs, not just a matrix.** Plan §15 asks for the confusion
  matrix specifically "to catch systematically confused sign pairs". A 500×500
  matrix does not communicate that; a ranked list of the pairs a model actually
  mixes up does.
* **Sign error rate, not sequence accuracy alone.** A continuous utterance with one
  wrong sign in eight is not equivalent to one that is entirely wrong, and exact
  sequence match scores them the same.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..errors import SignSyncError

__all__ = [
    "ClassificationReport",
    "classification_report",
    "confusion_matrix",
    "confusion_pairs",
    "edit_distance",
    "ErrorRate",
    "sign_error_rate",
    "word_error_rate",
    "sequence_accuracy",
    "bleu",
    "rouge_l",
    "motion_smoothness",
    "trajectory_error",
]


# --------------------------------------------------------------------------- classification


@dataclass
class ClassificationReport:
    """Per-class and overall recognition metrics (plan §15, isolated signs)."""

    labels: tuple[str, ...]
    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    support: dict[str, int]
    abstentions: int = 0
    """Predictions that returned ``<unknown>``.

    Counted separately from errors because they are a different product outcome: an
    abstention prompts the signer to repeat, a wrong answer is spoken aloud as fact
    (plan §16.3)."""

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1.

        Reported alongside accuracy because a vocabulary with a few very common
        signs lets a model score well on accuracy while failing every rare sign —
        and rare signs are not less important to the person who needs one."""
        return sum(self.f1.values()) / len(self.f1) if self.f1 else 0.0

    @property
    def worst_classes(self) -> list[tuple[str, float]]:
        return sorted(self.f1.items(), key=lambda kv: kv[1])[:5]

    def summary(self) -> str:
        lines = [
            f"accuracy   : {self.accuracy:.1%}",
            f"macro F1   : {self.macro_f1:.1%}",
            f"classes    : {len(self.labels)}",
        ]
        if self.abstentions:
            lines.append(f"abstentions: {self.abstentions}")
        if self.worst_classes:
            worst = ", ".join(f"{g} ({f:.0%})" for g, f in self.worst_classes)
            lines.append(f"weakest    : {worst}")
        return "\n".join(lines)


def classification_report(
    truth: list[str], predicted: list[str], *, unknown: str = "<unknown>"
) -> ClassificationReport:
    """Accuracy, precision, recall and F1 per class."""
    if len(truth) != len(predicted):
        raise SignSyncError(f"{len(truth)} true labels but {len(predicted)} predictions")
    if not truth:
        raise SignSyncError("cannot evaluate an empty set")

    labels = tuple(sorted(set(truth) | (set(predicted) - {unknown})))
    true_positive = Counter[str]()
    false_positive = Counter[str]()
    false_negative = Counter[str]()
    support = Counter[str]()
    abstentions = 0

    for actual, guess in zip(truth, predicted, strict=True):
        support[actual] += 1
        if guess == unknown:
            abstentions += 1
            false_negative[actual] += 1
            continue
        if guess == actual:
            true_positive[actual] += 1
        else:
            false_positive[guess] += 1
            false_negative[actual] += 1

    precision, recall, f1 = {}, {}, {}
    for label in labels:
        tp, fp, fn = true_positive[label], false_positive[label], false_negative[label]
        precision[label] = tp / (tp + fp) if tp + fp else 0.0
        recall[label] = tp / (tp + fn) if tp + fn else 0.0
        denominator = precision[label] + recall[label]
        f1[label] = 2 * precision[label] * recall[label] / denominator if denominator else 0.0

    correct = sum(1 for a, b in zip(truth, predicted, strict=True) if a == b)
    return ClassificationReport(
        labels=labels,
        accuracy=correct / len(truth),
        precision=precision,
        recall=recall,
        f1=f1,
        support=dict(support),
        abstentions=abstentions,
    )


def confusion_matrix(
    truth: list[str], predicted: list[str], labels: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Confusion counts and their label order."""
    order = labels or sorted(set(truth) | set(predicted))
    index = {label: i for i, label in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=np.int64)
    for actual, guess in zip(truth, predicted, strict=True):
        if actual in index and guess in index:
            matrix[index[actual], index[guess]] += 1
    return matrix, order


def confusion_pairs(
    truth: list[str], predicted: list[str], *, limit: int = 10
) -> list[tuple[str, str, int]]:
    """The sign pairs a model actually confuses, most frequent first (plan §15)."""
    counts = Counter(
        (actual, guess)
        for actual, guess in zip(truth, predicted, strict=True)
        if actual != guess
    )
    return [(a, b, n) for (a, b), n in counts.most_common(limit)]


# --------------------------------------------------------------------------- sequences


def edit_distance(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    """Levenshtein distance as ``(distance, substitutions, deletions, insertions)``.

    The breakdown matters for diagnosis: deletions usually mean the segmenter
    missed a sign boundary, insertions that it split one sign in two, and
    substitutions that recognition confused two similar signs. A single number
    cannot distinguish a segmentation problem from a recognition one.
    """
    n, m = len(reference), len(hypothesis)
    # (distance, subs, dels, ins) per cell.
    previous: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(m + 1)]

    for i in range(1, n + 1):
        current: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            dele = previous[j]
            ins = current[j - 1]
            best = min(sub[0], dele[0], ins[0])
            if best == sub[0]:
                current.append((sub[0] + 1, sub[1] + 1, sub[2], sub[3]))
            elif best == dele[0]:
                current.append((dele[0] + 1, dele[1], dele[2] + 1, dele[3]))
            else:
                current.append((ins[0] + 1, ins[1], ins[2], ins[3] + 1))
        previous = current
    return previous[m]


@dataclass(frozen=True)
class ErrorRate:
    """An error rate with the breakdown that explains it."""

    rate: float
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    def summary(self) -> str:
        return (
            f"{self.rate:.1%} ({self.substitutions} sub, {self.deletions} del, "
            f"{self.insertions} ins over {self.reference_length} tokens)"
        )


def _error_rate(references: list[list[str]], hypotheses: list[list[str]]) -> ErrorRate:
    if len(references) != len(hypotheses):
        raise SignSyncError(f"{len(references)} references but {len(hypotheses)} hypotheses")

    total = subs = dels = ins = length = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        distance, s, d, i = edit_distance(reference, hypothesis)
        total += distance
        subs, dels, ins = subs + s, dels + d, ins + i
        length += len(reference)

    return ErrorRate(
        rate=total / length if length else 0.0,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_length=length,
    )


def sign_error_rate(references: list[list[str]], hypotheses: list[list[str]]) -> ErrorRate:
    """Sign error rate for continuous recognition (plan §15)."""
    return _error_rate(references, hypotheses)


def word_error_rate(references: list[str], hypotheses: list[str]) -> ErrorRate:
    """Word error rate for speech recognition (plan §15, §8.5)."""
    return _error_rate(
        [r.lower().split() for r in references], [h.lower().split() for h in hypotheses]
    )


def sequence_accuracy(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    """Fraction of sequences recognised exactly.

    Reported *with* sign error rate, never instead of it: exact match scores "one
    sign wrong in eight" the same as "entirely wrong".
    """
    if not references:
        raise SignSyncError("cannot evaluate an empty set")
    matches = sum(1 for r, h in zip(references, hypotheses, strict=True) if r == h)
    return matches / len(references)


# --------------------------------------------------------------------------- translation


def bleu(references: list[str], hypotheses: list[str], *, max_n: int = 4) -> float:
    """Corpus BLEU with a brevity penalty and add-one smoothing.

    Plan §15 lists BLEU for USL→English, and plan §8.4 immediately adds that
    automatic metrics are *not sufficient* for sign-language translation quality.
    Both are true: BLEU catches regressions cheaply, and it cannot tell whether a
    fluent sentence means the right thing. Report it next to the human evaluation,
    never in place of it.
    """
    if len(references) != len(hypotheses):
        raise SignSyncError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    if not references:
        raise SignSyncError("cannot evaluate an empty set")

    clipped = [0] * max_n
    totals = [0] * max_n
    reference_length = hypothesis_length = 0

    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        reference_length += len(ref_tokens)
        hypothesis_length += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_counts = Counter(_ngrams(ref_tokens, n))
            hyp_counts = Counter(_ngrams(hyp_tokens, n))
            totals[n - 1] += max(sum(hyp_counts.values()), 0)
            clipped[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())

    if hypothesis_length == 0:
        return 0.0

    # Add-one smoothing so a single short sentence with no 4-gram match does not
    # zero the whole corpus score — the corpora here are small (plan §9.2).
    precisions = [
        (clipped[i] + 1) / (totals[i] + 1) if totals[i] else 0.0 for i in range(max_n)
    ]
    if any(p == 0 for p in precisions):
        return 0.0

    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity = 1.0 if hypothesis_length > reference_length else math.exp(
        1 - reference_length / max(hypothesis_length, 1)
    )
    return geometric_mean * brevity


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_l(references: list[str], hypotheses: list[str]) -> float:
    """Mean ROUGE-L F1 over longest common subsequences.

    Complements BLEU: BLEU rewards matching phrases, ROUGE-L rewards covering the
    reference's content in order, which is closer to "did the meaning survive".
    """
    if len(references) != len(hypotheses):
        raise SignSyncError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    if not references:
        raise SignSyncError("cannot evaluate an empty set")

    scores = []
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref = reference.lower().split()
        hyp = hypothesis.lower().split()
        if not ref or not hyp:
            scores.append(0.0)
            continue
        lcs = _lcs_length(ref, hyp)
        precision = lcs / len(hyp)
        recall = lcs / len(ref)
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


def _lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1]))
        previous = current
    return previous[len(b)]


# --------------------------------------------------------------------------- motion


def motion_smoothness(positions: np.ndarray, fps: float = 30.0) -> float:
    """Normalised mean jerk — lower is smoother (plan §15, sign motion generation).

    Jerk (the third derivative of position) is the standard measure of motion
    naturalness: human limb movement minimises it, and interpolation artefacts show
    up there long before they are visible in position or velocity. It is a
    *screening* metric — plan §15 still requires human realism ratings, because
    smooth and wrong is a thing an avatar can easily be.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SignSyncError(f"expected (T, 3) positions, got shape {positions.shape}")
    if len(positions) < 4:
        return 0.0

    dt = 1.0 / fps
    jerk = np.diff(positions, n=3, axis=0) / dt**3
    magnitude = float(np.linalg.norm(jerk, axis=1).mean())

    travel = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    if travel < 1e-9:
        return 0.0
    duration = len(positions) / fps
    # Scale-free: dimensionless, so clips of different length and size compare.
    return magnitude * duration**3 / max(travel, 1e-9) / 1e3


def trajectory_error(reference: np.ndarray, generated: np.ndarray) -> float:
    """Mean per-frame distance between two trajectories, after length matching."""
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if reference.shape[1:] != generated.shape[1:]:
        raise SignSyncError(
            f"trajectories have different shapes: {reference.shape} vs {generated.shape}"
        )
    if len(reference) == 0 or len(generated) == 0:
        raise SignSyncError("cannot compare an empty trajectory")

    # Resample the generated trajectory onto the reference's timeline: signing speed
    # differs between a recording and a generated clip, and comparing frame i to
    # frame i would measure tempo rather than trajectory.
    source = np.linspace(0.0, 1.0, len(generated))
    target = np.linspace(0.0, 1.0, len(reference))
    aligned = np.stack(
        [np.interp(target, source, generated[:, d]) for d in range(generated.shape[1])], axis=1
    )
    return float(np.linalg.norm(reference - aligned, axis=1).mean())
