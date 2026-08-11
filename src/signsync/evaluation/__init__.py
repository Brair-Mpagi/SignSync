"""Evaluation (plan §15).

    automatic metrics ──┐
                        ├──▶ EvaluationReport ──▶ can_claim_success
    human evaluation ───┘         (False without a certified human round)

Plan §15 states that accuracy alone is not sufficient evidence of success, and plan
§19 makes the Deaf community's verdict the criterion that overrides the others. Both
are enforced here rather than left to discipline: a report without a certified human
round refuses to support a success claim, and a round whose panel lacks Deaf
evaluators cannot be certified.
"""

from __future__ import annotations

from .human import (
    PASS_THRESHOLD,
    SCALE,
    Criterion,
    EvaluationItem,
    EvaluationRound,
    Evaluator,
    EvaluatorRole,
    Panel,
    Rating,
    RoundResult,
)
from .metrics import (
    ClassificationReport,
    ErrorRate,
    bleu,
    classification_report,
    confusion_matrix,
    confusion_pairs,
    edit_distance,
    motion_smoothness,
    rouge_l,
    sequence_accuracy,
    sign_error_rate,
    trajectory_error,
    word_error_rate,
)
from .report import EvaluationReport

__all__ = [
    "ClassificationReport",
    "Criterion",
    "ErrorRate",
    "EvaluationItem",
    "EvaluationReport",
    "EvaluationRound",
    "Evaluator",
    "EvaluatorRole",
    "PASS_THRESHOLD",
    "Panel",
    "Rating",
    "RoundResult",
    "SCALE",
    "bleu",
    "classification_report",
    "confusion_matrix",
    "confusion_pairs",
    "edit_distance",
    "motion_smoothness",
    "rouge_l",
    "sequence_accuracy",
    "sign_error_rate",
    "trajectory_error",
    "word_error_rate",
]
