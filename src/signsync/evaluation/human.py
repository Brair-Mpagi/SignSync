"""Human evaluation (plan §15, §6).

Plan §15: **"Human evaluation is mandatory, not optional"**, drawn from fluent Deaf
signers, UNASLI-certified interpreters, and hearing users unfamiliar with the
system. Plan §19 goes further — criterion 10, positive evaluation from the Deaf
community itself, "overrides all others if there is a conflict".

This module makes those requirements checkable in code:

* :class:`Panel` refuses to certify a round whose evaluators do not include Deaf
  signers. Plan §6's "nothing about us without us" is a structural rule, and a panel
  of hearing interpreters reviewing sign quality is the specific failure it names.
* :meth:`EvaluationRound.result` reports the Deaf panellists' verdict *separately*
  from the aggregate, so criterion 10 can actually override rather than being
  averaged away by a larger group of hearing raters.
* Agreement between raters is computed, because a mean rating from raters who
  disagree completely is not evidence of anything.

None of this replaces running the sessions. It is the bookkeeping that makes the
outcome of those sessions auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..errors import SignSyncError

__all__ = [
    "EvaluatorRole",
    "Evaluator",
    "Panel",
    "Criterion",
    "EvaluationItem",
    "Rating",
    "EvaluationRound",
    "RoundResult",
]

#: Rating scale used throughout. Five points, anchored, so ratings mean the same
#: thing to different panellists and across rounds.
SCALE: dict[int, str] = {
    1: "not understandable / meaning lost",
    2: "poor — needed several attempts to interpret",
    3: "understandable with effort",
    4: "good — clear, minor issues",
    5: "natural — as a fluent signer would produce",
}

#: Minimum mean rating for a criterion to pass. Plan §4.3 sets the objective
#: thresholds; this is the per-criterion floor those objectives assume.
PASS_THRESHOLD = 3.5


class EvaluatorRole(str, Enum):
    """Who is rating. Plan §15 requires all three."""

    DEAF_SIGNER = "deaf_signer"
    INTERPRETER = "interpreter"
    """UNASLI-certified interpreter."""

    HEARING_USER = "hearing_user"
    """Unfamiliar with the system — tests genuine usability, not insider approval."""

    LINGUIST = "linguist"


class Criterion(str, Enum):
    """What is being rated (plan §15)."""

    MEANING_PRESERVED = "meaning_preserved"
    GRAMMATICALITY = "grammaticality"
    SIGN_INTELLIGIBILITY = "sign_intelligibility"
    MOTION_NATURALNESS = "motion_naturalness"
    NON_MANUAL_CORRECTNESS = "non_manual_correctness"
    OVERALL_USABILITY = "overall_usability"


@dataclass(frozen=True)
class Evaluator:
    """A panellist, pseudonymised.

    ``compensated`` is recorded because plan §9.3 and §13 make fair payment
    non-negotiable — signers and interpreters contributing linguistic labour are
    domain experts, not free crowd-sourced clicks — and an unpaid round should be
    visible in the record rather than discovered later.
    """

    evaluator_id: str
    role: EvaluatorRole
    is_deaf: bool = False
    usl_fluent: bool = False
    compensated: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.role is EvaluatorRole.DEAF_SIGNER and not self.is_deaf:
            raise SignSyncError(
                f"{self.evaluator_id}: role is deaf_signer but is_deaf is False"
            )


@dataclass
class Panel:
    """The group evaluating a round."""

    evaluators: list[Evaluator] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.evaluators)

    def add(self, evaluator: Evaluator) -> None:
        self.evaluators.append(evaluator)

    def with_role(self, role: EvaluatorRole) -> list[Evaluator]:
        return [e for e in self.evaluators if e.role is role]

    @property
    def deaf_evaluators(self) -> list[Evaluator]:
        return [e for e in self.evaluators if e.is_deaf]

    def problems(self, *, minimum_deaf: int = 3) -> list[str]:
        """Reasons this panel cannot certify a result.

        Returned as a list rather than raising, so a round can be *run* with an
        incomplete panel — pilots and dry runs are legitimate — while
        :meth:`EvaluationRound.result` still refuses to mark it certified.
        """
        problems: list[str] = []
        if len(self.deaf_evaluators) < minimum_deaf:
            problems.append(
                f"only {len(self.deaf_evaluators)} Deaf evaluator(s); at least {minimum_deaf} "
                "are required — sign quality cannot be certified by hearing reviewers "
                "(plan §6, §15)"
            )
        if not self.with_role(EvaluatorRole.INTERPRETER):
            problems.append("no UNASLI interpreter on the panel (plan §15)")
        if not self.with_role(EvaluatorRole.HEARING_USER):
            problems.append(
                "no hearing user unfamiliar with the system; without one the round "
                "measures insider approval rather than usability (plan §15)"
            )
        uncompensated = [e.evaluator_id for e in self.evaluators if not e.compensated]
        if uncompensated:
            problems.append(
                f"evaluator(s) {uncompensated} recorded as uncompensated (plan §9.3, §13)"
            )
        return problems


@dataclass(frozen=True)
class EvaluationItem:
    """One thing being rated: a translation, or a stretch of avatar motion."""

    item_id: str
    kind: str
    """``"translation"``, ``"avatar"``, ``"conversation"``."""

    source: str
    """What went in — a gloss sequence or an English sentence."""

    output: str
    """What came out."""

    reference: str = ""
    """Gold-standard version, where one exists."""

    criteria: tuple[Criterion, ...] = (Criterion.MEANING_PRESERVED,)
    context: str = ""


@dataclass(frozen=True)
class Rating:
    """One evaluator's rating of one item on one criterion."""

    item_id: str
    evaluator_id: str
    criterion: Criterion
    score: int
    comment: str = ""

    def __post_init__(self) -> None:
        if self.score not in SCALE:
            raise SignSyncError(
                f"score {self.score} outside the scale {sorted(SCALE)}; see SCALE for anchors"
            )


@dataclass
class RoundResult:
    """The outcome of an evaluation round."""

    round_id: str
    n_items: int
    n_ratings: int
    by_criterion: dict[str, float]
    by_criterion_deaf: dict[str, float]
    agreement: float
    panel_problems: tuple[str, ...]
    free_text: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        """Whether this round can support a claim about system quality.

        Requires a valid panel *and* Deaf panellists' ratings clearing the
        threshold. Plan §19 criterion 10 makes the Deaf community's verdict
        override the others, so a high aggregate cannot rescue a round the Deaf
        evaluators rated poorly.
        """
        if self.panel_problems:
            return False
        if not self.by_criterion_deaf:
            return False
        return all(score >= PASS_THRESHOLD for score in self.by_criterion_deaf.values())

    def failing_criteria(self) -> list[str]:
        return sorted(k for k, v in self.by_criterion_deaf.items() if v < PASS_THRESHOLD)

    def summary(self) -> str:
        lines = [f"round {self.round_id}: {self.n_ratings} ratings on {self.n_items} items"]
        for criterion, score in sorted(self.by_criterion.items()):
            deaf = self.by_criterion_deaf.get(criterion)
            suffix = f"  (Deaf panellists: {deaf:.2f})" if deaf is not None else ""
            lines.append(f"  {criterion:<24} {score:.2f}{suffix}")
        lines.append(f"  inter-rater agreement    {self.agreement:.2f}")

        if self.panel_problems:
            lines.append("\nNOT CERTIFIED — panel composition:")
            lines.extend(f"  ! {problem}" for problem in self.panel_problems)
        elif not self.certified:
            lines.append(
                f"\nNOT CERTIFIED — Deaf panellists rated below {PASS_THRESHOLD} on: "
                + ", ".join(self.failing_criteria())
            )
        else:
            lines.append("\nCERTIFIED by this round.")
        return "\n".join(lines)


@dataclass
class EvaluationRound:
    """A set of items, a panel, and the ratings they produced (plan §15)."""

    round_id: str
    items: list[EvaluationItem] = field(default_factory=list)
    panel: Panel = field(default_factory=Panel)
    ratings: list[Rating] = field(default_factory=list)
    run_on: date = field(default_factory=date.today)
    notes: str = ""

    def add_rating(self, rating: Rating) -> None:
        known_items = {item.item_id for item in self.items}
        if rating.item_id not in known_items:
            raise SignSyncError(f"rating refers to unknown item {rating.item_id!r}")
        if rating.evaluator_id not in {e.evaluator_id for e in self.panel.evaluators}:
            raise SignSyncError(f"rating from unknown evaluator {rating.evaluator_id!r}")
        self.ratings.append(rating)

    def coverage(self) -> float:
        """Fraction of (item, criterion, evaluator) cells that were actually rated."""
        expected = sum(len(item.criteria) for item in self.items) * len(self.panel)
        if expected == 0:
            return 0.0
        return min(1.0, len(self.ratings) / expected)

    def result(self) -> RoundResult:
        if not self.ratings:
            raise SignSyncError(f"round {self.round_id!r} has no ratings yet")

        deaf_ids = {e.evaluator_id for e in self.panel.deaf_evaluators}

        by_criterion: dict[str, list[int]] = {}
        by_criterion_deaf: dict[str, list[int]] = {}
        for rating in self.ratings:
            by_criterion.setdefault(rating.criterion.value, []).append(rating.score)
            if rating.evaluator_id in deaf_ids:
                by_criterion_deaf.setdefault(rating.criterion.value, []).append(rating.score)

        return RoundResult(
            round_id=self.round_id,
            n_items=len(self.items),
            n_ratings=len(self.ratings),
            by_criterion={k: mean(v) for k, v in by_criterion.items()},
            by_criterion_deaf={k: mean(v) for k, v in by_criterion_deaf.items()},
            agreement=self.agreement(),
            panel_problems=tuple(self.panel.problems()),
            free_text=tuple(r.comment for r in self.ratings if r.comment.strip()),
        )

    def agreement(self) -> float:
        """How much the raters agree, as 1 − (mean spread / scale range).

        A simple dispersion measure rather than a chance-corrected coefficient:
        panels here are small enough that kappa-family statistics are unstable, and
        the question being asked is blunt — "did the raters see the same thing?".
        A mean score from raters who disagree completely is not evidence.
        """
        grouped: dict[tuple[str, str], list[int]] = {}
        for rating in self.ratings:
            grouped.setdefault((rating.item_id, rating.criterion.value), []).append(rating.score)

        spreads = [pstdev(scores) for scores in grouped.values() if len(scores) > 1]
        if not spreads:
            return 0.0
        # Max possible standard deviation on a 1-5 scale is 2.0.
        return max(0.0, 1.0 - mean(spreads) / 2.0)

    # ------------------------------------------------------------------ forms

    def to_form(self) -> dict[str, Any]:
        """A blank rating form, for whatever tool the session actually uses.

        Exported as data rather than rendered, because the sessions plan §15
        describes happen in person, in USL, often on paper — the software's job is
        to define the items and ingest the results, not to dictate the room.
        """
        return {
            "round_id": self.round_id,
            "scale": {str(k): v for k, v in SCALE.items()},
            "instructions": (
                "Rate each item on each criterion. Please add a comment whenever a "
                "score is 3 or below — the comments are more useful than the numbers."
            ),
            "items": [
                {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "source": item.source,
                    "output": item.output,
                    "reference": item.reference,
                    "criteria": [c.value for c in item.criteria],
                    "context": item.context,
                }
                for item in self.items
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "round_id": self.round_id,
            "run_on": self.run_on.isoformat(),
            "notes": self.notes,
            "panel": [
                {
                    "evaluator_id": e.evaluator_id,
                    "role": e.role.value,
                    "is_deaf": e.is_deaf,
                    "usl_fluent": e.usl_fluent,
                    "compensated": e.compensated,
                    "notes": e.notes,
                }
                for e in self.panel.evaluators
            ],
            "items": self.to_form()["items"],
            "ratings": [
                {
                    "item_id": r.item_id,
                    "evaluator_id": r.evaluator_id,
                    "criterion": r.criterion.value,
                    "score": r.score,
                    "comment": r.comment,
                }
                for r in self.ratings
            ],
        }
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> EvaluationRound:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        round_ = cls(
            round_id=data["round_id"],
            run_on=date.fromisoformat(data["run_on"]),
            notes=data.get("notes", ""),
        )
        for entry in data.get("panel", []):
            round_.panel.add(
                Evaluator(
                    evaluator_id=entry["evaluator_id"],
                    role=EvaluatorRole(entry["role"]),
                    is_deaf=bool(entry.get("is_deaf", False)),
                    usl_fluent=bool(entry.get("usl_fluent", False)),
                    compensated=bool(entry.get("compensated", True)),
                    notes=entry.get("notes", ""),
                )
            )
        for entry in data.get("items", []):
            round_.items.append(
                EvaluationItem(
                    item_id=entry["item_id"],
                    kind=entry["kind"],
                    source=entry["source"],
                    output=entry["output"],
                    reference=entry.get("reference", ""),
                    criteria=tuple(Criterion(c) for c in entry.get("criteria", [])),
                    context=entry.get("context", ""),
                )
            )
        for entry in data.get("ratings", []):
            round_.add_rating(
                Rating(
                    item_id=entry["item_id"],
                    evaluator_id=entry["evaluator_id"],
                    criterion=Criterion(entry["criterion"]),
                    score=int(entry["score"]),
                    comment=entry.get("comment", ""),
                )
            )
        return round_
