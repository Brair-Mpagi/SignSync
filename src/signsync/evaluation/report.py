"""Combined evaluation report (plan §15, §19).

The single job of this module: make it impossible to claim the system works on
automatic metrics alone.

Plan §15 states that "accuracy alone is not sufficient evidence of success — this is
stated explicitly to prevent the project from declaring victory on a benchmark that
doesn't reflect real communication", and plan §19 criterion 10 makes the Deaf
community's evaluation the one that overrides the rest. So
:attr:`EvaluationReport.can_claim_success` is False whenever human evaluation is
absent or uncertified, no matter how good the numbers are, and
:meth:`EvaluationReport.summary` says which requirement is unmet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .human import RoundResult
from .metrics import ClassificationReport, ErrorRate

__all__ = ["EvaluationReport"]


@dataclass
class EvaluationReport:
    """Automatic metrics and human evaluation, together, with the latter required."""

    system: str = "signsync"
    generated_on: date = field(default_factory=date.today)

    recognition: ClassificationReport | None = None
    continuous: ErrorRate | None = None
    speech: ErrorRate | None = None
    bleu: float | None = None
    rouge_l: float | None = None
    motion_smoothness: float | None = None

    human_rounds: list[RoundResult] = field(default_factory=list)

    signer_independent: bool = False
    """Whether the recognition numbers come from a signer-independent split.

    Recorded explicitly because a number from a clip-level split measures
    memorisation, and reporting it as accuracy is the failure plan §14 Risk 2
    describes."""

    corpus_note: str = ""
    notes: str = ""

    # ------------------------------------------------------------------ gating

    @property
    def has_human_evaluation(self) -> bool:
        return any(r.certified for r in self.human_rounds)

    @property
    def can_claim_success(self) -> bool:
        """Whether this report supports a claim that the system works.

        Automatic metrics are necessary and not sufficient (plan §15). A certified
        human round and a signer-independent split are both required.
        """
        return self.has_human_evaluation and self.signer_independent

    def blockers(self) -> list[str]:
        """What stands between this report and a defensible success claim."""
        blockers: list[str] = []
        if not self.human_rounds:
            blockers.append(
                "no human evaluation has been run. Plan §15: human evaluation is "
                "mandatory, not optional — automatic metrics cannot establish that a "
                "translation means the right thing."
            )
        elif not self.has_human_evaluation:
            for round_result in self.human_rounds:
                if round_result.panel_problems:
                    blockers.extend(
                        f"round {round_result.round_id}: {problem}"
                        for problem in round_result.panel_problems
                    )
                elif not round_result.certified:
                    blockers.append(
                        f"round {round_result.round_id}: Deaf panellists rated below "
                        f"threshold on {', '.join(round_result.failing_criteria())} "
                        "(plan §19 criterion 10 makes this decisive)"
                    )
        if not self.signer_independent:
            blockers.append(
                "recognition results are not from a signer-independent split, so they "
                "measure memorisation of training signers (plan §8.3, §14 Risk 2)."
            )
        return blockers

    # ------------------------------------------------------------------ output

    def summary(self) -> str:
        lines = [
            f"SignSync evaluation — {self.system}, {self.generated_on.isoformat()}",
            "=" * 58,
            "",
            "Automatic metrics",
        ]
        if self.recognition is not None:
            lines.append(
                f"  isolated recognition : {self.recognition.accuracy:.1%} accuracy, "
                f"{self.recognition.macro_f1:.1%} macro F1"
            )
            if self.recognition.worst_classes:
                weakest = ", ".join(f"{g} {f:.0%}" for g, f in self.recognition.worst_classes[:3])
                lines.append(f"    weakest signs      : {weakest}")
        if self.continuous is not None:
            lines.append(f"  sign error rate      : {self.continuous.summary()}")
        if self.speech is not None:
            lines.append(f"  word error rate      : {self.speech.summary()}")
        if self.bleu is not None:
            lines.append(f"  BLEU                 : {self.bleu:.3f}")
        if self.rouge_l is not None:
            lines.append(f"  ROUGE-L              : {self.rouge_l:.3f}")
        if self.motion_smoothness is not None:
            lines.append(
                f"  motion smoothness    : {self.motion_smoothness:.3f} (lower is smoother)"
            )
        if len(lines) == 4:
            lines.append("  (none recorded)")

        split_note = "signer-independent" if self.signer_independent else "NOT signer-independent"
        lines += ["", f"Split: {split_note}"]
        if self.corpus_note:
            lines.append(f"Corpus: {self.corpus_note}")

        lines += ["", "Human evaluation (plan §15 — mandatory)"]
        if not self.human_rounds:
            lines.append("  none run")
        for round_result in self.human_rounds:
            lines.append("  " + round_result.summary().replace("\n", "\n  "))

        lines += ["", "=" * 58]
        if self.can_claim_success:
            lines.append("This report supports a claim that the system works as evaluated.")
        else:
            lines.append("This report does NOT support a success claim. Outstanding:")
            lines.extend(f"  - {blocker}" for blocker in self.blockers())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "generated_on": self.generated_on.isoformat(),
            "signer_independent": self.signer_independent,
            "corpus_note": self.corpus_note,
            "automatic": {
                "accuracy": self.recognition.accuracy if self.recognition else None,
                "macro_f1": self.recognition.macro_f1 if self.recognition else None,
                "sign_error_rate": self.continuous.rate if self.continuous else None,
                "word_error_rate": self.speech.rate if self.speech else None,
                "bleu": self.bleu,
                "rouge_l": self.rouge_l,
                "motion_smoothness": self.motion_smoothness,
            },
            "human": [
                {
                    "round_id": r.round_id,
                    "certified": r.certified,
                    "by_criterion": r.by_criterion,
                    "by_criterion_deaf": r.by_criterion_deaf,
                    "agreement": r.agreement,
                    "panel_problems": list(r.panel_problems),
                }
                for r in self.human_rounds
            ],
            "can_claim_success": self.can_claim_success,
            "blockers": self.blockers(),
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target
