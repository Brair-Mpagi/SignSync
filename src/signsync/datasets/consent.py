"""Consent enforcement (plan §16, ``docs/data-protection.md``).

Uganda's Data Protection and Privacy Act (2019) governs every recording this
project makes. The rule this module implements:

    No clip enters training, evaluation, export or a demo without a consent record
    that is present, unexpired, un-withdrawn, and that covers the specific use.

Two design decisions worth stating, because both look inconvenient:

**Consent is a set of scopes, not a boolean.** A signer may agree to model training
and refuse public release. Defaulting an unlisted scope to "granted" would turn a
narrow agreement into a broad one, which is exactly the extractive-research failure
plan §14 warns about. Unlisted scopes are therefore denied.

**A missing record raises, it does not skip.** Silently dropping an unconsented clip
hides a compliance failure just as effectively as silently using it creates one. The
caller who wants lenient behaviour asks for it explicitly via :meth:`ConsentRegistry.audit`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from ..errors import ConsentError

__all__ = [
    "ConsentScope",
    "ConsentStatus",
    "ConsentRecord",
    "ConsentRegistry",
    "ConsentAudit",
]


class ConsentScope(str, Enum):
    """Uses a participant may separately grant or withhold."""

    TRAINING = "training"
    EVALUATION = "evaluation"
    PUBLICATION = "publication"
    OPEN_RELEASE = "open_release"
    COMMERCIAL = "commercial"

    @classmethod
    def parse(cls, value: str | ConsentScope) -> ConsentScope:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise ConsentError(
                f"unknown consent scope {value!r}; known scopes: "
                f"{', '.join(s.value for s in cls)}"
            ) from None


class ConsentStatus(str, Enum):
    """Why a clip is or is not usable, for audit output."""

    GRANTED = "granted"
    NO_RECORD = "no_record"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    SCOPE_NOT_GRANTED = "scope_not_granted"
    NOT_YET_EFFECTIVE = "not_yet_effective"


@dataclass(frozen=True)
class ConsentRecord:
    """One participant's consent, as agreed in USL and recorded here.

    ``delivered_by`` and ``language`` exist because plan §16.2 requires the consent
    conversation to happen *in USL, with a fluent signer*. A record that cannot say
    who delivered it in which language is not evidence of informed consent, so the
    constructor rejects it.
    """

    participant_id: str
    granted_on: date
    retention_until: date
    scopes: frozenset[ConsentScope]
    delivered_by: str
    language: str = "USL"
    withdrawn_on: date | None = None
    is_minor: bool = False
    guardian_name: str | None = None
    guardian_relationship: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ConsentError("consent record needs a participant_id")
        if not self.delivered_by:
            raise ConsentError(
                f"{self.participant_id}: consent record must name who delivered the consent "
                "conversation (plan §16.2 requires a fluent signer, not an English form)"
            )
        if self.retention_until <= self.granted_on:
            raise ConsentError(
                f"{self.participant_id}: retention_until ({self.retention_until}) must be after "
                f"granted_on ({self.granted_on})"
            )
        if self.is_minor and not (self.guardian_name and self.guardian_relationship):
            raise ConsentError(
                f"{self.participant_id}: a participant under 18 requires a named guardian and "
                "their relationship (plan §16.1)"
            )
        if self.withdrawn_on is not None and self.withdrawn_on < self.granted_on:
            raise ConsentError(
                f"{self.participant_id}: withdrawn_on precedes granted_on"
            )

    def status(self, scope: ConsentScope, on: date | None = None) -> ConsentStatus:
        """Status of one scope on a given date (default: today)."""
        today = on or date.today()
        if today < self.granted_on:
            return ConsentStatus.NOT_YET_EFFECTIVE
        if self.withdrawn_on is not None and today >= self.withdrawn_on:
            return ConsentStatus.WITHDRAWN
        if today > self.retention_until:
            return ConsentStatus.EXPIRED
        if scope not in self.scopes:
            return ConsentStatus.SCOPE_NOT_GRANTED
        return ConsentStatus.GRANTED

    def permits(self, scope: ConsentScope, on: date | None = None) -> bool:
        return self.status(scope, on) is ConsentStatus.GRANTED

    def withdraw(self, on: date | None = None) -> ConsentRecord:
        """Return a withdrawn copy.

        Withdrawal is retroactive by design: every later load excludes this
        participant's clips. It is a data subject right under the Act, so it cannot
        be a support ticket someone gets to the following week.
        """
        from dataclasses import replace

        return replace(self, withdrawn_on=on or date.today())

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "granted_on": self.granted_on.isoformat(),
            "retention_until": self.retention_until.isoformat(),
            "scopes": sorted(s.value for s in self.scopes),
            "delivered_by": self.delivered_by,
            "language": self.language,
            "withdrawn_on": self.withdrawn_on.isoformat() if self.withdrawn_on else None,
            "is_minor": self.is_minor,
            "guardian_name": self.guardian_name,
            "guardian_relationship": self.guardian_relationship,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsentRecord:
        withdrawn = data.get("withdrawn_on")
        return cls(
            participant_id=data["participant_id"],
            granted_on=date.fromisoformat(data["granted_on"]),
            retention_until=date.fromisoformat(data["retention_until"]),
            scopes=frozenset(ConsentScope.parse(s) for s in data.get("scopes", [])),
            delivered_by=data["delivered_by"],
            language=data.get("language", "USL"),
            withdrawn_on=date.fromisoformat(withdrawn) if withdrawn else None,
            is_minor=bool(data.get("is_minor", False)),
            guardian_name=data.get("guardian_name"),
            guardian_relationship=data.get("guardian_relationship"),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class ConsentAudit:
    """Result of checking a set of participants against a scope."""

    scope: ConsentScope
    on: date
    permitted: tuple[str, ...]
    refused: tuple[tuple[str, ConsentStatus], ...]

    @property
    def ok(self) -> bool:
        return not self.refused

    def summary(self) -> str:
        if self.ok:
            return f"all {len(self.permitted)} participants permit {self.scope.value}"
        counts: dict[str, int] = {}
        for _, status in self.refused:
            counts[status.value] = counts.get(status.value, 0) + 1
        detail = ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items()))
        return (
            f"{len(self.permitted)} of {len(self.permitted) + len(self.refused)} participants "
            f"permit {self.scope.value} ({detail})"
        )


@dataclass
class ConsentRegistry:
    """The consent records for a corpus."""

    records: dict[str, ConsentRecord] = field(default_factory=dict)

    def add(self, record: ConsentRecord) -> None:
        self.records[record.participant_id] = record

    def get(self, participant_id: str) -> ConsentRecord | None:
        return self.records.get(participant_id)

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, participant_id: object) -> bool:
        return participant_id in self.records

    def status(
        self, participant_id: str, scope: ConsentScope, on: date | None = None
    ) -> ConsentStatus:
        record = self.records.get(participant_id)
        if record is None:
            return ConsentStatus.NO_RECORD
        return record.status(scope, on)

    def permits(
        self, participant_id: str, scope: ConsentScope | str, on: date | None = None
    ) -> bool:
        return self.status(participant_id, ConsentScope.parse(scope), on) is ConsentStatus.GRANTED

    def require(
        self, participant_id: str, scope: ConsentScope | str, on: date | None = None
    ) -> None:
        """Raise :class:`ConsentError` unless the use is permitted."""
        scope = ConsentScope.parse(scope)
        status = self.status(participant_id, scope, on)
        if status is ConsentStatus.GRANTED:
            return
        raise ConsentError(
            f"participant {participant_id!r} does not permit {scope.value}: {status.value}. "
            "See docs/data-protection.md."
        )

    def audit(
        self, participant_ids: list[str], scope: ConsentScope | str, on: date | None = None
    ) -> ConsentAudit:
        """Check many participants at once without raising.

        This is the query to run before a release or a public corpus drop, and after
        any withdrawal: it names which derived artefacts — splits, checkpoints,
        published clips — have to be regenerated.
        """
        scope = ConsentScope.parse(scope)
        today = on or date.today()
        permitted: list[str] = []
        refused: list[tuple[str, ConsentStatus]] = []
        for pid in dict.fromkeys(participant_ids):  # de-duplicate, keep order
            status = self.status(pid, scope, today)
            if status is ConsentStatus.GRANTED:
                permitted.append(pid)
            else:
                refused.append((pid, status))
        return ConsentAudit(
            scope=scope, on=today, permitted=tuple(permitted), refused=tuple(refused)
        )

    def withdrawn(self, on: date | None = None) -> list[str]:
        """Participants whose consent has been withdrawn as of ``on``."""
        today = on or date.today()
        return [
            pid
            for pid, rec in self.records.items()
            if rec.withdrawn_on is not None and today >= rec.withdrawn_on
        ]

    def expiring_before(self, deadline: date) -> list[str]:
        """Participants whose retention window closes before ``deadline``.

        Retention limits are statutory. Knowing which clips drop out next quarter is
        a planning input, not a surprise to discover when a training run shrinks.
        """
        return sorted(
            pid
            for pid, rec in self.records.items()
            if rec.withdrawn_on is None and rec.retention_until < deadline
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": [r.to_dict() for r in self.records.values()],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ConsentRegistry:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls()
        for entry in data.get("records", []):
            registry.add(ConsentRecord.from_dict(entry))
        return registry
