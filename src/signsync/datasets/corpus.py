"""Corpus loading, with consent enforced at the point of use.

Every path from disk into a model runs through :class:`CorpusLoader`. Putting the
consent check anywhere else — a review step, a script someone remembers to run —
means it eventually gets skipped, and the failure is invisible until it is a legal
and community-trust problem rather than a technical one (plan §16, §14).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..errors import ConsentError, CorpusError
from ..vision.schema import Channel, LandmarkSequence
from .consent import ConsentAudit, ConsentRegistry, ConsentScope
from .schema import ClipRecord, Corpus

__all__ = ["ClipData", "QualityGate", "CorpusLoader"]


@dataclass(frozen=True)
class ClipData:
    """An annotation record together with its landmark data."""

    record: ClipRecord
    sequence: LandmarkSequence

    @property
    def clip_id(self) -> str:
        return self.record.clip_id

    @property
    def signer_id(self) -> str:
        return self.record.signer_id


@dataclass(frozen=True)
class QualityGate:
    """Minimum tracking quality for a clip to be usable.

    A clip where the dominant hand is tracked in 40% of frames is a recording
    problem, not training data; letting it through teaches the model that tracking
    dropouts are part of signs.
    """

    min_dominant_hand_coverage: float = 0.6
    min_pose_coverage: float = 0.8
    min_frames: int = 4

    def check(self, record: ClipRecord, sequence: LandmarkSequence) -> str | None:
        """Return a rejection reason, or ``None`` if the clip passes."""
        if len(sequence) < self.min_frames:
            return f"only {len(sequence)} frames (minimum {self.min_frames})"
        coverage = sequence.coverage()
        if coverage["pose"] < self.min_pose_coverage:
            return f"pose tracked in {coverage['pose']:.0%} of frames"
        dominant = max(coverage["left_hand"], coverage["right_hand"])
        if dominant < self.min_dominant_hand_coverage:
            return f"dominant hand tracked in {dominant:.0%} of frames"
        return None


class CorpusLoader:
    """Consent-gated, quality-gated access to a corpus's landmark data."""

    def __init__(
        self,
        corpus: Corpus,
        registry: ConsentRegistry,
        *,
        scope: ConsentScope | str = ConsentScope.TRAINING,
        on: date | None = None,
        quality: QualityGate | None = None,
    ) -> None:
        self.corpus = corpus
        self.registry = registry
        self.scope = ConsentScope.parse(scope)
        self.on = on or date.today()
        self.quality = quality or QualityGate()

    def audit(self) -> ConsentAudit:
        """Consent status of every signer in the corpus for this loader's scope."""
        return self.registry.audit([c.signer_id for c in self.corpus.clips], self.scope, self.on)

    def permitted_clips(self) -> list[ClipRecord]:
        """Clips whose signer permits this scope.

        The explicit, non-raising path — use it to *report* what a release can
        include. :meth:`load` still raises for an individual unconsented clip,
        because asking for one by name is a different mistake from surveying.
        """
        allowed = set(self.audit().permitted)
        return [c for c in self.corpus.clips if c.signer_id in allowed]

    def resolve(self, record: ClipRecord) -> Path:
        """Absolute landmark path, confined to the corpus root.

        A manifest is data, and data can be wrong or hostile; a ``landmark_path`` of
        ``../../../etc/shadow`` should fail loudly rather than be read.
        """
        root = self.corpus.root.resolve()
        candidate = (root / record.landmark_path).resolve()
        if not candidate.is_relative_to(root):
            raise CorpusError(
                f"{record.clip_id}: landmark_path {record.landmark_path!r} escapes the corpus root"
            )
        return candidate

    def load(self, record: ClipRecord | str) -> ClipData:
        """Load one clip, raising if consent does not permit this scope."""
        if isinstance(record, str):
            record = self.corpus.clip(record)
        self.registry.require(record.signer_id, self.scope, self.on)

        path = self.resolve(record)
        if not path.exists():
            raise CorpusError(f"{record.clip_id}: landmark file missing at {path}")
        sequence = LandmarkSequence.load(path)
        return ClipData(record=record, sequence=sequence)

    def load_all(
        self,
        records: list[ClipRecord] | None = None,
        *,
        skip_failed_quality: bool = True,
    ) -> Iterator[ClipData]:
        """Load every permitted clip, skipping those that fail the quality gate.

        Consent refusals are *filtered* here rather than raised, because iterating a
        corpus is the surveying case; a refusal is a fact about the corpus, not an
        error in the caller's request. Quality rejections are reported through
        :attr:`rejected`.
        """
        candidates = records if records is not None else self.permitted_clips()
        self.rejected: list[tuple[str, str]] = []

        for record in candidates:
            try:
                data = self.load(record)
            except ConsentError as exc:
                self.rejected.append((record.clip_id, str(exc)))
                continue
            reason = self.quality.check(record, data.sequence)
            if reason is not None:
                self.rejected.append((record.clip_id, reason))
                if skip_failed_quality:
                    continue
            yield data

    def coverage_summary(self) -> dict[str, float]:
        """Mean per-channel tracking coverage across permitted clips.

        A corpus-level face coverage of 30% means the non-manual channel is mostly
        absent, and any model trained on it cannot represent negation or questions —
        worth knowing before, not after, training.
        """
        totals: dict[str, float] = dict.fromkeys(Channel.NAMES, 0.0)
        count = 0
        for data in self.load_all():
            for name, value in data.sequence.coverage().items():
                totals[name] += value
            count += 1
        if count == 0:
            return dict.fromkeys(Channel.NAMES, 0.0)
        return {name: value / count for name, value in totals.items()}
