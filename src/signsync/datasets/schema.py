"""Corpus schema — the annotation contract (plan §9.3).

Follows the gloss + phonetic-notation pattern plan §9.3 borrows from the Kenyan
Sign Language dataset work, so the corpus is reusable by other researchers rather
than locked to this codebase:

    video/landmarks + gloss + English translation + start/end + non-manual markers
    + signer metadata (consented) + optional HamNoSys

Three details that are easy to get wrong and expensive to fix later:

* **Signer metadata is a diversity instrument, not decoration.** Plan §9.3 recruits
  across age, gender, district and signing background specifically to attack the
  generalisation risk in §14. Those fields are therefore required, and
  :meth:`Corpus.diversity_report` measures whether recruitment actually achieved it.
* **Recording conditions are recorded.** "A model trained on one studio setup will
  fail on a phone camera in a clinic", so the conditions have to be visible in the
  data, not just varied and forgotten.
* **Non-manual markers are structured spans, not a free-text note.** They carry
  negation, question and conditional marking, and the generator downstream needs
  their timing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..errors import CorpusError

__all__ = [
    "AgeBand",
    "SigningBackground",
    "Handedness",
    "MarkerType",
    "NonManualMarker",
    "RecordingConditions",
    "SignerProfile",
    "ClipRecord",
    "Corpus",
    "CONCENTRATION_THRESHOLDS",
]

MANIFEST_VERSION = 1

#: Maximum share of clips one value of a dimension may hold before
#: :meth:`Corpus.diversity_warnings` complains.
#:
#: The limits differ because the dimensions do. District, age band and signing
#: background are recruitment targets under this project's control, so a third of
#: the corpus is already a concentration worth acting on. Handedness is not:
#: right-handed dominance is a property of the population, and only a corpus with
#: essentially no left-handed signers is a problem — but it is a real one, because
#: a left-handed signer mirrors every sign. Lighting and device track plan §9.3's
#: requirement to vary recording conditions deliberately.
CONCENTRATION_THRESHOLDS: dict[str, float] = {
    "signer": 0.35,
    "district": 0.40,
    "age_band": 0.45,
    "background": 0.45,
    "handedness": 0.95,
    "lighting": 0.60,
    "device": 0.60,
}


class AgeBand(str, Enum):
    """Coarse bands rather than dates of birth — the analysis needs the band, and
    storing less identifying detail is the point (data minimisation, plan §16)."""

    UNDER_18 = "under_18"
    A18_29 = "18_29"
    A30_44 = "30_44"
    A45_59 = "45_59"
    A60_PLUS = "60_plus"


class SigningBackground(str, Enum):
    """How the signer acquired USL.

    Plan §9.3 calls out school-taught versus home-sign-influenced signing as a
    recruitment axis; the two produce visibly different signing, and a corpus of
    only the former will not generalise to the latter.
    """

    NATIVE_DEAF_FAMILY = "native_deaf_family"
    SCHOOL_TAUGHT = "school_taught"
    HOME_SIGN_INFLUENCED = "home_sign_influenced"
    LATE_LEARNER = "late_learner"
    INTERPRETER = "interpreter"


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class MarkerType(str, Enum):
    """Non-manual markers with grammatical function (plan §8.7)."""

    BROW_RAISE = "brow_raise"  # yes/no questions, topics, conditionals
    BROW_FURROW = "brow_furrow"  # wh-questions
    HEAD_SHAKE = "head_shake"  # negation
    HEAD_NOD = "head_nod"  # affirmation, assertion
    HEAD_TILT = "head_tilt"  # topic marking, role shift
    BODY_SHIFT = "body_shift"  # role shift, spatial referencing
    EYE_GAZE = "eye_gaze"  # spatial referencing, addressee marking
    MOUTH_MORPHEME = "mouth_morpheme"  # adverbial ("mm", "th", "cha")
    MOUTHING = "mouthing"  # spoken-language mouth pattern
    PUFFED_CHEEKS = "puffed_cheeks"  # size/extent
    SQUINT = "squint"  # shared-knowledge marking


@dataclass(frozen=True)
class NonManualMarker:
    """A marker and the span of the utterance it scopes over.

    ``start``/``end`` are seconds from the clip start. The span matters: brow raise
    over the whole clause marks a yes/no question, while the same brow raise over
    one sign marks a topic.
    """

    marker: MarkerType
    start: float
    end: float
    intensity: float = 1.0
    scopes_glosses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise CorpusError(f"{self.marker.value}: end {self.end} precedes start {self.start}")
        if not 0.0 <= self.intensity <= 1.0:
            raise CorpusError(f"{self.marker.value}: intensity must be in [0, 1]")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker.value,
            "start": self.start,
            "end": self.end,
            "intensity": self.intensity,
            "scopes_glosses": list(self.scopes_glosses),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NonManualMarker:
        return cls(
            marker=MarkerType(data["marker"]),
            start=float(data["start"]),
            end=float(data["end"]),
            intensity=float(data.get("intensity", 1.0)),
            scopes_glosses=tuple(data.get("scopes_glosses", [])),
        )


@dataclass(frozen=True)
class RecordingConditions:
    """Deliberately varied capture conditions (plan §9.3)."""

    lighting: str = "unspecified"  # e.g. daylight, indoor_fluorescent, low_light
    background: str = "unspecified"  # e.g. plain, cluttered, outdoor
    camera_distance: str = "unspecified"  # e.g. close, medium, far
    device: str = "unspecified"  # e.g. phone, webcam, camcorder
    location: str = "unspecified"  # e.g. clinic, school, home
    resolution: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordingConditions:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SignerProfile:
    """A participating signer, pseudonymised.

    ``signer_id`` is a pseudonym and doubles as the consent ``participant_id``. Names
    and contact details live in the consent record and never enter the manifest.
    """

    signer_id: str
    age_band: AgeBand
    background: SigningBackground
    district: str
    handedness: Handedness = Handedness.RIGHT
    gender: str = "unspecified"
    is_deaf: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "age_band": self.age_band.value,
            "background": self.background.value,
            "district": self.district,
            "handedness": self.handedness.value,
            "gender": self.gender,
            "is_deaf": self.is_deaf,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignerProfile:
        return cls(
            signer_id=data["signer_id"],
            age_band=AgeBand(data["age_band"]),
            background=SigningBackground(data["background"]),
            district=data["district"],
            handedness=Handedness(data.get("handedness", "right")),
            gender=data.get("gender", "unspecified"),
            is_deaf=bool(data.get("is_deaf", True)),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class ClipRecord:
    """One annotated clip.

    Isolated signs carry a single-entry ``glosses``; continuous clips carry the full
    sequence plus per-gloss ``boundaries``. Keeping one record type for both means
    the isolated corpus (V1/V2) and the continuous corpus (V3/V4) share every
    loader, split and metric.
    """

    clip_id: str
    signer_id: str
    glosses: tuple[str, ...]
    english: str
    landmark_path: str
    duration: float
    fps: float = 30.0
    domain: str = "general"
    boundaries: tuple[tuple[float, float], ...] = ()
    markers: tuple[NonManualMarker, ...] = ()
    hamnosys: tuple[str, ...] = ()
    conditions: RecordingConditions = field(default_factory=RecordingConditions)
    annotator_id: str = "unknown"
    verified_by: str | None = None
    quality: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.glosses:
            raise CorpusError(f"{self.clip_id}: a clip needs at least one gloss")
        if self.boundaries and len(self.boundaries) != len(self.glosses):
            raise CorpusError(
                f"{self.clip_id}: {len(self.boundaries)} boundaries for {len(self.glosses)} glosses"
            )
        if self.hamnosys and len(self.hamnosys) != len(self.glosses):
            raise CorpusError(
                f"{self.clip_id}: {len(self.hamnosys)} HamNoSys entries for "
                f"{len(self.glosses)} glosses"
            )
        if self.duration <= 0:
            raise CorpusError(f"{self.clip_id}: duration must be positive")

    @property
    def is_continuous(self) -> bool:
        return len(self.glosses) > 1

    @property
    def gloss(self) -> str:
        """The single gloss of an isolated clip."""
        if self.is_continuous:
            raise CorpusError(f"{self.clip_id} is continuous; use .glosses")
        return self.glosses[0]

    def markers_of(self, marker: MarkerType) -> tuple[NonManualMarker, ...]:
        return tuple(m for m in self.markers if m.marker is marker)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "signer_id": self.signer_id,
            "glosses": list(self.glosses),
            "english": self.english,
            "landmark_path": self.landmark_path,
            "duration": self.duration,
            "fps": self.fps,
            "domain": self.domain,
            "boundaries": [list(b) for b in self.boundaries],
            "markers": [m.to_dict() for m in self.markers],
            "hamnosys": list(self.hamnosys),
            "conditions": self.conditions.to_dict(),
            "annotator_id": self.annotator_id,
            "verified_by": self.verified_by,
            "quality": self.quality,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipRecord:
        return cls(
            clip_id=data["clip_id"],
            signer_id=data["signer_id"],
            glosses=tuple(data["glosses"]),
            english=data.get("english", ""),
            landmark_path=data["landmark_path"],
            duration=float(data["duration"]),
            fps=float(data.get("fps", 30.0)),
            domain=data.get("domain", "general"),
            boundaries=tuple(tuple(b) for b in data.get("boundaries", [])),  # type: ignore[misc]
            markers=tuple(NonManualMarker.from_dict(m) for m in data.get("markers", [])),
            hamnosys=tuple(data.get("hamnosys", [])),
            conditions=RecordingConditions.from_dict(data.get("conditions", {})),
            annotator_id=data.get("annotator_id", "unknown"),
            verified_by=data.get("verified_by"),
            quality=float(data.get("quality", 1.0)),
            notes=data.get("notes", ""),
        )


@dataclass
class Corpus:
    """A manifest of clips and the signers who produced them.

    The manifest holds metadata only; landmark arrays live beside it on disk and are
    loaded on demand through :mod:`signsync.datasets.corpus`, which is where consent
    is enforced.
    """

    name: str
    root: Path
    clips: list[ClipRecord] = field(default_factory=list)
    signers: dict[str, SignerProfile] = field(default_factory=dict)
    notes: str = ""

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self):
        return iter(self.clips)

    def add_signer(self, profile: SignerProfile) -> None:
        self.signers[profile.signer_id] = profile

    def add_clip(self, clip: ClipRecord) -> None:
        if clip.signer_id not in self.signers:
            raise CorpusError(
                f"{clip.clip_id}: unknown signer {clip.signer_id!r}; add the profile first so "
                "diversity reporting and signer-independent splits stay meaningful"
            )
        self.clips.append(clip)

    def clip(self, clip_id: str) -> ClipRecord:
        for record in self.clips:
            if record.clip_id == clip_id:
                return record
        raise CorpusError(f"no clip {clip_id!r} in corpus {self.name!r}")

    def signer_ids(self) -> list[str]:
        return sorted({c.signer_id for c in self.clips})

    def vocabulary(self) -> list[str]:
        return sorted({g for c in self.clips for g in c.glosses})

    def isolated(self) -> list[ClipRecord]:
        return [c for c in self.clips if not c.is_continuous]

    def continuous(self) -> list[ClipRecord]:
        return [c for c in self.clips if c.is_continuous]

    def filter(self, **criteria: Any) -> list[ClipRecord]:
        """Select clips by attribute equality, e.g. ``filter(domain="health")``."""
        return [
            c
            for c in self.clips
            if all(getattr(c, key, None) == value for key, value in criteria.items())
        ]

    def gloss_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for clip in self.clips:
            for gloss in clip.glosses:
                counts[gloss] = counts.get(gloss, 0) + 1
        return dict(sorted(counts.items()))

    def diversity_report(self) -> dict[str, dict[str, int]]:
        """Clip counts by signer, district, age band, background and conditions.

        Plan §9.3 recruits for diversity; this is how you find out whether the
        recruitment actually happened, before a model trained on it is deployed to
        signers it has never seen.
        """
        report: dict[str, dict[str, int]] = {
            "signer": {},
            "district": {},
            "age_band": {},
            "background": {},
            "handedness": {},
            "lighting": {},
            "device": {},
            "domain": {},
        }
        for clip in self.clips:
            profile = self.signers.get(clip.signer_id)
            _bump(report["signer"], clip.signer_id)
            _bump(report["domain"], clip.domain)
            _bump(report["lighting"], clip.conditions.lighting)
            _bump(report["device"], clip.conditions.device)
            if profile is not None:
                _bump(report["district"], profile.district)
                _bump(report["age_band"], profile.age_band.value)
                _bump(report["background"], profile.background.value)
                _bump(report["handedness"], profile.handedness.value)
        return report

    def diversity_warnings(
        self, *, thresholds: dict[str, float] | None = None
    ) -> list[str]:
        """Flag recruitment dimensions dominated by a single value.

        A corpus where 80% of clips come from one district is not a diverse corpus,
        however many clips it has — and this is the failure that looks fine in
        training metrics and shows up only when the system meets a new signer.

        Only the dimensions in :data:`CONCENTRATION_THRESHOLDS` are checked, at the
        thresholds declared there. ``domain`` is excluded on purpose: plan §9.2
        deliberately *concentrates* the corpus on health and education, so flagging
        that would be warning about the plan working.
        """
        limits = thresholds or CONCENTRATION_THRESHOLDS
        warnings: list[str] = []
        total = len(self.clips)
        if total == 0:
            return ["corpus is empty"]
        for dimension, counts in self.diversity_report().items():
            limit = limits.get(dimension)
            if limit is None or not counts:
                continue
            value, count = max(counts.items(), key=lambda kv: kv[1])
            share = count / total
            if share > limit:
                warnings.append(
                    f"{share:.0%} of clips share {dimension}={value!r} "
                    f"({count}/{total}, limit {limit:.0%}); recruit wider (plan §9.3)"
                )
        return warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "name": self.name,
            "notes": self.notes,
            "signers": [s.to_dict() for s in self.signers.values()],
            "clips": [c.to_dict() for c in self.clips],
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.root / "manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        manifest = Path(path)
        if manifest.is_dir():
            manifest = manifest / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        version = data.get("version")
        if version != MANIFEST_VERSION:
            raise CorpusError(
                f"{manifest}: manifest version {version!r}, expected {MANIFEST_VERSION}"
            )
        corpus = cls(name=data["name"], root=manifest.parent, notes=data.get("notes", ""))
        for entry in data.get("signers", []):
            corpus.add_signer(SignerProfile.from_dict(entry))
        for entry in data.get("clips", []):
            corpus.add_clip(ClipRecord.from_dict(entry))
        return corpus


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1
