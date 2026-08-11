"""The semantic intermediate representation (plan §7.2).

Plan §7.2 is explicit that the system must not map ``SIGN → English word``:

    Video → Visual Features → Sign/Gloss Units → Semantic Representation → English
    English → Language Understanding → Semantic Representation → USL Gloss → Motion

:class:`SemanticFrame` is that middle box, and it is the only thing the two
translation directions share. Both directions are therefore forced through a
representation that is neither English word order nor gloss order, which is what
stops a dictionary substitution creeping back in.

The frame carries what the two grammars disagree about:

* **Speech act** — USL marks questions non-manually (brow position) and by moving
  the question word, English by word order and punctuation.
* **Polarity** — USL negation is a head shake scoping over a clause; English
  negation is a word inside the verb phrase.
* **Roles, not positions** — "who did what to whom" survives reordering; subject
  position does not.
* **Spatial loci** — USL establishes referents at points in signing space and
  points back at them. English uses pronouns and repeats nouns. Losing the locus
  loses the reference.

Every field is explicit about being unknown rather than defaulting to the English
answer, because a frame that quietly assumes present tense and positive polarity
will translate a negated past-tense sentence into a cheerful present-tense one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from ..datasets.schema import MarkerType

__all__ = [
    "SpeechAct",
    "Polarity",
    "Tense",
    "Aspect",
    "Role",
    "Entity",
    "SemanticFrame",
]


class SpeechAct(str, Enum):
    """What the utterance is doing."""

    STATEMENT = "statement"
    POLAR_QUESTION = "polar_question"  # yes/no — brow raise in USL
    CONTENT_QUESTION = "content_question"  # wh- — brow furrow in USL
    COMMAND = "command"
    CONDITIONAL = "conditional"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Tense(str, Enum):
    """USL marks time lexically (time signs, often utterance-initial) rather than
    on the verb; English marks it morphologically. The frame stores the fact, and
    each direction expresses it its own way."""

    UNSPECIFIED = "unspecified"
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"


class Aspect(str, Enum):
    UNSPECIFIED = "unspecified"
    SIMPLE = "simple"
    CONTINUOUS = "continuous"
    COMPLETED = "completed"
    HABITUAL = "habitual"


class Role(str, Enum):
    """Semantic roles. Deliberately not "subject" and "object" — those are
    positions in a sentence, and the whole point of the frame is to survive
    reordering between two languages that order differently."""

    AGENT = "agent"
    PATIENT = "patient"
    RECIPIENT = "recipient"
    LOCATION = "location"
    TIME = "time"
    MANNER = "manner"
    ATTRIBUTE = "attribute"
    TOPIC = "topic"


@dataclass(frozen=True)
class Entity:
    """A participant or circumstance in the frame."""

    gloss: str
    role: Role
    english: str = ""
    plural: bool = False
    definite: bool = True
    locus: str | None = None
    """Point in signing space this referent was established at (``"a"``, ``"b"``…).

    Kept because USL verbs agree with loci — the direction a verb moves says who
    did what to whom. Dropping the locus turns "she gives me" and "I give her"
    into the same frame."""

    confidence: float = 1.0

    def with_role(self, role: Role) -> Entity:
        return replace(self, role=role)


@dataclass(frozen=True)
class SemanticFrame:
    """One translated clause."""

    predicate: str | None = None
    """Gloss of the main verb or predicate, if the utterance has one."""

    entities: tuple[Entity, ...] = ()
    speech_act: SpeechAct = SpeechAct.STATEMENT
    polarity: Polarity = Polarity.POSITIVE
    tense: Tense = Tense.UNSPECIFIED
    aspect: Aspect = Aspect.UNSPECIFIED
    question_word: str | None = None
    """Gloss of the wh-word (``WHERE``, ``WHAT``…) for a content question."""

    markers: tuple[MarkerType, ...] = ()
    """Non-manual markers observed on input, or required on output."""

    confidence: float = 1.0
    """Lowest confidence of the evidence this frame was built from.

    Propagated end to end so the client can show it (plan §16.3). Taking the
    minimum rather than the mean is deliberate: one badly recognised sign can make
    the whole sentence wrong, and averaging it away hides exactly the case a user
    needs to be warned about."""

    unresolved: tuple[str, ...] = ()
    """Input tokens with no lexicon entry.

    Surfaced rather than dropped: "I could not translate these three words" is a
    usable message, whereas silently omitting them produces a fluent sentence that
    means something else."""

    notes: str = ""

    def entities_with(self, role: Role) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.role is role)

    def first(self, role: Role) -> Entity | None:
        found = self.entities_with(role)
        return found[0] if found else None

    @property
    def is_question(self) -> bool:
        return self.speech_act in (SpeechAct.POLAR_QUESTION, SpeechAct.CONTENT_QUESTION)

    @property
    def is_negative(self) -> bool:
        return self.polarity is Polarity.NEGATIVE

    def required_markers(self) -> tuple[MarkerType, ...]:
        """Non-manual markers this frame's meaning requires when signed.

        These are grammar, not emphasis: without a head shake the negation is not
        expressed at all, and without brow position a question reads as a statement
        (plan §8.7).
        """
        markers: list[MarkerType] = []
        if self.speech_act is SpeechAct.CONTENT_QUESTION:
            markers.append(MarkerType.BROW_FURROW)
        elif self.speech_act is SpeechAct.POLAR_QUESTION:
            markers.append(MarkerType.BROW_RAISE)
        elif self.speech_act is SpeechAct.CONDITIONAL:
            markers.append(MarkerType.BROW_RAISE)
        if self.polarity is Polarity.NEGATIVE:
            markers.append(MarkerType.HEAD_SHAKE)
        if self.entities_with(Role.TOPIC):
            markers.append(MarkerType.HEAD_TILT)
        return tuple(dict.fromkeys(markers))

    def with_confidence(self, *values: float) -> SemanticFrame:
        """Fold more evidence into the frame's confidence."""
        return replace(self, confidence=min([self.confidence, *values], default=self.confidence))

    def describe(self) -> str:
        """Compact debug rendering, used in logs and the API's trace output."""
        parts = [f"act={self.speech_act.value}"]
        if self.predicate:
            parts.append(f"pred={self.predicate}")
        for entity in self.entities:
            locus = f"@{entity.locus}" if entity.locus else ""
            parts.append(f"{entity.role.value}={entity.gloss}{locus}")
        if self.polarity is Polarity.NEGATIVE:
            parts.append("neg")
        if self.tense is not Tense.UNSPECIFIED:
            parts.append(f"tense={self.tense.value}")
        if self.question_word:
            parts.append(f"wh={self.question_word}")
        if self.unresolved:
            parts.append(f"unresolved={list(self.unresolved)}")
        return " ".join(parts)


@dataclass
class FrameBuilder:
    """Mutable accumulator used by both analysers."""

    predicate: str | None = None
    entities: list[Entity] = field(default_factory=list)
    speech_act: SpeechAct = SpeechAct.STATEMENT
    polarity: Polarity = Polarity.POSITIVE
    tense: Tense = Tense.UNSPECIFIED
    aspect: Aspect = Aspect.UNSPECIFIED
    question_word: str | None = None
    markers: list[MarkerType] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def add(self, entity: Entity) -> None:
        self.entities.append(entity)

    def build(self) -> SemanticFrame:
        return SemanticFrame(
            predicate=self.predicate,
            entities=tuple(self.entities),
            speech_act=self.speech_act,
            polarity=self.polarity,
            tense=self.tense,
            aspect=self.aspect,
            question_word=self.question_word,
            markers=tuple(dict.fromkeys(self.markers)),
            confidence=min(self.confidences) if self.confidences else 1.0,
            unresolved=tuple(self.unresolved),
        )
