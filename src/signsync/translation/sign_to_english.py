"""USL → English (plan §8.4).

    gloss sequence → SemanticFrame → English sentence

Two stages, because they fail differently and should be debuggable separately.
Parsing decides *what was said*; realisation decides *how to say it in English*. A
bad English sentence from a correct frame is a realiser bug; a fluent English
sentence that means the wrong thing is a parser bug, and that is the dangerous one.

Parsing rules follow the structural properties USL shares with most sign languages:
topic-comment ordering, utterance-initial time reference, clause-final question
words, post-predicate negation, no copula and no articles. **The specific rules
here are provisional** — plan §6 requires a USL linguist to review them, and
``docs/limitations.md`` records that they have not been.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..datasets.schema import MarkerType
from ..recognition.base import UNKNOWN_GLOSS, SignPrediction
from .lexicon import ENTITY_POS, PREDICATE_POS, LexEntry, Lexicon, default_lexicon
from .semantics import (
    Aspect,
    Entity,
    FrameBuilder,
    Polarity,
    Role,
    SemanticFrame,
    SpeechAct,
    Tense,
)

__all__ = ["TranslationResult", "SignToEnglish", "parse_glosses", "realise_english"]


@dataclass(frozen=True)
class TranslationResult:
    """Translated text plus the frame it came from."""

    text: str
    frame: SemanticFrame
    confidence: float
    unresolved: tuple[str, ...] = ()

    @property
    def is_reliable(self) -> bool:
        """Whether the output is safe to present without a warning.

        Plan §16.3 requires in-product transparency about limits, so the client has
        to be able to ask this rather than infer it from a number it has to know
        how to interpret.
        """
        return self.confidence >= 0.6 and not self.unresolved


def parse_glosses(
    glosses: list[str] | list[SignPrediction],
    lexicon: Lexicon | None = None,
    *,
    markers: tuple[MarkerType, ...] = (),
) -> SemanticFrame:
    """Build a :class:`SemanticFrame` from a recognised gloss sequence.

    Accepts raw gloss strings or :class:`SignPrediction`s; predictions carry
    confidence, which is propagated into the frame so the client can warn on a
    sentence built from shaky recognition.
    """
    lex = lexicon or default_lexicon()
    builder = FrameBuilder()

    tokens: list[tuple[str, float]] = []
    for item in glosses:
        if isinstance(item, SignPrediction):
            if item.gloss == UNKNOWN_GLOSS:
                builder.unresolved.append(UNKNOWN_GLOSS)
                builder.confidences.append(item.confidence)
                continue
            tokens.append((item.gloss.upper(), item.confidence))
        else:
            tokens.append((str(item).upper(), 1.0))

    # Non-manual markers observed on the input are grammar, so they set the speech
    # act and polarity before any word-order evidence is considered: a head shake
    # over a clause negates it even when no NOT sign appears (plan §8.7).
    builder.markers.extend(markers)
    if MarkerType.HEAD_SHAKE in markers:
        builder.polarity = Polarity.NEGATIVE
    if MarkerType.BROW_FURROW in markers:
        builder.speech_act = SpeechAct.CONTENT_QUESTION
    elif MarkerType.BROW_RAISE in markers:
        builder.speech_act = SpeechAct.POLAR_QUESTION

    predicate_entry: LexEntry | None = None
    pending: list[tuple[LexEntry, float]] = []

    for gloss, confidence in tokens:
        entry = lex.get(gloss)
        builder.confidences.append(confidence)
        if entry is None:
            builder.unresolved.append(gloss)
            continue

        if entry.pos == "wh":
            builder.speech_act = SpeechAct.CONTENT_QUESTION
            builder.question_word = entry.gloss
        elif entry.pos == "negator":
            builder.polarity = Polarity.NEGATIVE
        elif entry.pos == "time":
            builder.tense = entry.tense
            builder.add(Entity(entry.gloss, Role.TIME, entry.primary_english, confidence=confidence))
        elif entry.pos == "aspect":
            builder.aspect = entry.aspect
            if entry.aspect is Aspect.COMPLETED and builder.tense is Tense.UNSPECIFIED:
                builder.tense = Tense.PAST
        elif entry.pos in PREDICATE_POS and predicate_entry is None:
            predicate_entry = entry
            builder.predicate = entry.gloss
        elif entry.pos in ENTITY_POS or entry.pos in PREDICATE_POS:
            pending.append((entry, confidence))
        else:
            pending.append((entry, confidence))

    _assign_roles(builder, predicate_entry, pending)

    if builder.speech_act is SpeechAct.STATEMENT and builder.question_word:
        builder.speech_act = SpeechAct.CONTENT_QUESTION
    return builder.build()


def _assign_roles(
    builder: FrameBuilder, predicate: LexEntry | None, pending: list[tuple[LexEntry, float]]
) -> None:
    """Decide who did what to whom.

    Order matters but is not decisive: USL fronts topics, so the first noun is not
    reliably the agent. The heuristics used here are, in order — animate before
    inanimate for the agent slot, locations to LOCATION regardless of position, and
    an intransitive or adjectival predicate takes its single argument as the agent.
    """
    entities = [(entry, conf) for entry, conf in pending if entry.pos in ENTITY_POS]
    others = [(entry, conf) for entry, conf in pending if entry.pos not in ENTITY_POS]

    agent_assigned = False
    for entry, confidence in entities:
        if entry.location:
            role = Role.LOCATION
        elif not agent_assigned and (entry.pos == "pronoun" or entry.animate):
            role = Role.AGENT
            agent_assigned = True
        else:
            role = Role.PATIENT
        builder.add(Entity(entry.gloss, role, entry.primary_english, confidence=confidence))

    for entry, confidence in others:
        role = Role.ATTRIBUTE if entry.pos == "adjective" else entry.default_role()
        builder.add(Entity(entry.gloss, role, entry.primary_english, confidence=confidence))

    # An adjectival predicate ("ME SICK") has an experiencer, not a patient.
    if predicate is not None and predicate.pos == "adjective":
        patients = builder.entities and [e for e in builder.entities if e.role is Role.PATIENT]
        if not agent_assigned and patients:
            index = builder.entities.index(patients[0])
            builder.entities[index] = patients[0].with_role(Role.AGENT)


# --------------------------------------------------------------------------- realisation

_IRREGULAR_PAST = {
    "go": "went",
    "come": "came",
    "give": "gave",
    "see": "saw",
    "know": "knew",
    "understand": "understood",
    "feel": "felt",
    "hurt": "hurt",
}

_SUBJECT_FORMS = {"I": "I", "me": "I", "you": "you", "we": "we", "he": "he", "she": "she"}
_OBJECT_FORMS = {"I": "me", "me": "me", "you": "you", "we": "us", "he": "him", "she": "her"}

_LOCATION_PREPOSITION = {"go": "to", "come": "to", "wait": "at", "work": "at", "learn": "at"}


def realise_english(frame: SemanticFrame, lexicon: Lexicon | None = None) -> str:
    """Render a frame as an English sentence.

    English needs things USL does not mark: articles, a copula, agreement and
    auxiliary do-support. Those are generated here rather than being expected in
    the gloss sequence, which is the whole reason for the intermediate frame.
    """
    lex = lexicon or default_lexicon()

    agent = frame.first(Role.AGENT)
    patient = frame.first(Role.PATIENT)
    location = frame.first(Role.LOCATION)
    time = frame.first(Role.TIME)
    attribute = frame.first(Role.ATTRIBUTE)
    predicate_entry = lex.get(frame.predicate) if frame.predicate else None

    if frame.speech_act is SpeechAct.CONTENT_QUESTION:
        sentence = _realise_content_question(frame, lex, agent, patient, location, predicate_entry)
    else:
        sentence = _realise_clause(
            frame, lex, agent, patient, location, attribute, predicate_entry
        )

    if time is not None and time.english:
        sentence = f"{time.english.capitalize()}, {sentence}" if sentence else time.english

    sentence = sentence.strip()
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    return sentence + ("?" if frame.is_question else ".")


def _subject_word(entity: Entity | None) -> str:
    if entity is None:
        return ""
    return _SUBJECT_FORMS.get(entity.english, _noun_phrase(entity))


def _object_word(entity: Entity | None) -> str:
    if entity is None:
        return ""
    return _OBJECT_FORMS.get(entity.english, _noun_phrase(entity))


def _noun_phrase(entity: Entity) -> str:
    """Add the article USL does not have."""
    word = entity.english or entity.gloss.lower().replace("-", " ")
    if entity.plural or word in {"water", "food", "money", "pain", "medicine", "help"}:
        return word
    if word in _SUBJECT_FORMS or word in _OBJECT_FORMS:
        return word
    article = "the" if entity.definite else ("an" if word[:1] in "aeiou" else "a")
    return f"{article} {word}"


def _conjugate(verb: str, subject: str, frame: SemanticFrame) -> tuple[str, str]:
    """Return ``(auxiliary, verb form)`` for the tense, aspect and polarity."""
    third_person = subject not in {"I", "you", "we", "they"} and subject != ""
    negated = frame.is_negative

    if frame.tense is Tense.FUTURE:
        return ("will not" if negated else "will"), verb
    if frame.tense is Tense.PAST or frame.aspect is Aspect.COMPLETED:
        if negated:
            return "did not", verb
        return "", _IRREGULAR_PAST.get(verb, verb + ("d" if verb.endswith("e") else "ed"))
    if frame.aspect is Aspect.CONTINUOUS:
        be = "am" if subject == "I" else ("is" if third_person else "are")
        return (f"{be} not" if negated else be), verb + "ing"
    if negated:
        return ("does not" if third_person else "do not"), verb
    return "", (verb + "s" if third_person else verb)


def _realise_clause(
    frame: SemanticFrame,
    lex: Lexicon,
    agent: Entity | None,
    patient: Entity | None,
    location: Entity | None,
    attribute: Entity | None,
    predicate: object | None,
) -> str:
    subject = _subject_word(agent)
    parts: list[str] = []

    if predicate is None:
        # A bare greeting is a complete utterance in both languages and takes no
        # subject or copula — "HELLO" is "Hello", not "I am hello".
        if not subject and attribute is not None and lex.get(attribute.gloss) is not None:
            entry = lex.require(attribute.gloss)
            if entry.pos in ("greeting", "particle"):
                return entry.primary_english

        # No verb: USL has no copula, so "ME DEAF" and "ME TEACHER" arrive here and
        # English has to supply "am".
        complement = attribute or patient
        if subject and complement is not None:
            be = "am" if subject == "I" else ("is" if subject not in {"you", "we"} else "are")
            negation = " not" if frame.is_negative else ""
            return f"{subject} {be}{negation} {_noun_phrase(complement)}"
        if location is not None and subject:
            be = "am" if subject == "I" else ("is" if subject not in {"you", "we"} else "are")
            return f"{subject} {be} at {_noun_phrase(location)}"
        return subject or (_noun_phrase(patient) if patient else "")

    entry = predicate  # type: ignore[assignment]
    verb = entry.primary_english  # type: ignore[union-attr]

    if entry.pos == "adjective":  # type: ignore[union-attr]
        be = "am" if subject == "I" else ("is" if subject not in {"you", "we"} else "are")
        negation = " not" if frame.is_negative else ""
        return f"{subject} {be}{negation} {verb}".strip()

    auxiliary, form = _conjugate(verb, subject, frame)
    parts = [subject, auxiliary, form]
    if patient is not None:
        parts.append(_object_word(patient))
    if location is not None:
        preposition = _LOCATION_PREPOSITION.get(verb, "at")
        parts.append(f"{preposition} {_noun_phrase(location)}")
    return " ".join(p for p in parts if p)


def _realise_content_question(
    frame: SemanticFrame,
    lex: Lexicon,
    agent: Entity | None,
    patient: Entity | None,
    location: Entity | None,
    predicate: object | None,
) -> str:
    """Front the question word, as English requires and USL does not."""
    wh_entry = lex.get(frame.question_word) if frame.question_word else None
    wh = wh_entry.primary_english if wh_entry else "what"
    subject = _subject_word(agent)

    if wh == "where":
        subject_phrase = subject or (_noun_phrase(patient) if patient else "")
        if location is not None and not subject_phrase:
            subject_phrase = _noun_phrase(location)
        if predicate is not None and subject_phrase:
            verb = predicate.primary_english  # type: ignore[union-attr]
            auxiliary = "did" if frame.tense is Tense.PAST else "does"
            if subject_phrase in {"I", "you", "we"}:
                auxiliary = "did" if frame.tense is Tense.PAST else "do"
            return f"where {auxiliary} {subject_phrase} {verb}"
        be = "is" if not (patient and patient.plural) else "are"
        return f"where {be} {subject_phrase}" if subject_phrase else "where"

    if wh in ("who", "what") and predicate is None:
        # "YOU NAME WHAT" — the possessed noun is the topic.
        topic = patient or agent
        if topic is not None and agent is not None and agent is not topic:
            possessive = {"I": "my", "you": "your", "we": "our"}.get(agent.english, "the")
            return f"{wh} is {possessive} {topic.english}"
        if topic is not None:
            return f"{wh} is {_noun_phrase(topic)}"
        return wh

    verb = predicate.primary_english if predicate is not None else "do"  # type: ignore[union-attr]
    auxiliary = "did" if frame.tense is Tense.PAST else (
        "do" if subject in {"I", "you", "we"} or not subject else "does"
    )
    tail = f" {_object_word(patient)}" if patient is not None else ""
    return f"{wh} {auxiliary} {subject} {verb}{tail}".replace("  ", " ").strip()


class SignToEnglish:
    """Gloss sequence in, English out (plan §8.4)."""

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or default_lexicon()

    def translate(
        self,
        glosses: list[str] | list[SignPrediction],
        *,
        markers: tuple[MarkerType, ...] = (),
    ) -> TranslationResult:
        frame = parse_glosses(glosses, self.lexicon, markers=markers)
        return TranslationResult(
            text=realise_english(frame, self.lexicon),
            frame=frame,
            confidence=frame.confidence,
            unresolved=frame.unresolved,
        )
