"""English → USL (plan §8.6).

    English text → SemanticFrame → gloss sequence + non-manual markers

Plan §3 identifies this direction as the least mature across the region and the
place where the project can contribute most — and as the highest-risk component.
Plan §8.6 is explicit that it must learn USL structure rather than substituting a
sign per English word.

What the generator does, and why each step is not optional:

* **Drops English function words.** Articles, the copula and do-support have no
  signed equivalent. Signing them produces word salad, not emphasis.
* **Reorders to topic-comment.** Time reference goes first, then the topic, then
  the comment, with the question word clause-final. English word order signed
  directly is Signed Exact English, a manual code for English — not USL, and not
  what a Deaf USL signer reads fluently.
* **Attaches non-manual markers with scope.** Brow raise over a whole clause is a
  yes/no question; a head shake over the predicate is negation. Without them the
  sentence is not merely less expressive, it means something different.
* **Reports words it could not sign.** Silently dropping them yields a fluent
  sentence with different content.

The ordering rules are provisional and await linguist review (``docs/limitations.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..datasets.schema import MarkerType, NonManualMarker
from .lexicon import Lexicon, default_lexicon
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

__all__ = ["GlossSequence", "EnglishToSign", "analyse_english", "generate_glosses"]

#: English words with no signed equivalent. Removed, not translated.
_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "of", "to", "at", "in", "on", "for", "with", "and",
        "that", "this", "there", "it", "its", "please", "would", "could", "should",
        "some", "any", "very", "just", "so",
        # Tense auxiliaries: dropped because the frame already carries the tense,
        # which USL expresses with a time sign rather than on the verb.
        "will", "shall", "going", "gonna", "have", "has", "had",
    }
)

#: Modals deliberately NOT treated as function words.
#:
#: CAN, MUST and their kin are real signs carrying modality the frame cannot
#: currently represent. Listing them as function words would drop them silently and
#: turn "you must go" into "you go" — an instruction into a description. Leaving
#: them unresolved means the client is told the sentence was only partly translated.
UNSUPPORTED_MODALS = frozenset({"can", "cannot", "must", "may", "might", "need to", "ought"})

_NEGATIONS = frozenset({"not", "no", "never", "don't", "dont", "doesn't", "doesnt", "cannot", "can't", "cant"})
_PAST_MARKERS = frozenset({"was", "were", "did", "had", "yesterday", "ago", "earlier"})
_FUTURE_MARKERS = frozenset({"will", "shall", "tomorrow", "later", "soon", "going"})
_POLAR_OPENERS = frozenset({"do", "does", "did", "are", "is", "am", "can", "will", "would", "have", "has"})


@dataclass(frozen=True)
class GlossSequence:
    """A signable utterance: ordered glosses plus the markers that scope over them."""

    glosses: tuple[str, ...]
    markers: tuple[NonManualMarker, ...]
    frame: SemanticFrame
    unresolved: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.glosses)

    def __iter__(self):
        return iter(self.glosses)

    @property
    def is_complete(self) -> bool:
        """Whether every input word reached a sign."""
        return not self.unresolved

    def notation(self) -> str:
        """Gloss line with marker scopes, in the convention used in the literature.

        Markers are written as a bar over the glosses they scope, e.g.::

            _________wh
            HOSPITAL WHERE

        Rendered as text so it can go in logs, annotation tools and the API trace.
        """
        gloss_line = " ".join(self.glosses)
        if not self.markers:
            return gloss_line

        lines: list[str] = []
        for marker in self.markers:
            scoped = marker.scopes_glosses or self.glosses
            width = len(" ".join(scoped)) if scoped else len(gloss_line)
            start = gloss_line.find(" ".join(scoped)) if scoped else 0
            label = _MARKER_LABELS.get(marker.marker, marker.marker.value)
            lines.append(" " * max(start, 0) + "_" * max(width, 1) + label)
        lines.append(gloss_line)
        return "\n".join(lines)


_MARKER_LABELS = {
    MarkerType.BROW_RAISE: "y/n",
    MarkerType.BROW_FURROW: "wh",
    MarkerType.HEAD_SHAKE: "neg",
    MarkerType.HEAD_NOD: "aff",
    MarkerType.HEAD_TILT: "t",
    MarkerType.BODY_SHIFT: "rs",
}


def analyse_english(text: str, lexicon: Lexicon | None = None) -> SemanticFrame:
    """Parse an English sentence into a :class:`SemanticFrame`."""
    lex = lexicon or default_lexicon()
    builder = FrameBuilder()

    raw = text.strip()
    if not raw:
        return builder.build()

    lowered = raw.lower()
    words = re.findall(r"[a-z']+", lowered)
    if not words:
        return builder.build()

    if any(word in _NEGATIONS for word in words):
        builder.polarity = Polarity.NEGATIVE
    if any(word in _PAST_MARKERS for word in words):
        builder.tense = Tense.PAST
    elif any(word in _FUTURE_MARKERS for word in words):
        builder.tense = Tense.FUTURE
    if "already" in words or "finished" in words:
        builder.aspect = Aspect.COMPLETED

    question = raw.endswith("?")
    wh_entry = next(
        (entry for word in words if (entry := lex.lookup_english(word)) and entry.pos == "wh"),
        None,
    )
    if wh_entry is not None:
        builder.speech_act = SpeechAct.CONTENT_QUESTION
        builder.question_word = wh_entry.gloss
    elif question or words[0] in _POLAR_OPENERS:
        builder.speech_act = SpeechAct.POLAR_QUESTION if question else SpeechAct.STATEMENT

    matched = _match_tokens(lowered, lex)

    agent_assigned = False
    predicate_assigned = False
    for surface, entry in matched:
        if entry is None:
            if surface not in _FUNCTION_WORDS and surface not in _NEGATIONS:
                builder.unresolved.append(surface)
            continue
        if entry.pos in ("wh", "negator"):
            continue
        if entry.pos == "time":
            builder.tense = entry.tense if entry.tense is not Tense.UNSPECIFIED else builder.tense
            builder.add(Entity(entry.gloss, Role.TIME, entry.primary_english))
            continue
        if entry.pos == "aspect":
            builder.aspect = entry.aspect
            continue
        if entry.pos in ("verb", "adjective") and not predicate_assigned:
            builder.predicate = entry.gloss
            predicate_assigned = True
            continue
        if entry.pos in ("noun", "pronoun"):
            if entry.location:
                role = Role.LOCATION
            elif not agent_assigned and (entry.pos == "pronoun" or entry.animate):
                role, agent_assigned = Role.AGENT, True
            else:
                role = Role.PATIENT
            builder.add(Entity(entry.gloss, role, entry.primary_english, plural=entry.plural))
            continue
        builder.add(Entity(entry.gloss, entry.default_role(), entry.primary_english))

    return builder.build()


def _match_tokens(lowered: str, lex: Lexicon) -> list[tuple[str, object]]:
    """Tokenise, matching multi-word signs before single words.

    "thank you" is one sign; matching word by word would emit THANK-YOU's parts as
    two unrelated signs, or fail on both.
    """
    text = lowered
    for phrase in lex.phrases():
        if phrase in text:
            text = text.replace(phrase, phrase.replace(" ", "_"))

    tokens = re.findall(r"[a-z'_]+", text)
    matched: list[tuple[str, object]] = []
    for token in tokens:
        surface = token.replace("_", " ")
        matched.append((surface, lex.lookup_english(surface)))
    return matched


def generate_glosses(
    frame: SemanticFrame, lexicon: Lexicon | None = None, *, fps: float = 30.0
) -> GlossSequence:
    """Order a frame into USL gloss order and attach its non-manual markers.

    Order applied: TIME → TOPIC/LOCATION → AGENT → PREDICATE → PATIENT →
    NEGATION → WH. Question words go clause-final and time reference
    utterance-initial, which is where sign languages generally put them and where
    English does not.
    """
    lex = lexicon or default_lexicon()
    glosses: list[str] = []

    for entity in frame.entities_with(Role.TIME):
        glosses.append(entity.gloss)
    for entity in frame.entities_with(Role.TOPIC):
        glosses.append(entity.gloss)

    locations = frame.entities_with(Role.LOCATION)
    agents = frame.entities_with(Role.AGENT)
    patients = frame.entities_with(Role.PATIENT)
    attributes = frame.entities_with(Role.ATTRIBUTE)

    # A location is the topic of a "where" question — "HOSPITAL WHERE", not
    # "WHERE HOSPITAL".
    if frame.speech_act is SpeechAct.CONTENT_QUESTION and locations:
        glosses.extend(e.gloss for e in locations)
        locations = ()

    glosses.extend(e.gloss for e in agents)
    if frame.predicate:
        glosses.append(frame.predicate)
    glosses.extend(e.gloss for e in patients)
    glosses.extend(e.gloss for e in attributes)
    glosses.extend(e.gloss for e in locations)

    if frame.polarity is Polarity.NEGATIVE and "NOT" in lex:
        glosses.append("NOT")
    if frame.question_word:
        glosses.append(frame.question_word)

    # Drop duplicates while keeping order: an entity may be both topic and agent.
    ordered = list(dict.fromkeys(glosses))
    markers = _marker_spans(frame, ordered, fps=fps)

    return GlossSequence(
        glosses=tuple(ordered),
        markers=markers,
        frame=frame,
        unresolved=frame.unresolved,
    )


def _marker_spans(
    frame: SemanticFrame, glosses: list[str], *, fps: float, seconds_per_gloss: float = 0.7
) -> tuple[NonManualMarker, ...]:
    """Attach each required marker to the glosses it scopes over.

    Scope is the point. A brow raise over the whole clause is a yes/no question; the
    same brow raise over one sign marks that sign as the topic. Emitting markers
    without spans would leave the motion generator to guess, and it would guess the
    whole clause every time.
    """
    if not glosses:
        return ()

    total = len(glosses) * seconds_per_gloss
    markers: list[NonManualMarker] = []

    for marker in frame.required_markers():
        if marker is MarkerType.HEAD_SHAKE:
            # Negation scopes over the predicate and what follows it.
            start_index = glosses.index(frame.predicate) if frame.predicate in glosses else 0
            scoped = tuple(glosses[start_index:])
            start = start_index * seconds_per_gloss
        elif marker is MarkerType.HEAD_TILT:
            scoped = (glosses[0],)
            start = 0.0
        else:
            scoped = tuple(glosses)
            start = 0.0
        markers.append(
            NonManualMarker(
                marker=marker,
                start=start,
                end=total,
                intensity=0.9 if marker is MarkerType.HEAD_SHAKE else 0.8,
                scopes_glosses=scoped,
            )
        )
    return tuple(markers)


class EnglishToSign:
    """English text in, signable gloss sequence out (plan §8.6)."""

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or default_lexicon()

    def translate(self, text: str, *, fps: float = 30.0) -> GlossSequence:
        frame = analyse_english(text, self.lexicon)
        return generate_glosses(frame, self.lexicon, fps=fps)
