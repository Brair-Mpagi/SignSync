"""The gloss lexicon, and the loud warning attached to it.

Plan §2 notes that UNAD and Kyambogo have produced print references — the *Manual
of Ugandan Signs* and the *Uganda Sign Language Dictionary* — which are the right
source for a real lexicon. This module is the loader for that data; the file it
currently ships (``resources/usl_lexicon.json``) is a **placeholder written to
exercise the pipeline**, and :meth:`Lexicon.is_validated` reports that fact rather
than letting it be forgotten.

A lexicon entry is not a translation pair. It carries the grammatical properties
the two translation directions need — part of speech, whether a verb agrees with
spatial loci, whether a noun is a location or animate — because those, not the
English gloss label, are what decides where a sign goes in a USL sentence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..errors import SignSyncError
from .semantics import Aspect, Role, Tense

__all__ = ["LexEntry", "Lexicon", "default_lexicon", "PLACEHOLDER_WARNING"]

PLACEHOLDER_WARNING = (
    "The bundled USL lexicon is a placeholder for pipeline testing and has NOT been "
    "reviewed by a USL linguist or the Deaf Advisory Board. Do not present its output "
    "as Ugandan Sign Language. See docs/limitations.md."
)

_RESOURCE = Path(__file__).resolve().parent.parent / "resources" / "usl_lexicon.json"

#: Parts of speech that can head a predicate.
PREDICATE_POS = frozenset({"verb", "adjective"})

#: Parts of speech that can fill an entity slot.
ENTITY_POS = frozenset({"noun", "pronoun"})


@dataclass(frozen=True)
class LexEntry:
    """One sign, with the grammar the translators need."""

    gloss: str
    pos: str
    english: tuple[str, ...]
    plural: bool = False
    person: int | None = None
    transitive: bool = False
    directional: bool = False
    agreeing: bool = False
    """Verb agrees with spatial loci — its movement encodes who acts on whom."""

    location: bool = False
    animate: bool = False
    mass: bool = False
    tense: Tense = Tense.UNSPECIFIED
    aspect: Aspect = Aspect.UNSPECIFIED

    @property
    def primary_english(self) -> str:
        return self.english[0] if self.english else self.gloss.lower().replace("-", " ")

    def default_role(self) -> Role:
        """Role this sign takes when nothing else determines it."""
        if self.pos == "time":
            return Role.TIME
        if self.location:
            return Role.LOCATION
        if self.pos in ENTITY_POS:
            return Role.PATIENT
        if self.pos == "adjective":
            return Role.ATTRIBUTE
        if self.pos == "verb":
            # A second verb in a clause is the first one's complement — "NEED HELP"
            # is "need help", not two predicates. Falling through to TOPIC here put
            # it at the front of the gloss order and dropped it from the English.
            return Role.PATIENT
        if self.pos in ("greeting", "particle"):
            return Role.ATTRIBUTE
        return Role.TOPIC

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LexEntry:
        return cls(
            gloss=data["gloss"],
            pos=data["pos"],
            english=tuple(data.get("english", [])),
            plural=bool(data.get("plural", False)),
            person=data.get("person"),
            transitive=bool(data.get("transitive", False)),
            directional=bool(data.get("directional", False)),
            agreeing=bool(data.get("agreeing", False)),
            location=bool(data.get("location", False)),
            animate=bool(data.get("animate", False)),
            mass=bool(data.get("mass", False)),
            tense=Tense(data.get("tense", "unspecified")),
            aspect=Aspect(data.get("aspect", "unspecified")),
        )


@dataclass
class Lexicon:
    """Gloss lookup in both directions."""

    entries: dict[str, LexEntry] = field(default_factory=dict)
    status: str = "PLACEHOLDER"
    reviewed_by: str | None = None
    warning: str = PLACEHOLDER_WARNING
    _english_index: dict[str, str] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, gloss: object) -> bool:
        return str(gloss).upper() in self.entries

    @property
    def is_validated(self) -> bool:
        """Whether a linguist has signed this lexicon off.

        The API and clients read this to decide whether to show the provisional
        banner. It stays False until someone is named in ``reviewed_by`` — plan §6
        makes linguistic sign-off a requirement, and "we meant to check it" is not a
        state the software should be unable to detect.
        """
        return self.status.upper() == "VALIDATED" and bool(self.reviewed_by)

    def get(self, gloss: str) -> LexEntry | None:
        return self.entries.get(gloss.upper())

    def require(self, gloss: str) -> LexEntry:
        entry = self.get(gloss)
        if entry is None:
            raise SignSyncError(f"no lexicon entry for gloss {gloss!r}")
        return entry

    def glosses(self) -> list[str]:
        return sorted(self.entries)

    def by_pos(self, pos: str) -> list[LexEntry]:
        return [e for e in self.entries.values() if e.pos == pos]

    def lookup_english(self, word: str) -> LexEntry | None:
        """Find the sign for an English word or short phrase.

        Longest-match phrases are handled by the caller (``english_to_sign``);
        this is a plain lookup over the normalised surface forms plus a small
        morphological fallback, since "needs", "needed" and "needing" should all
        reach NEED without the lexicon listing every inflection.
        """
        key = word.strip().lower()
        if not key:
            return None
        key = _IRREGULAR_FORMS.get(key, key)
        gloss = self._english_index.get(key)
        if gloss is None:
            gloss = self._english_index.get(_stem(key))
        if gloss is None:
            for candidate in _inflection_candidates(key):
                gloss = self._english_index.get(candidate)
                if gloss is not None:
                    break
        return self.entries.get(gloss) if gloss else None

    def add(self, entry: LexEntry) -> None:
        self.entries[entry.gloss.upper()] = entry
        for surface in entry.english:
            self._english_index.setdefault(surface.strip().lower(), entry.gloss.upper())
            self._english_index.setdefault(_stem(surface.strip().lower()), entry.gloss.upper())

    def phrases(self) -> list[str]:
        """Multi-word English surface forms, longest first.

        English→USL needs these before word-level lookup: "thank you" is one sign,
        and translating it as two would produce a sentence no signer would make.
        """
        return sorted(
            (s for s in self._english_index if " " in s), key=lambda s: -len(s.split())
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> Lexicon:
        source = Path(path) if path else _RESOURCE
        data = json.loads(source.read_text(encoding="utf-8"))
        lexicon = cls(
            status=data.get("status", "PLACEHOLDER"),
            reviewed_by=data.get("reviewed_by"),
            warning=data.get("warning", PLACEHOLDER_WARNING),
        )
        for raw in data.get("entries", []):
            lexicon.add(LexEntry.from_dict(raw))
        if not lexicon.entries:
            raise SignSyncError(f"{source}: lexicon contains no entries")
        return lexicon


@lru_cache(maxsize=1)
def default_lexicon() -> Lexicon:
    """The bundled lexicon, loaded once."""
    return Lexicon.load()


#: Irregular English forms the suffix stripper cannot reach.
#:
#: Without these, "gave" and "went" are reported as untranslatable and the verb
#: vanishes from the gloss sequence — leaving a fluent-looking utterance with no
#: predicate, which is a worse failure than admitting the word was not understood.
_IRREGULAR_FORMS: dict[str, str] = {
    "gave": "give",
    "given": "give",
    "went": "go",
    "gone": "go",
    "came": "come",
    "saw": "see",
    "seen": "see",
    "knew": "know",
    "known": "know",
    "understood": "understand",
    "felt": "feel",
    "told": "tell",
    "said": "say",
    "took": "take",
    "got": "get",
    "taught": "teach",
    "sent": "send",
    "paid": "pay",
    "left": "leave",
}


def _stem(word: str) -> str:
    """Crude suffix stripping.

    Deliberately crude: a real morphological analyser is not the bottleneck here,
    the lexicon is. This exists so "hospitals" and "needed" reach their entries
    instead of being reported as untranslatable.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            stem = word[: -len(suffix)]
            if suffix == "ing" and stem.endswith(stem[-1] * 2 if stem else ""):
                stem = stem[:-1]
            return stem
    return word


def _inflection_candidates(word: str) -> tuple[str, ...]:
    """Alternative surface forms to try before giving up on a word."""
    stem = _stem(word)
    return (stem, f"{stem}e", word.rstrip("."), word.replace("'", ""))
