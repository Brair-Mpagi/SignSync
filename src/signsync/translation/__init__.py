"""Translation in both directions, through a shared semantic frame (plan §7.2).

    glosses ──▶ parse_glosses ──▶ SemanticFrame ──▶ realise_english ──▶ English
    English ──▶ analyse_english ─▶ SemanticFrame ──▶ generate_glosses ─▶ glosses + NMM

Neither direction may shortcut the frame. That constraint is the difference between
a translator and a dictionary lookup, and it is why negation, question type, tense
and semantic roles survive the trip in both directions.
"""

from __future__ import annotations

from .english_to_sign import EnglishToSign, GlossSequence, analyse_english, generate_glosses
from .lexicon import PLACEHOLDER_WARNING, LexEntry, Lexicon, default_lexicon
from .semantics import (
    Aspect,
    Entity,
    Polarity,
    Role,
    SemanticFrame,
    SpeechAct,
    Tense,
)
from .sign_to_english import (
    SignToEnglish,
    TranslationResult,
    parse_glosses,
    realise_english,
)

__all__ = [
    "Aspect",
    "EnglishToSign",
    "Entity",
    "GlossSequence",
    "LexEntry",
    "Lexicon",
    "PLACEHOLDER_WARNING",
    "Polarity",
    "Role",
    "SemanticFrame",
    "SignToEnglish",
    "SpeechAct",
    "Tense",
    "TranslationResult",
    "analyse_english",
    "default_lexicon",
    "generate_glosses",
    "parse_glosses",
    "realise_english",
]
