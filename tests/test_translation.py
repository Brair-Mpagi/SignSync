from __future__ import annotations

import pytest

from signsync.datasets.schema import MarkerType
from signsync.errors import SignSyncError
from signsync.recognition.base import UNKNOWN_GLOSS, SignPrediction
from signsync.translation import (
    EnglishToSign,
    Lexicon,
    Polarity,
    Role,
    SignToEnglish,
    SpeechAct,
    Tense,
    analyse_english,
    default_lexicon,
    generate_glosses,
    parse_glosses,
    realise_english,
)


@pytest.fixture(scope="module")
def s2e():
    return SignToEnglish()


@pytest.fixture(scope="module")
def e2s():
    return EnglishToSign()


# --------------------------------------------------------------------------- lexicon


def test_bundled_lexicon_is_flagged_as_unvalidated():
    """Plan §6: linguistic sign-off is required, so "we meant to check" must be visible."""
    lexicon = default_lexicon()
    assert lexicon.is_validated is False
    assert lexicon.reviewed_by is None
    assert "not" in lexicon.warning.lower()


def test_lexicon_carries_grammar_not_just_translations():
    entry = default_lexicon().require("HOSPITAL")
    assert entry.location is True
    assert default_lexicon().require("GIVE").agreeing is True
    assert default_lexicon().require("DOCTOR").animate is True


def test_lexicon_lookup_handles_inflections_and_irregulars():
    lexicon = default_lexicon()
    assert lexicon.lookup_english("hospitals").gloss == "HOSPITAL"
    assert lexicon.lookup_english("needed").gloss == "NEED"
    assert lexicon.lookup_english("gave").gloss == "GIVE"
    assert lexicon.lookup_english("went").gloss == "GO"
    assert lexicon.lookup_english("zzzz") is None
    assert lexicon.lookup_english("") is None


def test_lexicon_knows_multi_word_signs():
    assert default_lexicon().lookup_english("thank you").gloss == "THANK-YOU"
    assert "thank you" in default_lexicon().phrases()


def test_missing_gloss_raises_with_the_gloss_named():
    with pytest.raises(SignSyncError, match="NOSUCHSIGN"):
        default_lexicon().require("NOSUCHSIGN")


def test_empty_lexicon_file_is_rejected(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{"version": 1, "entries": []}', encoding="utf-8")
    with pytest.raises(SignSyncError, match="no entries"):
        Lexicon.load(path)


# --------------------------------------------------------------------------- USL -> English


@pytest.mark.parametrize(
    ("glosses", "expected"),
    [
        (["HOSPITAL", "WHERE"], "Where is the hospital?"),
        (["ME", "NEED", "HELP"], "I need help."),
        (["ME", "SICK"], "I am sick."),
        (["ME", "NOT", "UNDERSTAND"], "I do not understand."),
        (["YOU", "NAME", "WHAT"], "What is your name?"),
        (["DOCTOR", "HELP", "ME"], "The doctor helps me."),
        (["HELLO"], "Hello."),
        (["ME", "DEAF"], "I am deaf."),
    ],
)
def test_gloss_sequences_become_english(s2e, glosses, expected):
    assert s2e.translate(glosses).text == expected


def test_time_sign_sets_tense_without_a_verb_inflection(s2e):
    result = s2e.translate(["YESTERDAY", "ME", "GO", "HOSPITAL"])
    assert result.frame.tense is Tense.PAST
    assert "went" in result.text


def test_parse_extracts_roles_not_positions():
    frame = parse_glosses(["DOCTOR", "GIVE", "MEDICINE"])
    assert frame.predicate == "GIVE"
    assert frame.first(Role.AGENT).gloss == "DOCTOR"
    assert frame.first(Role.PATIENT).gloss == "MEDICINE"


def test_location_is_a_location_wherever_it_appears():
    fronted = parse_glosses(["HOSPITAL", "ME", "GO"])
    trailing = parse_glosses(["ME", "GO", "HOSPITAL"])
    assert fronted.first(Role.LOCATION).gloss == "HOSPITAL"
    assert trailing.first(Role.LOCATION).gloss == "HOSPITAL"


def test_head_shake_negates_without_a_negation_sign():
    """Plan §8.7: non-manual markers carry grammar, not emphasis."""
    frame = parse_glosses(["ME", "UNDERSTAND"], markers=(MarkerType.HEAD_SHAKE,))
    assert frame.polarity is Polarity.NEGATIVE
    assert "not" in realise_english(frame)


def test_brow_raise_makes_a_statement_a_question():
    frame = parse_glosses(["YOU", "UNDERSTAND"], markers=(MarkerType.BROW_RAISE,))
    assert frame.speech_act is SpeechAct.POLAR_QUESTION
    assert realise_english(frame).endswith("?")


def test_confidence_is_the_weakest_link_not_the_average():
    """One badly recognised sign can make the whole sentence wrong (plan §16.3)."""
    frame = parse_glosses(
        [
            SignPrediction("ME", 0.99),
            SignPrediction("NEED", 0.31),
            SignPrediction("HELP", 0.98),
        ]
    )
    assert frame.confidence == pytest.approx(0.31)


def test_unrecognised_signs_are_reported_not_dropped(s2e):
    result = s2e.translate([SignPrediction(UNKNOWN_GLOSS, 0.2), "ME", "NEED"])
    assert UNKNOWN_GLOSS in result.unresolved
    assert not result.is_reliable


def test_unknown_gloss_is_surfaced(s2e):
    result = s2e.translate(["ME", "FLIBBERTIGIBBET"])
    assert "FLIBBERTIGIBBET" in result.unresolved


def test_empty_input_yields_empty_output(s2e):
    assert s2e.translate([]).text == ""


# --------------------------------------------------------------------------- English -> USL


@pytest.mark.parametrize(
    ("english", "expected"),
    [
        ("Where is the hospital?", ("HOSPITAL", "WHERE")),
        ("I need help.", ("ME", "NEED", "HELP")),
        ("I am not sick.", ("ME", "SICK", "NOT")),
        ("Thank you.", ("THANK-YOU",)),
        ("I will go to school tomorrow.", ("TOMORROW", "ME", "GO", "SCHOOL")),
        ("She gave me the medicine yesterday.", ("YESTERDAY", "HE-SHE", "GIVE", "ME", "MEDICINE")),
    ],
)
def test_english_becomes_usl_gloss_order(e2s, english, expected):
    assert e2s.translate(english).glosses == expected


def test_function_words_are_dropped_not_signed(e2s):
    """Signing "the" and "is" produces word salad, not emphasis."""
    glosses = e2s.translate("The doctor is at the hospital.").glosses
    assert "THE" not in glosses and "IS" not in glosses


def test_question_word_goes_last_not_first(e2s):
    glosses = e2s.translate("Where is the toilet?").glosses
    assert glosses[-1] == "WHERE", f"question word must be clause-final, got {glosses}"


def test_time_reference_goes_first(e2s):
    glosses = e2s.translate("I am going to the hospital tomorrow.").glosses
    assert glosses[0] == "TOMORROW"


def test_negation_follows_the_predicate(e2s):
    glosses = e2s.translate("I do not know.").glosses
    assert glosses.index("NOT") > glosses.index("KNOW")


def test_markers_are_generated_with_scope(e2s):
    result = e2s.translate("Where is the hospital?")
    assert [m.marker for m in result.markers] == [MarkerType.BROW_FURROW]
    assert result.markers[0].scopes_glosses == result.glosses
    assert result.markers[0].end > result.markers[0].start


def test_negation_marker_scopes_the_predicate_onward(e2s):
    result = e2s.translate("I do not understand.")
    shake = next(m for m in result.markers if m.marker is MarkerType.HEAD_SHAKE)
    assert "UNDERSTAND" in shake.scopes_glosses
    assert "ME" not in shake.scopes_glosses


def test_polar_question_gets_a_brow_raise(e2s):
    result = e2s.translate("Do you need medicine?")
    assert MarkerType.BROW_RAISE in [m.marker for m in result.markers]


def test_untranslatable_modals_are_reported_rather_than_silently_dropped(e2s):
    """"You must go" must not become "you go" — an instruction into a description."""
    result = e2s.translate("You must go to the hospital.")
    assert "must" in result.unresolved
    assert not result.is_complete


def test_tense_auxiliaries_are_not_reported_as_untranslatable(e2s):
    """The frame carries the tense, so "will" is expressed as a time sign."""
    assert e2s.translate("I will go to school tomorrow.").unresolved == ()


def test_notation_renders_marker_scope(e2s):
    notation = e2s.translate("Where is the hospital?").notation()
    assert "wh" in notation
    assert "HOSPITAL WHERE" in notation
    assert "_" in notation


def test_notation_without_markers_is_just_glosses(e2s):
    assert e2s.translate("I need water").notation() == "ME NEED WATER"


def test_empty_english_yields_no_glosses(e2s):
    assert e2s.translate("").glosses == ()
    assert e2s.translate("!!!").glosses == ()


# --------------------------------------------------------------------------- round trip


@pytest.mark.parametrize(
    "english",
    [
        "I need help.",
        "Where is the hospital?",
        "I am not sick.",
        "I need water.",
        "I do not understand.",
    ],
)
def test_round_trip_preserves_meaning(e2s, s2e, english):
    """English → frame → glosses → frame → English should not drift."""
    signed = e2s.translate(english)
    back = s2e.translate(list(signed.glosses), markers=tuple(m.marker for m in signed.markers))

    original = analyse_english(english)
    assert back.frame.speech_act is original.speech_act
    assert back.frame.polarity is original.polarity
    assert back.frame.predicate == original.predicate


def test_round_trip_keeps_negation_that_a_word_for_word_map_would_lose(e2s, s2e):
    signed = e2s.translate("I am not sick.")
    back = s2e.translate(list(signed.glosses), markers=tuple(m.marker for m in signed.markers))
    assert "not" in back.text.lower()


def test_generate_from_a_hand_built_frame():
    frame = analyse_english("The doctor helps me.")
    sequence = generate_glosses(frame)
    assert sequence.glosses[0] == "DOCTOR"
    assert "HELP" in sequence.glosses
