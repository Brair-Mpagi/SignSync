from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from signsync.errors import SignSyncError
from signsync.evaluation import (
    PASS_THRESHOLD,
    Criterion,
    EvaluationItem,
    EvaluationReport,
    EvaluationRound,
    Evaluator,
    EvaluatorRole,
    Panel,
    Rating,
    bleu,
    classification_report,
    confusion_matrix,
    confusion_pairs,
    edit_distance,
    motion_smoothness,
    rouge_l,
    sequence_accuracy,
    sign_error_rate,
    trajectory_error,
    word_error_rate,
)


# --------------------------------------------------------------------------- classification


def test_classification_report_scores_per_class():
    report = classification_report(
        ["HELLO", "HELLO", "WATER", "HELP"], ["HELLO", "WATER", "WATER", "HELP"]
    )
    assert report.accuracy == pytest.approx(0.75)
    assert report.recall["HELLO"] == pytest.approx(0.5)
    assert report.precision["WATER"] == pytest.approx(0.5)
    assert "accuracy" in report.summary()


def test_macro_f1_exposes_failure_on_rare_signs():
    """Accuracy hides it; a rare sign matters to whoever needs it."""
    truth = ["COMMON"] * 18 + ["RARE", "RARE"]
    predicted = ["COMMON"] * 20
    report = classification_report(truth, predicted)

    assert report.accuracy == pytest.approx(0.9)
    assert report.macro_f1 < 0.6
    assert report.worst_classes[0][0] == "RARE"


def test_abstentions_are_counted_separately_from_errors():
    """An abstention asks the signer to repeat; a wrong answer is spoken as fact."""
    report = classification_report(["HELP", "HELP"], ["<unknown>", "WATER"])
    assert report.abstentions == 1
    assert report.accuracy == 0.0


def test_confusion_pairs_name_the_signs_that_get_mixed_up():
    truth = ["PAIN", "PAIN", "PAIN", "WATER"]
    predicted = ["PAINT", "PAINT", "PAIN", "WATER"]
    assert confusion_pairs(truth, predicted)[0] == ("PAIN", "PAINT", 2)


def test_confusion_matrix_shape_and_diagonal():
    matrix, labels = confusion_matrix(["A", "B"], ["A", "A"])
    assert matrix.shape == (2, 2)
    assert matrix[labels.index("A"), labels.index("A")] == 1


def test_classification_validates_its_inputs():
    with pytest.raises(SignSyncError, match="predictions"):
        classification_report(["A"], ["A", "B"])
    with pytest.raises(SignSyncError, match="empty"):
        classification_report([], [])


# --------------------------------------------------------------------------- sequences


def test_edit_distance_breaks_errors_down_by_type():
    """Deletions point at the segmenter, substitutions at the recogniser."""
    distance, subs, dels, ins = edit_distance(["A", "B", "C"], ["A", "X", "C", "D"])
    assert distance == 2
    assert subs == 1 and ins == 1 and dels == 0


def test_edit_distance_of_identical_sequences_is_zero():
    assert edit_distance(["A", "B"], ["A", "B"])[0] == 0


def test_sign_error_rate_reports_the_breakdown():
    rate = sign_error_rate([["ME", "NEED", "HELP"]], [["ME", "HELP"]])
    assert rate.rate == pytest.approx(1 / 3)
    assert rate.deletions == 1
    assert "del" in rate.summary()


def test_sequence_accuracy_is_stricter_than_error_rate():
    """One wrong sign in eight is not the same as entirely wrong."""
    references = [["A", "B", "C", "D", "E", "F", "G", "H"]]
    hypotheses = [["A", "B", "C", "D", "E", "F", "G", "X"]]

    assert sequence_accuracy(references, hypotheses) == 0.0
    assert sign_error_rate(references, hypotheses).rate == pytest.approx(0.125)


def test_word_error_rate_is_case_insensitive():
    assert word_error_rate(["I need help"], ["i need help"]).rate == 0.0
    assert word_error_rate(["I need help"], ["I need"]).rate == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- translation


def test_bleu_rewards_an_exact_match():
    perfect = bleu(["where is the hospital"], ["where is the hospital"])
    wrong = bleu(["where is the hospital"], ["i need water now"])
    assert perfect > 0.9
    assert wrong < perfect


def test_bleu_penalises_a_too_short_hypothesis():
    """Brevity penalty: dropping half the sentence is not a good translation."""
    full = bleu(["i need help at the hospital"], ["i need help at the hospital"])
    short = bleu(["i need help at the hospital"], ["i need"])
    assert short < full


def test_rouge_l_rewards_covering_the_reference_in_order():
    assert rouge_l(["i need help"], ["i need help"]) == pytest.approx(1.0)
    assert rouge_l(["i need help"], ["help need i"]) < 1.0
    assert rouge_l(["i need help"], [""]) == 0.0


def test_translation_metrics_validate_input_lengths():
    with pytest.raises(SignSyncError, match="hypotheses"):
        bleu(["a"], ["a", "b"])
    with pytest.raises(SignSyncError, match="empty"):
        rouge_l([], [])


# --------------------------------------------------------------------------- motion


def test_smoothness_prefers_smooth_motion_over_jerky():
    t = np.linspace(0, 1, 60)
    smooth = np.stack([t, np.sin(np.pi * t), np.zeros_like(t)], axis=1)
    jerky = smooth.copy()
    jerky[::4, 1] += 0.35

    assert motion_smoothness(smooth) < motion_smoothness(jerky)


def test_smoothness_of_a_static_clip_is_zero():
    assert motion_smoothness(np.zeros((30, 3))) == 0.0


def test_smoothness_rejects_the_wrong_shape():
    with pytest.raises(SignSyncError, match=r"\(T, 3\)"):
        motion_smoothness(np.zeros((10, 2)))


def test_trajectory_error_aligns_different_tempos():
    """Comparing frame i to frame i would measure signing speed, not trajectory."""
    reference = np.stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)], axis=1)
    slower = np.stack([np.linspace(0, 1, 80), np.zeros(80), np.zeros(80)], axis=1)
    assert trajectory_error(reference, slower) < 1e-6

    shifted = reference + np.array([0.0, 0.5, 0.0])
    assert trajectory_error(reference, shifted) == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- human


def full_panel() -> Panel:
    panel = Panel()
    for i in range(3):
        panel.add(
            Evaluator(f"deaf-{i}", EvaluatorRole.DEAF_SIGNER, is_deaf=True, usl_fluent=True)
        )
    panel.add(Evaluator("interp-0", EvaluatorRole.INTERPRETER, usl_fluent=True))
    panel.add(Evaluator("hearing-0", EvaluatorRole.HEARING_USER))
    return panel


def build_round(scores: dict[str, int], panel: Panel | None = None) -> EvaluationRound:
    round_ = EvaluationRound(round_id="r1", panel=panel or full_panel())
    round_.items.append(
        EvaluationItem(
            item_id="i1",
            kind="translation",
            source="HOSPITAL WHERE",
            output="Where is the hospital?",
            criteria=(Criterion.MEANING_PRESERVED,),
        )
    )
    for evaluator_id, score in scores.items():
        round_.add_rating(Rating("i1", evaluator_id, Criterion.MEANING_PRESERVED, score))
    return round_


def test_panel_without_deaf_evaluators_cannot_certify():
    """Plan §6: sign quality cannot be signed off by hearing reviewers."""
    panel = Panel()
    panel.add(Evaluator("interp-0", EvaluatorRole.INTERPRETER))
    panel.add(Evaluator("hearing-0", EvaluatorRole.HEARING_USER))

    problems = panel.problems()
    assert any("Deaf evaluator" in p for p in problems)
    assert not build_round({"interp-0": 5, "hearing-0": 5}, panel).result().certified


def test_panel_needs_an_outside_hearing_user():
    """Otherwise the round measures insider approval, not usability (plan §15)."""
    panel = Panel()
    for i in range(3):
        panel.add(Evaluator(f"deaf-{i}", EvaluatorRole.DEAF_SIGNER, is_deaf=True))
    panel.add(Evaluator("interp-0", EvaluatorRole.INTERPRETER))
    assert any("hearing user" in p for p in panel.problems())


def test_uncompensated_evaluators_are_flagged():
    """Plan §9.3, §13: fair payment is non-negotiable, not an afterthought."""
    panel = full_panel()
    panel.add(Evaluator("deaf-x", EvaluatorRole.DEAF_SIGNER, is_deaf=True, compensated=False))
    assert any("uncompensated" in p for p in panel.problems())


def test_deaf_verdict_overrides_a_high_aggregate():
    """Plan §19 criterion 10: the Deaf community's verdict is decisive."""
    result = build_round(
        {"deaf-0": 2, "deaf-1": 2, "deaf-2": 2, "interp-0": 5, "hearing-0": 5}
    ).result()

    assert result.by_criterion["meaning_preserved"] > 2.5, "aggregate is pulled up by hearing raters"
    assert result.by_criterion_deaf["meaning_preserved"] < PASS_THRESHOLD
    assert not result.certified
    assert "meaning_preserved" in result.failing_criteria()


def test_a_good_round_with_a_valid_panel_certifies():
    result = build_round(
        {"deaf-0": 5, "deaf-1": 4, "deaf-2": 4, "interp-0": 4, "hearing-0": 4}
    ).result()
    assert result.certified
    assert "CERTIFIED" in result.summary()


def test_agreement_falls_when_raters_disagree():
    agree = build_round({"deaf-0": 4, "deaf-1": 4, "deaf-2": 4, "interp-0": 4, "hearing-0": 4})
    disagree = build_round({"deaf-0": 1, "deaf-1": 5, "deaf-2": 1, "interp-0": 5, "hearing-0": 3})
    assert agree.agreement() > disagree.agreement()


def test_ratings_must_reference_known_items_and_evaluators():
    round_ = build_round({"deaf-0": 4})
    with pytest.raises(SignSyncError, match="unknown item"):
        round_.add_rating(Rating("nope", "deaf-0", Criterion.MEANING_PRESERVED, 4))
    with pytest.raises(SignSyncError, match="unknown evaluator"):
        round_.add_rating(Rating("i1", "ghost", Criterion.MEANING_PRESERVED, 4))


def test_scores_outside_the_scale_are_rejected():
    with pytest.raises(SignSyncError, match="scale"):
        Rating("i1", "deaf-0", Criterion.MEANING_PRESERVED, 9)


def test_deaf_signer_role_requires_is_deaf():
    with pytest.raises(SignSyncError, match="is_deaf"):
        Evaluator("x", EvaluatorRole.DEAF_SIGNER, is_deaf=False)


def test_round_roundtrips_through_disk(tmp_path):
    original = build_round({"deaf-0": 4, "deaf-1": 4, "deaf-2": 5, "interp-0": 4, "hearing-0": 4})
    restored = EvaluationRound.load(original.save(tmp_path / "round.json"))

    assert restored.round_id == original.round_id
    assert len(restored.ratings) == len(original.ratings)
    assert restored.result().certified == original.result().certified


def test_blank_form_carries_the_scale_anchors():
    form = build_round({"deaf-0": 4}).to_form()
    assert form["items"][0]["source"] == "HOSPITAL WHERE"
    assert "1" in form["scale"] and "5" in form["scale"]
    assert "comment" in form["instructions"]


def test_result_needs_ratings():
    with pytest.raises(SignSyncError, match="no ratings"):
        EvaluationRound(round_id="empty", panel=full_panel()).result()


# --------------------------------------------------------------------------- report


def test_report_refuses_success_on_automatic_metrics_alone():
    """Plan §15's whole purpose: no declaring victory on a benchmark."""
    report = EvaluationReport(
        recognition=classification_report(["A", "B"], ["A", "B"]),
        bleu=0.99,
        signer_independent=True,
    )
    assert not report.can_claim_success
    assert any("human evaluation is mandatory" in b for b in report.blockers())
    assert "does NOT support a success claim" in report.summary()


def test_report_refuses_success_without_a_signer_independent_split():
    good_round = build_round(
        {"deaf-0": 5, "deaf-1": 4, "deaf-2": 4, "interp-0": 4, "hearing-0": 4}
    ).result()
    report = EvaluationReport(human_rounds=[good_round], signer_independent=False)

    assert not report.can_claim_success
    assert any("memorisation" in b for b in report.blockers())


def test_report_supports_success_when_both_requirements_are_met():
    good_round = build_round(
        {"deaf-0": 5, "deaf-1": 4, "deaf-2": 4, "interp-0": 4, "hearing-0": 4}
    ).result()
    report = EvaluationReport(
        recognition=classification_report(["A", "B"], ["A", "B"]),
        human_rounds=[good_round],
        signer_independent=True,
    )

    assert report.can_claim_success
    assert report.blockers() == []
    assert "supports a claim" in report.summary()


def test_report_serialises_its_blockers(tmp_path):
    import json

    report = EvaluationReport(bleu=0.5, generated_on=date(2026, 8, 10))
    data = json.loads(report.save(tmp_path / "report.json").read_text())

    assert data["can_claim_success"] is False
    assert data["blockers"]
    assert data["automatic"]["bleu"] == 0.5
