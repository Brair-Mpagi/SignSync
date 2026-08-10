from __future__ import annotations

import pytest

from signsync.datasets.schema import (
    AgeBand,
    ClipRecord,
    Corpus,
    Handedness,
    SignerProfile,
    SigningBackground,
)
from signsync.datasets.splits import (
    SPLIT_NAMES,
    Split,
    signer_independent_split,
    signer_kfold,
    validate_split,
)
from signsync.errors import SplitError


def make_corpus(tmp_path, signer_clips: dict[str, list[str]]) -> Corpus:
    corpus = Corpus(name="t", root=tmp_path)
    for i, (signer, glosses) in enumerate(signer_clips.items()):
        corpus.add_signer(
            SignerProfile(
                signer_id=signer,
                age_band=list(AgeBand)[i % len(AgeBand)],
                background=list(SigningBackground)[i % len(SigningBackground)],
                district=f"district-{i}",
                handedness=Handedness.RIGHT,
            )
        )
        for j, gloss in enumerate(glosses):
            corpus.add_clip(
                ClipRecord(
                    clip_id=f"{signer}-{j}",
                    signer_id=signer,
                    glosses=(gloss,),
                    english=gloss.lower(),
                    landmark_path=f"{signer}-{j}.npz",
                    duration=1.0,
                )
            )
    return corpus


@pytest.fixture
def corpus(tmp_path):
    vocab = ["HELLO", "HELP", "WATER"]
    return make_corpus(tmp_path, {f"s{i}": vocab * 2 for i in range(8)})


def test_no_signer_appears_on_two_sides(corpus):
    """Plan §8.3: split by signer identity, never by clip."""
    split = signer_independent_split(corpus)
    sides = [set(split.signers[name]) for name in SPLIT_NAMES]
    assert sides[0] & sides[1] == set()
    assert sides[0] & sides[2] == set()
    assert sides[1] & sides[2] == set()
    assert sum(len(s) for s in sides) == len(corpus.signer_ids())


def test_every_clip_is_used_exactly_once(corpus):
    split = signer_independent_split(corpus)
    all_ids = [cid for name in SPLIT_NAMES for cid in split[name]]
    assert sorted(all_ids) == sorted(c.clip_id for c in corpus.clips)


def test_split_is_deterministic_for_a_seed(corpus):
    assert signer_independent_split(corpus, seed=7) == signer_independent_split(corpus, seed=7)


def test_split_roughly_honours_the_requested_ratios(corpus):
    split = signer_independent_split(corpus, ratios=(0.5, 0.25, 0.25))
    sizes = split.sizes()
    total = sum(sizes.values())
    assert 0.3 <= sizes["train"] / total <= 0.7
    assert sizes["val"] > 0 and sizes["test"] > 0


def test_uneven_signers_still_fill_every_requested_side(tmp_path):
    """One prolific signer must not leave val or test empty."""
    corpus = make_corpus(
        tmp_path,
        {
            "big": ["HELLO", "HELP", "WATER"] * 10,
            "small-a": ["HELLO", "HELP", "WATER"],
            "small-b": ["HELLO", "HELP", "WATER"],
        },
    )
    split = signer_independent_split(corpus)
    assert all(len(split[name]) > 0 for name in SPLIT_NAMES)


@pytest.mark.parametrize("n_signers", [3, 4, 5, 6, 9])
def test_every_requested_side_gets_a_signer_at_realistic_corpus_sizes(tmp_path, n_signers):
    """Early corpora have barely more signers than splits (plan §9.2)."""
    corpus = make_corpus(
        tmp_path, {f"s{i}": ["HELLO", "HELP", "WATER"] for i in range(n_signers)}
    )
    split = signer_independent_split(corpus)
    assert all(len(split[name]) > 0 for name in SPLIT_NAMES), split.summary()


def test_too_few_signers_is_a_clear_error(tmp_path):
    corpus = make_corpus(tmp_path, {"only": ["HELLO", "HELP"]})
    with pytest.raises(SplitError, match="Recruit more signers"):
        signer_independent_split(corpus)


def test_empty_corpus_cannot_be_split(tmp_path):
    with pytest.raises(SplitError, match="empty corpus"):
        signer_independent_split(Corpus(name="e", root=tmp_path))


def test_two_way_split_is_allowed(corpus):
    split = signer_independent_split(corpus, ratios=(0.8, 0.0, 0.2))
    assert split.val == ()
    assert split.train and split.test


def test_validate_rejects_a_signer_dependent_split(corpus):
    """The check that stops memorisation being reported as accuracy."""
    clips = corpus.clips
    same_signer = [c for c in clips if c.signer_id == "s0"]
    bad = Split(
        train=(same_signer[0].clip_id,),
        val=(),
        test=(same_signer[1].clip_id,),
    )
    with pytest.raises(SplitError, match="appear in both train and test"):
        validate_split(corpus, bad, require_full_vocabulary=False)


def test_validate_rejects_a_duplicated_clip(corpus):
    cid = corpus.clips[0].clip_id
    with pytest.raises(SplitError, match="appears in both"):
        validate_split(corpus, Split(train=(cid,), val=(cid,), test=()))


def test_validate_rejects_unknown_clip(corpus):
    with pytest.raises(SplitError, match="unknown clip"):
        validate_split(corpus, Split(train=("ghost",), val=(), test=()))


def test_gloss_missing_from_train_is_rejected_by_default(tmp_path):
    corpus = make_corpus(
        tmp_path,
        {"a": ["HELLO", "HELLO"], "b": ["HELLO", "HELLO"], "c": ["RARE", "RARE"]},
    )
    with pytest.raises(SplitError, match="never in train"):
        signer_independent_split(corpus, ratios=(0.5, 0.0, 0.5), seed=3)


def test_few_shot_split_is_allowed_when_explicit(tmp_path):
    corpus = make_corpus(
        tmp_path,
        {"a": ["HELLO", "HELLO"], "b": ["HELLO", "HELLO"], "c": ["RARE", "RARE"]},
    )
    split = signer_independent_split(
        corpus, ratios=(0.5, 0.0, 0.5), seed=3, require_full_vocabulary=False
    )
    assert split.train and split.test


def test_kfold_holds_out_disjoint_signer_groups(corpus):
    folds = signer_kfold(corpus, k=4)
    assert len(folds) == 4

    held_out = [set(f.signers["test"]) for f in folds]
    assert set().union(*held_out) == set(corpus.signer_ids())
    for i, left in enumerate(held_out):
        assert left, "every fold must hold out at least one signer"
        for right in held_out[i + 1 :]:
            assert left & right == set()
    for fold in folds:
        assert set(fold.signers["train"]) & set(fold.signers["test"]) == set()


def test_kfold_rejects_impossible_k(corpus):
    with pytest.raises(SplitError, match="folds"):
        signer_kfold(corpus, k=99)
    with pytest.raises(SplitError, match="at least 2"):
        signer_kfold(corpus, k=1)


def test_split_summary_is_readable(corpus):
    summary = signer_independent_split(corpus).summary()
    assert "train:" in summary and "signers" in summary


def test_split_indexing_rejects_a_typo(corpus):
    split = signer_independent_split(corpus)
    with pytest.raises(KeyError):
        split["trian"]
