from __future__ import annotations

from datetime import date

import pytest

from signsync.datasets.consent import ConsentScope
from signsync.datasets.corpus import CorpusLoader, QualityGate
from signsync.datasets.schema import (
    ClipRecord,
    Corpus,
    MarkerType,
    NonManualMarker,
    SignerProfile,
)
from signsync.datasets.synthetic import SyntheticCorpusSpec, build_synthetic_corpus
from signsync.errors import ConsentError, CorpusError

TODAY = date(2026, 8, 10)
SMALL = SyntheticCorpusSpec(n_signers=6, vocabulary=("HELLO", "HELP"), repeats_per_gloss=1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    return build_synthetic_corpus(root, SMALL, today=TODAY)


def test_synthetic_corpus_is_complete(built):
    corpus, registry = built
    assert len(corpus) > 0
    assert {"HELLO", "HELP"} <= set(corpus.vocabulary())
    assert len(corpus.signers) == 6
    assert len(registry) == 6
    assert corpus.continuous(), "sentences must be present for the segmenter tests"


def test_manifest_roundtrips(built, tmp_path):
    corpus, _ = built
    restored = Corpus.load(corpus.root)
    assert len(restored) == len(corpus)
    assert restored.vocabulary() == corpus.vocabulary()
    assert restored.clips[0].conditions.device == corpus.clips[0].conditions.device


def test_loader_excludes_withdrawn_and_expired_signers(built):
    """The sample corpus deliberately contains both, so gating gets exercised."""
    corpus, registry = built
    loader = CorpusLoader(corpus, registry, scope=ConsentScope.TRAINING, on=TODAY)

    audit = loader.audit()
    refused = dict(audit.refused)
    assert "signer-04" in refused  # withdrawn
    assert "signer-05" in refused  # retention lapsed

    permitted_signers = {c.signer_id for c in loader.permitted_clips()}
    assert "signer-04" not in permitted_signers
    assert "signer-05" not in permitted_signers
    assert "signer-00" in permitted_signers


def test_loading_an_unconsented_clip_by_name_raises(built):
    corpus, registry = built
    loader = CorpusLoader(corpus, registry, on=TODAY)
    withdrawn_clip = next(c for c in corpus.clips if c.signer_id == "signer-04")

    with pytest.raises(ConsentError, match="withdrawn"):
        loader.load(withdrawn_clip)


def test_narrower_scope_permits_fewer_clips(built):
    corpus, registry = built
    training = CorpusLoader(corpus, registry, scope=ConsentScope.TRAINING, on=TODAY)
    release = CorpusLoader(corpus, registry, scope=ConsentScope.OPEN_RELEASE, on=TODAY)
    assert len(release.permitted_clips()) < len(training.permitted_clips())


def test_load_all_yields_data_and_records_rejections(built):
    corpus, registry = built
    loader = CorpusLoader(corpus, registry, on=TODAY)
    loaded = list(loader.load_all())

    assert loaded
    assert all(len(d.sequence) > 0 for d in loaded)
    assert loader.rejected == [] or all(isinstance(r, tuple) for r in loader.rejected)


def test_quality_gate_rejects_a_short_clip(built):
    corpus, registry = built
    loader = CorpusLoader(corpus, registry, on=TODAY, quality=QualityGate(min_frames=10_000))
    assert list(loader.load_all()) == []
    assert len(loader.rejected) == len(loader.permitted_clips())


def test_coverage_summary_reports_channels(built):
    corpus, registry = built
    coverage = CorpusLoader(corpus, registry, on=TODAY).coverage_summary()
    assert set(coverage) == {"pose", "left_hand", "right_hand", "face"}
    assert coverage["pose"] > 0.9


def test_landmark_path_cannot_escape_the_corpus_root(built):
    corpus, registry = built
    loader = CorpusLoader(corpus, registry, on=TODAY)
    hostile = ClipRecord(
        clip_id="evil",
        signer_id="signer-00",
        glosses=("HELLO",),
        english="hello",
        landmark_path="../../../etc/passwd",
        duration=1.0,
    )
    with pytest.raises(CorpusError, match="escapes the corpus root"):
        loader.resolve(hostile)


def test_clip_requires_a_known_signer():
    corpus = Corpus(name="c", root=None)  # type: ignore[arg-type]
    with pytest.raises(CorpusError, match="unknown signer"):
        corpus.add_clip(
            ClipRecord(
                clip_id="x",
                signer_id="ghost",
                glosses=("HELLO",),
                english="hello",
                landmark_path="a.npz",
                duration=1.0,
            )
        )


def test_clip_validates_its_own_annotation():
    with pytest.raises(CorpusError, match="at least one gloss"):
        ClipRecord("c", "s", (), "", "a.npz", 1.0)
    with pytest.raises(CorpusError, match="boundaries"):
        ClipRecord("c", "s", ("A", "B"), "", "a.npz", 1.0, boundaries=((0.0, 1.0),))
    with pytest.raises(CorpusError, match="HamNoSys"):
        ClipRecord("c", "s", ("A", "B"), "", "a.npz", 1.0, hamnosys=("x",))


def test_marker_span_is_validated():
    with pytest.raises(CorpusError, match="precedes start"):
        NonManualMarker(MarkerType.BROW_RAISE, 1.0, 0.5)
    with pytest.raises(CorpusError, match="intensity"):
        NonManualMarker(MarkerType.BROW_RAISE, 0.0, 1.0, intensity=2.0)


def test_question_clips_carry_a_brow_marker(built):
    """Non-manual marking is grammar, not decoration (plan §8.7)."""
    corpus, _ = built
    where = next(c for c in corpus.clips if "WHERE" in c.glosses)
    assert where.markers_of(MarkerType.BROW_FURROW)


def test_diversity_report_and_warnings(built):
    corpus, _ = built
    report = corpus.diversity_report()
    assert len(report["district"]) > 1
    assert len(report["age_band"]) > 1
    assert corpus.diversity_warnings() == []


def test_diversity_warnings_flag_a_single_district_corpus(tmp_path):
    corpus = Corpus(name="narrow", root=tmp_path)
    from signsync.datasets.schema import AgeBand, Handedness, SigningBackground

    for i in range(3):
        corpus.add_signer(
            SignerProfile(
                signer_id=f"s{i}",
                age_band=AgeBand.A18_29,
                background=SigningBackground.SCHOOL_TAUGHT,
                district="Kampala",
                handedness=Handedness.RIGHT,
            )
        )
        corpus.add_clip(
            ClipRecord(f"c{i}", f"s{i}", ("HELLO",), "hello", f"{i}.npz", 1.0)
        )

    warnings = corpus.diversity_warnings()
    assert any("district" in w for w in warnings)
    assert any("plan §9.3" in w for w in warnings)


def test_domain_concentration_is_not_a_warning(tmp_path):
    """Plan §9.2 deliberately concentrates on health and education."""
    from signsync.datasets.schema import CONCENTRATION_THRESHOLDS

    assert "domain" not in CONCENTRATION_THRESHOLDS


def test_empty_corpus_warns(tmp_path):
    assert Corpus(name="e", root=tmp_path).diversity_warnings() == ["corpus is empty"]


def test_isolated_clip_exposes_a_single_gloss(built):
    corpus, _ = built
    isolated = corpus.isolated()[0]
    assert isolated.gloss == isolated.glosses[0]
    with pytest.raises(CorpusError, match="continuous"):
        _ = corpus.continuous()[0].gloss
