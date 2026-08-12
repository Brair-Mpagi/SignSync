from __future__ import annotations

from datetime import date, timedelta

import pytest

from signsync.datasets.consent import (
    ConsentRecord,
    ConsentRegistry,
    ConsentScope,
    ConsentStatus,
)
from signsync.errors import ConsentError

TODAY = date(2026, 8, 10)


def make_record(**overrides) -> ConsentRecord:
    defaults = {
        "participant_id": "p1",
        "granted_on": TODAY - timedelta(days=30),
        "retention_until": TODAY + timedelta(days=365),
        "scopes": frozenset({ConsentScope.TRAINING, ConsentScope.EVALUATION}),
        "delivered_by": "fluent-signer-1",
    }
    defaults.update(overrides)
    return ConsentRecord(**defaults)  # type: ignore[arg-type]


def test_unlisted_scope_is_denied_not_assumed():
    """A narrow agreement must not silently widen (plan §16.3)."""
    record = make_record()
    assert record.permits(ConsentScope.TRAINING, TODAY)
    assert not record.permits(ConsentScope.OPEN_RELEASE, TODAY)
    assert record.status(ConsentScope.COMMERCIAL, TODAY) is ConsentStatus.SCOPE_NOT_GRANTED


def test_withdrawal_is_retroactive():
    record = make_record().withdraw(TODAY - timedelta(days=1))
    assert record.status(ConsentScope.TRAINING, TODAY) is ConsentStatus.WITHDRAWN


def test_expired_retention_stops_the_clip_loading():
    record = make_record(retention_until=TODAY - timedelta(days=1))
    assert record.status(ConsentScope.TRAINING, TODAY) is ConsentStatus.EXPIRED


def test_consent_before_its_start_date_is_not_effective():
    record = make_record(granted_on=TODAY + timedelta(days=5))
    assert record.status(ConsentScope.TRAINING, TODAY) is ConsentStatus.NOT_YET_EFFECTIVE


def test_minor_requires_a_named_guardian():
    with pytest.raises(ConsentError, match="guardian"):
        make_record(is_minor=True)
    assert make_record(
        is_minor=True, guardian_name="A. Guardian", guardian_relationship="parent"
    ).is_minor


def test_record_must_name_who_delivered_the_consent_conversation():
    """Plan §16.2: consent is delivered in USL by a fluent signer."""
    with pytest.raises(ConsentError, match="fluent signer"):
        make_record(delivered_by="")


def test_retention_must_follow_the_grant():
    with pytest.raises(ConsentError, match="retention_until"):
        make_record(retention_until=TODAY - timedelta(days=365))


def test_missing_record_is_an_error_not_a_silent_skip():
    registry = ConsentRegistry()
    assert registry.status("nobody", ConsentScope.TRAINING, TODAY) is ConsentStatus.NO_RECORD
    with pytest.raises(ConsentError, match="no_record"):
        registry.require("nobody", ConsentScope.TRAINING, TODAY)


def test_audit_reports_without_raising():
    registry = ConsentRegistry()
    registry.add(make_record(participant_id="ok"))
    registry.add(make_record(participant_id="gone").withdraw(TODAY - timedelta(days=2)))

    audit = registry.audit(["ok", "gone", "unknown"], ConsentScope.TRAINING, TODAY)

    assert audit.permitted == ("ok",)
    assert dict(audit.refused) == {
        "gone": ConsentStatus.WITHDRAWN,
        "unknown": ConsentStatus.NO_RECORD,
    }
    assert not audit.ok
    assert "1 of 3" in audit.summary()


def test_audit_deduplicates_participants():
    registry = ConsentRegistry()
    registry.add(make_record(participant_id="ok"))
    audit = registry.audit(["ok", "ok", "ok"], ConsentScope.TRAINING, TODAY)
    assert audit.permitted == ("ok",)
    assert audit.ok


def test_expiring_before_supports_retention_planning():
    registry = ConsentRegistry()
    registry.add(make_record(participant_id="soon", retention_until=TODAY + timedelta(days=10)))
    registry.add(make_record(participant_id="later"))
    assert registry.expiring_before(TODAY + timedelta(days=30)) == ["soon"]


def test_withdrawn_lists_participants_for_artefact_regeneration():
    registry = ConsentRegistry()
    registry.add(make_record(participant_id="gone").withdraw(TODAY))
    registry.add(make_record(participant_id="stays"))
    assert registry.withdrawn(TODAY) == ["gone"]


def test_registry_roundtrips_through_json(tmp_path):
    registry = ConsentRegistry()
    registry.add(make_record(participant_id="p1"))
    registry.add(
        make_record(
            participant_id="p2",
            is_minor=True,
            guardian_name="G",
            guardian_relationship="aunt",
            scopes=frozenset({ConsentScope.TRAINING}),
        )
    )
    path = registry.save(tmp_path / "consent.json")
    restored = ConsentRegistry.load(path)

    assert len(restored) == 2
    assert restored.get("p2").guardian_relationship == "aunt"
    assert restored.permits("p1", ConsentScope.EVALUATION, TODAY)
    assert not restored.permits("p2", ConsentScope.EVALUATION, TODAY)


def test_unknown_scope_string_is_rejected():
    with pytest.raises(ConsentError, match="unknown consent scope"):
        ConsentScope.parse("whatever")
