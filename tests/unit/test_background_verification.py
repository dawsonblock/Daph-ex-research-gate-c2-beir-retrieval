"""BACKGROUND_VERIFICATION_V1 acceptance tests V1-V12.

Per configs/background_verification_v1_design.json.

    CORE RULE: verification APPENDS EVIDENCE ABOUT a claim.
               It never mutates the claim in place.

The milestone is NOT "make the memory know what is true". It is the narrower
claim that verification can run asynchronously, append auditable results,
survive replay/restart, preserve disagreement, and derive a stable current
status without mutating canonical memory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.experiment_integrity.certified_memory import (  # noqa: E402
    assert_certified_memory_v1_unchanged, pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.memory_write import (  # noqa: E402
    ClaimStore, LifecycleState, VerificationStatus)
from hrm_adaptive_memory.memory_write.verification import (  # noqa: E402
    DeterministicMemoryConsistencyChecker, EvidenceResolutionError, VerificationResult,
    priority_signals)

E1, E2 = "Wren pressure assembly", "Auk relay unit"
REL = "operating tier"


def _store(tmp_path, name="store", **kw) -> ClaimStore:
    return ClaimStore(tmp_path / name, **kw)


def _claim(store, entity=E1, value="Tier 4", source="A", **kw):
    return store.ingest(subject=entity, relation=REL, value=value, source_id=source, **kw).record


def _emit(store, rid, result, evidence=(), checker="c1", conf=1.0, **kw):
    return store.append_verification(
        claim_record_id=rid, checker_id=checker, checker_type="TEST",
        method="m", method_version="1", evidence_ids=evidence,
        result=result, confidence=conf, **kw)


class TestV1IngestLeavesClaimUnverified:
    def test_novel_claim_is_unverified(self, tmp_path):
        store = _store(tmp_path)
        r = _claim(store)
        assert store.verification_status(r.record_id) is VerificationStatus.UNVERIFIED
        assert r.lifecycle_state is LifecycleState.ACTIVE

    def test_supporting_and_conflicting_ingests_still_leave_both_unverified(self, tmp_path):
        """Ingest OBSERVES a relationship; it records no opinion about it."""
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        b = _claim(store, value="Tier 4", source="B")   # SUPPORT observation
        c = _claim(store, value="Tier 9", source="C")   # CONFLICT observation
        for r in (a, b, c):
            assert store.verification_status(r.record_id) is VerificationStatus.UNVERIFIED
            assert store.get(r.record_id).lifecycle_state is LifecycleState.ACTIVE


class TestV2SupportedWithoutChangingClaimBytes:
    def test_derives_supported_and_claim_log_line_is_byte_identical(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        _claim(store, value="Tier 4", source="B")

        def claim_line():
            for line in store.log_path.read_text().splitlines():
                ev = json.loads(line)
                if ev["event"] == "INGEST" and ev["record"]["record_id"] == a.record_id:
                    return line
            raise AssertionError("claim event not found")

        before = claim_line()
        DeterministicMemoryConsistencyChecker(store).verify(a.record_id)
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED
        assert claim_line() == before, "the claim's own bytes must not change"
        assert store.get(a.record_id).lifecycle_state is LifecycleState.ACTIVE


class TestV3ContradictedPreservesBothSides:
    def test_both_sides_remain_active_and_retrievable(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        b = _claim(store, value="Tier 9", source="B")
        w = DeterministicMemoryConsistencyChecker(store)
        w.verify(a.record_id)
        w.verify(b.record_id)
        assert store.verification_status(a.record_id) is VerificationStatus.CONTRADICTED
        assert store.verification_status(b.record_id) is VerificationStatus.CONTRADICTED
        ids = {r.evidence_id for r in store.retrievable_index_records()}
        assert {a.record_id, b.record_id} <= ids, "neither side may be dropped"


class TestV4HistoryIsNeverErased:
    def test_later_verification_does_not_remove_earlier_events(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        e1 = _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c1")
        e2 = _emit(store, a.record_id, VerificationResult.CONTRADICTED, checker="c2")
        hist = store.verification_events(a.record_id)
        assert [e.verification_event_id for e in hist] == [
            e1.verification_event_id, e2.verification_event_id]
        raw = store.log_path.read_text()
        assert e1.verification_event_id in raw and e2.verification_event_id in raw

    def test_explicit_retirement_keeps_the_retired_event_in_history(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        e1 = _emit(store, a.record_id, VerificationResult.CONTRADICTED, checker="c1")
        _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c2",
              supersedes_verification=e1.verification_event_id)
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED
        assert len(store.verification_events(a.record_id)) == 2, "history retained"


class TestV5ReplayEqualsIncremental:
    def test_genesis_replay_derives_the_same_statuses(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        b = _claim(store, value="Tier 9", source="B")
        w = DeterministicMemoryConsistencyChecker(store)
        w.verify(a.record_id)
        w.verify(b.record_id)
        _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c9")
        live = {r.record_id: store.verification_status(r.record_id) for r in store.retrievable()}

        replayed_store = ClaimStore(tmp_path / "store")
        replayed = {r.record_id: replayed_store.verification_status(r.record_id)
                    for r in replayed_store.retrievable()}
        assert replayed == live
        assert replayed_store.consolidated_state().state_hash() == \
            store.consolidated_state().state_hash()


class TestV6LifecycleIndependentOfVerification:
    def test_a_record_can_be_supported_and_superseded_simultaneously(self, tmp_path):
        """The state that the pre-split single enum could not express."""
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        _claim(store, value="Tier 4", source="B")
        DeterministicMemoryConsistencyChecker(store).verify(a.record_id)
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED

        store.ingest(subject=E1, relation=REL, value="Tier 7", source_id="C",
                     supersedes=a.record_id)
        assert store.get(a.record_id).lifecycle_state is LifecycleState.SUPERSEDED
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED, \
            "supersession must not alter verification history"

    def test_retraction_does_not_alter_verification_status(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED)
        store.retract(a.record_id, reason="withdrawn")
        assert store.get(a.record_id).lifecycle_state is LifecycleState.RETRACTED
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED

    def test_verification_does_not_change_retrievability(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        before = {r.evidence_id for r in store.retrievable_index_records()}
        _emit(store, a.record_id, VerificationResult.CONTRADICTED)
        assert {r.evidence_id for r in store.retrievable_index_records()} == before


class TestV7DisagreementIsExplicit:
    def test_disagreeing_verifiers_derive_INCONCLUSIVE(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c1", conf=0.99)
        _emit(store, a.record_id, VerificationResult.FALSIFIED, checker="c2", conf=0.01)
        assert store.verification_status(a.record_id) is VerificationStatus.INCONCLUSIVE, \
            "confidence must NOT arbitrate between disagreeing verifiers"

    def test_disagreement_is_queryable_not_averaged(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c1")
        _emit(store, a.record_id, VerificationResult.CONTRADICTED, checker="c2")
        dis = store.verification_disagreements()
        assert len(dis) == 1
        rid, events = dis[0]
        assert rid == a.record_id
        assert {e.result for e in events} == {VerificationResult.SUPPORTED,
                                              VerificationResult.CONTRADICTED}

    def test_recency_does_not_break_ties(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c1",
              observed_at_utc="2020-01-01T00:00:00+00:00")
        _emit(store, a.record_id, VerificationResult.CONTRADICTED, checker="c2",
              observed_at_utc="2099-01-01T00:00:00+00:00")
        assert store.verification_status(a.record_id) is VerificationStatus.INCONCLUSIVE, \
            "a later verification must not silently win"

    def test_unanimous_verifiers_derive_that_result(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c1")
        _emit(store, a.record_id, VerificationResult.SUPPORTED, checker="c2")
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED


class TestV8FailClosedOnBadEvidence:
    def test_unknown_evidence_id_is_refused(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        with pytest.raises(EvidenceResolutionError):
            _emit(store, a.record_id, VerificationResult.SUPPORTED, evidence=("vmw-nope",))
        assert store.verification_events(a.record_id) == []

    def test_unknown_claim_target_is_refused(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(EvidenceResolutionError):
            _emit(store, "vmw-nonexistent", VerificationResult.SUPPORTED)

    def test_unknown_supersedes_verification_is_refused(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        with pytest.raises(EvidenceResolutionError):
            _emit(store, a.record_id, VerificationResult.SUPPORTED,
                  supersedes_verification="vfy-nope")


class TestV9ProvenanceAndAuditability:
    def test_status_is_traceable_to_the_exact_events_that_produced_it(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        b = _claim(store, value="Tier 4", source="B")
        evt = DeterministicMemoryConsistencyChecker(store).verify(a.record_id)
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED
        assert evt.evidence_ids == (b.record_id,), "evidence must cite the actual record"
        for field in ("checker_id", "checker_type", "method", "method_version",
                      "observed_at_utc", "confidence"):
            assert getattr(evt, field) is not None
        assert evt.verification_event_id in store.log_path.read_text()

    def test_event_id_is_content_addressed(self, tmp_path):
        store = _store(tmp_path)
        # claims must be pinned in time too, or their provenance-addressed
        # record_ids differ and the comparison tests nothing
        a = _claim(store, observed_at_utc="2019-01-01T00:00:00+00:00")
        e1 = _emit(store, a.record_id, VerificationResult.SUPPORTED,
                   observed_at_utc="2020-01-01T00:00:00+00:00")
        store2 = _store(tmp_path, "other")
        a2 = _claim(store2, observed_at_utc="2019-01-01T00:00:00+00:00")
        e2 = _emit(store2, a2.record_id, VerificationResult.SUPPORTED,
                   observed_at_utc="2020-01-01T00:00:00+00:00")
        assert a.record_id == a2.record_id
        assert e1.verification_event_id == e2.verification_event_id, \
            "same determination -> same id"


class TestV10OptionalRetrievalFiltering:
    def test_filtering_is_opt_in_and_changes_nothing_by_default(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        b = _claim(store, E2, value="Tier 1", source="A")
        _emit(store, a.record_id, VerificationResult.SUPPORTED)

        default_ids = {r.evidence_id for r in store.retrievable_index_records()}
        assert default_ids == {a.record_id, b.record_id}

        only_supported = {r.evidence_id for r in store.retrievable_index_records(
            verification_filter={VerificationStatus.SUPPORTED})}
        assert only_supported == {a.record_id}

    def test_filtering_does_not_change_canonical_storage(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED)
        digest = store.log_sha256()
        store.retrievable_index_records(verification_filter={VerificationStatus.UNVERIFIED})
        assert store.log_sha256() == digest


class TestV11CrashRestartSafety:
    def test_identical_determination_is_idempotent_not_a_second_opinion(self, tmp_path):
        """A worker that crashes after appending re-derives the SAME event id
        on restart, so replay drops it rather than recording a duplicate."""
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        _claim(store, value="Tier 4", source="B")
        w = DeterministicMemoryConsistencyChecker(store)
        e1 = w.verify(a.record_id, observed_at_utc="2020-01-01T00:00:00+00:00")
        e2 = w.verify(a.record_id, observed_at_utc="2020-01-01T00:00:00+00:00")
        assert e1.verification_event_id == e2.verification_event_id
        assert len(store.verification_events(a.record_id)) == 1, "no duplicate opinion"
        assert store.verification_status(a.record_id) is VerificationStatus.SUPPORTED

    def test_committed_verification_survives_restart(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED)
        reloaded = ClaimStore(tmp_path / "store")
        assert reloaded.verification_status(a.record_id) is VerificationStatus.SUPPORTED
        assert len(reloaded.verification_events(a.record_id)) == 1

    def test_torn_verification_append_is_dropped_not_fatal(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.SUPPORTED)
        good = store.verification_status(a.record_id)
        with store.log_path.open("a") as fh:
            fh.write('{"event": "VERIFICATION", "verification": {"verificat')
        reloaded = ClaimStore(tmp_path / "store", auto_snapshot=False)
        assert reloaded.truncated_tail is True
        assert reloaded.verification_status(a.record_id) is good


class TestV12CertifiedReaderUnchanged:
    def test_certified_identity_unchanged_across_verification(self, tmp_path):
        pin_certified_memory_v1_boundary_policy()
        before = assert_certified_memory_v1_unchanged()
        store = _store(tmp_path)
        a = _claim(store, value="Tier 4", source="A")
        _claim(store, value="Tier 9", source="B")
        w = DeterministicMemoryConsistencyChecker(store)
        for r in store.retrievable():
            w.verify(r.record_id)
        assert assert_certified_memory_v1_unchanged().canonical_sha256() == \
            before.canonical_sha256()

    def test_verification_status_is_not_exposed_to_the_reader_by_default(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        _emit(store, a.record_id, VerificationResult.CONTRADICTED)
        rec = store.retrievable_index_records()[0]
        assert "verification_status" not in rec.metadata, \
            "V12: verification status must not reach the certified reader yet"

    def test_verification_layer_does_not_import_the_reasoning_path(self):
        src = (ROOT / "hrm_adaptive_memory/memory_write/verification.py").read_text()
        for forbidden in ("g2_paths", "packet_composition", "packet_stage",
                          "selector_v2", "runtime_graph", "generate_with_confidence"):
            assert forbidden not in src


class TestFirstWorkerAndPriority:
    def test_worker_never_emits_falsified(self, tmp_path):
        """Memory consistency cannot establish falsity -- that needs evidence
        from outside the store (deferred to V2)."""
        store = _store(tmp_path)
        _claim(store, value="Tier 4", source="A")
        _claim(store, value="Tier 9", source="B")
        w = DeterministicMemoryConsistencyChecker(store)
        for r in store.retrievable():
            assert w.determine(r.record_id)[0] is not VerificationResult.FALSIFIED

    def test_isolated_claim_is_inconclusive_not_supported(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        DeterministicMemoryConsistencyChecker(store).verify(a.record_id)
        assert store.verification_status(a.record_id) is VerificationStatus.INCONCLUSIVE

    def test_determine_is_side_effect_free(self, tmp_path):
        store = _store(tmp_path)
        a = _claim(store)
        digest = store.log_sha256()
        DeterministicMemoryConsistencyChecker(store).determine(a.record_id)
        assert store.log_sha256() == digest

    def test_queue_prioritizes_conflicted_claims(self, tmp_path):
        store = _store(tmp_path)
        quiet = _claim(store, E2, value="Tier 1", source="A")
        a = _claim(store, E1, value="Tier 4", source="A")
        b = _claim(store, E1, value="Tier 9", source="B")
        q = DeterministicMemoryConsistencyChecker(store).queue()
        top = {rid for prio, rid in q if prio == 1}
        assert top == {a.record_id, b.record_id}
        assert quiet.record_id not in top

    def test_priority_schema_records_unused_signals(self, tmp_path):
        p = priority_signals(conflict_present=False, claim_importance=0.9,
                             source_trust=0.5, novelty=0.2,
                             age_staleness_seconds=10.0, dependent_count=3)
        for k in ("claim_importance", "source_trust", "novelty",
                  "age_staleness_seconds", "dependent_count"):
            assert p[k] is not None
        assert p["v1_effective_priority"] == 0, "only conflict_present drives V1 priority"
