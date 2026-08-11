"""VERIFIED_MEMORY_WRITE_V1 MILESTONE_1 acceptance tests.

Per configs/verified_memory_write_v1_design.json ACCEPTANCE_TESTS_FALSIFIABLE.

Steps 1-7 of the milestone are machinery -- they can all work and still be
useless. Steps 8 and 9 are the real claims, and they are T1/T2/T3 here:
a write must actually BECOME RETRIEVABLE through the UNMODIFIED certified
retrieval path, and a retraction/supersession must actually change what that
path returns.

T6 is the guard against the most likely false PASS: the certified reader
caches backends under a CONTENT-BLIND key, so a write could succeed while
retrieval silently answers from a stale index.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.contracts import IndexRecord  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.certified_memory import (  # noqa: E402
    assert_certified_memory_v1_unchanged, pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.memory_write import (  # noqa: E402
    ClaimStore, ConflictOutcome, LifecycleState, NotNativelyParseableError)

SUBJECT = "Wren pressure assembly"
RELATION = "operating tier"


def _store(tmp_path) -> ClaimStore:
    return ClaimStore(tmp_path / "store")


def _retrieve_ids(store: ClaimStore, query: str, k: int = 5) -> list[str]:
    """Query through the UNMODIFIED certified retrieval path, over a backend
    obtained AFTER the current corpus state."""
    records = store.retrievable_index_records()
    if not records:
        return []
    backend = get_cached_backend(CanonicalRetrievalMode.BM25, records)
    res = asyncio.run(backend.search(query, k=k))
    return [e.evidence_id for e in res.evidence]


class TestT1WriteBecomesRetrievable:
    """MILESTONE_1 step 8. Both directions matter: the PRE-write failure is
    what makes the post-write success meaningful."""

    def test_query_fails_before_write_and_succeeds_after(self, tmp_path):
        store = _store(tmp_path)
        query = f"What is the {RELATION} for {SUBJECT}?"

        assert _retrieve_ids(store, query) == [], "empty store must retrieve nothing"

        r = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4",
                         source_id="src-A")
        assert r.outcome is ConflictOutcome.NOVEL
        # V1: ingest records a LIFECYCLE state only. Verification status is
        # derived from verification events and defaults to UNVERIFIED.
        assert r.record.lifecycle_state is LifecycleState.ACTIVE

        got = _retrieve_ids(store, query)
        assert r.record.record_id in got, "written record must be retrievable"

    def test_written_content_is_natively_parseable(self, tmp_path):
        store = _store(tmp_path)
        r = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        from hrm_adaptive_memory.c4.bridge_extraction import extract_v4_entities
        assert SUBJECT in extract_v4_entities(r.record.content)

    def test_unparseable_subject_is_refused(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(NotNativelyParseableError):
            store.ingest(subject="x", relation=RELATION, value="Tier 4", source_id="src-A")


class TestT2RetractionRemovesReachability:
    """MILESTONE_1 step 9 (retraction half)."""

    def test_retracted_record_is_unreachable_but_history_survives(self, tmp_path):
        store = _store(tmp_path)
        query = f"What is the {RELATION} for {SUBJECT}?"
        r = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        rid = r.record.record_id
        assert rid in _retrieve_ids(store, query)

        store.retract(rid, reason="source withdrawn")
        assert rid not in _retrieve_ids(store, query), "retracted record must be unreachable"

        kept = store.get(rid)
        assert kept is not None, "history must survive retraction"
        assert kept.lifecycle_state is LifecycleState.RETRACTED
        assert kept.content  # bytes intact


class TestT3SupersessionReturnsTheNewValue:
    """MILESTONE_1 step 9 (supersession half)."""

    def test_supersession_swaps_which_value_is_reachable(self, tmp_path):
        store = _store(tmp_path)
        query = f"What is the {RELATION} for {SUBJECT}?"
        old = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4",
                           source_id="src-A").record
        new = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 7",
                           source_id="src-B", supersedes=old.record_id)
        assert new.outcome is ConflictOutcome.SUPERSESSION

        ids = _retrieve_ids(store, query)
        assert new.record.record_id in ids
        assert old.record_id not in ids

        assert store.get(old.record_id).lifecycle_state is LifecycleState.SUPERSEDED
        assert store.get(old.record_id).superseded_by == new.record.record_id
        assert store.get(new.record.record_id).supersedes == old.record_id

    def test_supersedes_unknown_record_is_refused(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(KeyError):
            store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 7",
                         source_id="src-B", supersedes="vmw-does-not-exist")


class TestT4ConflictIsRepresentedNotResolved:
    def test_contradicting_claims_both_persist_and_are_reported(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        b = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 7", source_id="src-B")
        assert b.outcome is ConflictOutcome.CONFLICT
        # CHANGED BY BACKGROUND_VERIFICATION_V1 (V1/V6): ingest OBSERVES the
        # conflict but no longer writes a verification opinion. Both records
        # stay lifecycle ACTIVE; deriving CONTRADICTED is the verification
        # layer's job, tested in test_background_verification.py::TestV3.
        for rid in (a.record.record_id, b.record.record_id):
            assert store.get(rid).lifecycle_state is LifecycleState.ACTIVE

        ids = _retrieve_ids(store, f"What is the {RELATION} for {SUBJECT}?")
        assert a.record.record_id in ids and b.record.record_id in ids, \
            "neither side of a conflict may be silently dropped"

        dis = store.disagreements()
        assert len(dis) == 1
        assert {r.value for r in dis[0][1]} == {"Tier 4", "Tier 7"}

    def test_later_timestamp_alone_is_not_supersession(self, tmp_path):
        """Frozen conservative temporal policy."""
        store = _store(tmp_path)
        store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4",
                     source_id="src-A", observed_at_utc="2020-01-01T00:00:00+00:00")
        later = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 7",
                             source_id="src-B", observed_at_utc="2099-01-01T00:00:00+00:00")
        assert later.outcome is ConflictOutcome.CONFLICT, \
            "a later timestamp alone must NOT supersede"

    def test_same_claim_from_a_second_source_is_support(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        b = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-B")
        assert b.outcome is ConflictOutcome.SUPPORT
        # CHANGED BY BACKGROUND_VERIFICATION_V1 (V1): the SUPPORT observation
        # no longer promotes either record's status. See
        # test_background_verification.py::TestV2 for the derived SUPPORTED.
        assert store.get(a.record.record_id).lifecycle_state is LifecycleState.ACTIVE
        assert store.get(b.record.record_id).lifecycle_state is LifecycleState.ACTIVE

    def test_identical_reingest_is_duplicate_and_adds_nothing(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4",
                         source_id="src-A", observed_at_utc="2020-01-01T00:00:00+00:00")
        n_before = len(store.all_records())
        b = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4",
                         source_id="src-A", observed_at_utc="2020-01-01T00:00:00+00:00")
        assert b.outcome is ConflictOutcome.DUPLICATE
        assert len(store.all_records()) == n_before
        assert b.record.record_id == a.record.record_id


class TestT5CertifiedMemoryUnchanged:
    def test_identity_holds_across_write_operations(self, tmp_path):
        pin_certified_memory_v1_boundary_policy()
        before = assert_certified_memory_v1_unchanged()
        store = _store(tmp_path)
        r = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 7",
                     source_id="src-B", supersedes=r.record.record_id)
        store.retract(r.record.record_id, reason="test")
        after = assert_certified_memory_v1_unchanged()
        assert before.canonical_sha256() == after.canonical_sha256()

    def test_write_layer_does_not_import_the_reasoning_path(self):
        src = (ROOT / "hrm_adaptive_memory/memory_write/claim_store.py").read_text()
        for forbidden in ("g2_paths", "packet_composition", "packet_stage",
                          "selector_v2", "runtime_graph", "generate_with_confidence"):
            assert forbidden not in src, (
                f"write layer must not touch the certified reasoning path ({forbidden})")


class TestT6StaleCacheWouldHaveLied:
    """The guard on T1/T2/T3. The certified reader caches backends under
    (mode, frozenset(evidence_id)) -- content-blind. If ids could be reused
    across a content change, retrieval would answer from a stale index."""

    def test_content_blind_cache_returns_a_stale_backend_on_id_reuse(self):
        shared_id = "vmw-stale-cache-demo"
        old = [IndexRecord(evidence_id=shared_id, source_id="s", content="subject=Wren pressure assembly; operating tier=Tier 4",
                           token_count=6, source_type="t", metadata={})]
        new = [IndexRecord(evidence_id=shared_id, source_id="s", content="subject=Wren pressure assembly; operating tier=Tier 7",
                           token_count=6, source_type="t", metadata={})]
        b_old = get_cached_backend(CanonicalRetrievalMode.BM25, old)
        b_new = get_cached_backend(CanonicalRetrievalMode.BM25, new)
        assert b_old is b_new, (
            "demonstrates the hazard: identical id sets return the SAME cached "
            "backend even though content changed")

    def test_provenance_addressing_makes_the_hazard_unreachable(self, tmp_path):
        """Because record_id derives from content+source+observed_at, any
        change to what a record says yields a different id -- so the
        content-blind cache key necessarily changes too."""
        store = _store(tmp_path)
        a = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        ids_before = frozenset(r.evidence_id for r in store.retrievable_index_records())

        b = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 7",
                         source_id="src-A", supersedes=a.record.record_id)
        ids_after = frozenset(r.evidence_id for r in store.retrievable_index_records())

        # same subject, same relation, same source -- only the VALUE changed,
        # which is exactly the case a content-blind cache would have missed
        assert a.record.record_id != b.record.record_id
        assert ids_before != ids_after, (
            "the reader's cache key is frozenset(evidence_id); it must change when "
            "a record's content changes, or retrieval would answer from a stale index")
        assert get_cached_backend(CanonicalRetrievalMode.BM25,
                                  store.retrievable_index_records()) is not get_cached_backend(
            CanonicalRetrievalMode.BM25,
            [r for r in store.retrievable_index_records() if r.evidence_id in ids_before])


class TestPersistenceIsAppendOnly:
    def test_state_survives_reload_from_disk(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        store.retract(a.record.record_id, reason="test")
        v = store.corpus_version

        reloaded = ClaimStore(tmp_path / "store")
        assert reloaded.corpus_version == v
        assert reloaded.get(a.record.record_id).lifecycle_state is LifecycleState.RETRACTED
        assert reloaded.retrievable() == []

    def test_corpus_version_is_monotonic(self, tmp_path):
        store = _store(tmp_path)
        seen = [store.corpus_version]
        store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        seen.append(store.corpus_version)
        store.ingest(subject=SUBJECT, relation="thermal rating", value="High", source_id="src-A")
        seen.append(store.corpus_version)
        assert seen == sorted(seen) and len(set(seen)) == len(seen)

    def test_history_is_never_destroyed(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=SUBJECT, relation=RELATION, value="Tier 4", source_id="src-A")
        store.retract(a.record.record_id, reason="test")
        raw = (tmp_path / "store" / "claims_events.jsonl").read_text()
        assert "INGEST" in raw and "STATE_CHANGE" in raw
        assert a.record.record_id in raw
