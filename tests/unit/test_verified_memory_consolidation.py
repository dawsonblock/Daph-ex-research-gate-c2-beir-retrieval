"""VERIFIED_MEMORY_CONSOLIDATION_V1 acceptance tests C1-C12.

Per configs/verified_memory_consolidation_v1_design.json. Consolidation
REORGANIZES memory; it does not decide truth.

C9 is the headline: incremental consolidation == full replay consolidation,
compared across two genuinely different implementations.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.certified_memory import (  # noqa: E402
    assert_certified_memory_v1_unchanged, pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.memory_write import ClaimStore, ConflictOutcome, LifecycleState  # noqa: E402
from hrm_adaptive_memory.memory_write.consolidation import (  # noqa: E402
    ALIAS_RELATION, consolidate_from_scratch)

E1, E2, E3 = "Wren pressure assembly", "Auk relay unit", "Finch control module"
REL = "operating tier"


def _store(tmp_path, name="store", **kw) -> ClaimStore:
    return ClaimStore(tmp_path / name, **kw)


def _full(store):
    return consolidate_from_scratch(store.all_records(), store.corpus_version)


def _mixed_workload(store) -> dict:
    """New claims, exact duplicates, aliases, conflicts, supersessions,
    retractions -- every event type consolidation must handle."""
    ids = {}
    ids["a"] = store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="A").record.record_id
    # support: same claim, different source
    ids["b"] = store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="B").record.record_id
    # conflict on another entity
    ids["c"] = store.ingest(subject=E2, relation=REL, value="Tier 1", source_id="A").record.record_id
    ids["d"] = store.ingest(subject=E2, relation=REL, value="Tier 9", source_id="B").record.record_id
    # alias claim
    ids["alias"] = store.ingest(subject=E3, relation=ALIAS_RELATION, value=E1,
                                source_id="A").record.record_id
    # supersession
    ids["e"] = store.ingest(subject=E3, relation="thermal rating", value="High",
                            source_id="A").record.record_id
    ids["f"] = store.ingest(subject=E3, relation="thermal rating", value="Low",
                            source_id="A", supersedes=ids["e"]).record.record_id
    # retraction
    ids["g"] = store.ingest(subject=E2, relation="thermal rating", value="Mid",
                            source_id="C").record.record_id
    store.retract(ids["g"], reason="withdrawn")
    return ids


class TestC1DeterministicReplay:
    def test_replaying_the_same_log_twice_gives_identical_hashes(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        h1 = ClaimStore(tmp_path / "store").consolidated_state().state_hash()
        h2 = ClaimStore(tmp_path / "store").consolidated_state().state_hash()
        assert h1 == h2

    def test_hash_is_independent_of_dict_iteration_order(self, tmp_path):
        """Canonical ordering: the hash must depend only on content."""
        s1 = _store(tmp_path, "s1")
        s2 = _store(tmp_path, "s2")
        for s, order in ((s1, [E1, E2]), (s2, [E2, E1])):
            for e in order:
                s.ingest(subject=e, relation=REL, value="Tier 4", source_id="A",
                         observed_at_utc="2020-01-01T00:00:00+00:00")
        assert s1.consolidated_state().state_hash() == s2.consolidated_state().state_hash()


class TestC2DerivedStateFullyRebuildable:
    def test_deleting_all_derived_artifacts_and_rebuilding_reproduces_state(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        before = store.consolidated_state().state_hash()

        (tmp_path / "store" / "consolidated_snapshot.json").unlink()
        (tmp_path / "store" / "MANIFEST.json").unlink()

        rebuilt = ClaimStore(tmp_path / "store")
        assert rebuilt.consolidated_state().state_hash() == before
        assert rebuilt.snapshot_path.is_file(), "snapshot must be republished on rebuild"


class TestC3CanonicalRecordsNeverMutated:
    def test_consolidation_does_not_touch_the_event_log(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        digest = store.log_sha256()
        for _ in range(3):
            store.consolidated_state()
            _full(store)
        assert store.log_sha256() == digest

    def test_snapshots_do_not_alias_mutable_state(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        a = store.consolidated_state()
        store.ingest(subject="Osprey sensor array", relation=REL, value="Tier 2", source_id="D")
        b = store.consolidated_state()
        assert a.state_hash() != b.state_hash(), "earlier snapshot must not mutate with the store"


class TestC4ContradictionsPreserved:
    def test_conflicting_claims_stay_active_and_grouped(self, tmp_path):
        store = _store(tmp_path)
        ids = _mixed_workload(store)
        state = store.consolidated_state()
        groups = dict(state.contradiction_groups)
        key = f"{E2.lower()}|{REL}"
        assert key in groups
        assert set(groups[key]) == {ids["c"], ids["d"]}
        assert ids["c"] in state.active_record_ids and ids["d"] in state.active_record_ids

    def test_consolidation_never_reduces_a_contradiction_to_one_member(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        for _key, members in store.consolidated_state().contradiction_groups:
            assert len(members) >= 2


class TestC5ExplicitSupersessionOnly:
    def test_later_timestamp_alone_does_not_supersede_in_derived_state(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="A",
                         observed_at_utc="2020-01-01T00:00:00+00:00")
        b = store.ingest(subject=E1, relation=REL, value="Tier 9", source_id="B",
                         observed_at_utc="2099-01-01T00:00:00+00:00")
        assert b.outcome is ConflictOutcome.CONFLICT
        state = store.consolidated_state()
        assert a.record.record_id in state.active_record_ids
        assert b.record.record_id in state.active_record_ids
        assert len(state.contradiction_groups) == 1


class TestC6RetractionsRemovedFromActiveRetrieval:
    def test_retracted_absent_from_projection_and_index_but_present_in_log(self, tmp_path):
        store = _store(tmp_path)
        ids = _mixed_workload(store)
        state = store.consolidated_state()
        assert ids["g"] not in state.active_record_ids
        assert ids["g"] not in {r.evidence_id for r in store.retrievable_index_records()}
        assert store.get(ids["g"]).lifecycle_state is LifecycleState.RETRACTED
        assert ids["g"] in store.log_path.read_text()

    def test_superseded_also_absent_from_projection(self, tmp_path):
        store = _store(tmp_path)
        ids = _mixed_workload(store)
        state = store.consolidated_state()
        assert ids["e"] not in state.active_record_ids
        assert ids["f"] in state.active_record_ids


class TestC7ProvenanceSurvivesDuplicateClustering:
    def test_cluster_members_retain_individual_provenance(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="A",
                         observed_at_utc="2020-01-01T00:00:00+00:00")
        b = store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="B",
                         observed_at_utc="2021-06-06T00:00:00+00:00")
        clusters = store.consolidated_state().duplicate_clusters
        assert len(clusters) == 1
        members = clusters[0]
        assert set(members) == {a.record.record_id, b.record.record_id}
        # every member individually recoverable with its own provenance
        srcs = {store.get(m).source_id for m in members}
        obs = {store.get(m).observed_at_utc for m in members}
        assert srcs == {"A", "B"}
        assert obs == {"2020-01-01T00:00:00+00:00", "2021-06-06T00:00:00+00:00"}


class TestC8CertifiedReaderUnchanged:
    def test_identity_holds_across_consolidation(self, tmp_path):
        pin_certified_memory_v1_boundary_policy()
        before = assert_certified_memory_v1_unchanged()
        store = _store(tmp_path)
        _mixed_workload(store)
        store.consolidated_state()
        _full(store)
        assert assert_certified_memory_v1_unchanged().canonical_sha256() == before.canonical_sha256()

    def test_consolidation_does_not_import_the_reasoning_path(self):
        src = (ROOT / "hrm_adaptive_memory/memory_write/consolidation.py").read_text()
        for forbidden in ("g2_paths", "packet_composition", "packet_stage",
                          "selector_v2", "runtime_graph", "generate_with_confidence"):
            assert forbidden not in src


class TestC9IncrementalEqualsFullRebuild:
    """THE HEADLINE INVARIANT."""

    def test_incremental_matches_full_scan_after_mixed_workload(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        assert store.consolidated_state().state_hash() == _full(store).state_hash()

    def test_invariant_holds_at_every_step(self, tmp_path):
        """Checked after EVERY event, so a divergence cannot be masked by a
        later compensating change."""
        store = _store(tmp_path, auto_snapshot=False)
        ops = [
            lambda: store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="A"),
            lambda: store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="B"),
            lambda: store.ingest(subject=E2, relation=REL, value="Tier 1", source_id="A"),
            lambda: store.ingest(subject=E2, relation=REL, value="Tier 9", source_id="B"),
            lambda: store.ingest(subject=E3, relation=ALIAS_RELATION, value=E1, source_id="A"),
        ]
        for op in ops:
            op()
            assert store.consolidated_state().state_hash() == _full(store).state_hash()

    def test_retraction_dissolves_groups_identically_in_both_paths(self, tmp_path):
        """The most likely consolidation bug: a retracted record left behind
        in a support/contradiction/duplicate group."""
        store = _store(tmp_path)
        a = store.ingest(subject=E2, relation=REL, value="Tier 1", source_id="A")
        b = store.ingest(subject=E2, relation=REL, value="Tier 9", source_id="B")
        assert len(store.consolidated_state().contradiction_groups) == 1
        store.retract(b.record.record_id, reason="withdrawn")
        inc = store.consolidated_state()
        assert inc.state_hash() == _full(store).state_hash()
        assert inc.contradiction_groups == (), "contradiction must dissolve when one side retracts"
        assert a.record.record_id in inc.active_record_ids

    def test_reloaded_store_matches_in_process_store(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        assert ClaimStore(tmp_path / "store").consolidated_state().state_hash() == \
            store.consolidated_state().state_hash()


class TestC10CacheInvalidationFollowsStateHash:
    def test_active_membership_change_changes_state_hash_and_cache_key(self, tmp_path):
        store = _store(tmp_path)
        a = store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="A")
        h_before = store.consolidated_state().state_hash()
        ids_before = frozenset(r.evidence_id for r in store.retrievable_index_records())

        store.retract(a.record.record_id, reason="withdrawn")
        assert store.consolidated_state().state_hash() != h_before
        assert frozenset(r.evidence_id for r in store.retrievable_index_records()) != ids_before

    def test_stale_index_would_have_served_a_removed_record(self, tmp_path):
        """Demonstrates the hazard is real, not hypothetical."""
        store = _store(tmp_path)
        a = store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="A")
        stale_records = store.retrievable_index_records()
        store.retract(a.record.record_id, reason="withdrawn")

        fresh = store.retrievable_index_records()
        assert fresh == []
        # querying the STALE index still returns the retracted record
        backend = get_cached_backend(CanonicalRetrievalMode.BM25, stale_records)
        hits = asyncio.run(backend.search(f"What is the {REL} for {E1}?", k=5)).evidence
        assert a.record.record_id in {e.evidence_id for e in hits}, (
            "a stale index WOULD have served the retracted record -- which is why "
            "the index must be rebound whenever the state hash changes")


class TestC11CrashRestartRecovery:
    def test_crash_after_append_before_snapshot(self, tmp_path):
        """Snapshot missing entirely: the log alone must suffice."""
        store = _store(tmp_path, auto_snapshot=False)
        _mixed_workload(store)
        expected = store.consolidated_state().state_hash()
        assert not store.snapshot_path.is_file()
        assert ClaimStore(tmp_path / "store").consolidated_state().state_hash() == expected

    def test_partial_snapshot_is_never_published(self, tmp_path):
        """Atomic publish: a temp file must never be mistaken for a snapshot."""
        store = _store(tmp_path)
        _mixed_workload(store)
        tmp = store.snapshot_path.with_suffix(store.snapshot_path.suffix + ".tmp")
        tmp.write_text('{"state": "HALF-WRI')
        reloaded = ClaimStore(tmp_path / "store")
        assert reloaded.consolidated_state().state_hash() == store.consolidated_state().state_hash()
        assert reloaded.snapshot_is_valid()

    def test_snapshot_describing_a_different_log_is_discarded(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        snap = json.loads(store.snapshot_path.read_text())
        snap["derived_from_log_sha256"] = "0" * 64
        store.snapshot_path.write_text(json.dumps(snap))
        reloaded = ClaimStore(tmp_path / "store", auto_snapshot=False)
        assert not reloaded.snapshot_is_valid(), "mismatched snapshot must be rejected"
        # and the canonical log still yields the correct state
        assert reloaded.consolidated_state().state_hash() == \
            consolidate_from_scratch(reloaded.all_records(), reloaded.corpus_version).state_hash()

    def test_truncated_final_log_line_is_dropped_not_fatal(self, tmp_path):
        """A torn append (crash during stage 1) was never committed."""
        store = _store(tmp_path)
        _mixed_workload(store)
        good = store.consolidated_state().state_hash()
        n_events = store.corpus_version
        with store.log_path.open("a") as fh:
            fh.write('{"event": "INGEST", "record": {"record_i')  # torn

        reloaded = ClaimStore(tmp_path / "store", auto_snapshot=False)
        assert reloaded.truncated_tail is True
        assert reloaded.corpus_version == n_events
        assert reloaded.consolidated_state().state_hash() == good


class TestC12EventOrderReplayInvariants:
    def test_log_is_append_only_prefix_stable(self, tmp_path):
        store = _store(tmp_path)
        store.ingest(subject=E1, relation=REL, value="Tier 4", source_id="A")
        prefix = store.log_path.read_bytes()
        store.ingest(subject=E2, relation=REL, value="Tier 1", source_id="A")
        assert store.log_path.read_bytes().startswith(prefix), \
            "earlier log bytes must never be rewritten"

    def test_events_replay_in_log_order(self, tmp_path):
        store = _store(tmp_path)
        ids = _mixed_workload(store)
        events = [json.loads(l) for l in store.log_path.read_text().splitlines() if l.strip()]
        ingests = [e["record"]["record_id"] for e in events if e["event"] == "INGEST"]
        assert ingests[0] == ids["a"], "first ingest must be first in the log"
        # the supersession STATE_CHANGE must follow the superseding INGEST
        i_f = ingests.index(ids["f"])
        sc = [i for i, e in enumerate(events)
              if e["event"] == "STATE_CHANGE" and e["record_id"] == ids["e"]
              and e["lifecycle_state"] == "SUPERSEDED"]
        i_ingest_f = [i for i, e in enumerate(events)
                      if e["event"] == "INGEST" and e["record"]["record_id"] == ids["f"]][0]
        assert sc and sc[0] > i_ingest_f and i_f >= 0

    def test_corpus_version_equals_event_count(self, tmp_path):
        store = _store(tmp_path)
        _mixed_workload(store)
        n = len([l for l in store.log_path.read_text().splitlines() if l.strip()])
        assert store.corpus_version == n
