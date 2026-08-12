"""Adversarial regressions for the audited verified-memory substrate."""
from __future__ import annotations

import hashlib
import json

import pytest

from hrm_adaptive_memory.memory_write import ClaimStore, VerificationStatus
from hrm_adaptive_memory.memory_write.claim_store import IntegrityError
from hrm_adaptive_memory.memory_write.verification import (
    DeterministicMemoryConsistencyChecker, VerificationResult)


def _claim(store, *, entity="Curlew control module", relation="ownership tier",
           value="Tier 4", source="source-a"):
    return store.ingest(subject=entity, relation=relation, value=value,
                        source_id=source).record


def _reseal_manifest(store):
    manifest = json.loads(store.manifest_path.read_text())
    manifest["event_log_sha256"] = hashlib.sha256(store.log_path.read_bytes()).hexdigest()
    store.manifest_path.write_text(json.dumps(manifest, sort_keys=True))


def test_tampered_log_is_rejected_before_replay(tmp_path):
    store = ClaimStore(tmp_path / "store")
    record = _claim(store)
    events = [json.loads(line) for line in store.log_path.read_text().splitlines()]
    events[0]["record"]["value"] = "Tier 9"
    events[0]["record"]["content"] = events[0]["record"]["content"].replace("Tier 4", "Tier 9")
    store.log_path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")
    with pytest.raises(IntegrityError, match="committed manifest"):
        ClaimStore(tmp_path / "store")
    assert record.record_id


def test_record_id_is_rederived_even_if_attacker_reseals_manifest(tmp_path):
    store = ClaimStore(tmp_path / "store")
    _claim(store)
    events = [json.loads(line) for line in store.log_path.read_text().splitlines()]
    events[0]["record"]["value"] = "Tier 9"
    events[0]["record"]["content"] = events[0]["record"]["content"].replace("Tier 4", "Tier 9")
    store.log_path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")
    _reseal_manifest(store)
    with pytest.raises(IntegrityError, match="record_id"):
        ClaimStore(tmp_path / "store")


def test_unknown_event_type_fails_closed(tmp_path):
    store = ClaimStore(tmp_path / "store")
    store.log_path.write_text('{"event":"BOGUS"}\n')
    with pytest.raises(IntegrityError, match="unknown canonical event"):
        ClaimStore(tmp_path / "store")


def test_verification_retries_are_exactly_once_without_timestamp_pinning(tmp_path):
    store = ClaimStore(tmp_path / "store")
    claim = _claim(store)
    _claim(store, source="source-b")
    worker = DeterministicMemoryConsistencyChecker(store)
    first = worker.verify(claim.record_id)
    second = worker.verify(claim.record_id)
    assert first.verification_event_id == second.verification_event_id
    assert len(store.verification_events(claim.record_id)) == 1


def test_same_source_is_not_independent_corroboration(tmp_path):
    store = ClaimStore(tmp_path / "store")
    claim = _claim(store)
    _claim(store, source="source-a")
    event = DeterministicMemoryConsistencyChecker(store).verify(claim.record_id)
    assert event.result is VerificationResult.INCONCLUSIVE


def test_supersession_requires_same_canonical_claim(tmp_path):
    store = ClaimStore(tmp_path / "store")
    prior = _claim(store)
    with pytest.raises(ValueError, match="same canonical entity and relation"):
        _claim(store, entity="Jacana pressure assembly", relation="relay code",
               value="Blue", source="source-b")
        store.ingest(subject="Jacana pressure assembly", relation="relay code",
                     value="Blue", source_id="source-b", supersedes=prior.record_id)
    assert store.get(prior.record_id).lifecycle_state.value == "ACTIVE"


def test_retracted_supporting_evidence_no_longer_drives_current_status(tmp_path):
    store = ClaimStore(tmp_path / "store")
    claim = _claim(store)
    support = _claim(store, source="source-b")
    worker = DeterministicMemoryConsistencyChecker(store)
    worker.verify(claim.record_id)
    assert store.verification_status(claim.record_id) is VerificationStatus.SUPPORTED
    store.retract(support.record_id, "withdrawn")
    assert store.verification_status(claim.record_id) is VerificationStatus.UNVERIFIED


def test_queue_does_not_requeue_currently_verified_claims(tmp_path):
    store = ClaimStore(tmp_path / "store")
    claim = _claim(store)
    _claim(store, source="source-b")
    worker = DeterministicMemoryConsistencyChecker(store)
    worker.verify(claim.record_id)
    assert claim.record_id not in {record_id for _priority, record_id in worker.queue()}
