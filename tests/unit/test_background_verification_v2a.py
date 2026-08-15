"""Offline V2A acceptance tests: immutable external evidence, no live network."""
from __future__ import annotations

from pathlib import Path

from hrm_adaptive_memory.experiment_integrity.certified_memory import (
    assert_certified_memory_v1_unchanged, pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.external_verification import (
    AcquisitionRequest, EvidenceStore, ExternalVerificationWorker,
    HTTPStructuredDataAcquirer, LocalStructuredFixtureAcquirer, SourceType, VerificationQueue,
    derive_current_status, explain_claim)
from hrm_adaptive_memory.memory_write import ClaimStore, VerificationStatus


FIXTURES = Path(__file__).parents[1] / "fixtures" / "external_verification_v2a"


def _system(tmp_path):
    claims = ClaimStore(tmp_path / "claims")
    evidence = EvidenceStore(tmp_path / "evidence")
    queue = VerificationQueue(tmp_path / "queue")
    acquirer = LocalStructuredFixtureAcquirer.from_directory(FIXTURES)
    return claims, evidence, queue, acquirer


def _claim(claims, value="6"):
    return claims.ingest(subject="Carbon atom", relation="atomic_number", value=value,
                         source_id="test-claim", observed_at_utc="2026-08-10T00:00:00+00:00").record


def _enqueue(queue, claim, uri, *, priority=1, policy="v2a-test"):
    return queue.enqueue(
        claim_id=claim.record_id, priority=priority, reason="acceptance-test",
        created_at="2026-08-11T00:00:00+00:00", verification_policy_id=policy,
        verification_policy_version="1.0.0",
        request=AcquisitionRequest(source_uri=uri,
                                   source_type=SourceType.AUTHORITATIVE_STRUCTURED_DATA))


def test_v2a_t1_default_state_is_unverified(tmp_path):
    claims, evidence, _queue, _acquirer = _system(tmp_path)
    claim = _claim(claims)
    assert claims.verification_status(claim.record_id) is VerificationStatus.UNVERIFIED
    assert derive_current_status(claims, evidence, claim.record_id) is VerificationStatus.UNVERIFIED


def test_v2a_t2_support_immutable_claim_and_t3_falsification(tmp_path):
    claims, evidence, queue, acquirer = _system(tmp_path)
    supported = _claim(claims, "6")
    original_claim_line = claims.log_path.read_bytes()
    _enqueue(queue, supported, "fixture://authority/chemistry/carbon-v1")
    assert ExternalVerificationWorker(claims, evidence, queue, acquirer).run_all()[0].result.value == "SUPPORTED"
    assert claims.log_path.read_bytes().startswith(original_claim_line)
    assert claims.verification_status(supported.record_id) is VerificationStatus.SUPPORTED

    falsified = _claim(claims, "8")
    _enqueue(queue, falsified, "fixture://authority/chemistry/carbon-v1")
    ExternalVerificationWorker(claims, evidence, queue, acquirer).run_all()
    event = claims.verification_events(falsified.record_id)[0]
    assert event.result.value == "FALSIFIED"
    assert event.method_version == "1.0.0"
    assert event.reason_code == "AUTHORITATIVE_EXACT_FIELD_MISMATCH"


def test_v2a_t4_disagreement_is_preserved_not_arbitrated(tmp_path):
    claims, evidence, queue, acquirer = _system(tmp_path)
    claim = _claim(claims)
    _enqueue(queue, claim, "fixture://authority/chemistry/carbon-v1", policy="source-a")
    _enqueue(queue, claim, "fixture://authority/chemistry/carbon-v2", policy="source-b")
    ExternalVerificationWorker(claims, evidence, queue, acquirer).run_all()
    events = claims.verification_events(claim.record_id)
    assert {event.result.value for event in events} == {"SUPPORTED", "FALSIFIED"}
    assert claims.verification_status(claim.record_id) is VerificationStatus.INCONCLUSIVE


def test_v2a_t5_historical_snapshot_is_stable_and_t9_later_evidence_appends(tmp_path):
    claims, evidence, queue, acquirer = _system(tmp_path)
    claim = _claim(claims)
    _enqueue(queue, claim, "fixture://authority/chemistry/carbon-v1", policy="snapshot-a")
    worker = ExternalVerificationWorker(claims, evidence, queue, acquirer)
    worker.run_all()
    first = next(evidence.stream())
    first_raw_hash = first.raw_content_hash
    _enqueue(queue, claim, "fixture://authority/chemistry/carbon-v2", policy="snapshot-b")
    worker.run_all()
    assert evidence.get(first.evidence_id).raw_content_hash == first_raw_hash
    assert len(list(evidence.stream())) == 2
    assert len(claims.verification_events(claim.record_id)) == 2


def test_v2a_t6_mirrors_share_one_lineage(tmp_path):
    claims, evidence, _queue, acquirer = _system(tmp_path)
    claim = _claim(claims)
    base = acquirer.acquire(AcquisitionRequest("fixture://authority/chemistry/carbon-v1"))
    records = []
    for number in range(3):
        request = AcquisitionRequest(f"fixture://mirror/{number}")
        mirrored = base.__class__(
            **{**base.__dict__, "request": request, "fetched_at": f"2026-08-1{number}T00:00:00+00:00",
               "source_lineage_id": "lin-syndicated-carbon"})
        records.append(evidence.append_acquisition(
            mirrored, claim_record_id=claim.record_id, acquisition_method="fixture",
            acquisition_version="1", provenance={}))
    lineage = evidence.source_lineage("lin-syndicated-carbon")
    assert len(lineage.member_evidence_ids) == 3
    assert len({record.source_lineage_id for record in records}) == 1


def test_v2a_t7_failures_are_inconclusive(tmp_path):
    claims, evidence, queue, acquirer = _system(tmp_path)
    claim = _claim(claims)
    _enqueue(queue, claim, "fixture://authority/chemistry/carbon-missing", policy="missing")
    ExternalVerificationWorker(claims, evidence, queue, acquirer).run_all()
    assert claims.verification_events(claim.record_id)[0].result.value == "INCONCLUSIVE"
    other = _claim(claims, "7")
    _enqueue(queue, other, "fixture://authority/network/failure", policy="network")
    ExternalVerificationWorker(claims, evidence, queue, acquirer).run_all()
    assert claims.verification_events(other.record_id)[0].reason_code == "ACQUISITION_NETWORK_ERROR"


def test_generic_http_cannot_promote_caller_asserted_authority():
    request = AcquisitionRequest(
        "https://example.com/data.json",
        source_type=SourceType.AUTHORITATIVE_STRUCTURED_DATA)
    result = HTTPStructuredDataAcquirer().acquire(request)
    assert result.status.value == "INVALID_RESPONSE"
    assert "cannot establish authoritative" in result.detail


def test_generic_http_blocks_private_network_capture():
    request = AcquisitionRequest(
        "https://127.0.0.1/data.json",
        source_type=SourceType.UNTRUSTED_CAPTURE_ONLY)
    result = HTTPStructuredDataAcquirer().acquire(request)
    assert result.status.value == "NETWORK_ERROR"


def test_v2a_t8_replay_hash_t10_retraction_t12_explainability(tmp_path):
    claims, evidence, queue, acquirer = _system(tmp_path)
    claim = _claim(claims)
    _enqueue(queue, claim, "fixture://authority/chemistry/carbon-v1")
    ExternalVerificationWorker(claims, evidence, queue, acquirer).run_all()
    original_hashes = claims.verification_state_hash(), evidence.state_hash()
    event = claims.verification_events(claim.record_id)[0]
    evidence_id = event.evidence_ids[0]
    explanation = explain_claim(claims, evidence, claim.record_id)
    assert explanation["verification_events"][0]["evidence"][0]["raw_content_hash"]
    assert explanation["verification_events"][0]["method_version"] == "1.0.0"

    evidence.retract(evidence_id, reason="fixture withdrawal", observed_at="2026-08-12T00:00:00+00:00",
                     provenance={"test": "V2A-T10"})
    assert evidence.get(evidence_id) is not None and not evidence.is_active(evidence_id)
    assert derive_current_status(claims, evidence, claim.record_id) is VerificationStatus.UNVERIFIED

    replayed_claims = ClaimStore(tmp_path / "claims")
    replayed_evidence = EvidenceStore(tmp_path / "evidence")
    assert replayed_claims.verification_state_hash() == original_hashes[0]
    assert replayed_evidence.state_hash() != original_hashes[1]  # retraction is a current-state change
    assert replayed_evidence.validate_hashes(evidence_id)


def test_v2a_t11_crash_boundaries_are_idempotent(tmp_path):
    claims, evidence, queue, acquirer = _system(tmp_path)
    claim = _claim(claims)
    job = _enqueue(queue, claim, "fixture://authority/chemistry/carbon-v1")
    # Simulate a crash after immutable evidence append but before event append.
    acquisition = acquirer.acquire(job.request)
    evidence.append_acquisition(acquisition, claim_record_id=claim.record_id,
                                acquisition_method="fixture", acquisition_version="1",
                                provenance={"verification_job_id": job.job_id})
    worker = ExternalVerificationWorker(claims, evidence, queue, acquirer)
    worker.run_all()
    assert len(claims.verification_events(claim.record_id)) == 1
    # Simulate restart after event append but before acknowledgement.
    queue._acknowledged.clear()
    restarted = VerificationQueue(tmp_path / "queue")
    ExternalVerificationWorker(claims, evidence, restarted, acquirer).run_all()
    assert len(claims.verification_events(claim.record_id)) == 1


def test_v2a_t13_certified_reader_invariance_t14_v1_regression_t15_equivalence_t16_bounded(tmp_path):
    pin_certified_memory_v1_boundary_policy()
    before = assert_certified_memory_v1_unchanged().canonical_sha256()
    claims, evidence, queue, acquirer = _system(tmp_path)
    claim = _claim(claims)
    _enqueue(queue, claim, "fixture://authority/chemistry/carbon-v1")
    ExternalVerificationWorker(claims, evidence, queue, acquirer).run_all()
    pin_certified_memory_v1_boundary_policy()
    assert assert_certified_memory_v1_unchanged().canonical_sha256() == before
    assert ClaimStore(tmp_path / "claims").verification_state_hash() == claims.verification_state_hash()
    assert not hasattr(evidence, "_events")
