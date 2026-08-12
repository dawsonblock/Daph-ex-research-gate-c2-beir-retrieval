#!/usr/bin/env python3
"""Record permanent adversarial regression evidence for V2A qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.external_verification import (  # noqa: E402
    AcquisitionRequest, AcquisitionResult, AcquisitionStatus,
    DeterministicExactFieldVerifier, EvidenceStore, SourceType,
    derive_current_status)
from hrm_adaptive_memory.memory_write import (  # noqa: E402
    ClaimStore, IntegrityError, LifecycleState, VerificationStatus)
from hrm_adaptive_memory.memory_write.verification import (  # noqa: E402
    DeterministicMemoryConsistencyChecker)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim(store: ClaimStore, *, entity: str = "Curlew control module",
           relation: str = "ownership tier", value: str = "Tier 4",
           source: str = "source-a"):
    return store.ingest(
        subject=entity, relation=relation, value=value, source_id=source,
        observed_at_utc=f"2026-08-11T00:00:{store.corpus_version:02d}+00:00").record


def _expect_error(case: str, expected, operation) -> dict:
    try:
        operation()
    except expected as exc:
        return {"case": case, "passed": True, "observed": type(exc).__name__,
                "detail": str(exc)}
    return {"case": case, "passed": False,
            "detail": f"expected {expected.__name__}"}


def _tamper_log(root: Path, *, reseal: bool) -> None:
    store = ClaimStore(root)
    _claim(store)
    events = [json.loads(line) for line in store.log_path.read_text().splitlines()]
    events[0]["record"]["value"] = "Tier 9"
    events[0]["record"]["content"] = events[0]["record"]["content"].replace(
        "Tier 4", "Tier 9")
    store.log_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")
    if reseal:
        manifest = json.loads(store.manifest_path.read_text())
        manifest["event_log_sha256"] = hashlib.sha256(
            store.log_path.read_bytes()).hexdigest()
        store.manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    ClaimStore(root)


def run_cases(base: Path) -> list[dict]:
    results: list[dict] = []

    results.append(_expect_error(
        "tamper_canonical_log", IntegrityError,
        lambda: _tamper_log(base / "tamper-manifest", reseal=False)))
    results.append(_expect_error(
        "tamper_record_content_retain_id_resealed_manifest", IntegrityError,
        lambda: _tamper_log(base / "tamper-id", reseal=True)))

    def unknown_event():
        root = base / "unknown-event"
        store = ClaimStore(root)
        store.log_path.write_text('{"event":"BOGUS"}\n')
        ClaimStore(root)

    results.append(_expect_error(
        "unknown_event_type", IntegrityError, unknown_event))

    root = base / "retry"
    store = ClaimStore(root)
    claim = _claim(store)
    _claim(store, source="source-b")
    worker = DeterministicMemoryConsistencyChecker(store)
    first, second = worker.verify(claim.record_id), worker.verify(claim.record_id)
    results.append({
        "case": "retry_after_committed_verification",
        "passed": first.verification_event_id == second.verification_event_id
                  and len(store.verification_events(claim.record_id)) == 1,
        "event_id": first.verification_event_id,
        "event_count": len(store.verification_events(claim.record_id)),
    })

    root = base / "same-source"
    store = ClaimStore(root)
    claim = _claim(store)
    _claim(store, source="source-a")
    event = DeterministicMemoryConsistencyChecker(store).verify(claim.record_id)
    results.append({
        "case": "same_source_repeated_twice",
        "passed": event.result.value == "INCONCLUSIVE",
        "observed_result": event.result.value,
        "evidence_ids": list(event.evidence_ids),
    })

    root = base / "mirrors"
    store = ClaimStore(root / "claims")
    evidence = EvidenceStore(root / "evidence")
    claim = _claim(store)
    lineage_id = "lin-one-upstream"
    mirror_ids = []
    for index in range(2):
        request = AcquisitionRequest(
            f"https://mirror-{index}.invalid/carbon",
            canonical_source_uri="https://authority.invalid/carbon",
            source_type=SourceType.AUTHORITATIVE_STRUCTURED_DATA)
        acquisition = AcquisitionResult(
            AcquisitionStatus.SUCCESS, request,
            raw_content=json.dumps({"entity": "Curlew control module",
                                    "ownership tier": "Tier 4"}).encode(),
            fetched_at=f"2026-08-11T00:00:0{index}+00:00",
            extracted_fields={"entity": "Curlew control module",
                              "ownership tier": "Tier 4"},
            upstream_source_id="one-upstream", source_lineage_id=lineage_id)
        mirror_ids.append(evidence.append_acquisition(
            acquisition, claim_record_id=claim.record_id,
            acquisition_method="adversarial-mirror",
            acquisition_version="1.0.0", provenance={}).evidence_id)
    lineage = evidence.source_lineage(lineage_id)
    results.append({
        "case": "same_upstream_through_two_mirrors",
        "passed": lineage is not None
                  and len(lineage.member_evidence_ids) == 2
                  and lineage.lineage_id == lineage_id,
        "lineage_id": lineage.lineage_id if lineage else None,
        "mirror_evidence_ids": mirror_ids,
        "independent_lineage_count": 1 if lineage else 0,
    })

    root = base / "cross-supersession"
    store = ClaimStore(root)
    prior = _claim(store)
    result = _expect_error(
        "cross_claim_supersession_attempt", ValueError,
        lambda: store.ingest(
            subject="Jacana pressure assembly", relation="relay code",
            value="Blue", source_id="source-b", supersedes=prior.record_id))
    result["prior_state_after_attempt"] = store.get(prior.record_id).lifecycle_state.value
    result["passed"] = result["passed"] and (
        store.get(prior.record_id).lifecycle_state is LifecycleState.ACTIVE)
    results.append(result)

    root = base / "withdraw-support"
    claims = ClaimStore(root / "claims")
    evidence = EvidenceStore(root / "evidence")
    claim = _claim(claims)
    request = AcquisitionRequest(
        "fixture://authority/curlew",
        source_type=SourceType.AUTHORITATIVE_STRUCTURED_DATA)
    acquisition = AcquisitionResult(
        AcquisitionStatus.SUCCESS, request,
        raw_content=b'{"entity":"Curlew control module","ownership tier":"Tier 4"}',
        fetched_at="2026-08-11T00:00:00+00:00",
        extracted_fields={"entity": "Curlew control module",
                          "ownership tier": "Tier 4"},
        source_lineage_id="lin-authority-curlew")
    captured = evidence.append_acquisition(
        acquisition, claim_record_id=claim.record_id,
        acquisition_method="adversarial-fixture", acquisition_version="1.0.0",
        provenance={"verification_job_id": "withdraw-job"})
    verifier = DeterministicExactFieldVerifier()
    decision = verifier.verify(claim, captured)
    claims.append_external_verification(
        claim_record_id=claim.record_id, checker_id="adversarial-v2a",
        checker_type=verifier.CHECKER_TYPE, method=decision.method,
        method_version=decision.method_version,
        evidence_ids=decision.evidence_ids, evidence_resolver=evidence.get,
        result=decision.result, confidence=1.0,
        reason_code=decision.reason_code,
        source_lineage_ids=(captured.source_lineage_id,),
        receipt_hash=decision.receipt_hash, verification_job_id="withdraw-job",
        observed_at_utc="2026-08-11T00:00:00+00:00")
    before = derive_current_status(claims, evidence, claim.record_id)
    evidence.retract(
        captured.evidence_id, reason="authority withdrew record",
        observed_at="2026-08-12T00:00:00+00:00", provenance={})
    after = derive_current_status(claims, evidence, claim.record_id)
    results.append({
        "case": "withdraw_all_supporting_evidence",
        "passed": before is VerificationStatus.SUPPORTED
                  and after is VerificationStatus.UNVERIFIED
                  and len(claims.verification_events(claim.record_id)) == 1,
        "before": before.value, "after": after.value,
        "historical_event_count": len(claims.verification_events(claim.record_id)),
    })

    def corrupt_middle():
        root = base / "mid-log"
        store = ClaimStore(root)
        _claim(store)
        _claim(store, entity="Auk relay unit", source="source-b")
        lines = store.log_path.read_text().splitlines(keepends=True)
        store.log_path.write_text(lines[0] + '{"event":\n' + "".join(lines[1:]))
        store.manifest_path.unlink()
        ClaimStore(root)

    results.append(_expect_error(
        "mid_log_corruption", json.JSONDecodeError, corrupt_middle))

    root = base / "torn-tail"
    store = ClaimStore(root)
    claim = _claim(store)
    before_hash = store.consolidated_state().state_hash()
    with store.log_path.open("a") as handle:
        handle.write('{"event":"INGEST","record":')
    replayed = ClaimStore(root)
    results.append({
        "case": "torn_final_append",
        "passed": replayed.truncated_tail
                  and replayed.consolidated_state().state_hash() == before_hash
                  and replayed.get(claim.record_id) is not None,
        "truncated_tail": replayed.truncated_tail,
        "state_hash_equal": replayed.consolidated_state().state_hash() == before_hash,
    })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "evidence/background_verification_v2a/adversarial/adversarial.json"))
    parser.add_argument("--skip-litlogger", action="store_true")
    args = parser.parse_args()
    if _git("status", "--porcelain"):
        raise RuntimeError("adversarial qualification refuses a dirty source tree")
    source_commit = _git("rev-parse", "HEAD")
    source_tree_hash = _git("rev-parse", "HEAD^{tree}")
    experiment = None
    if not args.skip_litlogger:
        import litlogger
        experiment = litlogger.init(
            name=f"v2a-adversarial-{source_commit[:7]}",
            teamspace="deep-gpu-acceleration-project",
            metadata={"protocol": "BACKGROUND_VERIFICATION_V2A",
                      "source_commit": source_commit, "source_tree_hash": source_tree_hash,
                      "location": "US", "altitude": "1334"})
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="daph-v2a-adversarial-") as directory:
        cases = run_cases(Path(directory))
    run_valid = len(cases) == 10 and all(case["passed"] for case in cases)
    receipt = {
        "artifact": "BACKGROUND_VERIFICATION_V2A_ADVERSARIAL_REGRESSIONS",
        "run_valid": run_valid,
        "mechanism_status": "REGRESSION_EVIDENCE_RECORDED",
        "scientific_verdict": "PASS" if run_valid else "FAIL",
        "source_commit": source_commit,
        "source_tree_hash": source_tree_hash,
        "branch": _git("branch", "--show-current"),
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_s": time.perf_counter() - started,
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{_sha256(out)}  {out.name}\n")
    if experiment is not None:
        for case in cases:
            experiment["case_pass"].append(case["passed"])
        experiment["run_valid"] = str(run_valid).lower()
        experiment.finalize()
    print(json.dumps({"run_valid": run_valid, "cases": len(cases),
                      "receipt": str(out)}, sort_keys=True))
    return 0 if run_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
