#!/usr/bin/env python3
"""Qualification pressure ladder for BACKGROUND_VERIFICATION_V2A.

The workload exercises the durable queue, immutable evidence store, source
lineage indexes, deterministic exact-field verifier, verification event log,
evidence withdrawal, and cold genesis replay. It freezes no performance
threshold after observing results: run validity requires only complete
receipts and incremental/genesis state-hash equality at every rung.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.external_verification import (  # noqa: E402
    AcquisitionRequest, AcquisitionResult, AcquisitionStatus,
    DeterministicExactFieldVerifier, EvidenceStore, SourceType,
    VerificationQueue)
from hrm_adaptive_memory.memory_write import ClaimStore  # noqa: E402


LADDER = (1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
PROTOCOL = "BACKGROUND_VERIFICATION_V2A"
PROTOCOL_VERSION = "1.0.0"
RUN_SEED = 20260811
RAW_TEMPLATE_COUNT = 1_000
MIRRORS_PER_LINEAGE = 4
RETRACTION_INTERVAL = 20


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def _rss_bytes() -> int:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024


def _index_bytes(mapping) -> int:
    size = sys.getsizeof(mapping)
    for key, value in mapping.items():
        size += sys.getsizeof(key) + sys.getsizeof(value)
        if isinstance(value, (list, tuple, set, frozenset)):
            size += sum(sys.getsizeof(item) for item in value)
    return size


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _start_experiment(name: str, metadata: dict[str, str], disabled: bool):
    if disabled:
        return None
    import litlogger
    return litlogger.init(
        name=name, teamspace="deep-gpu-acceleration-project",
        metadata=metadata, print_url=True)


def _metric(experiment, key: str, value: Any) -> None:
    if experiment is not None:
        experiment[key].append(value)


def _finalize(experiment, status: str) -> None:
    if experiment is not None:
        experiment["run_status"] = status
        experiment.finalize()


def _replay_receipt(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    t0 = time.perf_counter()
    claims = ClaimStore(root / "claims", auto_snapshot=False)
    claim_hash = claims.verification_state_hash()
    claim_events = sum(len(events) for events in claims._verifications.values())
    claim_replay_s = time.perf_counter() - t0
    del claims
    gc.collect()

    t0 = time.perf_counter()
    evidence = EvidenceStore(root / "evidence", auto_snapshot=False)
    evidence_hash = evidence.state_hash()
    evidence_records = len(evidence._records)
    evidence_events = evidence._event_count
    evidence_replay_s = time.perf_counter() - t0
    del evidence
    gc.collect()

    t0 = time.perf_counter()
    queue = VerificationQueue(root / "queue")
    queue_pending = len(queue._jobs)
    queue_acknowledged = len(queue._acknowledged)
    queue_replay_s = time.perf_counter() - t0
    return {
        "claim_verification_state_hash": claim_hash,
        "evidence_state_hash": evidence_hash,
        "claim_verification_events": claim_events,
        "evidence_records": evidence_records,
        "evidence_events_including_retractions": evidence_events,
        "queue_pending": queue_pending,
        "queue_acknowledged": queue_acknowledged,
        "claim_replay_s": claim_replay_s,
        "evidence_replay_s": evidence_replay_s,
        "queue_replay_s": queue_replay_s,
        "total_replay_s": time.perf_counter() - started,
    }


def _subprocess_replay(root: Path) -> dict[str, Any]:
    raw = subprocess.check_output([
        sys.executable, str(Path(__file__).resolve()),
        "--replay-only", "--root", str(root), "--skip-litlogger",
    ], cwd=ROOT, text=True)
    return json.loads(raw)


def _instrument_atomic_writes(store, counter: dict[str, int], key: str) -> None:
    original = store._atomic_write

    def measured(path: Path, text: str) -> None:
        counter[key] += len(text.encode())
        original(path, text)

    store._atomic_write = measured


def run(args) -> int:
    source_commit = _git("rev-parse", "HEAD")
    source_tree_hash = _git("rev-parse", "HEAD^{tree}")
    if _git("status", "--porcelain"):
        raise RuntimeError("pressure qualification refuses a dirty source tree")

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    experiment = _start_experiment(
        f"v2a-external-pressure-{args.max_events}",
        {
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "source_commit": source_commit,
            "source_tree_hash": source_tree_hash,
            "max_events": str(args.max_events),
            "location": "US",
            "altitude": "1334",
        },
        args.skip_litlogger)

    status = "FAILED"
    try:
        claims = ClaimStore(root / "claims", auto_snapshot=False)
        evidence = EvidenceStore(root / "evidence", auto_snapshot=False)
        queue = VerificationQueue(root / "queue")
        if claims.corpus_version or evidence._event_count or queue._jobs or queue._acknowledged:
            raise RuntimeError("pressure root must be empty")
        verifier = DeterministicExactFieldVerifier()
        claim = claims.ingest(
            subject="Carbon atom", relation="atomic_number", value="6",
            source_id="pressure-claim",
            observed_at_utc="2026-08-11T00:00:00+00:00").record

        derived_written = {"claims": 0, "evidence": 0}
        _instrument_atomic_writes(claims, derived_written, "claims")
        _instrument_atomic_writes(evidence, derived_written, "evidence")
        cumulative = {
            "queue_enqueue_s": 0.0,
            "evidence_append_s": 0.0,
            "verification_decision_s": 0.0,
            "verification_event_append_s": 0.0,
            "queue_ack_s": 0.0,
            "evidence_retraction_s": 0.0,
        }
        withdrawn: list[str] = []
        results = []
        completed = 0
        started_at = datetime.now(UTC).isoformat()
        run_started = time.perf_counter()

        for target in LADDER:
            if target > args.max_events:
                break
            rung_started = time.perf_counter()
            while completed < target:
                index = completed
                created_at = (
                    datetime(2026, 8, 11, tzinfo=UTC) + timedelta(seconds=index)
                ).isoformat()
                request = AcquisitionRequest(
                    source_uri=f"fixture://pressure/mirror/{index}",
                    canonical_source_uri=f"fixture://pressure/upstream/{index // MIRRORS_PER_LINEAGE}",
                    source_type=SourceType.AUTHORITATIVE_STRUCTURED_DATA)

                t0 = time.perf_counter()
                job = queue.enqueue(
                    claim_id=claim.record_id, priority=index % 3,
                    reason="v2a-pressure", created_at=created_at,
                    verification_policy_id="v2a-pressure",
                    verification_policy_version="1.0.0", request=request)
                selected = queue.next_pending()
                if selected is None or selected.job_id != job.job_id:
                    raise RuntimeError("durable queue did not return the enqueued job")
                cumulative["queue_enqueue_s"] += time.perf_counter() - t0

                template = index % RAW_TEMPLATE_COUNT
                raw = _canonical({
                    "entity": "Carbon atom", "atomic_number": "6",
                    "stable_template": template,
                }).encode()
                acquisition = AcquisitionResult(
                    AcquisitionStatus.SUCCESS, request, raw_content=raw,
                    fetched_at=created_at,
                    extracted_fields={"entity": "Carbon atom", "atomic_number": "6"},
                    publisher="V2A pressure fixture",
                    publisher_domain="fixture.invalid",
                    upstream_source_id=f"upstream-{index // MIRRORS_PER_LINEAGE}",
                    source_lineage_id=f"lin-pressure-{index // MIRRORS_PER_LINEAGE:08d}")

                t0 = time.perf_counter()
                record = evidence.append_acquisition(
                    acquisition, claim_record_id=claim.record_id,
                    acquisition_method="pressure_fixture",
                    acquisition_version="1.0.0",
                    provenance={"verification_job_id": job.job_id})
                cumulative["evidence_append_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                decision = verifier.verify(claim, record)
                cumulative["verification_decision_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                claims.append_external_verification(
                    claim_record_id=claim.record_id,
                    checker_id="external-v2a-pressure",
                    checker_type=verifier.CHECKER_TYPE,
                    method=decision.method,
                    method_version=decision.method_version,
                    evidence_ids=decision.evidence_ids,
                    evidence_resolver=evidence.get,
                    result=decision.result, confidence=1.0,
                    reason_code=decision.reason_code,
                    source_lineage_ids=(record.source_lineage_id,),
                    receipt_hash=decision.receipt_hash,
                    verification_job_id=job.job_id,
                    notes="pressure qualification",
                    observed_at_utc=created_at)
                cumulative["verification_event_append_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                queue.acknowledge(job.job_id)
                cumulative["queue_ack_s"] += time.perf_counter() - t0

                if index and index % RETRACTION_INTERVAL == 0:
                    t0 = time.perf_counter()
                    evidence.retract(
                        record.evidence_id, reason="pressure-withdrawal",
                        observed_at=created_at, provenance={"index": index})
                    cumulative["evidence_retraction_s"] += time.perf_counter() - t0
                    withdrawn.append(record.evidence_id)
                completed += 1

            checkpoint_started = time.perf_counter()
            claims.publish_snapshot()
            evidence.publish_snapshot()
            incremental = {
                "claim_verification_state_hash": claims.verification_state_hash(),
                "evidence_state_hash": evidence.state_hash(),
            }
            replay = _subprocess_replay(root)
            equality = (
                replay["claim_verification_state_hash"]
                == incremental["claim_verification_state_hash"]
                and replay["evidence_state_hash"] == incremental["evidence_state_hash"]
                and replay["queue_pending"] == 0
                and replay["queue_acknowledged"] == target
            )
            raw_dir_bytes = _dir_bytes(evidence.raw_dir)
            canonical_bytes = sum(
                path.stat().st_size for path in
                (claims.log_path, evidence.log_path, queue.log_path))
            row = {
                "events": target,
                "rung_elapsed_s": time.perf_counter() - rung_started,
                "total_elapsed_s": time.perf_counter() - run_started,
                "checkpoint_and_replay_s": time.perf_counter() - checkpoint_started,
                "rss_bytes": _rss_bytes(),
                "evidence_append_events_per_s": (
                    target / cumulative["evidence_append_s"]),
                "queue_enqueue_events_per_s": (
                    target / cumulative["queue_enqueue_s"]),
                "queue_ack_events_per_s": target / cumulative["queue_ack_s"],
                "verification_events_per_s": (
                    target / cumulative["verification_event_append_s"]),
                "verification_decisions_per_s": (
                    target / cumulative["verification_decision_s"]),
                "retraction_events": len(withdrawn),
                "claim_log_bytes": claims.log_path.stat().st_size,
                "verification_log_bytes": claims.log_path.stat().st_size,
                "evidence_log_bytes": evidence.log_path.stat().st_size,
                "queue_log_bytes": queue.log_path.stat().st_size,
                "raw_snapshot_bytes": raw_dir_bytes,
                "raw_snapshot_files": sum(1 for path in evidence.raw_dir.iterdir()
                                            if path.is_file()),
                "claim_accelerator_snapshot_bytes": claims.snapshot_path.stat().st_size,
                "evidence_accelerator_snapshot_bytes": evidence.snapshot_path.stat().st_size,
                "lineage_count": len(evidence._lineage_members),
                "lineage_index_estimated_bytes": _index_bytes(evidence._lineage_members),
                "evidence_job_index_estimated_bytes": _index_bytes(evidence._by_job),
                "verification_history_events_resident": sum(
                    len(events) for events in claims._verifications.values()),
                "verification_history_estimated_bytes": (
                    _index_bytes(claims._verifications)
                    + _index_bytes({"ids": claims._verification_ids})
                    + _index_bytes({"jobs": claims._verification_job_ids})),
                "queue_pending_jobs_resident": len(queue._jobs),
                "queue_acknowledged_ids_resident": len(queue._acknowledged),
                "queue_state_estimated_bytes": (
                    _index_bytes(queue._jobs)
                    + _index_bytes({"acked": queue._acknowledged})
                    + sys.getsizeof(queue._pending_heap)),
                "derived_bytes_written_cumulative": sum(derived_written.values()),
                "write_amplification": (
                    sum(derived_written.values()) / canonical_bytes),
                "incremental_equals_genesis": equality,
                "incremental_hashes": incremental,
                "genesis_replay": replay,
            }
            results.append(row)
            for key in (
                "events", "rss_bytes", "evidence_append_events_per_s",
                "queue_enqueue_events_per_s", "verification_events_per_s",
                "lineage_count", "write_amplification"):
                _metric(experiment, key, row[key])
            print(
                f"{target:>9} events  RSS={row['rss_bytes']/1e9:6.2f} GB  "
                f"evidence={row['evidence_append_events_per_s']:8.1f}/s  "
                f"verify={row['verification_events_per_s']:8.1f}/s  "
                f"replay={replay['total_replay_s']:7.2f}s  equality={equality}",
                flush=True)
            if not equality:
                raise RuntimeError(f"incremental/genesis mismatch at {target}")
            if row["rss_bytes"] > args.rss_limit_gb * 1e9:
                raise MemoryError(
                    f"RSS {row['rss_bytes']/1e9:.2f} GB exceeded "
                    f"limit {args.rss_limit_gb:.2f} GB")

        receipt = {
            "artifact": "BACKGROUND_VERIFICATION_V2A_EXTERNAL_EVIDENCE_PRESSURE",
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "run_valid": len(results) > 0 and results[-1]["events"] == args.max_events
                         and all(row["incremental_equals_genesis"] for row in results),
            "mechanism_status": "MEASURED_NOT_YET_QUALIFIED",
            "scientific_verdict": "MEASUREMENT",
            "source_commit": source_commit,
            "source_tree_hash": source_tree_hash,
            "branch": _git("branch", "--show-current"),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "ladder": list(LADDER),
            "max_events": args.max_events,
            "workload": {
                "seed": RUN_SEED,
                "raw_template_count": RAW_TEMPLATE_COUNT,
                "mirrors_per_lineage": MIRRORS_PER_LINEAGE,
                "retraction_interval": RETRACTION_INTERVAL,
                "claim_count": 1,
                "all_verifications_expected": "SUPPORTED",
            },
            "write_amplification_definition": (
                "cumulative bytes atomically written to claim/evidence manifests "
                "and accelerator snapshots divided by current canonical claim, "
                "evidence, and queue log bytes"),
            "results": results,
        }
        out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        digest_path = out.with_suffix(out.suffix + ".sha256")
        digest_path.write_text(f"{_sha256_file(out)}  {out.name}\n")
        status = "PASS" if receipt["run_valid"] else "FAILED"
        _metric(experiment, "run_valid", receipt["run_valid"])
        print(f"receipt={out}\nsha256={digest_path}", flush=True)
        return 0 if receipt["run_valid"] else 2
    except Exception as exc:
        _metric(experiment, "failure_type", type(exc).__name__)
        _metric(experiment, "failure_message", str(exc)[:500])
        raise
    finally:
        _finalize(experiment, status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=1_000_000,
                        choices=LADDER)
    parser.add_argument("--rss-limit-gb", type=float, default=12.0)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out",
                        default=str(ROOT / "evidence/background_verification_v2a/pressure/pressure.json"))
    parser.add_argument("--skip-litlogger", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    if args.replay_only:
        print(json.dumps(_replay_receipt(Path(args.root).resolve()), sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
