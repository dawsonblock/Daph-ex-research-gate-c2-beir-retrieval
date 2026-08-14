#!/usr/bin/env python3
"""Tiny recorded V2A network integration smoke with offline reproduction."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from hrm_adaptive_memory.external_verification.qualification import qualification_identity
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.external_verification import (  # noqa: E402
    AcquisitionRequest, AcquisitionResult, AcquisitionStatus,
    DeterministicExactFieldVerifier, EvidenceStore, ExternalVerificationWorker,
    SourceType, VerificationQueue, derive_current_status)
from hrm_adaptive_memory.memory_write import ClaimStore, VerificationStatus  # noqa: E402


WORLD_BANK = "https://api.worldbank.org/v2/country/CA?format=json"
WORLD_BANK_JAPAN = "https://api.worldbank.org/v2/country/JP?format=json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordedAuthoritativeAcquirer:
    """Two-route adapter with frozen deterministic extraction rules."""

    ACQUISITION_METHOD = "v2a_network_smoke_authoritative"
    ACQUISITION_VERSION = "1.0.0"

    def __init__(self):
        self.fetch_count = 0
        self.routes: dict[str, tuple[Callable[[bytes], dict[str, Any]], str, str, str]] = {
            WORLD_BANK: (
                self._world_bank_fields, "World Bank", "api.worldbank.org",
                "lin-world-bank-country-canada"),
            WORLD_BANK_JAPAN: (
                self._world_bank_japan_fields, "World Bank", "api.worldbank.org",
                "lin-world-bank-country-japan"),
        }

    @staticmethod
    def _world_bank_fields(raw: bytes) -> dict[str, Any]:
        payload = json.loads(raw)
        record = payload[1][0]
        if record.get("iso2Code") != "CA" or record.get("name") != "Canada":
            raise ValueError("World Bank response is not the frozen Canada record")
        return {"entity": "Canada country", "capital city": record["capitalCity"]}

    @staticmethod
    def _world_bank_japan_fields(raw: bytes) -> dict[str, Any]:
        payload = json.loads(raw)
        record = payload[1][0]
        if record.get("iso2Code") != "JP" or record.get("name") != "Japan":
            raise ValueError("World Bank response is not the frozen Japan record")
        return {"entity": "Japan country", "capital city": record["capitalCity"]}

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        route = self.routes.get(request.source_uri)
        if route is None:
            return AcquisitionResult(
                AcquisitionStatus.NOT_FOUND, request, detail="route not frozen")
        extractor, publisher, domain, lineage = route
        self.fetch_count += 1
        try:
            http_request = urllib.request.Request(
                request.source_uri,
                headers={"User-Agent": "DAPH-V2A-network-smoke/1.0"})
            with urllib.request.urlopen(http_request, timeout=20) as response:  # nosec B310
                raw = response.read()
                fields = extractor(raw)
                return AcquisitionResult(
                    AcquisitionStatus.SUCCESS, request, raw_content=raw,
                    content_type=response.headers.get_content_type(),
                    character_encoding=response.headers.get_content_charset() or "utf-8",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    extracted_fields=fields, publisher=publisher,
                    publisher_domain=domain,
                    upstream_source_id=domain, source_lineage_id=lineage,
                    response_metadata={
                        "http_status": response.status,
                        "final_url": response.geturl(),
                    })
        except urllib.error.HTTPError as exc:
            return AcquisitionResult(
                AcquisitionStatus.RATE_LIMITED if exc.code == 429
                else AcquisitionStatus.NOT_FOUND,
                request, detail=f"HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError) as exc:
            return AcquisitionResult(
                AcquisitionStatus.NETWORK_ERROR, request, detail=str(exc))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return AcquisitionResult(
                AcquisitionStatus.PARSE_ERROR, request, detail=str(exc))


class NetworkForbiddenAcquirer:
    def __init__(self):
        self.calls = 0

    def acquire(self, request):
        self.calls += 1
        raise AssertionError(f"offline replay attempted network access: {request.source_uri}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "evidence/background_verification_v2a/network_smoke/network_smoke.json"))
    parser.add_argument("--skip-litlogger", action="store_true")
    args = parser.parse_args()
    if _git("status", "--porcelain"):
        raise RuntimeError("network smoke refuses a dirty source tree")
    source_commit = _git("rev-parse", "HEAD")
    source_tree_hash = _git("rev-parse", "HEAD^{tree}")
    identity = qualification_identity(ROOT)
    experiment = None
    if not args.skip_litlogger:
        import litlogger
        experiment = litlogger.init(
            name=f"v2a-network-smoke-{source_commit[:7]}",
            teamspace="deep-gpu-acceleration-project",
            metadata={
                "protocol": "BACKGROUND_VERIFICATION_V2A",
                "source_commit": source_commit,
                "source_tree_hash": source_tree_hash,
                "network_scope": "World Bank Canada + Japan records only",
                "qualification_role": "integration smoke, not scientific evaluation",
                "location": "US", "altitude": "1334",
            })

    with tempfile.TemporaryDirectory(prefix="daph-v2a-network-") as directory:
        root = Path(directory)
        claims = ClaimStore(root / "claims")
        evidence = EvidenceStore(root / "evidence")
        queue = VerificationQueue(root / "queue")
        live = RecordedAuthoritativeAcquirer()
        fixtures = [
            ("Canada country", "capital city", "Ottawa", "claim-world-bank", WORLD_BANK,
             SourceType.AUTHORITATIVE_STRUCTURED_DATA),
            ("Japan country", "capital city", "Tokyo", "claim-world-bank-japan",
             WORLD_BANK_JAPAN, SourceType.AUTHORITATIVE_STRUCTURED_DATA),
        ]
        records = []
        for index, (entity, relation, value, source_id, url, source_type) in enumerate(fixtures):
            claim = claims.ingest(
                subject=entity, relation=relation, value=value, source_id=source_id,
                observed_at_utc=f"2026-08-11T00:00:0{index}+00:00").record
            queue.enqueue(
                claim_id=claim.record_id, priority=1, reason="network-smoke",
                created_at=f"2026-08-11T00:00:0{index}+00:00",
                verification_policy_id="v2a-network-smoke",
                verification_policy_version="1.0.0",
                request=AcquisitionRequest(url, source_type=source_type))
            records.append(claim)
        decisions = ExternalVerificationWorker(
            claims, evidence, queue, live).run_all()
        claims.publish_snapshot()
        evidence.publish_snapshot()
        live_pass = (
            live.fetch_count == 2
            and len(decisions) == 2
            and all(decision.result.value == "SUPPORTED" for decision in decisions)
            and all(derive_current_status(claims, evidence, claim.record_id)
                    is VerificationStatus.SUPPORTED for claim in records)
        )
        live_claim_hash = claims.verification_state_hash()
        live_evidence_hash = evidence.state_hash()
        live_event_ids = sorted(
            event.verification_event_id for claim in records
            for event in claims.verification_events(claim.record_id))
        live_events = [
            event for claim in records
            for event in claims.verification_events(claim.record_id)
        ]

        # Reconstruct from durable logs, with an acquirer that fails on any
        # attempted network access. Re-run every deterministic decision from
        # immutable captured bytes and compare it with the append-only event.
        replayed_claims = ClaimStore(root / "claims")
        replayed_evidence = EvidenceStore(root / "evidence")
        replayed_queue = VerificationQueue(root / "queue")
        forbidden = NetworkForbiddenAcquirer()
        replay_worker = ExternalVerificationWorker(
            replayed_claims, replayed_evidence, replayed_queue, forbidden)
        replay_decisions = replay_worker.run_all()
        verifier = DeterministicExactFieldVerifier()
        reproduced = []
        for claim in records:
            replayed_claim = replayed_claims.get(claim.record_id)
            event = replayed_claims.verification_events(claim.record_id)[0]
            captured = (
                replayed_evidence.get(event.evidence_ids[0])
                if event.evidence_ids else None)
            decision = verifier.verify(replayed_claim, captured) if captured else None
            reproduced.append({
                "claim_id": claim.record_id,
                "event_id": event.verification_event_id,
                "event_result": event.result.value,
                "reason_code": event.reason_code,
                "captured_evidence": captured is not None,
                "result_equal": decision is not None and decision.result is event.result,
                "receipt_hash_equal": (
                    decision is not None and decision.receipt_hash == event.receipt_hash),
                "raw_hash_valid": (
                    captured is not None
                    and replayed_evidence.validate_hashes(captured.evidence_id)),
            })
        offline_pass = (
            forbidden.calls == 0 and replay_decisions == []
            and replayed_claims.verification_state_hash() == live_claim_hash
            and replayed_evidence.state_hash() == live_evidence_hash
            and sorted(event.verification_event_id for claim in records
                       for event in replayed_claims.verification_events(claim.record_id))
                == live_event_ids
            and all(item["result_equal"] and item["receipt_hash_equal"]
                    and item["raw_hash_valid"] for item in reproduced)
        )
        evidence_receipts = [{
            "evidence_id": record.evidence_id,
            "source_uri": record.source_uri,
            "publisher": record.publisher,
            "publisher_domain": record.publisher_domain,
            "source_type": record.source_type.value,
            "source_lineage_id": record.source_lineage_id,
            "raw_content_hash": record.raw_content_hash,
            "normalized_content_hash": record.normalized_content_hash,
            "content_type": record.content_type,
            "fetched_at": record.fetched_at,
            "hashes_validate": replayed_evidence.validate_hashes(record.evidence_id),
        } for record in replayed_evidence.stream()]

    run_valid = live_pass and offline_pass
    receipt = {
        "artifact": "BACKGROUND_VERIFICATION_V2A_NETWORK_INTEGRATION_SMOKE",
        "run_valid": run_valid,
        "mechanism_status": "NETWORK_SMOKE_PASS" if run_valid else "NETWORK_SMOKE_FAIL",
        "scientific_verdict": "INTEGRATION_ONLY_NOT_SCIENTIFIC_EVALUATION",
        "source_commit": source_commit,
        "source_tree_hash": source_tree_hash,
        "qualification_identity": identity,
        "branch": _git("branch", "--show-current"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "network_fetch_count": live.fetch_count,
        "network_sources": [WORLD_BANK, WORLD_BANK_JAPAN],
        "source_record_count": 2,
        "publisher_count": 1,
        "independent_publisher_corroboration_claimed": False,
        "live_pipeline_pass": live_pass,
        "offline_replay_pass": offline_pass,
        "offline_network_calls": forbidden.calls,
        "claim_verification_state_hash": live_claim_hash,
        "evidence_state_hash": live_evidence_hash,
        "verification_event_ids": live_event_ids,
        "live_events": [{
            "event_id": event.verification_event_id,
            "result": event.result.value,
            "reason_code": event.reason_code,
            "evidence_ids": list(event.evidence_ids),
        } for event in live_events],
        "offline_reproductions": reproduced,
        "evidence": evidence_receipts,
        "bounded_claim": (
            "Two fixed authoritative sources traversed live fetch -> immutable snapshot "
            "-> hashes -> declared lineage -> deterministic exact-field verification -> "
            "append event -> genesis replay; captured bytes reproduced the decisions offline."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{_sha256(out)}  {out.name}\n")
    if experiment is not None:
        experiment["source_pass"].append(live_pass)
        experiment["offline_replay_pass"].append(offline_pass)
        experiment["network_fetch_count"].append(live.fetch_count)
        experiment["run_valid"] = str(run_valid).lower()
        experiment.finalize()
    print(json.dumps({
        "run_valid": run_valid, "live_pipeline_pass": live_pass,
        "offline_replay_pass": offline_pass, "receipt": str(out),
    }, sort_keys=True))
    return 0 if run_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
