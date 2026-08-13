"""Deterministic, append-only external verification for V2A.

The module is intentionally compact but deliberately separates acquisition,
immutable snapshots, deterministic comparison, and event emission.  Tests use
``LocalStructuredFixtureAcquirer`` exclusively; HTTP belongs to a separately
invoked integration layer and is not required for correctness.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from hrm_adaptive_memory.memory_write.claim_store import ClaimRecord, ClaimStore
from hrm_adaptive_memory.memory_write.states import VerificationStatus
from hrm_adaptive_memory.memory_write.verification import (
    VerificationResult, derive_status, retired_ids)
from .network import NetworkPolicyError, NetworkTransportError, PeerBoundHTTPSClient


PROTOCOL_ID = "BACKGROUND_VERIFICATION_V2A"
PROTOCOL_VERSION = "1.0.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalise_content(raw: bytes, encoding: str) -> str:
    text = raw.decode(encoding)
    try:
        return _canonical_json(json.loads(text))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


class SourceType(str, Enum):
    UNTRUSTED_CAPTURE_ONLY = "UNTRUSTED_CAPTURE_ONLY"
    AUTHORITATIVE_STRUCTURED_DATA = "AUTHORITATIVE_STRUCTURED_DATA"
    OFFICIAL_PRIMARY_SOURCE = "OFFICIAL_PRIMARY_SOURCE"
    PRIMARY_SCIENTIFIC_LITERATURE = "PRIMARY_SCIENTIFIC_LITERATURE"
    PREPRINT = "PREPRINT"
    SECONDARY_TECHNICAL_SOURCE = "SECONDARY_TECHNICAL_SOURCE"
    GENERAL_WEB = "GENERAL_WEB"


class AcquisitionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NOT_FOUND = "NOT_FOUND"
    NETWORK_ERROR = "NETWORK_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    UNSUPPORTED_CONTENT = "UNSUPPORTED_CONTENT"
    PARSE_ERROR = "PARSE_ERROR"
    RATE_LIMITED = "RATE_LIMITED"


@dataclass(frozen=True)
class AcquisitionRequest:
    source_uri: str
    canonical_source_uri: str | None = None
    query_used: str = ""
    request_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_type: SourceType = SourceType.AUTHORITATIVE_STRUCTURED_DATA


@dataclass(frozen=True)
class AcquisitionResult:
    status: AcquisitionStatus
    request: AcquisitionRequest
    raw_content: bytes = b""
    content_type: str = "application/json"
    character_encoding: str = "utf-8"
    fetched_at: str = ""
    extracted_fields: Mapping[str, Any] = field(default_factory=dict)
    publisher: str = ""
    publisher_domain: str = ""
    published_at: str | None = None
    observed_at: str | None = None
    upstream_source_id: str | None = None
    source_lineage_id: str | None = None
    response_metadata: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True)
class SourceLineage:
    lineage_id: str
    root_source: str
    member_evidence_ids: tuple[str, ...]
    lineage_detection_method: str
    lineage_detection_version: str
    status: str = "DETERMINISTIC"


@dataclass(frozen=True)
class ExternalEvidenceRecord:
    evidence_id: str
    claim_record_id: str | None
    source_uri: str
    canonical_source_uri: str
    source_type: SourceType
    publisher: str
    publisher_domain: str
    fetched_at: str
    published_at: str | None
    observed_at: str | None
    raw_content_hash: str
    normalized_content_hash: str
    raw_snapshot_location: str
    content_type: str
    character_encoding: str
    normalized_content: str
    extracted_fields: Mapping[str, Any]
    acquisition_method: str
    acquisition_version: str
    query_used: str
    request_metadata_hash: str
    response_metadata_hash: str
    source_lineage_id: str
    upstream_source_id: str | None
    lifecycle_state: str
    provenance: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = self.source_type.value
        return data


class EvidenceStore:
    """Durable immutable evidence records plus content-addressed snapshots.

    History stays in ``evidence_events.jsonl`` and is streamed during replay;
    only the active record map and retraction ids are resident in memory.
    """

    def __init__(self, root: str | Path, auto_snapshot: bool = True,
                 snapshot_interval: int = 128):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "evidence_events.jsonl"
        self.manifest_path = self.root / "EVIDENCE_MANIFEST.json"
        self.snapshot_path = self.root / "evidence_snapshot.json"
        self.raw_dir = self.root / "raw_snapshots"
        self.raw_dir.mkdir(exist_ok=True)
        self.auto_snapshot = auto_snapshot
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")
        self.snapshot_interval = snapshot_interval
        self.truncated_tail = False
        self._records: dict[str, ExternalEvidenceRecord] = {}
        self._retracted: set[str] = set()
        self._by_job: dict[str, list[str]] = {}
        self._lineage_members: dict[str, list[str]] = {}
        self._event_count = 0
        self._log_hasher = self._committed_log_hasher()
        self._verify_manifest_before_replay()
        self._replay()
        if self.auto_snapshot and self._event_count and not self.snapshot_is_valid():
            self.publish_snapshot()

    def _stream_events(self):
        if not self.log_path.exists():
            return
        with self.log_path.open() as handle:
            while True:
                line = handle.readline()
                if line == "":
                    return
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if handle.readline() == "":
                        self.truncated_tail = True
                        return
                    raise

    def _committed_log_hasher(self):
        hasher = hashlib.sha256()
        if not self.log_path.exists():
            return hasher
        with self.log_path.open("rb") as handle:
            lines = handle.readlines()
        for index, line in enumerate(lines):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
            hasher.update(line)
        return hasher

    def _verify_manifest_before_replay(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("EVIDENCE_MANIFEST.json is not valid JSON") from exc
        if manifest.get("event_log_sha256") != self.log_sha256():
            raise ValueError(
                "external evidence log differs from its committed manifest; "
                "refusing to replay unauthenticated history")

    @staticmethod
    def _record_from_json(data: Mapping[str, Any]) -> ExternalEvidenceRecord:
        raw = dict(data)
        raw["source_type"] = SourceType(raw["source_type"])
        return ExternalEvidenceRecord(**raw)

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        if event["event"] == "EVIDENCE_APPENDED":
            record = self._record_from_json(event["record"])
            if record.evidence_id not in self._records:
                self._records[record.evidence_id] = record
                job_id = str(record.provenance.get("verification_job_id", ""))
                if job_id:
                    self._by_job.setdefault(job_id, []).append(record.evidence_id)
                self._lineage_members.setdefault(
                    record.source_lineage_id, []).append(record.evidence_id)
        elif event["event"] == "EVIDENCE_RETRACTED":
            self._retracted.add(event["evidence_id"])
        else:
            raise ValueError(f"unknown evidence event {event['event']!r}")

    def _replay(self) -> None:
        self._records, self._retracted, self._event_count = {}, set(), 0
        self._by_job, self._lineage_members = {}, {}
        for event in self._stream_events():
            self._event_count += 1
            self._apply_event(event)

    def _atomic_write(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _atomic_write_bytes(self, path: Path, value: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _append(self, event: Mapping[str, Any]) -> None:
        line = _canonical_json(event) + "\n"
        with self.log_path.open("a") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._log_hasher.update(line.encode("utf-8"))
        self._event_count += 1
        self._apply_event(event)
        self._write_manifest()
        # A full state hash scans the active evidence map. Publishing on every
        # append would therefore turn an otherwise O(1) append into O(n²)
        # across a run. Snapshots are accelerators only; bounded periodic
        # publication plus canonical-log replay has the same crash semantics.
        if self.auto_snapshot and self._event_count % self.snapshot_interval == 0:
            self.publish_snapshot()

    def log_sha256(self) -> str:
        return self._log_hasher.hexdigest()

    def _write_manifest(self, *, include_state: bool = False) -> None:
        manifest = {
            "store": "EXTERNAL_EVIDENCE_V2A.1",
            "event_count": self._event_count,
            "event_log_sha256": self.log_sha256(),
        }
        if include_state:
            manifest["state_hash"] = self.state_hash()
        self._atomic_write(self.manifest_path,
                           json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    def state_hash(self) -> str:
        hasher = hashlib.sha256()
        for evidence_id in sorted(self._records):
            record = self._records[evidence_id]
            hasher.update(_canonical_json({
                "evidence_id": evidence_id,
                "normalized_content_hash": record.normalized_content_hash,
                "source_lineage_id": record.source_lineage_id,
                "active": evidence_id not in self._retracted,
            }).encode())
            hasher.update(b"\n")
        return hasher.hexdigest()

    def publish_snapshot(self) -> Path:
        self._atomic_write(self.snapshot_path, json.dumps({
            "event_count": self._event_count, "log_sha256": self.log_sha256(),
            "state_hash": self.state_hash(),
        }, sort_keys=True, indent=2) + "\n")
        self._write_manifest(include_state=True)
        return self.snapshot_path

    def snapshot_is_valid(self) -> bool:
        try:
            snapshot = json.loads(self.snapshot_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        return (snapshot.get("event_count") == self._event_count
                and snapshot.get("log_sha256") == self.log_sha256()
                and snapshot.get("state_hash") == self.state_hash())

    def append_acquisition(self, result: AcquisitionResult, *, claim_record_id: str | None,
                           acquisition_method: str, acquisition_version: str,
                           provenance: Mapping[str, Any] | None = None) -> ExternalEvidenceRecord:
        if result.status is not AcquisitionStatus.SUCCESS:
            raise ValueError("only successful acquisition can produce immutable evidence")
        fetched_at = result.fetched_at or _utc_now()
        normalized = _normalise_content(result.raw_content, result.character_encoding)
        raw_hash, normalized_hash = _sha256(result.raw_content), _sha256(normalized)
        canonical_uri = result.request.canonical_source_uri or result.request.source_uri
        lineage_id = result.source_lineage_id or "lin-" + _sha256(_canonical_json({
            "canonical_source_uri": canonical_uri,
            "upstream_source_id": result.upstream_source_id or "",
            "raw_content_hash": raw_hash,
        }))[:20]
        identity = _canonical_json({
            "normalized_content": normalized, "canonical_source_uri": canonical_uri,
            "acquisition_version": acquisition_version, "fetched_at": fetched_at,
        })
        evidence_id = "evd-" + _sha256(identity)[:20]
        snapshot_path = self.raw_dir / f"{raw_hash}.bin"
        if not snapshot_path.exists():
            self._atomic_write_bytes(snapshot_path, result.raw_content)
        record = ExternalEvidenceRecord(
            evidence_id=evidence_id, claim_record_id=claim_record_id,
            source_uri=result.request.source_uri, canonical_source_uri=canonical_uri,
            source_type=result.request.source_type, publisher=result.publisher,
            publisher_domain=result.publisher_domain, fetched_at=fetched_at,
            published_at=result.published_at, observed_at=result.observed_at,
            raw_content_hash=raw_hash, normalized_content_hash=normalized_hash,
            raw_snapshot_location=str(snapshot_path.relative_to(self.root)),
            content_type=result.content_type, character_encoding=result.character_encoding,
            normalized_content=normalized, extracted_fields=dict(result.extracted_fields),
            acquisition_method=acquisition_method, acquisition_version=acquisition_version,
            query_used=result.request.query_used,
            request_metadata_hash=_sha256(_canonical_json(dict(result.request.request_metadata))),
            response_metadata_hash=_sha256(_canonical_json(dict(result.response_metadata))),
            source_lineage_id=lineage_id, upstream_source_id=result.upstream_source_id,
            lifecycle_state="ACTIVE", provenance=dict(provenance or {}))
        if evidence_id not in self._records:
            self._append({"event": "EVIDENCE_APPENDED", "at": fetched_at, "record": record.to_json()})
        return self._records[evidence_id]

    def retract(self, evidence_id: str, *, reason: str, observed_at: str,
                provenance: Mapping[str, Any]) -> None:
        if evidence_id not in self._records:
            raise KeyError(f"unknown evidence_id {evidence_id!r}")
        if evidence_id not in self._retracted:
            self._append({"event": "EVIDENCE_RETRACTED", "evidence_id": evidence_id,
                          "reason": reason, "observed_at": observed_at,
                          "provenance": dict(provenance)})

    def get(self, evidence_id: str) -> ExternalEvidenceRecord | None:
        return self._records.get(evidence_id)

    def is_active(self, evidence_id: str) -> bool:
        return evidence_id in self._records and evidence_id not in self._retracted

    def stream(self) -> Iterable[ExternalEvidenceRecord]:
        for evidence_id in sorted(self._records):
            yield self._records[evidence_id]

    def by_job(self, job_id: str) -> list[ExternalEvidenceRecord]:
        return [self._records[evidence_id]
                for evidence_id in self._by_job.get(job_id, ())]

    def validate_hashes(self, evidence_id: str) -> bool:
        record = self._records[evidence_id]
        raw = (self.root / record.raw_snapshot_location).read_bytes()
        return (_sha256(raw) == record.raw_content_hash
                and _sha256(_normalise_content(raw, record.character_encoding))
                == record.normalized_content_hash)

    def source_lineage(self, lineage_id: str) -> SourceLineage | None:
        members = tuple(self._lineage_members.get(lineage_id, ()))
        if not members:
            return None
        first = self._records[members[0]]
        return SourceLineage(lineage_id, first.canonical_source_uri, members,
                             "declared_or_content_hash", "1.0.0")


def derive_current_status(claims: ClaimStore, evidence: EvidenceStore,
                          claim_id: str) -> VerificationStatus:
    """Derive a V2A-aware current view without altering V1 derivation.

    A retracted external snapshot remains in history but cannot contribute to
    the current external-evidence view. V1 events and acquisition-failure
    events have no external evidence ids and retain their existing behavior.
    """
    events = claims.verification_events(claim_id)
    live = [event for event in events if not (
        event.protocol_id == PROTOCOL_ID and event.evidence_ids
        and not any(evidence.is_active(evidence_id) for evidence_id in event.evidence_ids)
    )]
    return derive_status(live, retired_ids(live))


def explain_claim(claims: ClaimStore, evidence: EvidenceStore,
                  claim_id: str) -> dict[str, Any]:
    """Return the auditable inputs behind the current V2A status."""
    events = claims.verification_events(claim_id)
    return {
        "claim_id": claim_id,
        "current_status": derive_current_status(claims, evidence, claim_id).value,
        "verification_events": [{
            "verification_event_id": event.verification_event_id,
            "evidence_ids": list(event.evidence_ids),
            "source_lineage_ids": list(event.source_lineage_ids),
            "method": event.method, "method_version": event.method_version,
            "verified_at": event.observed_at_utc, "reason_code": event.reason_code,
            "receipt_hash": event.receipt_hash,
            "evidence": [{
                "evidence_id": record.evidence_id, "source_uri": record.source_uri,
                "raw_content_hash": record.raw_content_hash,
                "normalized_content_hash": record.normalized_content_hash,
                "active": evidence.is_active(record.evidence_id),
            } for evidence_id in event.evidence_ids
              if (record := evidence.get(evidence_id)) is not None],
        } for event in events],
    }


class EvidenceAcquirer(Protocol):
    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult: ...


class LocalStructuredFixtureAcquirer:
    """Offline-only adapter with immutable fixture payloads.

    Each fixture is a mapping with status, raw JSON/text, fixed fetched_at,
    structured fields, and optional deterministic lineage metadata.
    """

    ACQUISITION_METHOD = "local_structured_fixture"
    ACQUISITION_VERSION = "1.0.0"

    def __init__(self, fixtures: Mapping[str, Mapping[str, Any]]):
        self.fixtures = fixtures

    @classmethod
    def from_directory(cls, directory: str | Path) -> "LocalStructuredFixtureAcquirer":
        fixtures = {}
        for path in sorted(Path(directory).glob("*.json")):
            payload = json.loads(path.read_text())
            fixtures[payload["source_uri"]] = payload
        return cls(fixtures)

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        fixture = self.fixtures.get(request.source_uri)
        if fixture is None:
            return AcquisitionResult(AcquisitionStatus.NOT_FOUND, request, detail="fixture absent")
        status = AcquisitionStatus(fixture.get("status", "SUCCESS"))
        raw = fixture.get("raw_content", "")
        if isinstance(raw, (dict, list)):
            raw = _canonical_json(raw)
        return AcquisitionResult(
            status=status, request=request, raw_content=str(raw).encode("utf-8"),
            content_type=fixture.get("content_type", "application/json"),
            fetched_at=fixture.get("fetched_at", ""),
            extracted_fields=fixture.get("extracted_fields", {}),
            publisher=fixture.get("publisher", ""), publisher_domain=fixture.get("publisher_domain", ""),
            published_at=fixture.get("published_at"), observed_at=fixture.get("observed_at"),
            upstream_source_id=fixture.get("upstream_source_id"),
            source_lineage_id=fixture.get("source_lineage_id"),
            response_metadata=fixture.get("response_metadata", {}), detail=fixture.get("detail", ""))


class HTTPStructuredDataAcquirer:
    """Untrusted public-HTTPS capture for non-qualification integration runs.

    This generic adapter cannot create truth-bearing authoritative evidence.
    Production authority requires a separately frozen endpoint/extractor
    registry; callers cannot acquire authority by labeling a request.
    """

    ACQUISITION_METHOD = "http_structured_data"
    ACQUISITION_VERSION = "1.2.0-peer-bound-untrusted-capture"
    MAX_RESPONSE_BYTES = 8 * 1024 * 1024

    def __init__(self, client: PeerBoundHTTPSClient | None = None):
        self.client = client or PeerBoundHTTPSClient()

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        if request.source_type is not SourceType.UNTRUSTED_CAPTURE_ONLY:
            return AcquisitionResult(
                AcquisitionStatus.INVALID_RESPONSE, request,
                detail="generic HTTP cannot establish authoritative source identity")
        try:
            response = self.client.fetch(request.source_uri)
            if response.status >= 400:
                status = AcquisitionStatus.RATE_LIMITED if response.status == 429 else AcquisitionStatus.NOT_FOUND
                return AcquisitionResult(status, request, detail=f"HTTP {response.status}")
            raw, content_type = response.body, response.content_type
            if content_type not in {"application/json", "text/csv"}:
                return AcquisitionResult(AcquisitionStatus.UNSUPPORTED_CONTENT, request,
                                         raw_content=raw, content_type=content_type)
            try:
                extracted = json.loads(raw) if content_type == "application/json" else {}
            except json.JSONDecodeError:
                return AcquisitionResult(AcquisitionStatus.PARSE_ERROR, request,
                                         raw_content=raw, content_type=content_type)
            return AcquisitionResult(AcquisitionStatus.SUCCESS, request, raw_content=raw,
                                     content_type=content_type, fetched_at=_utc_now(),
                                     extracted_fields=extracted,
                                     response_metadata={"http_status": response.status,
                                                        "final_uri": response.final_uri,
                                                        "peer_ip": response.peer_ip})
        except (NetworkPolicyError, NetworkTransportError):
            return AcquisitionResult(AcquisitionStatus.NETWORK_ERROR, request)


class OfficialTextAcquirer:
    """Capture official HTML/text for explicitly invoked integration runs.

    V2A does not infer fields from prose. This adapter snapshots bytes for
    audit and therefore can only yield an INCONCLUSIVE decision until a
    separately frozen deterministic extractor is added.
    """

    ACQUISITION_METHOD = "official_html_text"
    ACQUISITION_VERSION = "1.2.0-peer-bound-untrusted-capture"
    MAX_RESPONSE_BYTES = 8 * 1024 * 1024

    def __init__(self, client: PeerBoundHTTPSClient | None = None):
        self.client = client or PeerBoundHTTPSClient()

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        if request.source_type is not SourceType.UNTRUSTED_CAPTURE_ONLY:
            return AcquisitionResult(
                AcquisitionStatus.INVALID_RESPONSE, request,
                detail="generic HTTP cannot establish official source identity")
        try:
            response = self.client.fetch(request.source_uri)
            if response.status >= 400:
                status = AcquisitionStatus.RATE_LIMITED if response.status == 429 else AcquisitionStatus.NOT_FOUND
                return AcquisitionResult(status, request, detail=f"HTTP {response.status}")
            raw, content_type = response.body, response.content_type
            if content_type not in {"text/html", "text/plain"}:
                return AcquisitionResult(AcquisitionStatus.UNSUPPORTED_CONTENT, request,
                                         raw_content=raw, content_type=content_type)
            return AcquisitionResult(AcquisitionStatus.SUCCESS, request, raw_content=raw,
                                     content_type=content_type, fetched_at=_utc_now(),
                                     response_metadata={"http_status": response.status,
                                                        "final_uri": response.final_uri,
                                                        "peer_ip": response.peer_ip})
        except (NetworkPolicyError, NetworkTransportError):
            return AcquisitionResult(AcquisitionStatus.NETWORK_ERROR, request)


@dataclass(frozen=True)
class VerificationDecision:
    result: VerificationResult
    method: str
    method_version: str
    claim_id: str
    evidence_ids: tuple[str, ...]
    comparison_fields: Mapping[str, Any]
    reason_code: str
    receipt_hash: str


class DeterministicExactFieldVerifier:
    """The sole V2A truth-bearing comparison: exact authoritative fields."""

    CHECKER_TYPE = "DETERMINISTIC_EXTERNAL_EXACT_FIELD"
    METHOD = "authoritative_exact_field"
    METHOD_VERSION = "1.0.0"

    def verify(self, claim: ClaimRecord, evidence: ExternalEvidenceRecord) -> VerificationDecision:
        fields = dict(evidence.extracted_fields)
        field = claim.canonical_relation
        base = {"entity": claim.canonical_entity, "field": field,
                "claim_value": claim.value, "evidence_id": evidence.evidence_id}
        if evidence.source_type not in {
            SourceType.AUTHORITATIVE_STRUCTURED_DATA, SourceType.OFFICIAL_PRIMARY_SOURCE
        }:
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  "SOURCE_CLASS_NOT_QUALIFIED")
        if _norm(fields.get("entity", "")) != _norm(claim.canonical_entity):
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  "ENTITY_MISMATCH_OR_AMBIGUOUS")
        if field not in fields:
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  "MISSING_COMPARISON_FIELD")
        base["evidence_value"] = fields[field]
        if _norm(fields[field]) == _norm(claim.value):
            return self._decision(VerificationResult.SUPPORTED, claim, evidence, base,
                                  "AUTHORITATIVE_EXACT_FIELD_MATCH")
        result = (VerificationResult.FALSIFIED
                  if evidence.source_type is SourceType.AUTHORITATIVE_STRUCTURED_DATA
                  else VerificationResult.CONTRADICTED)
        return self._decision(result, claim, evidence, base,
                              "AUTHORITATIVE_EXACT_FIELD_MISMATCH")

    def _decision(self, result: VerificationResult, claim: ClaimRecord,
                  evidence: ExternalEvidenceRecord, fields: Mapping[str, Any],
                  reason_code: str) -> VerificationDecision:
        receipt = _sha256(_canonical_json({"result": result.value, "method": self.METHOD,
                                           "version": self.METHOD_VERSION,
                                           "claim": claim.record_id, "evidence": evidence.evidence_id,
                                           "fields": dict(fields), "reason_code": reason_code}))
        return VerificationDecision(result, self.METHOD, self.METHOD_VERSION,
                                    claim.record_id, (evidence.evidence_id,), dict(fields),
                                    reason_code, receipt)


@dataclass(frozen=True)
class VerificationJob:
    job_id: str
    claim_id: str
    priority: int
    reason: str
    created_at: str
    attempt_count: int
    next_attempt_at: str
    verification_policy_id: str
    verification_policy_version: str
    request: AcquisitionRequest

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["request"]["source_type"] = self.request.source_type.value
        return data


class VerificationQueue:
    """Durable, replayable, deterministic job queue (no learned scheduling)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "verification_queue.jsonl"
        self._jobs: dict[str, VerificationJob] = {}
        self._acknowledged: set[str] = set()
        self._pending_heap: list[tuple[int, str, str]] = []
        self._replay()

    def _stream(self):
        if not self.log_path.exists():
            return
        with self.log_path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def _append(self, event: Mapping[str, Any]) -> None:
        with self.log_path.open("a") as handle:
            handle.write(_canonical_json(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _replay(self) -> None:
        for event in self._stream():
            if event["event"] == "ENQUEUED":
                raw = dict(event["job"])
                raw["request"] = AcquisitionRequest(
                    **{**raw["request"], "source_type": SourceType(raw["request"]["source_type"])})
                self._jobs.setdefault(raw["job_id"], VerificationJob(**raw))
            elif event["event"] == "ACKNOWLEDGED":
                self._acknowledged.add(event["job_id"])
            else:
                raise ValueError(f"unknown verification queue event {event.get('event')!r}")
        self._pending_heap = [
            (-job.priority, job.next_attempt_at, job.job_id)
            for job_id, job in self._jobs.items()
            if job_id not in self._acknowledged
        ]
        for job_id in self._acknowledged:
            self._jobs.pop(job_id, None)
        heapq.heapify(self._pending_heap)

    def enqueue(self, *, claim_id: str, priority: int, reason: str, created_at: str,
                verification_policy_id: str, verification_policy_version: str,
                request: AcquisitionRequest) -> VerificationJob:
        identity = _canonical_json({"claim_id": claim_id, "policy": verification_policy_id,
                                    "version": verification_policy_version,
                                    "source_uri": request.source_uri, "query": request.query_used})
        job_id = "vjob-" + _sha256(identity)[:20]
        candidate = VerificationJob(
            job_id, claim_id, priority, reason, created_at, 0, created_at,
            verification_policy_id, verification_policy_version, request)
        if job_id not in self._jobs and job_id not in self._acknowledged:
            job = candidate
            self._append({"event": "ENQUEUED", "job": job.to_json()})
            self._jobs[job_id] = job
            heapq.heappush(self._pending_heap,
                           (-job.priority, job.next_attempt_at, job.job_id))
        return self._jobs.get(job_id, candidate)

    def acknowledge(self, job_id: str) -> None:
        if job_id in self._acknowledged:
            return
        if job_id not in self._jobs:
            raise KeyError(job_id)
        if job_id not in self._acknowledged:
            self._append({"event": "ACKNOWLEDGED", "job_id": job_id})
            self._acknowledged.add(job_id)
            self._jobs.pop(job_id, None)

    def pending(self) -> list[VerificationJob]:
        return sorted((job for job_id, job in self._jobs.items() if job_id not in self._acknowledged),
                      key=lambda job: (-job.priority, job.next_attempt_at, job.job_id))

    def next_pending(self) -> VerificationJob | None:
        while self._pending_heap:
            _priority, _next_attempt, job_id = self._pending_heap[0]
            if job_id in self._acknowledged or job_id not in self._jobs:
                heapq.heappop(self._pending_heap)
                continue
            return self._jobs[job_id]
        return None


class ExternalVerificationWorker:
    """Restart-safe V2A orchestration: acquire -> snapshot -> verify -> append."""

    CHECKER_ID = "external-v2a-exact-field-1"

    def __init__(self, claims: ClaimStore, evidence: EvidenceStore, queue: VerificationQueue,
                 acquirer: EvidenceAcquirer, verifier: DeterministicExactFieldVerifier | None = None):
        self.claims, self.evidence, self.queue = claims, evidence, queue
        self.acquirer, self.verifier = acquirer, verifier or DeterministicExactFieldVerifier()

    def _committed(self, job_id: str) -> bool:
        return self.claims.verification_job_committed(job_id)

    def run_next(self) -> VerificationDecision | None:
        job = self.queue.next_pending()
        if job is None:
            return None
        if self._committed(job.job_id):
            self.queue.acknowledge(job.job_id)
            return None
        claim = self.claims.get(job.claim_id)
        if claim is None:
            raise KeyError(f"queued claim is missing: {job.claim_id}")
        observed = job.created_at
        captured = self.evidence.by_job(job.job_id)
        if captured:
            evidence = captured[0]
            decision = self.verifier.verify(claim, evidence)
        else:
            result = self.acquirer.acquire(job.request)
            if result.status is not AcquisitionStatus.SUCCESS:
                receipt = _sha256(_canonical_json({"job": job.job_id, "status": result.status.value,
                                                   "detail": result.detail}))
                decision = VerificationDecision(VerificationResult.INCONCLUSIVE,
                    "acquisition_failure", "1.0.0", claim.record_id, (), {},
                    f"ACQUISITION_{result.status.value}", receipt)
                evidence = None
            else:
                method = getattr(self.acquirer, "ACQUISITION_METHOD", "unknown_acquirer")
                version = getattr(self.acquirer, "ACQUISITION_VERSION", "1.0.0")
                evidence = self.evidence.append_acquisition(
                    result, claim_record_id=claim.record_id, acquisition_method=method,
                    acquisition_version=version, provenance={"verification_job_id": job.job_id})
                decision = self.verifier.verify(claim, evidence)
        lineage_ids = (evidence.source_lineage_id,) if evidence is not None else ()
        event = self.claims.append_external_verification(
            claim_record_id=claim.record_id, checker_id=self.CHECKER_ID,
            checker_type=self.verifier.CHECKER_TYPE, method=decision.method,
            method_version=decision.method_version, evidence_ids=decision.evidence_ids,
            evidence_resolver=self.evidence.get, result=decision.result, confidence=1.0,
            reason_code=decision.reason_code, source_lineage_ids=lineage_ids,
            receipt_hash=decision.receipt_hash, verification_job_id=job.job_id,
            notes=_canonical_json({"comparison_fields": decision.comparison_fields}),
            observed_at_utc=observed)
        self.queue.acknowledge(job.job_id)
        return decision

    def run_all(self) -> list[VerificationDecision]:
        decisions = []
        while self.queue.next_pending() is not None:
            decision = self.run_next()
            if decision is not None:
                decisions.append(decision)
        return decisions
