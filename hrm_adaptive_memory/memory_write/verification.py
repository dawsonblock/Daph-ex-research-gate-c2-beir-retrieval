"""Verification events and status derivation for BACKGROUND_VERIFICATION_V1.

Per configs/background_verification_v1_design.json.

    CORE RULE: verification APPENDS EVIDENCE ABOUT a claim.
               It never mutates the claim in place.

A claim's verification status is therefore never stored on the claim. It is
DERIVED by replaying the verification events that cite it, exactly as
consolidated state is derived from claim events. Storing it would make it
mutable state rather than a replay conclusion, which is the thing the core
rule forbids.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .states import VerificationStatus


class VerificationResult(str, Enum):
    """What a SINGLE verification event concluded. Distinct from
    VerificationStatus, which is the DERIVED conclusion over all of them --
    notably, no single event can conclude INCONCLUSIVE-by-disagreement,
    because disagreement is a property of the set."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    FALSIFIED = "FALSIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceResolutionError(ValueError):
    """V8: a verification event cited evidence that does not resolve. Fail
    closed -- storing a dangling pointer would make the result unauditable,
    which defeats the purpose of recording evidence at all."""


@dataclass(frozen=True)
class VerificationEvent:
    verification_event_id: str
    claim_record_id: str
    checker_id: str
    checker_type: str
    method: str
    method_version: str
    evidence_ids: tuple[str, ...]
    observed_at_utc: str
    result: VerificationResult
    confidence: float
    notes: str = ""
    supersedes_verification: str | None = None
    # Optional V2A provenance. Empty defaults preserve every V1 event's
    # serialized shape and content-addressed identity.
    source_lineage_ids: tuple[str, ...] = ()
    reason_code: str = ""
    protocol_id: str = ""
    protocol_version: str = ""
    execution_identity: str = ""
    receipt_hash: str = ""
    verification_job_id: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["result"] = self.result.value
        d["evidence_ids"] = list(self.evidence_ids)
        # Keep V1 events in their original serialized schema. V2A provenance
        # is emitted only when this is actually an external determination.
        for key in ("source_lineage_ids", "reason_code", "protocol_id",
                    "protocol_version", "execution_identity", "receipt_hash",
                    "verification_job_id"):
            if not d[key]:
                del d[key]
        if "source_lineage_ids" in d:
            d["source_lineage_ids"] = list(self.source_lineage_ids)
        return d


def verification_event_id(*, claim_record_id: str, checker_id: str, method: str,
                          method_version: str, evidence_ids: Sequence[str],
                          result: str, observed_at_utc: str,
                          checker_type: str = "", confidence: float | None = None,
                          notes: str = "", supersedes_verification: str | None = None,
                          source_lineage_ids: Sequence[str] = (),
                          reason_code: str = "", protocol_id: str = "",
                          protocol_version: str = "", execution_identity: str = "",
                          receipt_hash: str = "",
                          verification_job_id: str = "") -> str:
    """Content-addressed over the full determination.

    This gives idempotency for free, which is what V11 needs: a worker that
    crashes after appending but before recording completion will, on restart,
    recompute the SAME determination and produce the SAME id -- so the replay
    sees a duplicate to drop rather than a second, spurious opinion.
    """
    fields = {
        "claim_record_id": claim_record_id, "checker_id": checker_id,
        "method": method, "method_version": method_version,
        "evidence_ids": sorted(evidence_ids), "result": result,
        "checker_type": checker_type, "confidence": confidence, "notes": notes,
        "supersedes_verification": supersedes_verification,
    }
    # Retry-safe jobs are identity-addressed. Their wall-clock observation time
    # is provenance, not a new determination, so it must not defeat exactly-once
    # delivery after a crash. Ad-hoc V1 calls retain timestamp identity.
    if verification_job_id:
        fields["verification_job_id"] = verification_job_id
    else:
        fields["observed_at_utc"] = observed_at_utc
    # Omit all new fields when empty so V1 event ids remain byte-for-byte
    # identical to ids computed before V2A existed.
    extras = {
        "source_lineage_ids": sorted(source_lineage_ids),
        "reason_code": reason_code, "protocol_id": protocol_id,
        "protocol_version": protocol_version, "execution_identity": execution_identity,
        "receipt_hash": receipt_hash,
    }
    fields.update({key: value for key, value in extras.items() if value})
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return "vfy-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def derive_status(events: Iterable[VerificationEvent],
                  retired: Iterable[str] = ()) -> VerificationStatus:
    """THE FROZEN DERIVATION RULE.

        no live events        -> UNVERIFIED
        all agree on a result -> that result
        they disagree         -> INCONCLUSIVE

    Two deliberate non-behaviours, both required by the design:

    * RECENCY DOES NOT BREAK TIES. A later verification does not override an
      earlier one. This mirrors the claim-level policy that a later timestamp
      alone is not supersession, and for the same reasons -- clock skew,
      backfill, and corrections-vs-updates make recency an unsafe proxy for
      authority. Retiring a verdict requires an explicit
      supersedes_verification link.
    * CONFIDENCE DOES NOT BREAK TIES. Averaging or arg-maxing confidence
      across disagreeing verifiers is precisely the "silently averaged away"
      failure V7 forbids. Confidence is recorded for later use and is not
      consulted here.
    """
    retired = set(retired)
    live = [e for e in events if e.verification_event_id not in retired]
    if not live:
        return VerificationStatus.UNVERIFIED
    results = {e.result for e in live}
    if len(results) == 1:
        return VerificationStatus(next(iter(results)).value)
    return VerificationStatus.INCONCLUSIVE


def retired_ids(events: Iterable[VerificationEvent]) -> set[str]:
    """Verification events explicitly retired by a later one."""
    return {e.supersedes_verification for e in events if e.supersedes_verification}


def priority_signals(*, conflict_present: bool, claim_importance: float | None = None,
                     source_trust: float | None = None, novelty: float | None = None,
                     age_staleness_seconds: float | None = None,
                     dependent_count: int | None = None) -> dict:
    """The frozen queue-priority SCHEMA.

    V1 POLICY USES conflict_present ONLY. The remaining fields are recorded
    and deliberately unused: weighting them before there is evidence about
    what actually deserves verification compute would be the same post-hoc
    move this project avoids elsewhere. The schema exists so a later
    executive has somewhere to put that decision.
    """
    return {
        "conflict_present": bool(conflict_present),
        "claim_importance": claim_importance,
        "source_trust": source_trust,
        "novelty": novelty,
        "age_staleness_seconds": age_staleness_seconds,
        "dependent_count": dependent_count,
        "v1_effective_priority": 1 if conflict_present else 0,
        "v1_policy": "conflict_present only; other signals recorded, not used",
    }


def _normalise_value(value: str) -> str:
    return " ".join(value.strip().lower().split())


class DeterministicMemoryConsistencyChecker:
    """The first worker: verification against EXISTING MEMORY only.

    No web, no LLM, no external source -- fully deterministic and therefore
    checkable by replay. It inherits the classification that used to happen
    inline during ingest (which V1 forbids), so the migration and the first
    worker are the same piece of work rather than two.

    It never emits FALSIFIED. Memory consistency can show that other records
    agree or disagree, but "this claim is false" requires evidence from
    outside the store, which is BACKGROUND_VERIFICATION_V2's job. Emitting
    FALSIFIED here would overstate what a consistency check can know.
    """

    CHECKER_TYPE = "DETERMINISTIC_MEMORY_CONSISTENCY"
    METHOD = "same_claim_key_value_agreement"
    METHOD_VERSION = "1.0.0"

    def __init__(self, store, checker_id: str = "det-mem-consistency-1"):
        self.store = store
        self.checker_id = checker_id

    def determine(self, record_id: str) -> tuple[VerificationResult, tuple[str, ...], str]:
        """Return (result, evidence_ids, notes) WITHOUT appending anything --
        separated from emission so the determination can be inspected and
        tested independently of the log."""
        rec = self.store.get(record_id)
        if rec is None:
            raise KeyError(f"unknown claim record {record_id!r}")
        # Repeated observations from one source are not independent
        # corroboration. V2A upgrades this to declared lineage; V1 uses the
        # available source identifier conservatively.
        peers = [r for r in self.store.retrievable()
                 if r.claim_key == rec.claim_key and r.record_id != record_id
                 and r.source_id != rec.source_id]
        same = [r for r in peers if _normalise_value(r.value) == _normalise_value(rec.value)]
        diff = [r for r in peers if _normalise_value(r.value) != _normalise_value(rec.value)]
        if diff:
            return (VerificationResult.CONTRADICTED,
                    tuple(sorted(r.record_id for r in diff)),
                    f"{len(diff)} active record(s) assert a different value for this claim key")
        if same:
            return (VerificationResult.SUPPORTED,
                    tuple(sorted(r.record_id for r in same)),
                    f"{len(same)} active record(s) independently assert the same value")
        return (VerificationResult.INCONCLUSIVE, (),
                "no other active record addresses this claim key")

    def verify(self, record_id: str, observed_at_utc: str | None = None) -> VerificationEvent:
        result, evidence, notes = self.determine(record_id)
        job_material = json.dumps({
            "claim_record_id": record_id, "checker_id": self.checker_id,
            "method": self.METHOD, "method_version": self.METHOD_VERSION,
            "evidence_ids": sorted(evidence), "result": result.value,
        }, sort_keys=True, separators=(",", ":"))
        job_id = "memvfy-" + hashlib.sha256(job_material.encode()).hexdigest()[:20]
        return self.store.append_verification(
            claim_record_id=record_id, checker_id=self.checker_id,
            checker_type=self.CHECKER_TYPE, method=self.METHOD,
            method_version=self.METHOD_VERSION, evidence_ids=evidence,
            result=result, confidence=1.0, notes=notes,
            observed_at_utc=observed_at_utc, verification_job_id=job_id)

    def queue(self) -> list[tuple[int, str]]:
        """Claims awaiting verification, highest priority first. V1 policy:
        claims participating in a contradiction go first."""
        conflicted: set[str] = set()
        for _key, group in self.store.consolidated_state().contradiction_groups:
            conflicted.update(group)
        out = []
        for r in self.store.retrievable():
            if self.store.verification_status(r.record_id) is not VerificationStatus.UNVERIFIED:
                continue
            p = priority_signals(conflict_present=r.record_id in conflicted)
            out.append((p["v1_effective_priority"], r.record_id))
        return sorted(out, key=lambda t: (-t[0], t[1]))
