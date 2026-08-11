"""Append-only verified claim store for VERIFIED_MEMORY_WRITE_V1 MILESTONE_1.

Implements steps 1-7 of the frozen milestone: ingest, canonicalize, detect
duplicate/support/conflict/supersession, attach provenance, assign a
verification state, persist, and expose records for retrieval rebuild.
Steps 8-9 (a write becoming retrievable; retraction/supersession changing
retrieval) are the acceptance tests in tests/unit/test_verified_memory_write.py.

Three frozen policies from the design, implemented literally:

  NEVER SILENTLY DELETE. Storage is an append-only EVENT LOG; current state
  is derived by replay. Retraction and supersession change RETRIEVABILITY
  and STATE, never history.

  CONTRADICTED IS A TERMINAL STATE, not an error. A memory that cannot say
  "two sources disagree and nothing resolves it" is lying about what it
  knows, so conflicting claims are BOTH retained and BOTH marked.

  A LATER TIMESTAMP ALONE IS NOT SUPERSESSION. An explicit supersedes link
  is required. Timestamp-based supersession has real failure modes (clock
  skew, backfill, corrections-vs-updates) and is a separately frozen
  follow-up.

WHY record_id IS PROVENANCE-ADDRESSED (and why that is load-bearing):
the certified reader caches retrieval backends under
``(mode, frozenset(evidence_id))`` -- a CONTENT-BLIND key. If a record's
content could change while keeping its id, the reader would silently answer
from a stale index. Deriving record_id from content+source+observed_at makes
that impossible: any change to what a record says produces a different id,
so the cache key necessarily changes. Test T6 demonstrates that a
content-blind reuse of ids WOULD have lied, so this is proven to be doing
work rather than assumed.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..contracts import IndexRecord
from .consolidation import ConsolidationIndex
from .states import (  # noqa: F401
    NON_RETRIEVABLE_STATES, ConflictOutcome, LifecycleState, VerificationStatus,
    lifecycle_from_event)
from .verification import (  # noqa: F401
    EvidenceResolutionError, VerificationEvent, VerificationResult, derive_status,
    retired_ids, verification_event_id)

#: The write template. Chosen because it is a real b3 content shape and
#: round-trips through the certified extract_v4_entities unchanged --
#: verified per-record at ingest time, fail-closed.
CONTENT_TEMPLATE = "subject={subject}; {relation}={value}"
EXTRACTION_METHOD = "structured_claim_v1+grammar_v4_verify"




class NotNativelyParseableError(ValueError):
    """The rendered content did not round-trip through the certified entity
    extractor. Fail closed: writing a record the reader cannot parse would
    silently poison the corpus."""


@dataclass(frozen=True)
class ClaimRecord:
    record_id: str
    content: str
    canonical_entity: str
    canonical_relation: str
    value: str
    source_id: str
    ingested_at_utc: str
    observed_at_utc: str
    extraction_method: str
    lifecycle_state: LifecycleState
    corpus_version: int
    corroborating_record_ids: tuple[str, ...] = ()
    contradicting_record_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    superseded_by: str | None = None

    @property
    def claim_key(self) -> tuple[str, str]:
        return (self.canonical_entity.strip().lower(), self.canonical_relation.strip().lower())

    def to_json(self) -> dict:
        d = asdict(self)
        d["lifecycle_state"] = self.lifecycle_state.value
        d["corroborating_record_ids"] = list(self.corroborating_record_ids)
        d["contradicting_record_ids"] = list(self.contradicting_record_ids)
        return d


@dataclass
class IngestResult:
    outcome: ConflictOutcome
    record: ClaimRecord | None
    affected_record_ids: tuple[str, ...] = ()
    note: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verify_native(content: str, subject: str) -> str:
    """Canonicalize the entity using the CERTIFIED extractor and assert the
    rendered content round-trips. Returns the canonical entity."""
    from ..c4.bridge_extraction import extract_v4_entities
    ents = extract_v4_entities(content)
    want = subject.strip()
    for e in ents:
        if e.strip().lower() == want.lower():
            return e.strip()
    raise NotNativelyParseableError(
        f"rendered content {content!r} did not yield subject {want!r} through the "
        f"certified extract_v4_entities (got {ents!r}). Refusing to write a record "
        "the reader cannot parse.")


class ClaimStore:
    """Append-only claim store. Current state is derived by replaying the
    event log, so no mutation ever destroys history."""

    def __init__(self, root: str | Path, auto_snapshot: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "claims_events.jsonl"
        self.manifest_path = self.root / "MANIFEST.json"
        self.snapshot_path = self.root / "consolidated_snapshot.json"
        self.auto_snapshot = auto_snapshot
        #: True if the final log line was a partial/torn append (a crash
        #: during step 1 of the commit sequence). The truncated line is
        #: DISCARDED -- an incomplete event was never committed -- and
        #: replay proceeds from the intact prefix.
        self.truncated_tail = False
        #: BOUNDED_MEMORY_RUNTIME_V1: history lives on DISK. Only a counter is
        #: retained, never the events themselves. Measured at 100k events, the
        #: resident event list was 57.7% of all accounted python-object memory
        #: (168.8 MB of 293 MB) -- by far the largest single structure, and
        #: pure duplication of bytes already durable in the log.
        self._corpus_version = 0
        self._records: dict[str, ClaimRecord] = {}
        self._index = ConsolidationIndex()
        #: Raw (entity, relation) -> ACTIVE record ids. Ingest-time conflict
        #: detection is LOCAL and cheap by design, so it must not scan the
        #: whole corpus: without this index every ingest was O(n), making
        #: bulk ingestion O(n^2) (measured: per-event cost doubled with each
        #: doubling of n, 5.8ms/event at only 4k events).
        self._active_by_claim_key: dict[tuple[str, str], set[str]] = {}
        #: claim_record_id -> verification events citing it, in log order.
        self._verifications: dict[str, list[VerificationEvent]] = {}
        #: every verification_event_id ever seen, for V11 idempotency
        self._verification_ids: set[str] = set()
        #: Running hash of the canonical log. Recomputing it per append meant
        #: re-reading and re-hashing the entire file on every event -- the
        #: second of three O(n)-per-event costs found by the scaling probe.
        self._log_hasher = hashlib.sha256(
            self.log_path.read_bytes() if self.log_path.is_file() else b"")
        self._replay()
        # Per the frozen C11 recovery rule: a snapshot is only ever an
        # accelerator. If it is missing, torn, or describes a different log,
        # it is DISCARDED and REBUILT from the canonical log -- which is the
        # authoritative state we just replayed.
        if self.auto_snapshot and self._corpus_version and not self.snapshot_is_valid():
            self.publish_snapshot()

    def _stream_events(self):
        """Yield events from the canonical log one at a time.

        Explicit readline() rather than iterating the file object, so the
        one-line lookahead used to classify a decode failure cannot interact
        with the iterator's internal buffering. A torn line can ONLY be the
        final one -- an append that never committed -- so if nothing follows
        it, drop it and record truncated_tail; anything else is genuine
        mid-log corruption and must not be silently tolerated.
        """
        if not self.log_path.is_file():
            return
        with self.log_path.open() as fh:
            while True:
                line = fh.readline()
                if line == "":
                    return
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if fh.readline() == "":
                        self.truncated_tail = True
                        return
                    raise

    # --- derived state ---------------------------------------------------
    def _apply_event(self, ev: dict) -> None:
        """Apply ONE event to both the record map and the incremental
        consolidation index. Used by live appends and by full replay, so the
        two can never diverge in how an event is interpreted."""
        kind = ev["event"]
        if kind == "INGEST":
            r = ev["record"]
            raw = dict(r)
            # accept both the current encoding and pre-split logs (A6)
            legacy = raw.pop("verification_state", None)
            raw["lifecycle_state"] = lifecycle_from_event(
                raw.get("lifecycle_state", legacy) or LifecycleState.ACTIVE.value)
            raw["corroborating_record_ids"] = tuple(raw.get("corroborating_record_ids", ()))
            raw["contradicting_record_ids"] = tuple(raw.get("contradicting_record_ids", ()))
            rec = ClaimRecord(**raw)
            self._records[rec.record_id] = rec
            self._index.apply_ingest(rec, self._corpus_version)
            self._index_claim_key(rec)
        elif kind == "STATE_CHANGE":
            rid = ev["record_id"]
            cur = self._records[rid]
            rec = replace(
                cur,
                lifecycle_state=lifecycle_from_event(
                    ev.get("lifecycle_state", ev.get("verification_state"))),
                corroborating_record_ids=tuple(ev.get("corroborating_record_ids",
                                                      cur.corroborating_record_ids)),
                contradicting_record_ids=tuple(ev.get("contradicting_record_ids",
                                                      cur.contradicting_record_ids)),
                superseded_by=ev.get("superseded_by", cur.superseded_by))
            self._records[rid] = rec
            self._index.apply_state_change(rec, self._corpus_version)
            self._index_claim_key(rec)
        elif kind == "VERIFICATION":
            v = ev["verification"]
            evt = VerificationEvent(
                verification_event_id=v["verification_event_id"],
                claim_record_id=v["claim_record_id"], checker_id=v["checker_id"],
                checker_type=v["checker_type"], method=v["method"],
                method_version=v["method_version"],
                evidence_ids=tuple(v.get("evidence_ids", ())),
                observed_at_utc=v["observed_at_utc"],
                result=VerificationResult(v["result"]), confidence=float(v["confidence"]),
                notes=v.get("notes", ""),
                supersedes_verification=v.get("supersedes_verification"))
            # V11: identical determinations are idempotent, not second opinions
            if evt.verification_event_id in self._verification_ids:
                return
            self._verification_ids.add(evt.verification_event_id)
            self._verifications.setdefault(evt.claim_record_id, []).append(evt)

    def _index_claim_key(self, rec: ClaimRecord) -> None:
        """Keep the raw-claim-key index in step with a record's activity.
        Mirrors NON_RETRIEVABLE_STATES exactly, so it can never disagree with
        retrievable() about which records are active."""
        bucket = self._active_by_claim_key.setdefault(rec.claim_key, set())
        if rec.lifecycle_state in NON_RETRIEVABLE_STATES:
            bucket.discard(rec.record_id)
            if not bucket:
                del self._active_by_claim_key[rec.claim_key]
        else:
            bucket.add(rec.record_id)

    def _replay(self) -> None:
        """Rebuild all in-RAM state by STREAMING the canonical log. Peak
        memory is now O(current state), not O(history)."""
        self._records = {}
        self._index = ConsolidationIndex()
        self._active_by_claim_key = {}
        self._verifications = {}
        self._verification_ids = set()
        self._corpus_version = 0
        for ev in self._stream_events():
            self._corpus_version += 1
            self._apply_event(ev)

    @property
    def corpus_version(self) -> int:
        """Monotonic; every appended event increments it. The reader's
        backend cache must be keyed on this (or on a quantity that provably
        changes with it, as record_id does)."""
        return self._corpus_version

    def all_records(self) -> list[ClaimRecord]:
        """Every record ever written, including retracted and superseded --
        history is never destroyed."""
        return list(self._records.values())

    def get(self, record_id: str) -> ClaimRecord | None:
        return self._records.get(record_id)

    def retrievable(self) -> list[ClaimRecord]:
        return [r for r in self._records.values()
                if r.lifecycle_state not in NON_RETRIEVABLE_STATES]

    def retrievable_index_records(self, verification_filter=None) -> list[IndexRecord]:
        """What the CERTIFIED reader indexes. Excludes retracted and
        superseded records, so a retraction/supersession necessarily changes
        the evidence-id set the reader sees."""
        records = self.retrievable()
        if verification_filter is not None:
            # V10: OPT-IN only. Default behaviour is unchanged so
            # CERTIFIED_MEMORY_V1's identity and inputs are untouched (V12).
            allowed = frozenset(verification_filter)
            records = [r for r in records if self.verification_status(r.record_id) in allowed]
        return [IndexRecord(
            evidence_id=r.record_id, source_id=r.source_id, content=r.content,
            token_count=max(1, len(r.content.split())),
            source_type="verified_memory_write_v1",
            metadata={"lifecycle_state": r.lifecycle_state.value,
                      "canonical_entity": r.canonical_entity,
                      "canonical_relation": r.canonical_relation,
                      "observed_at_utc": r.observed_at_utc},
        ) for r in records]

    # --- persistence -----------------------------------------------------
    def _append(self, event: dict) -> None:
        """The frozen C11 commit sequence:
            1. append the canonical event
            2. fsync the log
            3. derive new consolidated state (incrementally)
            4. atomically publish the snapshot

        A crash between ANY two stages leaves the canonical log sufficient
        for recovery: the snapshot is only ever an accelerator, is validated
        against the log it claims to describe, and is discarded on mismatch.
        """
        line = json.dumps(event, sort_keys=True) + "\n"
        with self.log_path.open("a") as fh:          # 1
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())                    # 2
        self._log_hasher.update(line.encode())
        self._corpus_version += 1
        self._apply_event(event)                     # 3 -- O(1), not a full replay
        if self.auto_snapshot:
            self.publish_snapshot()                  # 4 (also refreshes the manifest)

    def _atomic_write(self, path: Path, text: str) -> None:
        """Write via temp + fsync + os.replace, which is atomic on POSIX, so
        a crash mid-write can never leave a half-written file that a naive
        reader would trust."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def log_sha256(self) -> str:
        """O(1): maintained incrementally as events are appended."""
        return self._log_hasher.hexdigest()

    def consolidated_state(self):
        """Derived state from the INCREMENTAL index."""
        return self._index.snapshot()

    def publish_snapshot(self) -> Path:
        """Stage 4. The snapshot records the corpus_version and log hash it
        was derived from, so a later load can tell whether it still
        describes this log."""
        state = self.consolidated_state()
        self._atomic_write(self.snapshot_path, json.dumps({
            "derived_from_corpus_version": self.corpus_version,
            "derived_from_log_sha256": self.log_sha256(),
            "state_hash": state.state_hash(),
            "state": state.to_json(),
        }, indent=2, sort_keys=True) + "\n")
        self._write_manifest()
        return self.snapshot_path

    def snapshot_is_valid(self) -> bool:
        """A snapshot is trusted ONLY if it describes exactly this log.
        Anything else -- missing, torn, or describing a different history --
        is discarded and rebuilt from the canonical log."""
        if not self.snapshot_path.is_file():
            return False
        try:
            snap = json.loads(self.snapshot_path.read_text())
        except json.JSONDecodeError:
            return False
        return (snap.get("derived_from_corpus_version") == self.corpus_version
                and snap.get("derived_from_log_sha256") == self.log_sha256())

    def _write_manifest(self) -> None:
        digest = self.log_sha256()
        self._atomic_write(self.manifest_path, json.dumps({
            "store": "VERIFIED_MEMORY_WRITE_V1",
            "design": "configs/verified_memory_write_v1_design.json",
            "corpus_version": self.corpus_version,
            "event_log_sha256": digest,
            "n_records_total": len(self._records),
            "n_retrievable": len(self.retrievable()),
            "lifecycle_counts": {s.value: sum(1 for r in self._records.values()
                                              if r.lifecycle_state == s) for s in LifecycleState},
            "consolidated_state_hash": self.consolidated_state().state_hash(),
        }, indent=2, sort_keys=True) + "\n")

    def _state_change(self, record_id: str, state: LifecycleState, **extra) -> None:
        self._append({"event": "STATE_CHANGE", "record_id": record_id,
                      "lifecycle_state": state.value, "at": _now(), **extra})

    # --- the write path --------------------------------------------------
    def ingest(self, *, subject: str, relation: str, value: str, source_id: str,
               observed_at_utc: str | None = None, supersedes: str | None = None) -> IngestResult:
        """MILESTONE_1 steps 1-6."""
        content = CONTENT_TEMPLATE.format(subject=subject, relation=relation, value=value)
        canonical_entity = _verify_native(content, subject)  # step 2, fail-closed
        observed = observed_at_utc or _now()
        record_id = "vmw-" + hashlib.sha256(
            f"{content}|{source_id}|{observed}".encode()).hexdigest()[:20]

        if record_id in self._records:
            return IngestResult(ConflictOutcome.DUPLICATE, self._records[record_id],
                                note="identical content, source and observation time")

        key = (canonical_entity.strip().lower(), relation.strip().lower())
        existing = [self._records[rid] for rid in sorted(self._active_by_claim_key.get(key, ()))]
        same_value = [r for r in existing if r.value.strip().lower() == value.strip().lower()]
        diff_value = [r for r in existing if r.value.strip().lower() != value.strip().lower()]

        # V1: ingest OBSERVES a relationship but records NO verification
        # opinion. Every ingested claim is lifecycle ACTIVE and, in
        # verification terms, UNVERIFIED until a verification event says
        # otherwise. Classifying support/conflict is the first verification
        # worker's job, not ingest's.
        if supersedes is not None:
            if self._records.get(supersedes) is None:
                raise KeyError(f"supersedes references unknown record_id {supersedes!r}")
            outcome = ConflictOutcome.SUPERSESSION
        elif same_value:
            outcome = ConflictOutcome.SUPPORT
        elif diff_value:
            # frozen policy: a later timestamp alone is NOT supersession
            outcome = ConflictOutcome.CONFLICT
        else:
            outcome = ConflictOutcome.NOVEL

        record = ClaimRecord(
            record_id=record_id, content=content, canonical_entity=canonical_entity,
            canonical_relation=relation.strip(), value=value.strip(), source_id=source_id,
            ingested_at_utc=_now(), observed_at_utc=observed,
            extraction_method=EXTRACTION_METHOD, lifecycle_state=LifecycleState.ACTIVE,
            corpus_version=self.corpus_version + 1,
            corroborating_record_ids=tuple(r.record_id for r in same_value),
            contradicting_record_ids=tuple(r.record_id for r in diff_value),
            supersedes=supersedes)
        self._append({"event": "INGEST", "at": record.ingested_at_utc,
                      "record": record.to_json()})

        # Only SUPERSESSION touches lifecycle. SUPPORT/CONFLICT used to emit
        # back-reference STATE_CHANGEs setting SUPPORTED/CONTRADICTED; that was
        # ingest writing a verification opinion, which V1 forbids. The
        # structural relation survives on the new record's
        # corroborating_/contradicting_record_ids, and consolidation derives
        # support/contradiction groups from claim VALUES regardless.
        affected = []
        if outcome is ConflictOutcome.SUPERSESSION:
            self._state_change(supersedes, LifecycleState.SUPERSEDED, superseded_by=record_id)
            affected.append(supersedes)

        return IngestResult(outcome, self._records[record_id], tuple(affected))

    def retract(self, record_id: str, reason: str) -> ClaimRecord:
        if record_id not in self._records:
            raise KeyError(f"unknown record_id {record_id!r}")
        self._state_change(record_id, LifecycleState.RETRACTED, reason=reason)
        return self._records[record_id]

    def append_verification(self, *, claim_record_id: str, checker_id: str,
                           checker_type: str, method: str, method_version: str,
                           evidence_ids, result: VerificationResult,
                           confidence: float, notes: str = "",
                           observed_at_utc: str | None = None,
                           supersedes_verification: str | None = None) -> VerificationEvent:
        """Append a verification event. NEVER touches the claim record.

        V8 fails closed twice: the claim itself and every cited evidence id
        must resolve to a real record. A dangling pointer would make the
        determination unauditable, which defeats the point of recording
        evidence at all.
        """
        if claim_record_id not in self._records:
            raise EvidenceResolutionError(
                f"verification targets unknown claim record {claim_record_id!r}")
        evidence = tuple(evidence_ids)
        unknown = [e for e in evidence if e not in self._records]
        if unknown:
            raise EvidenceResolutionError(
                f"verification cites unresolvable evidence ids {unknown!r}; refusing to "
                "store a dangling pointer")
        if supersedes_verification is not None and supersedes_verification not in self._verification_ids:
            raise EvidenceResolutionError(
                f"supersedes_verification references unknown verification event "
                f"{supersedes_verification!r}")
        observed = observed_at_utc or _now()
        vid = verification_event_id(
            claim_record_id=claim_record_id, checker_id=checker_id, method=method,
            method_version=method_version, evidence_ids=evidence,
            result=result.value, observed_at_utc=observed)
        evt = VerificationEvent(
            verification_event_id=vid, claim_record_id=claim_record_id,
            checker_id=checker_id, checker_type=checker_type, method=method,
            method_version=method_version, evidence_ids=evidence,
            observed_at_utc=observed, result=result, confidence=float(confidence),
            notes=notes, supersedes_verification=supersedes_verification)
        self._append({"event": "VERIFICATION", "at": _now(), "verification": evt.to_json()})
        return evt

    def verification_events(self, record_id: str) -> list[VerificationEvent]:
        """Full verification history for a claim, in log order. V4: later
        events never erase earlier ones."""
        return list(self._verifications.get(record_id, ()))

    def verification_status(self, record_id: str) -> VerificationStatus:
        """DERIVED, never stored. See verification.derive_status for the
        frozen rule."""
        events = self._verifications.get(record_id, ())
        return derive_status(events, retired_ids(events))

    def verification_disagreements(self) -> list[tuple[str, list[VerificationEvent]]]:
        """V7: claims whose live verification events disagree. Reported
        explicitly rather than averaged into a single verdict."""
        out = []
        for rid, events in sorted(self._verifications.items()):
            retired = retired_ids(events)
            live = [e for e in events if e.verification_event_id not in retired]
            if len({e.result for e in live}) > 1:
                out.append((rid, live))
        return out

    def disagreements(self) -> list[tuple[tuple[str, str], list[ClaimRecord]]]:
        """Claims where retrievable records assert conflicting values. The
        store must be able to REPORT disagreement rather than silently
        picking a winner."""
        by_key: dict[tuple[str, str], list[ClaimRecord]] = {}
        for r in self.retrievable():
            by_key.setdefault(r.claim_key, []).append(r)
        return [(k, rs) for k, rs in sorted(by_key.items())
                if len({x.value.strip().lower() for x in rs}) > 1]
