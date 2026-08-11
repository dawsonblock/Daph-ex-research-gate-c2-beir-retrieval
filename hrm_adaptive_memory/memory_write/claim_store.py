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
    NON_RETRIEVABLE_STATES, ConflictOutcome, VerificationState)

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
    verification_state: VerificationState
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
        d["verification_state"] = self.verification_state.value
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
        self._events: list[dict] = []
        if self.log_path.is_file():
            self._events = self._load_events(self.log_path)
        self._records: dict[str, ClaimRecord] = {}
        self._index = ConsolidationIndex()
        #: Raw (entity, relation) -> ACTIVE record ids. Ingest-time conflict
        #: detection is LOCAL and cheap by design, so it must not scan the
        #: whole corpus: without this index every ingest was O(n), making
        #: bulk ingestion O(n^2) (measured: per-event cost doubled with each
        #: doubling of n, 5.8ms/event at only 4k events).
        self._active_by_claim_key: dict[tuple[str, str], set[str]] = {}
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
        if self.auto_snapshot and self._events and not self.snapshot_is_valid():
            self.publish_snapshot()

    def _load_events(self, path: Path) -> list[dict]:
        events: list[dict] = []
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    # torn final append: never committed, safe to drop
                    self.truncated_tail = True
                    break
                raise
        return events

    # --- derived state ---------------------------------------------------
    def _apply_event(self, ev: dict) -> None:
        """Apply ONE event to both the record map and the incremental
        consolidation index. Used by live appends and by full replay, so the
        two can never diverge in how an event is interpreted."""
        kind = ev["event"]
        if kind == "INGEST":
            r = ev["record"]
            rec = ClaimRecord(
                **{**r, "verification_state": VerificationState(r["verification_state"]),
                   "corroborating_record_ids": tuple(r.get("corroborating_record_ids", ())),
                   "contradicting_record_ids": tuple(r.get("contradicting_record_ids", ()))})
            self._records[rec.record_id] = rec
            self._index.apply_ingest(rec, len(self._events))
            self._index_claim_key(rec)
        elif kind == "STATE_CHANGE":
            rid = ev["record_id"]
            cur = self._records[rid]
            rec = replace(
                cur,
                verification_state=VerificationState(ev["verification_state"]),
                corroborating_record_ids=tuple(ev.get("corroborating_record_ids",
                                                      cur.corroborating_record_ids)),
                contradicting_record_ids=tuple(ev.get("contradicting_record_ids",
                                                      cur.contradicting_record_ids)),
                superseded_by=ev.get("superseded_by", cur.superseded_by))
            self._records[rid] = rec
            self._index.apply_state_change(rec, len(self._events))
            self._index_claim_key(rec)

    def _index_claim_key(self, rec: ClaimRecord) -> None:
        """Keep the raw-claim-key index in step with a record's activity.
        Mirrors NON_RETRIEVABLE_STATES exactly, so it can never disagree with
        retrievable() about which records are active."""
        bucket = self._active_by_claim_key.setdefault(rec.claim_key, set())
        if rec.verification_state in NON_RETRIEVABLE_STATES:
            bucket.discard(rec.record_id)
            if not bucket:
                del self._active_by_claim_key[rec.claim_key]
        else:
            bucket.add(rec.record_id)

    def _replay(self) -> None:
        self._records = {}
        self._index = ConsolidationIndex()
        self._active_by_claim_key = {}
        for ev in self._events:
            self._apply_event(ev)

    @property
    def corpus_version(self) -> int:
        """Monotonic; every appended event increments it. The reader's
        backend cache must be keyed on this (or on a quantity that provably
        changes with it, as record_id does)."""
        return len(self._events)

    def all_records(self) -> list[ClaimRecord]:
        """Every record ever written, including retracted and superseded --
        history is never destroyed."""
        return list(self._records.values())

    def get(self, record_id: str) -> ClaimRecord | None:
        return self._records.get(record_id)

    def retrievable(self) -> list[ClaimRecord]:
        return [r for r in self._records.values()
                if r.verification_state not in NON_RETRIEVABLE_STATES]

    def retrievable_index_records(self) -> list[IndexRecord]:
        """What the CERTIFIED reader indexes. Excludes retracted and
        superseded records, so a retraction/supersession necessarily changes
        the evidence-id set the reader sees."""
        return [IndexRecord(
            evidence_id=r.record_id, source_id=r.source_id, content=r.content,
            token_count=max(1, len(r.content.split())),
            source_type="verified_memory_write_v1",
            metadata={"verification_state": r.verification_state.value,
                      "canonical_entity": r.canonical_entity,
                      "canonical_relation": r.canonical_relation,
                      "observed_at_utc": r.observed_at_utc},
        ) for r in self.retrievable()]

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
        self._events.append(event)
        line = json.dumps(event, sort_keys=True) + "\n"
        with self.log_path.open("a") as fh:          # 1
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())                    # 2
        self._log_hasher.update(line.encode())
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
            "state_counts": {s.value: sum(1 for r in self._records.values()
                                          if r.verification_state == s) for s in VerificationState},
            "consolidated_state_hash": self.consolidated_state().state_hash(),
        }, indent=2, sort_keys=True) + "\n")

    def _state_change(self, record_id: str, state: VerificationState, **extra) -> None:
        self._append({"event": "STATE_CHANGE", "record_id": record_id,
                      "verification_state": state.value, "at": _now(), **extra})

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

        # step 5: assign state
        if supersedes is not None:
            target = self._records.get(supersedes)
            if target is None:
                raise KeyError(f"supersedes references unknown record_id {supersedes!r}")
            outcome, state = ConflictOutcome.SUPERSESSION, VerificationState.UNVERIFIED
        elif same_value:
            outcome = ConflictOutcome.SUPPORT
            state = VerificationState.SUPPORTED
        elif diff_value:
            # frozen policy: a later timestamp alone is NOT supersession
            outcome, state = ConflictOutcome.CONFLICT, VerificationState.CONTRADICTED
        else:
            outcome, state = ConflictOutcome.NOVEL, VerificationState.UNVERIFIED

        record = ClaimRecord(
            record_id=record_id, content=content, canonical_entity=canonical_entity,
            canonical_relation=relation.strip(), value=value.strip(), source_id=source_id,
            ingested_at_utc=_now(), observed_at_utc=observed,
            extraction_method=EXTRACTION_METHOD, verification_state=state,
            corpus_version=self.corpus_version + 1,
            corroborating_record_ids=tuple(r.record_id for r in same_value),
            contradicting_record_ids=tuple(r.record_id for r in diff_value),
            supersedes=supersedes)
        self._append({"event": "INGEST", "at": record.ingested_at_utc,
                      "record": record.to_json()})

        affected = []
        if outcome is ConflictOutcome.SUPPORT:
            for r in same_value:
                self._state_change(
                    r.record_id, VerificationState.SUPPORTED,
                    corroborating_record_ids=list(r.corroborating_record_ids) + [record_id])
                affected.append(r.record_id)
        elif outcome is ConflictOutcome.CONFLICT:
            for r in diff_value:
                self._state_change(
                    r.record_id, VerificationState.CONTRADICTED,
                    contradicting_record_ids=list(r.contradicting_record_ids) + [record_id])
                affected.append(r.record_id)
        elif outcome is ConflictOutcome.SUPERSESSION:
            self._state_change(supersedes, VerificationState.SUPERSEDED, superseded_by=record_id)
            affected.append(supersedes)

        return IngestResult(outcome, self._records[record_id], tuple(affected))

    def retract(self, record_id: str, reason: str) -> ClaimRecord:
        if record_id not in self._records:
            raise KeyError(f"unknown record_id {record_id!r}")
        self._state_change(record_id, VerificationState.RETRACTED, reason=reason)
        return self._records[record_id]

    def disagreements(self) -> list[tuple[tuple[str, str], list[ClaimRecord]]]:
        """Claims where retrievable records assert conflicting values. The
        store must be able to REPORT disagreement rather than silently
        picking a winner."""
        by_key: dict[tuple[str, str], list[ClaimRecord]] = {}
        for r in self.retrievable():
            by_key.setdefault(r.claim_key, []).append(r)
        return [(k, rs) for k, rs in sorted(by_key.items())
                if len({x.value.strip().lower() for x in rs}) > 1]
