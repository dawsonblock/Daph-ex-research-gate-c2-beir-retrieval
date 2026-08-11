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
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

from ..contracts import IndexRecord

#: The write template. Chosen because it is a real b3 content shape and
#: round-trips through the certified extract_v4_entities unchanged --
#: verified per-record at ingest time, fail-closed.
CONTENT_TEMPLATE = "subject={subject}; {relation}={value}"
EXTRACTION_METHOD = "structured_claim_v1+grammar_v4_verify"


class VerificationState(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"


class ConflictOutcome(str, Enum):
    NOVEL = "NOVEL"
    DUPLICATE = "DUPLICATE"
    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"
    SUPERSESSION = "SUPERSESSION"


#: States whose records the READER must not see.
NON_RETRIEVABLE_STATES = frozenset({VerificationState.RETRACTED, VerificationState.SUPERSEDED})


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

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "claims_events.jsonl"
        self.manifest_path = self.root / "MANIFEST.json"
        self._events: list[dict] = []
        if self.log_path.is_file():
            self._events = [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]
        self._records: dict[str, ClaimRecord] = {}
        self._replay()

    # --- derived state ---------------------------------------------------
    def _replay(self) -> None:
        self._records = {}
        for ev in self._events:
            kind = ev["event"]
            if kind == "INGEST":
                r = ev["record"]
                self._records[r["record_id"]] = ClaimRecord(
                    **{**r, "verification_state": VerificationState(r["verification_state"]),
                       "corroborating_record_ids": tuple(r.get("corroborating_record_ids", ())),
                       "contradicting_record_ids": tuple(r.get("contradicting_record_ids", ()))})
            elif kind == "STATE_CHANGE":
                rid = ev["record_id"]
                cur = self._records[rid]
                self._records[rid] = replace(
                    cur,
                    verification_state=VerificationState(ev["verification_state"]),
                    corroborating_record_ids=tuple(ev.get("corroborating_record_ids",
                                                          cur.corroborating_record_ids)),
                    contradicting_record_ids=tuple(ev.get("contradicting_record_ids",
                                                          cur.contradicting_record_ids)),
                    superseded_by=ev.get("superseded_by", cur.superseded_by))

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
        self._events.append(event)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
        self._replay()
        self._write_manifest()

    def _write_manifest(self) -> None:
        digest = hashlib.sha256(self.log_path.read_bytes()).hexdigest() if self.log_path.is_file() else ""
        self.manifest_path.write_text(json.dumps({
            "store": "VERIFIED_MEMORY_WRITE_V1",
            "design": "configs/verified_memory_write_v1_design.json",
            "corpus_version": self.corpus_version,
            "event_log_sha256": digest,
            "n_records_total": len(self._records),
            "n_retrievable": len(self.retrievable()),
            "state_counts": {s.value: sum(1 for r in self._records.values()
                                          if r.verification_state == s) for s in VerificationState},
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
        existing = [r for r in self.retrievable() if r.claim_key == key]
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
