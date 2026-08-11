"""Derived consolidated state for VERIFIED_MEMORY_CONSOLIDATION_V1.

Per configs/verified_memory_consolidation_v1_design.json. Consolidation
REORGANIZES memory; it does not decide truth. The append-only event log
remains the only authoritative state -- everything here is derived and
throwaway by construction.

TWO INDEPENDENT IMPLEMENTATIONS, deliberately:

  ConsolidationIndex        maintained INCREMENTALLY, one event at a time,
                            with its own live indexes.
  consolidate_from_scratch  computes every group by a fresh full scan over
                            the replayed record map, using no incremental
                            structure at all.

The headline invariant (C9) asserts their state hashes agree. They are
separate code paths on purpose: comparing one function against itself would
prove nothing. What this actually catches is retraction failing to dissolve
a group, mutation aliasing between snapshots, non-deterministic iteration
order, stale index entries, and missed invalidation.

Nothing here resolves a contradiction, picks a winner, infers supersession,
or changes a verification state. Those are truth decisions and are out of
scope.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping

from .states import NON_RETRIEVABLE_STATES, VerificationState  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover
    from .claim_store import ClaimRecord

#: The one relation that expresses "these two names denote the same entity".
#: Alias clusters are built from EXPLICIT claims of this relation only --
#: never from string similarity, co-occurrence, or embedding proximity,
#: because inferring identity is a truth decision.
ALIAS_RELATION = "alias_of"


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


@dataclass(frozen=True)
class ConsolidatedState:
    """Derived state. Every collection is canonically ordered so the hash is
    a pure function of content, never of dict/set iteration order."""
    corpus_version: int
    active_record_ids: tuple[str, ...]
    duplicate_clusters: tuple[tuple[str, ...], ...]
    alias_clusters: tuple[tuple[str, ...], ...]
    support_groups: tuple[tuple[str, tuple[str, ...]], ...]
    contradiction_groups: tuple[tuple[str, tuple[str, ...]], ...]

    def to_json(self) -> dict:
        return {
            "corpus_version": self.corpus_version,
            "active_record_ids": list(self.active_record_ids),
            "duplicate_clusters": [list(c) for c in self.duplicate_clusters],
            "alias_clusters": [list(c) for c in self.alias_clusters],
            "support_groups": [[k, list(v)] for k, v in self.support_groups],
            "contradiction_groups": [[k, list(v)] for k, v in self.contradiction_groups],
        }

    def state_hash(self) -> str:
        """Canonical hash. corpus_version is deliberately EXCLUDED: the same
        active content reached by different event counts is the same derived
        state, and including the counter would make C9 pass trivially while
        hiding real divergence in the groups themselves."""
        payload = {k: v for k, v in self.to_json().items() if k != "corpus_version"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --- shared canonical grouping helpers ------------------------------------
# Used by BOTH implementations so the two agree on FORMAT while still
# computing membership independently. Formatting is not the thing under
# test; membership is.

def _union_find(edges: Iterable[tuple[str, str]], nodes: Iterable[str]) -> list[tuple[str, ...]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for n in nodes:
        find(n)
    for a, b in edges:
        union(a, b)
    clusters: dict[str, list[str]] = {}
    for n in sorted(parent):
        clusters.setdefault(find(n), []).append(n)
    return sorted(tuple(sorted(v)) for v in clusters.values() if len(v) > 1)


def _canonical_groups(groups: Mapping[str, Iterable[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted((k, tuple(sorted(v))) for k, v in groups.items()))


def _alias_key(entity: str, alias_parent: Mapping[str, str]) -> str:
    """Resolve an entity name to its alias-cluster representative."""
    return alias_parent.get(_norm(entity), _norm(entity))


def _build_alias_parent(records: "Iterable[ClaimRecord]") -> dict[str, str]:
    """Map every aliased entity name to a single deterministic representative
    (the lexicographically smallest name in its cluster)."""
    edges = []
    nodes = set()
    for r in records:
        nodes.add(_norm(r.canonical_entity))
        if _norm(r.canonical_relation) == ALIAS_RELATION:
            edges.append((_norm(r.canonical_entity), _norm(r.value)))
            nodes.add(_norm(r.value))
    parent: dict[str, str] = {}
    for cluster in _union_find(edges, nodes):
        rep = cluster[0]
        for name in cluster:
            parent[name] = rep
    return parent


def _active(records: "Iterable[ClaimRecord]") -> "list[ClaimRecord]":
    return [r for r in records if r.verification_state not in NON_RETRIEVABLE_STATES]


# --- IMPLEMENTATION A: from-scratch full scan -----------------------------

def consolidate_from_scratch(records: "Iterable[ClaimRecord]", corpus_version: int) -> ConsolidatedState:
    """Compute every group by scanning ALL records fresh. Uses no incremental
    structure. This is the reference implementation for C9."""
    all_records = list(records)
    active = _active(all_records)
    alias_parent = _build_alias_parent(active)

    # duplicate clusters: identical rendered content among ACTIVE records
    by_content: dict[str, list[str]] = {}
    for r in active:
        by_content.setdefault(r.content, []).append(r.record_id)
    duplicate_clusters = sorted(tuple(sorted(v)) for v in by_content.values() if len(v) > 1)

    # support / contradiction, alias-resolved
    by_claim: dict[str, list[ClaimRecord]] = {}
    for r in active:
        if _norm(r.canonical_relation) == ALIAS_RELATION:
            continue
        key = f"{_alias_key(r.canonical_entity, alias_parent)}|{_norm(r.canonical_relation)}"
        by_claim.setdefault(key, []).append(r)

    support: dict[str, list[str]] = {}
    contradiction: dict[str, list[str]] = {}
    for key, group in by_claim.items():
        values = {_norm(r.value) for r in group}
        if len(values) > 1:
            contradiction[key] = [r.record_id for r in group]
        elif len(group) > 1:
            support[key] = [r.record_id for r in group]

    return ConsolidatedState(
        corpus_version=corpus_version,
        active_record_ids=tuple(sorted(r.record_id for r in active)),
        duplicate_clusters=tuple(duplicate_clusters),
        alias_clusters=tuple(_union_find(
            [(_norm(r.canonical_entity), _norm(r.value)) for r in active
             if _norm(r.canonical_relation) == ALIAS_RELATION],
            [_norm(r.canonical_entity) for r in active])),
        support_groups=_canonical_groups(support),
        contradiction_groups=_canonical_groups(contradiction),
    )


# --- IMPLEMENTATION B: incremental ----------------------------------------

class ConsolidationIndex:
    """Maintained one event at a time. Never rescans the full record set.

    Retraction and supersession must correctly REMOVE a record from every
    index it participates in -- failing to do so is the single most likely
    consolidation bug, and is exactly what C9 catches.
    """

    def __init__(self):
        self._active: "dict[str, ClaimRecord]" = {}
        self._by_content: dict[str, set[str]] = {}
        self._alias_edges: set[tuple[str, str]] = set()
        self._entities: set[str] = set()
        self._corpus_version = 0

    # -- incremental mutations ------------------------------------------
    def apply_ingest(self, record: "ClaimRecord", corpus_version: int) -> None:
        self._corpus_version = corpus_version
        if record.verification_state in NON_RETRIEVABLE_STATES:
            return
        self._add(record)

    def apply_state_change(self, record: "ClaimRecord", corpus_version: int) -> None:
        self._corpus_version = corpus_version
        if record.verification_state in NON_RETRIEVABLE_STATES:
            self._remove(record.record_id)
        else:
            # state may have changed (e.g. UNVERIFIED -> SUPPORTED) without
            # affecting activity; refresh the stored copy so downstream
            # snapshots never serve a stale record object
            self._add(record)

    def _add(self, record: "ClaimRecord") -> None:
        self._active[record.record_id] = record
        self._by_content.setdefault(record.content, set()).add(record.record_id)
        self._entities.add(_norm(record.canonical_entity))
        if _norm(record.canonical_relation) == ALIAS_RELATION:
            self._alias_edges.add((_norm(record.canonical_entity), _norm(record.value)))
            self._entities.add(_norm(record.value))

    def _remove(self, record_id: str) -> None:
        record = self._active.pop(record_id, None)
        if record is None:
            return
        bucket = self._by_content.get(record.content)
        if bucket is not None:
            bucket.discard(record_id)
            if not bucket:
                del self._by_content[record.content]
        if _norm(record.canonical_relation) == ALIAS_RELATION:
            # an alias edge is only supported while at least one ACTIVE
            # record asserts it
            still = any(_norm(r.canonical_relation) == ALIAS_RELATION
                        and _norm(r.canonical_entity) == _norm(record.canonical_entity)
                        and _norm(r.value) == _norm(record.value)
                        for r in self._active.values())
            if not still:
                self._alias_edges.discard((_norm(record.canonical_entity), _norm(record.value)))
        # entity membership is rebuilt on snapshot from active records, so no
        # stale-entity bookkeeping is needed here

    # -- snapshot ---------------------------------------------------------
    def snapshot(self) -> ConsolidatedState:
        active = list(self._active.values())
        entities = {_norm(r.canonical_entity) for r in active}
        for a, b in self._alias_edges:
            entities.update((a, b))
        alias_clusters = _union_find(sorted(self._alias_edges), sorted(entities))
        alias_parent: dict[str, str] = {}
        for cluster in alias_clusters:
            for name in cluster:
                alias_parent[name] = cluster[0]

        duplicate_clusters = sorted(
            tuple(sorted(ids)) for ids in self._by_content.values() if len(ids) > 1)

        by_claim: "dict[str, list[ClaimRecord]]" = {}
        for r in active:
            if _norm(r.canonical_relation) == ALIAS_RELATION:
                continue
            key = f"{_alias_key(r.canonical_entity, alias_parent)}|{_norm(r.canonical_relation)}"
            by_claim.setdefault(key, []).append(r)

        support: dict[str, list[str]] = {}
        contradiction: dict[str, list[str]] = {}
        for key, group in by_claim.items():
            values = {_norm(r.value) for r in group}
            if len(values) > 1:
                contradiction[key] = [r.record_id for r in group]
            elif len(group) > 1:
                support[key] = [r.record_id for r in group]

        return ConsolidatedState(
            corpus_version=self._corpus_version,
            active_record_ids=tuple(sorted(self._active)),
            duplicate_clusters=tuple(duplicate_clusters),
            alias_clusters=tuple(alias_clusters),
            support_groups=_canonical_groups(support),
            contradiction_groups=_canonical_groups(contradiction),
        )
