"""G1 runtime evidence graph. Built ONLY from observable evidence text.

Two graphs exist in this project and they must never touch
---------------------------------------------------------
    evaluator proof graph : proof_edges, the answer node, the bridge label,
                            the required-evidence set. It is the answer key.
    runtime evidence graph: this module. Entities and relations recovered from
                            visible record text by existing runtime parsers.

The separation is enforced by an executable leakage test that strips docstrings
and comments before scanning, so this prose may name the forbidden fields in
order to explain them while no executable statement may read them. The test also
rejects indirect access through task-metadata dictionaries, because reading
``task["required_evidence_ids"]`` is the same leak wearing a subscript.

Deliberately minimal
--------------------
Five edge types, three node types in practice. Richer ontology (OWNED_BY,
LOCATED_AT, DEPENDS_ON, UPDATED_BY) is NOT added until the simple graph shows
value -- B4 established that adding structure without evidence that it
discriminates just produces a more elaborate way to fail.

Every edge carries provenance so that a later question of the form "why did
traversal surface this record?" is answerable from the artifact alone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..retrieval.canonicalization import _norm, extract_identity_links
from .bridge_extraction import extract_v4_entities

#: Bumped whenever entity/relation extraction changes, so provenance on a stored
#: edge identifies which parser produced it.
PARSER_VERSION = "g1-runtime-parsers-v1"

#: Hard traversal bound for G1. Not a default, a ceiling: `bounded_neighborhood`
#: raises above it. Unbounded walks are a different mechanism and would need
#: their own protocol.
MAX_HOPS = 2

NODE_TYPES = frozenset({"ENTITY", "RECORD", "RELATION", "TIMESTAMP", "ATTRIBUTE"})

EDGE_TYPES = frozenset({
    "RECORD_MENTIONS_ENTITY",
    "ENTITY_ALIASES_ENTITY",
    "ENTITY_RESOLVES_TO_ENTITY",
    "RECORD_EXPRESSES_RELATION",
    "ENTITY_LINKED_TO_ENTITY",
})


@dataclass(frozen=True)
class GraphEdge:
    """One runtime-derived edge, with enough provenance to audit it later."""
    edge_id: str
    edge_type: str
    source: str
    target: str
    source_record_id: str
    extraction_method: str
    source_span_hash: str
    parser_version: str = PARSER_VERSION

    def as_dict(self) -> dict[str, str]:
        return {
            "edge_id": self.edge_id, "edge_type": self.edge_type,
            "source": self.source, "target": self.target,
            "source_record_id": self.source_record_id,
            "extraction_method": self.extraction_method,
            "source_span_hash": self.source_span_hash,
            "parser_version": self.parser_version,
        }


def _span_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _edge_id(edge_type: str, source: str, target: str, record_id: str) -> str:
    payload = f"{edge_type}|{source}|{target}|{record_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class _Row:
    """Shim so a record can feed the qualified canonicalization parser."""

    def __init__(self, record_id: str, content: str):
        self.evidence_id = record_id
        self.content = content


@dataclass
class RuntimeGraph:
    """Ephemeral per-query graph. Rebuilt each run; nothing persists yet."""
    edges: list[GraphEdge] = field(default_factory=list)
    entities_by_record: dict[str, frozenset[str]] = field(default_factory=dict)
    records_by_entity: dict[str, set[str]] = field(default_factory=dict)
    relation_records: set[str] = field(default_factory=set)
    entity_links: dict[str, set[str]] = field(default_factory=dict)
    alias_links: dict[str, set[str]] = field(default_factory=dict)

    # --- observability -----------------------------------------------------
    def stats(self) -> dict[str, int]:
        by_type: dict[str, int] = {}
        for edge in self.edges:
            by_type[edge.edge_type] = by_type.get(edge.edge_type, 0) + 1
        return {
            "record_nodes": len(self.entities_by_record),
            "entity_nodes": len(self.records_by_entity),
            "edges_total": len(self.edges),
            "relation_bearing_records": len(self.relation_records),
            **{f"edges_{name.lower()}": count for name, count in sorted(by_type.items())},
        }

    def neighbours(self, entity: str) -> frozenset[str]:
        """One hop over ENTITY_LINKED_TO_ENTITY plus alias identity."""
        key = _norm(entity)
        out = set(self.entity_links.get(key, ()))
        out |= self.alias_links.get(key, set())
        out.discard(key)
        return frozenset(out)


def build_runtime_graph(*, record_ids: Sequence[str], texts: Mapping[str, str],
                        relation: str = "") -> RuntimeGraph:
    """Build the graph for one candidate pool from visible text only.

    ``relation`` is the target relation parsed from the question, used to mark
    RECORD_EXPRESSES_RELATION. It arrives as a string the question already
    contained -- no label from the generator is consulted.
    """
    graph = RuntimeGraph()
    relation_norm = _norm(relation) if relation else ""
    rows = [_Row(rid, texts.get(rid, "")) for rid in record_ids]

    for row in rows:
        content = row.content
        entities = extract_v4_entities(content)
        normalized = {_norm(e) for e in entities if _norm(e)}
        graph.entities_by_record[row.evidence_id] = frozenset(normalized)
        span = _span_hash(content)

        for entity in sorted(normalized):
            graph.records_by_entity.setdefault(entity, set()).add(row.evidence_id)
            graph.edges.append(GraphEdge(
                edge_id=_edge_id("RECORD_MENTIONS_ENTITY", row.evidence_id,
                                 entity, row.evidence_id),
                edge_type="RECORD_MENTIONS_ENTITY", source=row.evidence_id,
                target=entity, source_record_id=row.evidence_id,
                extraction_method="extract_v4_entities", source_span_hash=span))

        # Co-mention inside one record is the only link this version asserts.
        ordered = sorted(normalized)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                graph.entity_links.setdefault(left, set()).add(right)
                graph.entity_links.setdefault(right, set()).add(left)
                graph.edges.append(GraphEdge(
                    edge_id=_edge_id("ENTITY_LINKED_TO_ENTITY", left, right,
                                     row.evidence_id),
                    edge_type="ENTITY_LINKED_TO_ENTITY", source=left,
                    target=right, source_record_id=row.evidence_id,
                    extraction_method="co_mention_in_record",
                    source_span_hash=span))

        if relation_norm and relation_norm in _norm(content):
            graph.relation_records.add(row.evidence_id)
            graph.edges.append(GraphEdge(
                edge_id=_edge_id("RECORD_EXPRESSES_RELATION", row.evidence_id,
                                 relation_norm, row.evidence_id),
                edge_type="RECORD_EXPRESSES_RELATION", source=row.evidence_id,
                target=relation_norm, source_record_id=row.evidence_id,
                extraction_method="extract_target_relation_match",
                source_span_hash=span))

    for link in extract_identity_links(rows):
        left, right = _norm(link.surface), _norm(link.canonical)
        if not left or not right or left == right:
            continue
        graph.alias_links.setdefault(left, set()).add(right)
        graph.alias_links.setdefault(right, set()).add(left)
        record_id = getattr(link, "evidence_id", "") or ""
        span = _span_hash(texts.get(record_id, "") or f"{left}->{right}")
        for edge_type in ("ENTITY_ALIASES_ENTITY", "ENTITY_RESOLVES_TO_ENTITY"):
            graph.edges.append(GraphEdge(
                edge_id=_edge_id(edge_type, left, right, record_id),
                edge_type=edge_type, source=left, target=right,
                source_record_id=record_id,
                extraction_method="extract_identity_links",
                source_span_hash=span))
    return graph


def bounded_neighborhood(graph: RuntimeGraph, seeds: Iterable[str],
                         hops: int) -> tuple[frozenset[str], dict[str, int]]:
    """Entities within ``hops`` of ``seeds``. Refuses to exceed MAX_HOPS.

    Present for G2. G1's scored arm does not call it -- the typed-path arm walks
    one frozen template instead of a general frontier, and that difference is
    the hypothesis under test.
    """
    if hops > MAX_HOPS:
        raise ValueError(
            f"hops={hops} exceeds the frozen G1 bound MAX_HOPS={MAX_HOPS}; "
            "unbounded traversal needs its own protocol")
    frontier = {_norm(s) for s in seeds if _norm(s)}
    visited = set(frontier)
    edges_traversed = 0
    for _ in range(max(0, hops)):
        nxt: set[str] = set()
        for entity in sorted(frontier):
            for neighbour in sorted(graph.neighbours(entity)):
                edges_traversed += 1
                if neighbour not in visited:
                    nxt.add(neighbour)
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return frozenset(visited), {"nodes_visited": len(visited),
                                "edges_traversed": edges_traversed}
