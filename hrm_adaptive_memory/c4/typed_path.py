"""G1_TYPED_PATH: one frozen structural template, no general traversal.

The template, and nothing else
------------------------------
        canonical subject  ->  runtime-visible bridge  ->  target relation

This arm exists to answer a single question before any graph substrate is built:
is the missing middle layer really a graph, or merely typed path discrimination?
So it is deliberately NOT graph-lite-with-heuristics. It walks one explicit
pattern and fills any remaining capacity by retrieval rank.

Why this is narrower than the B4 prefilter that failed
------------------------------------------------------
B4's P2 kept any record mentioning any entity co-mentioned with the subject, then
ranked survivors by retrieval score -- so it shrank the pool while preserving the
retriever's original mistake, because bridge rank is the known weak signal.

Here a bridge entity only counts if the path CLOSES: some record must mention
that entity AND express the target relation. A co-mentioned entity that leads
nowhere relevant contributes no admissible records. That single constraint is
the whole difference between the two arms, which is what makes the comparison
interpretable.

Path tiers, frozen
------------------
    0  DIRECT    record mentions the subject and expresses the relation
    1  ENDPOINT  record mentions a closing bridge and expresses the relation
    2  LINK      record mentions the subject and a closing bridge

Tier 0 is required, not a convenience: a task whose answer sits directly on the
subject has no bridge to find, and an arm that could not represent it would fail
those tasks for a reason unrelated to the hypothesis.

Two hops maximum (subject -> bridge -> relation-bearing record), consistent with
the frozen MAX_HOPS bound.

Runtime safety: canonical subject, question-parsed relation, entities from
visible text, retrieval score and rank. Nothing from the evaluator proof graph,
directly or through a task-metadata dictionary. Enforced by an executable test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..retrieval.canonicalization import _norm
from .runtime_graph import MAX_HOPS, RuntimeGraph, build_runtime_graph

TIER_DIRECT = 0
TIER_ENDPOINT = 1
TIER_LINK = 2
TIER_FILLER = 3


@dataclass
class TypedPathResult:
    """Working set plus the diagnostics that explain WHY it looks like this."""
    kept: list[str] = field(default_factory=list)
    working_set_size: int = 0
    input_size: int = 0
    tier_counts: dict[int, int] = field(default_factory=dict)
    # --- path-hit diagnostics, per task ---------------------------------
    canonical_subject_found: bool = False
    target_relation_extracted: bool = False
    one_hop_bridge_candidates: int = 0
    closing_bridge_entities: int = 0
    typed_paths_found: int = 0
    typed_path_completed: bool = False
    path_participant_count: int = 0
    filler_count: int = 0
    graph_stats: dict[str, int] = field(default_factory=dict)
    traversal: dict[str, int] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, Any]:
        """Distinguishes 'no path found' from 'too many competing paths'.

        Those two failures point in opposite directions: the first says the path
        representation is absent, the second says selection among competing
        graph paths is the real problem -- which would itself argue for G2.
        """
        return {
            "canonical_subject_found": int(self.canonical_subject_found),
            "target_relation_extracted": int(self.target_relation_extracted),
            "one_hop_bridge_candidates": self.one_hop_bridge_candidates,
            "closing_bridge_entities": self.closing_bridge_entities,
            "typed_paths_found": self.typed_paths_found,
            "typed_path_completed": int(self.typed_path_completed),
            "path_participant_count": self.path_participant_count,
            "filler_count": self.filler_count,
            "working_set_size": self.working_set_size,
            "records_examined": self.input_size,
            **{f"graph_{k}": v for k, v in self.graph_stats.items()},
            **{f"traversal_{k}": v for k, v in self.traversal.items()},
        }


def typed_path_prefilter(
    *, candidate_ids: Sequence[str], texts: Mapping[str, str],
    canonical_subject: str | None, relation: str, working_set_size: int,
    fusion_scores: Mapping[str, float] | None = None,
    graph: RuntimeGraph | None = None,
) -> TypedPathResult:
    """Compress to ``working_set_size`` by walking one frozen typed path.

    Steps, frozen: resolve subject -> parse relation -> find one-hop linked
    entities -> keep only bridges whose path closes on the relation -> admit
    tiers 0/1/2 -> fill by deterministic retrieval rank -> truncate to M.
    """
    scores = fusion_scores or {}
    pool = list(candidate_ids)
    rank_of = {record_id: index for index, record_id in enumerate(pool, 1)}
    graph = graph or build_runtime_graph(
        record_ids=pool, texts=texts, relation=relation)

    result = TypedPathResult(
        input_size=len(pool),
        canonical_subject_found=bool(canonical_subject),
        target_relation_extracted=bool(relation),
        graph_stats=graph.stats())

    subject = _norm(canonical_subject or "")
    subject_records = {rid for rid in pool
                       if subject and subject in _norm(texts.get(rid, ""))}

    # --- hop 1: entities visible alongside the subject --------------------
    one_hop: set[str] = set()
    for record_id in sorted(subject_records):
        for entity in graph.entities_by_record.get(record_id, frozenset()):
            if entity and entity != subject:
                one_hop.add(entity)
    for alias in graph.alias_links.get(subject, set()):
        for record_id in sorted(graph.records_by_entity.get(alias, set())):
            for entity in graph.entities_by_record.get(record_id, frozenset()):
                if entity and entity != subject:
                    one_hop.add(entity)
    result.one_hop_bridge_candidates = len(one_hop)

    # --- hop 2: keep ONLY bridges whose path closes on the relation -------
    edges_traversed = 0
    closing: set[str] = set()
    endpoint_records: set[str] = set()
    for entity in sorted(one_hop):
        for record_id in sorted(graph.records_by_entity.get(entity, set())):
            edges_traversed += 1
            if record_id in graph.relation_records and record_id not in subject_records:
                closing.add(entity)
                endpoint_records.add(record_id)
    result.closing_bridge_entities = len(closing)
    result.traversal = {"hops": min(2, MAX_HOPS), "edges_traversed": edges_traversed,
                        "nodes_visited": len(one_hop) + len(subject_records)}

    direct_records = {rid for rid in subject_records if rid in graph.relation_records}
    link_records = {
        rid for rid in subject_records
        if graph.entities_by_record.get(rid, frozenset()) & closing}

    result.typed_paths_found = len(endpoint_records) + len(direct_records)
    result.typed_path_completed = bool(endpoint_records or direct_records)

    tier_of: dict[str, int] = {}
    for record_id in pool:
        if record_id in direct_records:
            tier_of[record_id] = TIER_DIRECT
        elif record_id in endpoint_records:
            tier_of[record_id] = TIER_ENDPOINT
        elif record_id in link_records:
            tier_of[record_id] = TIER_LINK
        else:
            tier_of[record_id] = TIER_FILLER

    def sort_key(record_id: str) -> tuple:
        return (
            tier_of[record_id],
            -(scores.get(record_id) or 0.0),
            rank_of.get(record_id, 10 ** 9),
            record_id,
        )

    kept = sorted(pool, key=sort_key)[:working_set_size]
    counts: dict[int, int] = {}
    for record_id in kept:
        counts[tier_of[record_id]] = counts.get(tier_of[record_id], 0) + 1
    result.kept = kept
    result.working_set_size = len(kept)
    result.tier_counts = counts
    result.filler_count = counts.get(TIER_FILLER, 0)
    result.path_participant_count = len(kept) - result.filler_count
    return result
