"""Serializer for semantic relation graphs.

Serializes the inferred relation graph into the format that
EvidenceSnapshot expects (supports/contradicts fields).
"""
from __future__ import annotations

from hrm_adaptive_memory.executive.semantic_relations.schema import (
    RelationGraph,
    RelationType,
)


def relation_graph_to_supports_contradicts(
    graph: RelationGraph,
) -> dict[str, dict[str, list[str]]]:
    """Convert a RelationGraph to per-evidence supports/contradicts lists.

    Returns:
        {evidence_id: {"supports": [h_id, ...], "contradicts": [h_id, ...]}}
    """
    result: dict[str, dict[str, list[str]]] = {}
    for rel in graph.relations:
        if rel.evidence_id not in result:
            result[rel.evidence_id] = {"supports": [], "contradicts": []}
        if rel.relation is RelationType.SUPPORT:
            result[rel.evidence_id]["supports"].append(rel.hypothesis_id)
        elif rel.relation is RelationType.CONTRADICT:
            result[rel.evidence_id]["contradicts"].append(rel.hypothesis_id)
    return result


def relation_graph_to_dict(graph: RelationGraph) -> dict:
    """Serialize a RelationGraph to a JSON-serializable dict."""
    return graph.as_dict()
