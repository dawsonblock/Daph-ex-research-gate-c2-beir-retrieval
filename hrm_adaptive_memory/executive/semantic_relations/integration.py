"""I3.12f: Integration of inferred semantic relations into EvidenceSnapshot.

This module provides the bridge between the semantic relation extractor
and the existing evidence benchmark pipeline.

The approach:
  1. Before build_evidence_snapshot, run the extractor on all visible
     evidence x hypothesis pairs.
  2. Replace oracle supports/contradicts with inferred ones.
  3. Build the snapshot as normal.

MDSG, T2, R1, A1, build_evidence_snapshot, and the executor are NOT
modified. Only the upstream evidence items change.

In S0 (oracle) condition, this step is skipped and the original
oracle supports/contradicts are used directly.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceRuntime, EvidenceSnapshot,
    build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.semantic_relations.extractor import (
    SemanticRelationExtractor,
)
from hrm_adaptive_memory.executive.semantic_relations.schema import (
    RelationGraph,
    RelationType,
)
from hrm_adaptive_memory.executive.semantic_relations.serializer import (
    relation_graph_to_supports_contradicts,
)


def infer_relations_for_runtime(
    runtime: EvidenceRuntime,
    extractor: SemanticRelationExtractor,
) -> tuple[EvidenceRuntime, RelationGraph]:
    """Run the extractor on all visible evidence and return a new runtime
    with inferred supports/contradicts.

    The original runtime is not modified. A new runtime is returned
    with EvidenceItem objects that have inferred supports/contradicts
    instead of oracle ones.

    Returns:
        (new_runtime, relation_graph)
    """
    task = runtime.task
    visible = runtime.visible_evidence

    # Build extractor inputs from visible evidence only
    evidence_items = [
        {"evidence_id": ev.evidence_id, "proposition": ev.proposition}
        for ev in visible
    ]
    hypotheses = [
        {"hypothesis_id": h.hypothesis_id, "proposition": h.proposition}
        for h in task.hypotheses
    ]

    # Extract the relation graph
    graph = extractor.extract_graph(
        task_id=task.task_id,
        evidence_items=evidence_items,
        hypotheses=hypotheses,
    )

    # Convert to supports/contradicts format
    inferred = relation_graph_to_supports_contradicts(graph)

    # Create new EvidenceItem objects with inferred relations
    new_evidence = []
    for ev in runtime.evidence:
        if ev.retrieved and ev.evidence_id in inferred:
            # Replace oracle supports/contradicts with inferred ones
            new_ev = replace(
                ev,
                supports=tuple(inferred[ev.evidence_id]["supports"]),
                contradicts=tuple(inferred[ev.evidence_id]["contradicts"]),
            )
        elif ev.retrieved:
            # Visible but no inferred relations (shouldn't happen, but handle it)
            new_ev = replace(ev, supports=(), contradicts=())
        else:
            # Hidden evidence - keep as is (not visible to controller)
            new_ev = ev
        new_evidence.append(new_ev)

    # Create new runtime with modified evidence
    new_runtime = replace(runtime, evidence=tuple(new_evidence))

    return new_runtime, graph


def build_evidence_snapshot_with_inferred_relations(
    runtime: EvidenceRuntime,
    extractor: SemanticRelationExtractor,
    *,
    prior_actions: tuple[str, ...] = (),
    prior_outcomes: tuple[str, ...] = (),
) -> tuple[EvidenceSnapshot, RelationGraph]:
    """Build a snapshot using inferred semantic relations.

    This is the S1 (raw proposition) condition. The extractor infers
    supports/contradicts from proposition text, and the snapshot is
    built with those inferred relations instead of oracle ones.

    Returns:
        (snapshot, relation_graph) - the relation graph is for provenance
    """
    new_runtime, graph = infer_relations_for_runtime(runtime, extractor)
    snapshot = build_evidence_snapshot(
        new_runtime,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )
    return snapshot, graph


def build_evidence_snapshot_oracle(
    runtime: EvidenceRuntime,
    *,
    prior_actions: tuple[str, ...] = (),
    prior_outcomes: tuple[str, ...] = (),
) -> EvidenceSnapshot:
    """Build a snapshot using oracle supports/contradicts.

    This is the S0 (oracle-structured) condition. It simply calls
    build_evidence_snapshot directly without any extraction.
    """
    return build_evidence_snapshot(
        runtime,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )
