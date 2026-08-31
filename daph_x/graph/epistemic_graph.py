"""Epistemic graph — primary state representation for DAPH-X.

The graph is the primary state representation. The canonical symbolic
topology is derived from the graph, not the other way around.

Graph structure:
  G = (V_H, V_E, V_C, V_S, E)

  V_H: hypothesis nodes
  V_E: evidence nodes
  V_C: claim nodes (derived assertions)
  V_S: source nodes (provenance)
  E: edges (supports, contradicts, depends_on, derived_from, tests, resolves)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class NodeType(str, Enum):
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    SOURCE = "source"
    SUBGOAL = "subgoal"


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    TESTS = "tests"
    RESOLVES = "resolves"
    RETRIEVED_BY = "retrieved_by"
    VERIFIED_BY = "verified_by"


@dataclass(frozen=True)
class EvidenceReliability:
    """Reliability attributes for a piece of evidence."""
    source_reliability: float = 1.0     # 0-1, how reliable is the source
    verification_confidence: float = 1.0  # 0-1, confidence in verification result
    independence_score: float = 1.0     # 0-1, how independent from other evidence
    ambiguity: float = 0.0              # 0-1, how ambiguous the evidence is
    freshness: float = 1.0              # 0-1, temporal freshness
    observation_noise: float = 0.0      # 0-1, observation noise


@dataclass(frozen=True)
class GraphNode:
    """A node in the epistemic graph."""
    node_id: str
    node_type: NodeType
    label: str = ""
    # For evidence nodes
    verification_state: str = "UNVERIFIED"
    temporal_status: str = "CURRENT"
    reliability: EvidenceReliability | None = None
    # For hypothesis nodes
    answer_action: str = ""
    # Provenance
    source_id: str | None = None
    derived_from: tuple[str, ...] = ()
    # Metadata
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """An edge in the epistemic graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpistemicGraph:
    """The epistemic graph — primary state representation for DAPH-X.

    This replaces the flat feature vector as the primary state.
    The canonical symbolic topology is derived from this graph.
    """
    nodes: Mapping[str, GraphNode]
    edges: tuple[GraphEdge, ...]
    # Resources
    steps_remaining: int = 10
    verify_remaining: int = 5
    retrieve_remaining: int = 3
    search_remaining: int = 3
    reasoning_tokens_remaining: int = 256
    elapsed_ms: int = 0
    max_elapsed_ms: int = 30000

    def graph_hash(self) -> str:
        """Compute a hash of the graph structure for provenance."""
        data = {
            "nodes": {k: {
                "type": v.node_type.value,
                "verification_state": v.verification_state,
                "temporal_status": v.temporal_status,
            } for k, v in sorted(self.nodes.items())},
            "edges": sorted(
                (e.source_id, e.target_id, e.edge_type.value)
                for e in self.edges
            ),
            "resources": {
                "steps_remaining": self.steps_remaining,
                "verify_remaining": self.verify_remaining,
                "retrieve_remaining": self.retrieve_remaining,
                "search_remaining": self.search_remaining,
            },
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

    def hypothesis_ids(self) -> list[str]:
        return [k for k, v in self.nodes.items() if v.node_type == NodeType.HYPOTHESIS]

    def evidence_ids(self) -> list[str]:
        return [k for k, v in self.nodes.items() if v.node_type == NodeType.EVIDENCE]

    def evidence_for_hypothesis(self, hyp_id: str) -> list[GraphNode]:
        """Get all evidence nodes connected to a hypothesis."""
        result = []
        for edge in self.edges:
            if edge.target_id == hyp_id or edge.source_id == hyp_id:
                for nid in (edge.source_id, edge.target_id):
                    if nid in self.nodes and self.nodes[nid].node_type == NodeType.EVIDENCE:
                        result.append(self.nodes[nid])
        return result

    def evidence_edges(self, evidence_id: str) -> list[GraphEdge]:
        """Get all edges involving an evidence node."""
        return [e for e in self.edges if e.source_id == evidence_id or e.target_id == evidence_id]

    def supports_hypothesis(self, evidence_id: str, hyp_id: str) -> bool:
        """Check if evidence supports a hypothesis."""
        for edge in self.edges:
            if (edge.source_id == evidence_id and edge.target_id == hyp_id
                    and edge.edge_type == EdgeType.SUPPORTS):
                return True
        return False

    def contradicts_hypothesis(self, evidence_id: str, hyp_id: str) -> bool:
        """Check if evidence contradicts a hypothesis."""
        for edge in self.edges:
            if (edge.source_id == evidence_id and edge.target_id == hyp_id
                    and edge.edge_type == EdgeType.CONTRADICTS):
                return True
        return False

    def to_legacy_evidence_items(self) -> list[dict]:
        """Convert to legacy EvidenceItem-compatible dicts for topology derivation."""
        items = []
        for nid, node in self.nodes.items():
            if node.node_type != NodeType.EVIDENCE:
                continue
            supports = [e.target_id for e in self.edges
                       if e.source_id == nid and e.edge_type == EdgeType.SUPPORTS]
            contradicts = [e.target_id for e in self.edges
                          if e.source_id == nid and e.edge_type == EdgeType.CONTRADICTS]
            items.append({
                "evidence_id": nid,
                "supports": tuple(supports),
                "contradicts": tuple(contradicts),
                "verification_state": node.verification_state,
                "temporal_status": node.temporal_status,
                "retrieved": True,
            })
        return items


def build_graph_from_evidence_task(task) -> EpistemicGraph:
    """Build an EpistemicGraph from a legacy EvidenceTask.

    This bridges the old task format to the new graph representation.
    """
    nodes = {}
    edges = []

    # Add hypothesis nodes
    for h in task.hypotheses:
        nodes[h.hypothesis_id] = GraphNode(
            node_id=h.hypothesis_id,
            node_type=NodeType.HYPOTHESIS,
            label=h.proposition,
            answer_action=h.answer_action.value,
        )

    # Add evidence nodes and edges
    for e in task.evidence_items:
        nodes[e.evidence_id] = GraphNode(
            node_id=e.evidence_id,
            node_type=NodeType.EVIDENCE,
            label=e.proposition,
            verification_state=e.verification_state.value,
            temporal_status=e.temporal_status.value,
            reliability=EvidenceReliability(
                source_reliability=1.0,  # Default: fully reliable in synthetic tasks
                verification_confidence=1.0 if e.verification_state.value != "UNVERIFIED" else 0.0,
                independence_score=1.0,  # Default: independent
                ambiguity=0.0,
                freshness=1.0 if e.temporal_status.value == "CURRENT" else 0.5,
                observation_noise=0.0,
            ),
        )
        for h_id in e.supports:
            edges.append(GraphEdge(
                source_id=e.evidence_id,
                target_id=h_id,
                edge_type=EdgeType.SUPPORTS,
            ))
        for h_id in e.contradicts:
            edges.append(GraphEdge(
                source_id=e.evidence_id,
                target_id=h_id,
                edge_type=EdgeType.CONTRADICTS,
            ))

    # Resources from task budget profile
    parts = task.budget_profile.split("_")
    steps = int(parts[1]) if len(parts) > 1 else 4
    verify = int(parts[2]) if len(parts) > 2 else 2
    search = int(parts[3]) if len(parts) > 3 else 0

    return EpistemicGraph(
        nodes=nodes,
        edges=tuple(edges),
        steps_remaining=steps,
        verify_remaining=verify,
        retrieve_remaining=0,
        search_remaining=search,
    )
