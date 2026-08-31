"""Three-level novelty signatures for DAPH-X M4.

Distinguishes:
  S_exact     — exact decision-state structure (ID-invariant)
  S_family    — topology/motif family (coarser than exact)
  S_mechanism — causal reason an action succeeds or fails

These signatures enforce disjoint splits at multiple granularity levels.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from daph_x.graph.epistemic_graph import (
    EpistemicGraph, GraphNode, GraphEdge, NodeType, EdgeType,
)


@dataclass(frozen=True)
class NoveltySignatures:
    """Three-level novelty signature for a generated state."""
    exact: str       # ID-invariant exact decision-state structure
    family: str      # Topology/motif family (coarser)
    mechanism: str   # Causal reason an action succeeds or fails

    def to_dict(self) -> dict:
        return {
            "exact": self.exact,
            "family": self.family,
            "mechanism": self.mechanism,
        }


def compute_exact_signature(graph: EpistemicGraph, resources: dict | None = None) -> str:
    """S_exact: ID-invariant exact decision-state structure.

    Replaces hypothesis IDs with H0, H1, ... and evidence IDs with
    E0, E1, ... in canonical order. Includes verification state,
    temporal status, edge types, and resource budgets.

    Two states with the same S_exact must produce identical decision
    landscapes under ID renaming.
    """
    hyp_ids = sorted([k for k, v in graph.nodes.items()
                      if v.node_type == NodeType.HYPOTHESIS])
    ev_ids = sorted([k for k, v in graph.nodes.items()
                     if v.node_type == NodeType.EVIDENCE])
    id_map = {}
    for i, h in enumerate(hyp_ids):
        id_map[h] = f"H{i}"
    for i, e in enumerate(ev_ids):
        id_map[e] = f"E{i}"

    nodes = {}
    for k, v in sorted(graph.nodes.items()):
        remapped = id_map.get(k, k)
        node_data = {
            "type": v.node_type.value,
            "vstate": v.verification_state,
            "tstatus": v.temporal_status,
        }
        if v.reliability is not None:
            # Bucket reliability to 2 decimal places for exact signature
            node_data["rel"] = {
                "sr": round(v.reliability.source_reliability, 2),
                "vc": round(v.reliability.verification_confidence, 2),
                "ind": round(v.reliability.independence_score, 2),
                "amb": round(v.reliability.ambiguity, 2),
                "fr": round(v.reliability.freshness, 2),
                "noise": round(v.reliability.observation_noise, 2),
            }
        nodes[remapped] = node_data

    edges = sorted(
        (id_map.get(e.source_id, e.source_id),
         id_map.get(e.target_id, e.target_id),
         e.edge_type.value)
        for e in graph.edges
    )

    res = resources or {
        "steps": graph.steps_remaining,
        "verify": graph.verify_remaining,
        "retrieve": graph.retrieve_remaining,
        "search": graph.search_remaining,
    }

    data = {"nodes": nodes, "edges": edges, "resources": res}
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def compute_family_signature(graph: EpistemicGraph) -> str:
    """S_family: topology/motif family (coarser than exact).

    Captures the decision-relevant graph motif without specific counts
    or resource budgets. Groups states that share the same abstract
    decision structure even if they differ in size or resources.

    Family dimensions:
      - n_hypotheses bucket (small=2-3, medium=4-6, large=7+)
      - n_evidence bucket
      - support pattern (unique_support, competing_support, no_support, mixed)
      - verification pattern (all_verified, some_verified, none_verified)
      - edge density bucket (sparse, medium, dense)
    """
    hyp_ids = [k for k, v in graph.nodes.items()
               if v.node_type == NodeType.HYPOTHESIS]
    ev_ids = [k for k, v in graph.nodes.items()
              if v.node_type == NodeType.EVIDENCE]
    n_hyp = len(hyp_ids)
    n_ev = len(ev_ids)

    # Buckets
    hyp_bucket = "small" if n_hyp <= 3 else ("medium" if n_hyp <= 6 else "large")
    ev_bucket = "small" if n_ev <= 3 else ("medium" if n_ev <= 6 else "large")

    # Support pattern
    n_support_edges = sum(1 for e in graph.edges if e.edge_type == EdgeType.SUPPORTS)
    n_contradict_edges = sum(1 for e in graph.edges if e.edge_type == EdgeType.CONTRADICTS)

    # Count supported hypotheses using CANONICAL topology
    # (not manual reconstruction — avoids FALSIFIED polarity bug)
    from daph.epistemic.topology import derive_hypothesis_topology
    from daph.epistemic.types import HypothesisState

    evidence_items = graph.to_legacy_evidence_items()
    topology = derive_hypothesis_topology(
        evidence_items=evidence_items,
        hypothesis_ids=hyp_ids,
    )
    supported_hyps = {
        h for h, state in topology.hypothesis_states.items()
        if state == HypothesisState.SUPPORTED
    }

    if len(supported_hyps) == 0:
        support_pattern = "no_support"
    elif len(supported_hyps) == 1:
        support_pattern = "unique_support"
    else:
        support_pattern = "competing_support"

    # Verification pattern
    n_verified = sum(1 for k, v in graph.nodes.items()
                     if v.node_type == NodeType.EVIDENCE
                     and v.verification_state != "UNVERIFIED")
    if n_verified == 0:
        verify_pattern = "none_verified"
    elif n_verified == n_ev:
        verify_pattern = "all_verified"
    else:
        verify_pattern = "some_verified"

    # Edge density
    total_possible = n_hyp * n_ev
    actual_edges = n_support_edges + n_contradict_edges
    if total_possible == 0:
        density_bucket = "empty"
    else:
        density = actual_edges / total_possible
        density_bucket = "sparse" if density < 0.4 else ("medium" if density < 0.7 else "dense")

    data = {
        "hyp_bucket": hyp_bucket,
        "ev_bucket": ev_bucket,
        "support_pattern": support_pattern,
        "verify_pattern": verify_pattern,
        "density": density_bucket,
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


def compute_mechanism_signature(
    graph: EpistemicGraph,
    correct_hypothesis_id: str,
    harm_mechanism: str,
) -> str:
    """S_mechanism: causal reason an action succeeds or fails.

    Identifies the underlying causal mechanism that determines whether
    an intervention is beneficial or harmful. This is held out separately
    from structure to test mechanism-level generalization.

    Mechanism types:
      - misleading_support      — verified evidence supports wrong hypothesis
      - bad_verify_target       — VERIFY picks irrelevant or misleading evidence
      - resource_depletion      — intervention wastes scarce resources
      - weak_evidence_dependence — evidence sources are correlated, belief overconfident
      - near_value_inversion    — ΔU is small, easy to get wrong direction
      - world_model_error       — transition probabilities are misleading
      - belief_overconfidence   — belief concentrates too hard on wrong hypothesis
      - novel_topology          — graph motif not seen in training
      - correct_clear           — no harm mechanism, intervention is clearly beneficial
    """
    data = {
        "mechanism": harm_mechanism,
        "correct_hyp_position": _hyp_position(graph, correct_hypothesis_id),
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


def _hyp_position(graph: EpistemicGraph, correct_hyp_id: str) -> str:
    """Classify the position of the correct hypothesis in the evidence landscape."""
    hyp_ids = sorted([k for k, v in graph.nodes.items()
                      if v.node_type == NodeType.HYPOTHESIS])

    # Count verified support/contradiction for the correct hypothesis
    n_support_correct = 0
    n_contradict_correct = 0
    n_support_other = 0
    n_contradict_other = 0

    for e in graph.edges:
        ev_node = graph.nodes.get(e.source_id)
        if not ev_node or ev_node.node_type != NodeType.EVIDENCE:
            continue
        if ev_node.verification_state == "UNVERIFIED":
            continue

        if e.edge_type == EdgeType.SUPPORTS:
            if e.target_id == correct_hyp_id:
                n_support_correct += 1
            else:
                n_support_other += 1
        elif e.edge_type == EdgeType.CONTRADICTS:
            if e.target_id == correct_hyp_id:
                n_contradict_correct += 1
            else:
                n_contradict_other += 1

    if n_support_correct > 0 and n_support_other == 0:
        return "correct_uniquely_supported"
    elif n_support_correct > 0 and n_support_other > 0:
        return "correct_competing"
    elif n_support_correct == 0 and n_support_other > 0:
        return "correct_unsupported_others_supported"
    elif n_contradict_correct > 0:
        return "correct_contradicted"
    else:
        return "correct_untested"


def compute_all_signatures(
    graph: EpistemicGraph,
    correct_hypothesis_id: str,
    harm_mechanism: str,
    resources: dict | None = None,
) -> NoveltySignatures:
    """Compute all three novelty signatures."""
    return NoveltySignatures(
        exact=compute_exact_signature(graph, resources),
        family=compute_family_signature(graph),
        mechanism=compute_mechanism_signature(graph, correct_hypothesis_id, harm_mechanism),
    )
