"""Graph-structural feature extraction for DAPH-X M4.

Extracts meaningful structural features from the epistemic graph and
canonical topology, replacing the meaningless topo_hash_prefix feature.

All features are PRE-DECISION: they use only information available
before the action is executed. No post-action outcomes, utilities,
or oracle fields are used.

Feature groups:
  1. Topology counts: n_supported, n_contradicted, n_untested, etc.
  2. Graph structure: edge density, support/contradiction degree distributions
  3. Evidence state: verified/unverified/falsified counts, verify ratio
  4. Reliability summaries: mean source_reliability, mean independence
  5. Belief state: entropy, top-two margin, confidence
  6. Resource ratios: verify_remaining/n_ev, steps_remaining/max_steps
  7. Action-target relational: features specific to the action being evaluated
"""
from __future__ import annotations

import math
from typing import Any

from daph_x.graph.epistemic_graph import (
    EpistemicGraph, GraphNode, GraphEdge, NodeType, EdgeType, EvidenceReliability,
)


def extract_graph_features(
    graph: EpistemicGraph,
    topology: Any,  # HypothesisTopology
    belief: Any,    # BeliefState
    action_type: str = "",
    action_target: str = "",
) -> dict[str, float]:
    """Extract rich structural features from the epistemic graph.

    Args:
        graph: The epistemic graph (nodes, edges, resources).
        topology: Canonical HypothesisTopology from derive_hypothesis_topology().
        belief: BeliefState from compute_belief_state().
        action_type: The action being evaluated (ANSWER, VERIFY, DEFER, etc.).
        action_target: The target of the action (hypothesis ID, evidence ID, etc.).

    Returns:
        Dictionary of feature name → float value.
        All features are pre-decision (available before action execution).
    """
    feats: dict[str, float] = {}

    hyp_ids = graph.hypothesis_ids()
    ev_ids = graph.evidence_ids()
    n_hyp = len(hyp_ids)
    n_ev = len(ev_ids)

    # ── 1. Topology counts ──
    feats["topo_n_supported"] = float(topology.n_viable_hypotheses)
    feats["topo_n_contradicted"] = float(topology.n_eliminated_hypotheses)
    feats["topo_n_untested"] = float(topology.n_untested_hypotheses)
    feats["topo_n_weakened"] = float(topology.n_weakened_hypotheses)
    feats["topo_n_stale"] = float(topology.n_stale_hypotheses)
    feats["topo_n_mixed_verified"] = float(topology.n_hyp_with_mixed_verified)
    feats["topo_has_unique_supported"] = 1.0 if topology.has_unique_verified_supported else 0.0
    feats["topo_has_competition"] = 1.0 if topology.has_verified_unresolved_competition else 0.0
    feats["topo_verification_complete"] = 1.0 if topology.verification_complete else 0.0
    feats["topo_unverified_exists"] = 1.0 if topology.unverified_evidence_exists else 0.0

    # ── 2. Graph structure ──
    support_edges = [e for e in graph.edges if e.edge_type == EdgeType.SUPPORTS]
    contradict_edges = [e for e in graph.edges if e.edge_type == EdgeType.CONTRADICTS]
    n_support = len(support_edges)
    n_contradict = len(contradict_edges)
    n_total_edges = n_support + n_contradict

    # Edge density: fraction of possible evidence→hypothesis edges that exist
    max_possible = n_hyp * n_ev if n_hyp > 0 and n_ev > 0 else 1
    feats["graph_edge_density"] = float(n_total_edges) / max_possible
    feats["graph_support_edge_ratio"] = float(n_support) / max(n_total_edges, 1)
    feats["graph_contradict_edge_ratio"] = float(n_contradict) / max(n_total_edges, 1)

    # Support degree distribution (how many support edges per hypothesis)
    support_degree = {}
    contradict_degree = {}
    for h_id in hyp_ids:
        support_degree[h_id] = sum(1 for e in support_edges if e.target_id == h_id)
        contradict_degree[h_id] = sum(1 for e in contradict_edges if e.target_id == h_id)

    sd_vals = list(support_degree.values())
    cd_vals = list(contradict_degree.values())
    feats["graph_mean_support_degree"] = float(np_mean(sd_vals)) if sd_vals else 0.0
    feats["graph_std_support_degree"] = float(np_std(sd_vals)) if sd_vals else 0.0
    feats["graph_mean_contradict_degree"] = float(np_mean(cd_vals)) if cd_vals else 0.0
    feats["graph_std_contradict_degree"] = float(np_std(cd_vals)) if cd_vals else 0.0

    # Support entropy: how evenly is support distributed across hypotheses?
    if sd_vals and sum(sd_vals) > 0:
        p = [v / sum(sd_vals) for v in sd_vals]
        feats["graph_support_entropy"] = float(-sum(pi * math.log2(pi) for pi in p if pi > 0))
    else:
        feats["graph_support_entropy"] = 0.0

    # ── 3. Evidence state ──
    n_verified = 0
    n_unverified = 0
    n_falsified = 0
    n_stale_ev = 0
    for nid in ev_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        vs = node.verification_state
        if vs == "SUFFICIENT":
            n_verified += 1
        elif vs == "UNVERIFIED":
            n_unverified += 1
        elif vs == "FALSIFIED":
            n_falsified += 1
        elif vs == "STALE":
            n_stale_ev += 1

    feats["ev_n_verified"] = float(n_verified)
    feats["ev_n_unverified"] = float(n_unverified)
    feats["ev_n_falsified"] = float(n_falsified)
    feats["ev_n_stale"] = float(n_stale_ev)
    feats["ev_verify_ratio"] = float(n_verified) / max(n_ev, 1)
    feats["ev_unverified_ratio"] = float(n_unverified) / max(n_ev, 1)

    # ── 4. Reliability summaries ──
    reliabilities = []
    independence_scores = []
    for nid in ev_ids:
        node = graph.nodes.get(nid)
        if node and node.reliability:
            reliabilities.append(node.reliability.source_reliability)
            independence_scores.append(node.reliability.independence_score)

    feats["ev_mean_source_reliability"] = float(np_mean(reliabilities)) if reliabilities else 1.0
    feats["ev_mean_independence"] = float(np_mean(independence_scores)) if independence_scores else 1.0
    feats["ev_std_source_reliability"] = float(np_std(reliabilities)) if reliabilities else 0.0

    # ── 5. Belief state ──
    if belief and hasattr(belief, "probabilities") and belief.probabilities:
        probs = sorted(belief.probabilities.values(), reverse=True)
        feats["belief_entropy"] = float(belief.entropy)
        feats["belief_confidence"] = float(belief.confidence())
        # Top-two margin
        if len(probs) >= 2:
            feats["belief_top_two_margin"] = float(probs[0] - probs[1])
        else:
            feats["belief_top_two_margin"] = float(probs[0]) if probs else 0.0
        # Normalized entropy (0 = certain, 1 = uniform)
        max_entropy = math.log2(n_hyp) if n_hyp > 1 else 1.0
        feats["belief_normalized_entropy"] = float(belief.entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        feats["belief_entropy"] = 0.0
        feats["belief_confidence"] = 0.0
        feats["belief_top_two_margin"] = 0.0
        feats["belief_normalized_entropy"] = 0.0

    # ── 6. Resource ratios ──
    feats["resource_verify_ratio"] = float(graph.verify_remaining) / max(n_ev, 1)
    feats["resource_steps_ratio"] = float(graph.steps_remaining) / 10.0  # max_steps=10 default
    feats["resource_search_ratio"] = float(graph.search_remaining) / max(n_ev, 1)
    feats["resource_verify_per_hyp"] = float(graph.verify_remaining) / max(n_hyp, 1)

    # ── 7. Action-target relational features ──
    # These describe the relationship between the action and the graph structure
    if action_type == "VERIFY" and action_target:
        _add_verify_target_features(feats, graph, action_target, hyp_ids)
    elif action_type == "ANSWER" and action_target:
        _add_answer_target_features(feats, graph, topology, belief, action_target)
    elif action_type == "COMPARE":
        _add_compare_features(feats, graph, topology)
    elif action_type == "DEFER":
        # DEFER features: describe the state that makes deferring reasonable
        feats["defer_n_competing"] = float(topology.n_viable_hypotheses)
        feats["defer_has_unverified"] = feats["topo_unverified_exists"]
        feats["defer_verify_available"] = 1.0 if graph.verify_remaining > 0 else 0.0
    elif action_type == "STOP":
        feats["stop_steps_remaining"] = float(graph.steps_remaining)
    else:
        # Default: no action-specific features
        pass

    # Ensure action-specific features exist for all action types (fill with 0)
    _ensure_action_features(feats, action_type)

    return feats


def _add_verify_target_features(
    feats: dict[str, float],
    graph: EpistemicGraph,
    target: str,
    hyp_ids: list[str],
) -> None:
    """Add features specific to VERIFY(target) actions."""
    node = graph.nodes.get(target)
    if node is None:
        feats["verify_target_exists"] = 0.0
        feats["verify_target_n_edges"] = 0.0
        feats["verify_target_is_unverified"] = 0.0
        feats["verify_target_discriminates"] = 0.0
        feats["verify_target_n_supports"] = 0.0
        feats["verify_target_n_contradicts"] = 0.0
        return

    feats["verify_target_exists"] = 1.0
    feats["verify_target_is_unverified"] = 1.0 if node.verification_state == "UNVERIFIED" else 0.0

    edges = graph.evidence_edges(target)
    n_edges = len(edges)
    feats["verify_target_n_edges"] = float(n_edges)

    n_supports = sum(1 for e in edges if e.edge_type == EdgeType.SUPPORTS)
    n_contradicts = sum(1 for e in edges if e.edge_type == EdgeType.CONTRADICTS)
    feats["verify_target_n_supports"] = float(n_supports)
    feats["verify_target_n_contradicts"] = float(n_contradicts)

    # Does this evidence discriminate between hypotheses?
    # Discriminating evidence connects to multiple hypotheses (or to a
    # SUPPORTED/CONTRADICTED hypothesis, changing the topology if verified)
    connected_hyps = set()
    for e in edges:
        if e.source_id == target:
            connected_hyps.add(e.target_id)
        elif e.target_id == target:
            connected_hyps.add(e.source_id)
    feats["verify_target_discriminates"] = 1.0 if len(connected_hyps) > 1 else 0.0
    feats["verify_target_n_connected_hyps"] = float(len(connected_hyps))

    # Reliability of the target evidence
    if node.reliability:
        feats["verify_target_reliability"] = float(node.reliability.source_reliability)
        feats["verify_target_independence"] = float(node.reliability.independence_score)
    else:
        feats["verify_target_reliability"] = 1.0
        feats["verify_target_independence"] = 1.0


def _add_answer_target_features(
    feats: dict[str, float],
    graph: EpistemicGraph,
    topology: Any,
    belief: Any,
    target: str,
) -> None:
    """Add features specific to ANSWER(target) actions."""
    # Is the target the unique supported hypothesis?
    feats["answer_is_unique_supported"] = 1.0 if (
        topology.unique_supported_hypothesis == target
    ) else 0.0

    # Support/contradiction counts for the target hypothesis
    vs = topology.verified_support_by_hypothesis.get(target, ())
    vc = topology.verified_contradiction_by_hypothesis.get(target, ())
    feats["answer_target_n_verified_support"] = float(len(vs))
    feats["answer_target_n_verified_contradict"] = float(len(vc))

    # Is the target contradicted?
    feats["answer_target_is_contradicted"] = 1.0 if len(vc) > 0 else 0.0

    # Belief probability for the target
    if belief and hasattr(belief, "probabilities"):
        feats["answer_target_belief"] = float(belief.probabilities.get(target, 0.0))
    else:
        feats["answer_target_belief"] = 0.0

    # Number of competing supported hypotheses
    supported = [h for h, s in topology.hypothesis_states.items()
                 if s.value == "SUPPORTED" and h != target]
    feats["answer_n_competing_supported"] = float(len(supported))


def _add_compare_features(
    feats: dict[str, float],
    graph: EpistemicGraph,
    topology: Any,
) -> None:
    """Add features specific to COMPARE actions."""
    feats["compare_n_supported"] = float(topology.n_viable_hypotheses)
    feats["compare_n_contradicted"] = float(topology.n_eliminated_hypotheses)
    feats["compare_has_competition"] = 1.0 if topology.has_verified_unresolved_competition else 0.0


def _ensure_action_features(feats: dict[str, float], action_type: str) -> None:
    """Ensure all action-specific features exist for all action types.

    This prevents missing-key errors when building feature matrices.
    Features not relevant to the current action are set to 0.0.
    """
    all_action_features = [
        # VERIFY
        "verify_target_exists", "verify_target_n_edges", "verify_target_is_unverified",
        "verify_target_discriminates", "verify_target_n_supports", "verify_target_n_contradicts",
        "verify_target_n_connected_hyps", "verify_target_reliability", "verify_target_independence",
        # ANSWER
        "answer_is_unique_supported", "answer_target_n_verified_support",
        "answer_target_n_verified_contradict", "answer_target_is_contradicted",
        "answer_target_belief", "answer_n_competing_supported",
        # COMPARE
        "compare_n_supported", "compare_n_contradicted", "compare_has_competition",
        # DEFER
        "defer_n_competing", "defer_has_unverified", "defer_verify_available",
        # STOP
        "stop_steps_remaining",
    ]
    for k in all_action_features:
        if k not in feats:
            feats[k] = 0.0


def np_mean(vals: list[float]) -> float:
    """Simple mean without numpy dependency."""
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def np_std(vals: list[float]) -> float:
    """Simple std without numpy dependency."""
    if len(vals) < 2:
        return 0.0
    m = np_mean(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)
