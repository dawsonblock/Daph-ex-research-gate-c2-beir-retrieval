"""Candidate action generator for DAPH-X.

Given the epistemic graph and resources, enumerate plausible legal actions.
Each action is parameterized by its target.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.actions.typed_actions import (
    Action, ActionType, answer, defer, verify, retrieve, search,
    test, compare, check_consistency, stop,
)
from daph_x.graph.epistemic_graph import EpistemicGraph, NodeType, EdgeType


def generate_candidates(graph: EpistemicGraph) -> list[Action]:
    """Generate all plausible legal actions from the current state.

    Returns a list of parameterized Action objects.
    """
    candidates = []

    # Import canonical topology for state classification
    from daph.epistemic.topology import derive_hypothesis_topology
    from daph.epistemic.types import TerminalReadiness

    hypothesis_ids = graph.hypothesis_ids()
    evidence_items = graph.to_legacy_evidence_items()

    topology = derive_hypothesis_topology(
        evidence_items=evidence_items,
        hypothesis_ids=hypothesis_ids,
    )

    # Determine action admissibility
    can_verify = graph.verify_remaining > 0 and topology.unverified_evidence_exists
    can_retrieve = graph.retrieve_remaining > 0
    can_search = graph.search_remaining > 0

    # ANSWER(h) — one candidate per SUPPORTED hypothesis with ANSWER action
    if topology.unique_supported_hypothesis:
        hyp_id = topology.unique_supported_hypothesis
        hyp_node = graph.nodes.get(hyp_id)
        if hyp_node and hyp_node.answer_action == "ANSWER":
            candidates.append(answer(hyp_id))

    # DEFER — always available as terminal
    candidates.append(defer("no_unique_justified_hypothesis"))

    # VERIFY(e) — one candidate per unverified evidence item
    if can_verify:
        for nid, node in graph.nodes.items():
            if (node.node_type == NodeType.EVIDENCE
                    and node.verification_state == "UNVERIFIED"):
                candidates.append(verify(nid))

    # RETRIEVE — placeholder (no hidden evidence in synthetic tasks yet)
    # if can_retrieve:
    #     candidates.append(retrieve(query="...", source_scope="hidden"))

    # SEARCH — placeholder
    # if can_search:
    #     candidates.append(search(query="...", source_scope="all"))

    # COMPARE(h1, h2) — one candidate per pair of viable hypotheses
    supported = [h for h, s in topology.hypothesis_states.items()
                 if s.value == "SUPPORTED"]
    if len(supported) >= 2:
        for i in range(len(supported)):
            for j in range(i + 1, len(supported)):
                candidates.append(compare(supported[i], supported[j]))

    # STOP — always available
    candidates.append(stop("executive_stop"))

    return candidates


def prune_candidates(candidates: list[Action], graph: EpistemicGraph) -> list[Action]:
    """Remove illegal, dominated, or redundant actions."""
    pruned = []
    seen = set()

    for action in candidates:
        # Deduplicate
        key = (action.action_type, action.target)
        if key in seen:
            continue
        seen.add(key)

        # Check resource constraints
        if action.action_type == ActionType.VERIFY and graph.verify_remaining <= 0:
            continue
        if action.action_type == ActionType.RETRIEVE and graph.retrieve_remaining <= 0:
            continue
        if action.action_type == ActionType.SEARCH and graph.search_remaining <= 0:
            continue

        pruned.append(action)

    return pruned


def generate_and_prune(graph: EpistemicGraph) -> list[Action]:
    """Generate and prune candidate actions."""
    candidates = generate_candidates(graph)
    return prune_candidates(candidates, graph)
