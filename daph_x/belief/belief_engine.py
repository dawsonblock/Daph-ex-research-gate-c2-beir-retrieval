"""Belief engine for DAPH-X.

Maintains a calibrated distribution P(H_i|E) over hypotheses.

Uses canonical topology + evidence reliability + provenance to derive
hypothesis probabilities. Does NOT ask the LLM for raw probabilities.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph.epistemic.topology import derive_hypothesis_topology
from daph.epistemic.types import HypothesisState, HypothesisTopology, TerminalReadiness
from daph_x.graph.epistemic_graph import EpistemicGraph, NodeType


@dataclass(frozen=True)
class BeliefState:
    """Calibrated belief state over hypotheses."""
    probabilities: Mapping[str, float]  # P(H_i|E)
    entropy: float                      # H(H|E)
    unique_supported: str | None        # Canonical unique supported hypothesis
    readiness: TerminalReadiness
    n_supported: int
    n_contradicted: int
    n_weakened: int
    n_untested: int

    def top_hypothesis(self) -> tuple[str, float] | None:
        """Return the highest-probability hypothesis."""
        if not self.probabilities:
            return None
        best = max(self.probabilities.items(), key=lambda x: x[1])
        return best

    def confidence(self) -> float:
        """Confidence in the top hypothesis."""
        top = self.top_hypothesis()
        return top[1] if top else 0.0


def compute_belief_state(graph: EpistemicGraph) -> BeliefState:
    """Compute calibrated belief state from the epistemic graph.

    Uses canonical topology for structural classification, then derives
    probabilities from evidence reliability and verification status.
    """
    hypothesis_ids = graph.hypothesis_ids()
    evidence_items = graph.to_legacy_evidence_items()

    topology = derive_hypothesis_topology(
        evidence_items=evidence_items,
        hypothesis_ids=hypothesis_ids,
    )

    # Derive probabilities from topology + evidence reliability
    scores = {}
    for h_id in hypothesis_ids:
        state = topology.hypothesis_states[h_id]

        if state == HypothesisState.CONTRADICTED:
            scores[h_id] = 0.01  # Nearly eliminated
        elif state == HypothesisState.SUPPORTED:
            # Score based on evidence quality
            support_evidence = topology.verified_support_by_hypothesis.get(h_id, ())
            evidence_quality = 1.0
            for ev_id in support_evidence:
                ev_node = graph.nodes.get(ev_id)
                if ev_node and ev_node.reliability:
                    evidence_quality *= ev_node.reliability.source_reliability
            scores[h_id] = 1.0 * evidence_quality
        elif state == HypothesisState.WEAKENED:
            scores[h_id] = 0.3  # Support failed, but not eliminated
        elif state == HypothesisState.UNTESTED:
            scores[h_id] = 0.1  # No evidence either way
        else:  # STALE
            scores[h_id] = 0.05  # Stale evidence

    # Normalize to probabilities
    total = sum(scores.values())
    if total > 0:
        probabilities = {h: s / total for h, s in scores.items()}
    else:
        # Uniform prior if no evidence
        probabilities = {h: 1.0 / len(hypothesis_ids) for h in hypothesis_ids}

    # Compute entropy
    entropy = 0.0
    for p in probabilities.values():
        if p > 0:
            entropy -= p * math.log2(p)

    # Determine readiness from canonical topology
    from daph.epistemic.topology import classify_terminal_readiness
    readiness = classify_terminal_readiness(
        topology,
        can_verify=graph.verify_remaining > 0 and topology.unverified_evidence_exists,
        can_retrieve=graph.retrieve_remaining > 0,
        can_search=graph.search_remaining > 0,
        has_unverified_discriminating_evidence=topology.unverified_evidence_exists,
        has_hidden_evidence=False,
        search_could_discriminate=graph.search_remaining > 0,
    )

    return BeliefState(
        probabilities=probabilities,
        entropy=entropy,
        unique_supported=topology.unique_supported_hypothesis,
        readiness=readiness,
        n_supported=topology.n_viable_hypotheses,
        n_contradicted=topology.n_eliminated_hypotheses,
        n_weakened=topology.n_weakened_hypotheses,
        n_untested=topology.n_untested_hypotheses,
    )
