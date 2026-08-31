"""World model for DAPH-X — P(o|s,a) transition layer.

Symbolic + empirical transition model. Where transitions are
deterministic, encode them exactly. Where outcomes are uncertain,
estimate empirical outcome distributions.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.actions.typed_actions import Action, ActionType
from daph_x.graph.epistemic_graph import (
    EpistemicGraph, GraphNode, GraphEdge, NodeType, EdgeType,
    EvidenceReliability,
)


class ObservationOutcome(str, Enum):
    """Possible outcomes of an action."""
    SUFFICIENT = "SUFFICIENT"
    FALSIFIED = "FALSIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NEW_EVIDENCE_FOUND = "NEW_EVIDENCE_FOUND"
    NO_NEW_EVIDENCE = "NO_NEW_EVIDENCE"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class Transition:
    """A possible transition: (outcome, probability, next_graph)."""
    outcome: ObservationOutcome
    probability: float
    next_graph: EpistemicGraph


def transition_model(graph: EpistemicGraph, action: Action) -> list[Transition]:
    """Compute possible transitions for an action.

    Returns a list of (outcome, probability, next_graph) tuples.
    Probabilities sum to 1.0.
    """
    if action.action_type == ActionType.ANSWER:
        return [Transition(
            outcome=ObservationOutcome.TERMINAL,
            probability=1.0,
            next_graph=graph,
        )]

    if action.action_type == ActionType.DEFER:
        return [Transition(
            outcome=ObservationOutcome.TERMINAL,
            probability=1.0,
            next_graph=graph,
        )]

    if action.action_type == ActionType.STOP:
        return [Transition(
            outcome=ObservationOutcome.TERMINAL,
            probability=1.0,
            next_graph=graph,
        )]

    if action.action_type == ActionType.VERIFY:
        return _verify_transitions(graph, action)

    if action.action_type == ActionType.SEARCH:
        return _search_transitions(graph, action)

    if action.action_type == ActionType.RETRIEVE:
        return _retrieve_transitions(graph, action)

    if action.action_type == ActionType.COMPARE:
        return [Transition(
            outcome=ObservationOutcome.NEW_EVIDENCE_FOUND,
            probability=1.0,
            next_graph=graph,  # COMPARE doesn't change evidence
        )]

    if action.action_type == ActionType.CHECK_CONSISTENCY:
        return [Transition(
            outcome=ObservationOutcome.NEW_EVIDENCE_FOUND,
            probability=1.0,
            next_graph=graph,
        )]

    # Default: no state change
    return [Transition(
        outcome=ObservationOutcome.INCONCLUSIVE,
        probability=1.0,
        next_graph=graph,
    )]


def _verify_transitions(graph: EpistemicGraph, action: Action) -> list[Transition]:
    """Compute transitions for VERIFY(e).

    The evidence node e has a verify_result that determines the outcome.
    In synthetic tasks, verify_result is set. In real tasks, it would
    be estimated from empirical frequencies.
    """
    evidence_id = action.target
    if not isinstance(evidence_id, str):
        return []

    node = graph.nodes.get(evidence_id)
    if not node or node.node_type != NodeType.EVIDENCE:
        return []

    # Check if we have a ground-truth verify_result (synthetic tasks)
    # In the graph, verify_result isn't stored directly — we infer from
    # the verification_state. For UNVERIFIED evidence, we need to model
    # possible outcomes.

    # For now, use a simple model:
    # - If the evidence supports a hypothesis: 70% SUFFICIENT, 20% FALSIFIED, 10% INCONCLUSIVE
    # - If the evidence contradicts a hypothesis: 70% SUFFICIENT, 20% FALSIFIED, 10% INCONCLUSIVE
    # These are placeholder frequencies — should be calibrated from data.

    # Determine what verification would produce
    transitions = []

    # Outcome 1: SUFFICIENT (verification succeeds)
    next_graph_sufficient = _apply_verify_outcome(graph, evidence_id, "SUFFICIENT")
    transitions.append(Transition(
        outcome=ObservationOutcome.SUFFICIENT,
        probability=0.7,  # Placeholder — should be calibrated
        next_graph=next_graph_sufficient,
    ))

    # Outcome 2: FALSIFIED (verification fails)
    next_graph_falsified = _apply_verify_outcome(graph, evidence_id, "FALSIFIED")
    transitions.append(Transition(
        outcome=ObservationOutcome.FALSIFIED,
        probability=0.2,  # Placeholder
        next_graph=next_graph_falsified,
    ))

    # Outcome 3: INCONCLUSIVE
    next_graph_inconclusive = _apply_verify_outcome(graph, evidence_id, "INCONCLUSIVE")
    transitions.append(Transition(
        outcome=ObservationOutcome.INCONCLUSIVE,
        probability=0.1,  # Placeholder
        next_graph=next_graph_inconclusive,
    ))

    return transitions


def _apply_verify_outcome(
    graph: EpistemicGraph, evidence_id: str, result: str,
) -> EpistemicGraph:
    """Apply a verification outcome to the graph."""
    new_nodes = dict(graph.nodes)
    node = new_nodes.get(evidence_id)
    if node is None:
        return graph

    if result == "SUFFICIENT":
        new_state = "SUFFICIENT"
    elif result == "FALSIFIED":
        new_state = "FALSIFIED"
    else:
        new_state = "UNVERIFIED"  # INCONCLUSIVE stays unverified

    new_nodes[evidence_id] = GraphNode(
        node_id=node.node_id,
        node_type=node.node_type,
        label=node.label,
        verification_state=new_state,
        temporal_status=node.temporal_status,
        reliability=node.reliability,
        answer_action=node.answer_action,
        source_id=node.source_id,
        derived_from=node.derived_from,
        metadata=node.metadata,
    )

    return EpistemicGraph(
        nodes=new_nodes,
        edges=graph.edges,
        steps_remaining=max(0, graph.steps_remaining - 1),
        verify_remaining=max(0, graph.verify_remaining - 1),
        retrieve_remaining=graph.retrieve_remaining,
        search_remaining=graph.search_remaining,
        reasoning_tokens_remaining=graph.reasoning_tokens_remaining,
        elapsed_ms=graph.elapsed_ms,
        max_elapsed_ms=graph.max_elapsed_ms,
    )


def _search_transitions(graph: EpistemicGraph, action: Action) -> list[Transition]:
    """Compute transitions for SEARCH."""
    next_graph = EpistemicGraph(
        nodes=graph.nodes,
        edges=graph.edges,
        steps_remaining=max(0, graph.steps_remaining - 1),
        verify_remaining=graph.verify_remaining,
        retrieve_remaining=graph.retrieve_remaining,
        search_remaining=max(0, graph.search_remaining - 1),
        reasoning_tokens_remaining=graph.reasoning_tokens_remaining,
        elapsed_ms=graph.elapsed_ms,
        max_elapsed_ms=graph.max_elapsed_ms,
    )
    return [Transition(
        outcome=ObservationOutcome.NEW_EVIDENCE_FOUND,
        probability=0.5,  # Placeholder
        next_graph=next_graph,
    ), Transition(
        outcome=ObservationOutcome.NO_NEW_EVIDENCE,
        probability=0.5,  # Placeholder
        next_graph=next_graph,
    )]


def _retrieve_transitions(graph: EpistemicGraph, action: Action) -> list[Transition]:
    """Compute transitions for RETRIEVE."""
    next_graph = EpistemicGraph(
        nodes=graph.nodes,
        edges=graph.edges,
        steps_remaining=max(0, graph.steps_remaining - 1),
        verify_remaining=graph.verify_remaining,
        retrieve_remaining=max(0, graph.retrieve_remaining - 1),
        search_remaining=graph.search_remaining,
        reasoning_tokens_remaining=graph.reasoning_tokens_remaining,
        elapsed_ms=graph.elapsed_ms,
        max_elapsed_ms=graph.max_elapsed_ms,
    )
    return [Transition(
        outcome=ObservationOutcome.NEW_EVIDENCE_FOUND,
        probability=0.5,  # Placeholder
        next_graph=next_graph,
    ), Transition(
        outcome=ObservationOutcome.NO_NEW_EVIDENCE,
        probability=0.5,
        next_graph=next_graph,
    )]
