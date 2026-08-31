"""Tests proving every action type executes without runtime errors.

This test was added after the COMPARE bug (ObservationOutcome.NEW_EVIDENCE
did not exist) was found to silently corrupt the causal corpus by raising
AttributeError, which was caught and turned into ERROR results with utility=0.

Every action type must produce valid transitions from a representative graph.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.actions.typed_actions import Action, ActionType
from daph_x.graph.epistemic_graph import (
    EpistemicGraph, GraphNode, GraphEdge, NodeType, EdgeType, EvidenceReliability,
)
from daph_x.world_model.transition_model import transition_model, ObservationOutcome


def _make_test_graph() -> EpistemicGraph:
    """Build a simple test graph with 2 hypotheses and 2 evidence items."""
    nodes = {
        "H1": GraphNode(
            node_id="H1", node_type=NodeType.HYPOTHESIS, label="Hyp A",
            answer_action="ANSWER",
        ),
        "H2": GraphNode(
            node_id="H2", node_type=NodeType.HYPOTHESIS, label="Hyp B",
            answer_action="ANSWER",
        ),
        "E1": GraphNode(
            node_id="E1", node_type=NodeType.EVIDENCE, label="Evidence 1",
            verification_state="UNVERIFIED", temporal_status="CURRENT",
            reliability=EvidenceReliability(),
        ),
        "E2": GraphNode(
            node_id="E2", node_type=NodeType.EVIDENCE, label="Evidence 2",
            verification_state="SUFFICIENT", temporal_status="CURRENT",
            reliability=EvidenceReliability(),
        ),
    }
    edges = (
        GraphEdge(source_id="E1", target_id="H1", edge_type=EdgeType.SUPPORTS),
        GraphEdge(source_id="E2", target_id="H2", edge_type=EdgeType.CONTRADICTS),
    )
    return EpistemicGraph(
        nodes=nodes, edges=edges,
        steps_remaining=5, verify_remaining=3,
        retrieve_remaining=2, search_remaining=2,
    )


@pytest.mark.parametrize("action_type,target", [
    (ActionType.ANSWER, "H1"),
    (ActionType.DEFER, None),
    (ActionType.STOP, None),
    (ActionType.VERIFY, "E1"),
    (ActionType.SEARCH, None),
    (ActionType.RETRIEVE, None),
    (ActionType.COMPARE, None),
    (ActionType.CHECK_CONSISTENCY, None),
])
def test_action_produces_valid_transitions(action_type, target):
    """Every action type must produce at least one valid transition."""
    graph = _make_test_graph()
    action = Action(action_type=action_type, target=target)
    transitions = transition_model(graph, action)

    assert len(transitions) > 0, f"Action {action_type} produced no transitions"
    assert all(isinstance(t.outcome, ObservationOutcome) for t in transitions), \
        f"Action {action_type} produced non-ObservationOutcome outcomes"

    # Probabilities must sum to ~1.0
    total_prob = sum(t.probability for t in transitions)
    assert abs(total_prob - 1.0) < 1e-6, \
        f"Action {action_type} probabilities sum to {total_prob}, not 1.0"


def test_compare_does_not_raise_attribute_error():
    """Specifically test that COMPARE doesn't use the nonexistent NEW_EVIDENCE.

    This is a regression test for the bug where COMPARE and CHECK_CONSISTENCY
    used ObservationOutcome.NEW_EVIDENCE which doesn't exist in the enum.
    """
    graph = _make_test_graph()
    action = Action(action_type=ActionType.COMPARE)
    transitions = transition_model(graph, action)

    assert len(transitions) == 1
    assert transitions[0].outcome == ObservationOutcome.NEW_EVIDENCE_FOUND


def test_check_consistency_does_not_raise_attribute_error():
    """Regression test for CHECK_CONSISTENCY using nonexistent NEW_EVIDENCE."""
    graph = _make_test_graph()
    action = Action(action_type=ActionType.CHECK_CONSISTENCY)
    transitions = transition_model(graph, action)

    assert len(transitions) == 1
    assert transitions[0].outcome == ObservationOutcome.NEW_EVIDENCE_FOUND


def test_all_outcomes_are_valid_enum_members():
    """Ensure no transition uses an invalid ObservationOutcome value."""
    graph = _make_test_graph()
    valid_outcomes = {o for o in ObservationOutcome}

    for action_type in ActionType:
        action = Action(action_type=action_type, target="E1" if action_type == ActionType.VERIFY else None)
        transitions = transition_model(graph, action)
        for t in transitions:
            assert t.outcome in valid_outcomes, \
                f"Action {action_type} produced invalid outcome {t.outcome}"
