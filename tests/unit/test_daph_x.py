"""Tests for DAPH-X core components."""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph_x.actions.typed_actions import (
    Action, ActionType, answer, defer, verify, retrieve, search,
    compare, check_consistency, stop,
)
from daph_x.authority.executive_decision import ExecutiveDecision, AuthorityMode
from daph_x.graph.epistemic_graph import (
    EpistemicGraph, GraphNode, GraphEdge, NodeType, EdgeType,
    EvidenceReliability, build_graph_from_evidence_task,
)
from daph_x.belief.belief_engine import compute_belief_state, BeliefState
from daph_x.actions.candidate_generator import generate_and_prune
from daph_x.authority.executive import select_action, ExecutiveConfig
from daph_x.world_model.transition_model import transition_model, ObservationOutcome

# Bridge from legacy task format
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState


def make_legacy_task(
    task_id: str = "test",
    hypotheses=None,
    evidence=None,
    correct_hypothesis="H1",
    expected_terminal="ANSWER",
    oracle_path=("ANSWER",),
) -> EvidenceTask:
    """Build a legacy EvidenceTask for testing."""
    if hypotheses is None:
        hypotheses = [
            ("H1", "type A", "ANSWER"),
            ("H2", "type B", "ANSWER"),
            ("H3", "type C", "DEFER"),
        ]
    if evidence is None:
        evidence = [
            ("E1", "Marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ]

    hyps = [EvidenceHypothesis(
        hypothesis_id=h_id, proposition=prop,
        answer_action=DecisionAction(action_str),
        answer_payload=f"{action_str}:{h_id}:{prop}",
    ) for h_id, prop, action_str in hypotheses]

    evs = [EvidenceItem(
        evidence_id=ev_id, proposition=prop, source_class="initial",
        supports=supports, contradicts=contradicts,
        verification_state=VerificationState(vs),
        temporal_status=TemporalStatus(ts),
        retrieved=True,
        verify_result=vs if vs != "UNVERIFIED" else None,
    ) for ev_id, prop, supports, contradicts, vs, ts in evidence]

    return EvidenceTask(
        task_id=task_id, split="test", category="TEST",
        task_summary="Test task", high_stakes=True,
        budget_profile="TEST_4_2_0",
        hypotheses=tuple(hyps), evidence_items=tuple(evs),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=oracle_path,
        expected_terminal=DecisionAction(expected_terminal),
        correct_hypothesis_id=correct_hypothesis,
    )


class TestTypedActions:
    def test_answer_action(self):
        a = answer("H1")
        assert a.action_type == ActionType.ANSWER
        assert a.target == "H1"
        assert not a.reversible
        assert str(a) == "ANSWER(H1)"

    def test_verify_action(self):
        v = verify("E1")
        assert v.action_type == ActionType.VERIFY
        assert v.target == "E1"
        assert not v.reversible
        assert str(v) == "VERIFY(E1)"

    def test_compare_action(self):
        c = compare("H1", "H2")
        assert c.action_type == ActionType.COMPARE
        assert c.target == ("H1", "H2")
        assert str(c) == "COMPARE(H1,H2)"

    def test_defer_action(self):
        d = defer("uncertain")
        assert d.action_type == ActionType.DEFER
        assert not d.reversible


class TestEpistemicGraph:
    def test_build_from_legacy_task(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        assert len(graph.nodes) == 6  # 3 hypotheses + 3 evidence
        assert len(graph.edges) == 3  # E1→H1, E2→H2, E3→H3

    def test_graph_hash_deterministic(self):
        task = make_legacy_task()
        g1 = build_graph_from_evidence_task(task)
        g2 = build_graph_from_evidence_task(task)
        assert g1.graph_hash() == g2.graph_hash()

    def test_hypothesis_ids(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        assert sorted(graph.hypothesis_ids()) == ["H1", "H2", "H3"]

    def test_supports_hypothesis(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        assert graph.supports_hypothesis("E1", "H1")
        assert not graph.supports_hypothesis("E1", "H2")

    def test_contradicts_hypothesis(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        assert graph.contradicts_hypothesis("E2", "H2")
        assert not graph.contradicts_hypothesis("E1", "H1")


class TestBeliefEngine:
    def test_unique_supported(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        belief = compute_belief_state(graph)
        assert belief.unique_supported == "H1"
        assert belief.readiness.value == "ANSWER_READY"

    def test_entropy_low_for_unique(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        belief = compute_belief_state(graph)
        assert belief.entropy < 2.0  # Low entropy when one is clearly supported

    def test_probabilities_sum_to_one(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        belief = compute_belief_state(graph)
        total = sum(belief.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_top_hypothesis(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        belief = compute_belief_state(graph)
        top = belief.top_hypothesis()
        assert top is not None
        assert top[0] == "H1"


class TestCandidateGenerator:
    def test_generates_answer_for_supported(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        candidates = generate_and_prune(graph)
        answers = [c for c in candidates if c.action_type == ActionType.ANSWER]
        assert len(answers) == 1
        assert answers[0].target == "H1"

    def test_generates_verify_for_unverified(self):
        task = make_legacy_task(evidence=[
            ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ])
        graph = build_graph_from_evidence_task(task)
        candidates = generate_and_prune(graph)
        verifies = [c for c in candidates if c.action_type == ActionType.VERIFY]
        assert len(verifies) == 1
        assert verifies[0].target == "E1"

    def test_no_verify_when_exhausted(self):
        task = make_legacy_task(evidence=[
            ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ])
        graph = build_graph_from_evidence_task(task)
        # Exhaust verify budget
        graph = EpistemicGraph(
            nodes=graph.nodes, edges=graph.edges,
            steps_remaining=graph.steps_remaining,
            verify_remaining=0,  # Exhausted
            retrieve_remaining=graph.retrieve_remaining,
            search_remaining=graph.search_remaining,
        )
        candidates = generate_and_prune(graph)
        verifies = [c for c in candidates if c.action_type == ActionType.VERIFY]
        assert len(verifies) == 0

    def test_always_has_defer(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        candidates = generate_and_prune(graph)
        defers = [c for c in candidates if c.action_type == ActionType.DEFER]
        assert len(defers) == 1


class TestWorldModel:
    def test_verify_transition_sufficient(self):
        task = make_legacy_task(evidence=[
            ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ])
        graph = build_graph_from_evidence_task(task)
        action = verify("E1")
        transitions = transition_model(graph, action)
        assert len(transitions) == 3  # SUFFICIENT, FALSIFIED, INCONCLUSIVE
        outcomes = [t.outcome for t in transitions]
        assert ObservationOutcome.SUFFICIENT in outcomes
        assert ObservationOutcome.FALSIFIED in outcomes

    def test_answer_transition_terminal(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        action = answer("H1")
        transitions = transition_model(graph, action)
        assert len(transitions) == 1
        assert transitions[0].outcome == ObservationOutcome.TERMINAL

    def test_verify_consumes_budget(self):
        task = make_legacy_task(evidence=[
            ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ])
        graph = build_graph_from_evidence_task(task)
        action = verify("E1")
        transitions = transition_model(graph, action)
        for t in transitions:
            assert t.next_graph.verify_remaining == graph.verify_remaining - 1


class TestExecutive:
    def test_selects_answer_when_ready(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        decision = select_action(graph)
        assert decision.selected_action.action_type == ActionType.ANSWER
        assert decision.selected_action.target == "H1"
        assert decision.structural_certificate
        assert decision.certificate_type == "unique_verified_support_answer"

    def test_has_candidates(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        decision = select_action(graph)
        assert decision.candidate_count > 0

    def test_state_hash_deterministic(self):
        task = make_legacy_task()
        graph = build_graph_from_evidence_task(task)
        d1 = select_action(graph)
        d2 = select_action(graph)
        assert d1.state_hash == d2.state_hash

    def test_abstain_when_no_candidates(self):
        """Empty graph should still produce a valid decision (DEFER or STOP)."""
        empty_graph = EpistemicGraph(nodes={}, edges=())
        decision = select_action(empty_graph)
        # With no hypotheses, DEFER and STOP are the only candidates
        assert decision.selected_action.action_type in (ActionType.DEFER, ActionType.STOP)
        assert decision.candidate_count > 0  # DEFER and STOP always present
