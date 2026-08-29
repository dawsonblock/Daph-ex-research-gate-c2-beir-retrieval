"""Tests for canonical topology conformance across all consumers.

Verifies that the executor, snapshot, and topology derivation all agree
on epistemic state, per EPISTEMIC_SEMANTICS_V1.md §13.
"""
import pytest
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState, TemporalStatus
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem, EvidenceRuntime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

from daph.epistemic import (
    derive_hypothesis_topology, is_answer_ready, HypothesisState,
)


def make_hyp(h_id, action=DecisionAction.ANSWER):
    return EvidenceHypothesis(h_id, f"Test {h_id}", action, f"{action.value}:{h_id}")


def make_ev(eid, supports=(), contradicts=(), vstate=VerificationState.UNVERIFIED,
            tstatus=TemporalStatus.CURRENT, retrieved=True):
    return EvidenceItem(
        eid, f"Evidence {eid}", "initial", supports, contradicts,
        vstate, tstatus, retrieved, None)


def make_task(hypotheses, evidence, expected=DecisionAction.ANSWER, correct="H1"):
    return EvidenceTask(
        task_id="test_task", split="test", category="test",
        task_summary="Test task", high_stakes=True, budget_profile="test",
        hypotheses=tuple(hypotheses), evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=(expected.value,),
        expected_terminal=expected, correct_hypothesis_id=correct)


def make_budget(steps=5, verify=3, retrieve=2, search=2, reasoning=256):
    return ResourceBudget(
        max_executive_steps=steps, max_retrieval_calls=retrieve,
        max_verification_calls=verify, max_search_calls=search,
        max_reasoning_tokens=reasoning, max_elapsed_ms=10000)


class TestExecutorUsesCanonicalTopology:
    """Verify executor ANSWER success uses canonical topology, not duplicated logic."""

    def test_answer_succeeds_unique_supported(self):
        """ANSWER succeeds when exactly one hypothesis is SUPPORTED and it's correct."""
        hyps = (make_hyp("H1"), make_hyp("H2", DecisionAction.DEFER))
        ev = (
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H2",), vstate=VerificationState.SUFFICIENT),
        )
        task = make_task(hyps, ev, expected=DecisionAction.ANSWER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1", "E2"), verified_evidence_ids=("E1", "E2"))
        executor = EvidenceExecutor()
        assert executor._check_answer_success(runtime) is True

    def test_answer_fails_competing_support(self):
        """ANSWER fails when two hypotheses have SUFFICIENT support (not unique)."""
        hyps = (make_hyp("H1"), make_hyp("H2", DecisionAction.DEFER))
        ev = (
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
        )
        task = make_task(hyps, ev, expected=DecisionAction.ANSWER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1", "E2"), verified_evidence_ids=("E1", "E2"))
        executor = EvidenceExecutor()
        assert executor._check_answer_success(runtime) is False

    def test_answer_fails_wrong_unique_supported(self):
        """ANSWER fails when unique supported hypothesis is NOT the correct one."""
        hyps = (make_hyp("H1"), make_hyp("H2", DecisionAction.DEFER))
        ev = (
            make_ev("E1", supports=("H2",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H1",), vstate=VerificationState.SUFFICIENT),
        )
        task = make_task(hyps, ev, expected=DecisionAction.ANSWER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1", "E2"), verified_evidence_ids=("E1", "E2"))
        executor = EvidenceExecutor()
        # H2 is uniquely supported, but correct is H1
        assert executor._check_answer_success(runtime) is False


class TestExecutorDeferUsesCanonicalTopology:
    """Verify executor DEFER success uses canonical topology."""

    def test_defer_succeeds_no_supported_no_continuation(self):
        """DEFER succeeds when no hypothesis is supported and no continuation available."""
        hyps = (make_hyp("H1", DecisionAction.DEFER), make_hyp("H2"))
        ev = (
            make_ev("E1", contradicts=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H2",), vstate=VerificationState.SUFFICIENT),
        )
        # Budget with no verify/retrieve/search remaining
        budget = ResourceBudget(
            max_executive_steps=1, max_retrieval_calls=0,
            max_verification_calls=0, max_search_calls=0,
            max_reasoning_tokens=0, max_elapsed_ms=10000)
        task = make_task(hyps, ev, expected=DecisionAction.DEFER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=budget),
            evidence=ev, retrieved_evidence_ids=("E1", "E2"), verified_evidence_ids=("E1", "E2"))
        executor = EvidenceExecutor()
        assert executor._check_defer_success(runtime) is True

    def test_defer_fails_when_answer_ready(self):
        """DEFER fails when state is ANSWER_READY (unique supported with ANSWER action)."""
        # Per EPISTEMIC_SEMANTICS_V1.md §6.1, ANSWER_READY requires the
        # uniquely supported hypothesis to have answer_action == ANSWER.
        # When the supported hypothesis has answer_action == DEFER, the
        # state is DEFER_READY, not ANSWER_READY.
        hyps = (make_hyp("H1", DecisionAction.ANSWER), make_hyp("H2"))
        ev = (
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
        )
        budget = ResourceBudget(
            max_executive_steps=1, max_retrieval_calls=0,
            max_verification_calls=0, max_search_calls=0,
            max_reasoning_tokens=0, max_elapsed_ms=10000)
        task = make_task(hyps, ev, expected=DecisionAction.DEFER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=budget),
            evidence=ev, retrieved_evidence_ids=("E1",), verified_evidence_ids=("E1",))
        executor = EvidenceExecutor()
        # H1 (ANSWER) is uniquely supported → ANSWER_READY → DEFER should fail
        assert executor._check_defer_success(runtime) is False

    def test_defer_succeeds_when_unique_supported_is_defer(self):
        """DEFER succeeds when uniquely supported hypothesis has answer_action=DEFER.

        Per EPISTEMIC_SEMANTICS_V1.md §6.1, ANSWER_READY requires the supported
        hypothesis to have answer_action == ANSWER. A uniquely supported DEFER
        hypothesis means the state is DEFER_READY, not ANSWER_READY.
        """
        hyps = (make_hyp("H1", DecisionAction.DEFER), make_hyp("H2"))
        ev = (
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
        )
        budget = ResourceBudget(
            max_executive_steps=1, max_retrieval_calls=0,
            max_verification_calls=0, max_search_calls=0,
            max_reasoning_tokens=0, max_elapsed_ms=10000)
        task = make_task(hyps, ev, expected=DecisionAction.DEFER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=budget),
            evidence=ev, retrieved_evidence_ids=("E1",), verified_evidence_ids=("E1",))
        executor = EvidenceExecutor()
        # H1 (DEFER) is uniquely supported → DEFER_READY → DEFER should succeed
        assert executor._check_defer_success(runtime) is True

    def test_defer_fails_when_continuation_available(self):
        """DEFER fails when a continuation could resolve the state."""
        hyps = (make_hyp("H1", DecisionAction.DEFER), make_hyp("H2"))
        ev = (
            make_ev("E1", supports=("H1",), vstate=VerificationState.UNVERIFIED),
            make_ev("E2", supports=("H2",), vstate=VerificationState.UNVERIFIED),
        )
        # Budget with verify available
        budget = make_budget(steps=3, verify=2, retrieve=0, search=0)
        task = make_task(hyps, ev, expected=DecisionAction.DEFER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=budget),
            evidence=ev, retrieved_evidence_ids=("E1", "E2"), verified_evidence_ids=())
        executor = EvidenceExecutor()
        # Unverified evidence + verify available → continuation possible → DEFER fails
        assert executor._check_defer_success(runtime) is False


class TestSnapshotContradictingCount:
    """Verify snapshot contradicting_count uses canonical SUFFICIENT+contradicts semantics."""

    def test_sufficient_contradicts_counted_as_contradicting(self):
        """SUFFICIENT + contradicts(H) → counted in contradicting_count."""
        hyps = (make_hyp("H1"), make_hyp("H2"))
        ev = (make_ev("E1", contradicts=("H1",), vstate=VerificationState.SUFFICIENT),)
        task = make_task(hyps, ev)
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1",), verified_evidence_ids=("E1",))
        snapshot = build_evidence_snapshot(runtime)
        assert snapshot.contradicting_count == 1

    def test_falsified_supports_not_counted_as_contradicting(self):
        """FALSIFIED + supports(H) → NOT counted in contradicting_count (legacy bug fix)."""
        hyps = (make_hyp("H1"), make_hyp("H2"))
        ev = (make_ev("E1", supports=("H1",), vstate=VerificationState.FALSIFIED),)
        task = make_task(hyps, ev)
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1",), verified_evidence_ids=("E1",))
        snapshot = build_evidence_snapshot(runtime)
        # Under legacy semantics this was 1; under canonical it's 0
        assert snapshot.contradicting_count == 0

    def test_sufficient_supports_counted_as_supporting(self):
        """SUFFICIENT + supports(H) → counted in supporting_count."""
        hyps = (make_hyp("H1"), make_hyp("H2"))
        ev = (make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),)
        task = make_task(hyps, ev)
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1",), verified_evidence_ids=("E1",))
        snapshot = build_evidence_snapshot(runtime)
        assert snapshot.supporting_count == 1


class TestConsumerAgreement:
    """Verify that topology, executor, and snapshot all agree on epistemic state."""

    def test_topology_executor_snapshot_agree_on_competing_support(self):
        """All three consumers agree: competing support → not answer-ready."""
        hyps = (make_hyp("H1"), make_hyp("H2"))
        ev = (
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
        )
        task = make_task(hyps, ev, expected=DecisionAction.ANSWER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1", "E2"),
            verified_evidence_ids=("E1", "E2"))

        # Topology: not answer-ready
        visible_ev = [
            {"evidence_id": e.evidence_id, "supports": list(e.supports),
             "contradicts": list(e.contradicts),
             "verification_state": e.verification_state,
             "temporal_status": e.temporal_status, "retrieved": e.retrieved}
            for e in ev
        ]
        topo = derive_hypothesis_topology(visible_ev, ["H1", "H2"])
        assert not is_answer_ready(topo)

        # Executor: ANSWER fails
        executor = EvidenceExecutor()
        assert executor._check_answer_success(runtime) is False

        # Snapshot: has supporting evidence for both
        snapshot = build_evidence_snapshot(runtime)
        assert snapshot.supporting_count == 2
        assert snapshot.contradicting_count == 0

    def test_topology_executor_agree_on_unique_resolution(self):
        """All three consumers agree: unique support → answer-ready."""
        hyps = (make_hyp("H1"), make_hyp("H2"))
        ev = (
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H2",), vstate=VerificationState.SUFFICIENT),
        )
        task = make_task(hyps, ev, expected=DecisionAction.ANSWER, correct="H1")
        runtime = EvidenceRuntime(
            task=task, resources=ResourceState(budget=make_budget()),
            evidence=ev, retrieved_evidence_ids=("E1", "E2"),
            verified_evidence_ids=("E1", "E2"))

        # Topology: answer-ready
        visible_ev = [
            {"evidence_id": e.evidence_id, "supports": list(e.supports),
             "contradicts": list(e.contradicts),
             "verification_state": e.verification_state,
             "temporal_status": e.temporal_status, "retrieved": e.retrieved}
            for e in ev
        ]
        topo = derive_hypothesis_topology(visible_ev, ["H1", "H2"])
        assert is_answer_ready(topo)
        assert topo.unique_supported_hypothesis == "H1"

        # Executor: ANSWER succeeds
        executor = EvidenceExecutor()
        assert executor._check_answer_success(runtime) is True

        # Snapshot: 1 supporting, 1 contradicting
        snapshot = build_evidence_snapshot(runtime)
        assert snapshot.supporting_count == 1
        assert snapshot.contradicting_count == 1
