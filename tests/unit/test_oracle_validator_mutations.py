"""Mutation tests for the oracle-path validator.

These tests intentionally create INVALID benchmark tasks and verify that
the canonical validator rejects them. A qualification validator is only
convincing if malformed examples are known to fail.

Mutations tested:
  A — competing verified support + ANSWER (must fail G_B3)
  B — FALSIFIED+supports treated as contradiction (must not pass)
  C — VERIFY from ANSWER_READY (must fail G_B2)
  D — DEFER from CONTINUE_REQUIRED (must fail G_B3)
  E — wrong correct hypothesis (must fail G_B4)
  F — illegal oracle action (must fail G_B1)
"""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

from validate_benchmark_oracles import validate_task_oracle


def make_task(
    task_id: str,
    category: str,
    hypotheses: list[tuple[str, str, str]],
    evidence: list[tuple[str, str, tuple, tuple, str, str]],
    correct_hypothesis: str,
    expected_terminal: str,
    oracle_path: tuple[str, ...],
    budget: dict | None = None,
) -> EvidenceTask:
    """Build a task from raw tuples."""
    hyps = []
    for h_id, prop, action_str in hypotheses:
        action = DecisionAction(action_str)
        hyps.append(EvidenceHypothesis(
            hypothesis_id=h_id,
            proposition=prop,
            answer_action=action,
            answer_payload=f"{action_str}:{h_id}:{prop}",
        ))

    ev_items = []
    for ev_id, prop, supports, contradicts, vstate_str, tstatus_str in evidence:
        ev_items.append(EvidenceItem(
            evidence_id=ev_id,
            proposition=prop,
            source_class="initial",
            supports=supports,
            contradicts=contradicts,
            verification_state=VerificationState(vstate_str),
            temporal_status=TemporalStatus(tstatus_str),
            retrieved=True,
            verify_result=vstate_str if vstate_str != "UNVERIFIED" else None,
        ))

    b = budget or {"steps": 4, "verify": 2, "retrieve": 0, "search": 0}
    rbudget = ResourceBudget(
        max_executive_steps=b["steps"],
        max_reasoning_tokens=256,
        max_retrieval_calls=b.get("retrieve", 0),
        max_verification_calls=b.get("verify", 2),
        max_search_calls=b.get("search", 0),
        max_elapsed_ms=10000,
    )

    return EvidenceTask(
        task_id=task_id,
        split="mutation_test",
        category=category,
        task_summary="Mutation test task",
        high_stakes=True,
        budget_profile=f"MUT_{b['steps']}_{b.get('verify', 2)}_{b.get('search', 0)}",
        hypotheses=tuple(hyps),
        evidence_items=tuple(ev_items),
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=oracle_path,
        expected_terminal=DecisionAction(expected_terminal),
        correct_hypothesis_id=correct_hypothesis,
    )


class TestMutationA_CompetingSupport:
    """Mutation A: competing verified support + ANSWER must fail G_B3."""

    def test_competing_support_answer_fails(self):
        """Two hypotheses with SUFFICIENT support, oracle says ANSWER."""
        task = make_task(
            task_id="mut_a_competing",
            category="MUTATION_A",
            hypotheses=[
                ("H1", "type A", "ANSWER"),
                ("H2", "type B", "ANSWER"),
                ("H3", "type C", "DEFER"),
            ],
            evidence=[
                ("E1", "Marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
                ("E2", "Marker for B", ("H2",), (), "SUFFICIENT", "CURRENT"),
                ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ],
            correct_hypothesis="H1",
            expected_terminal="ANSWER",
            oracle_path=("ANSWER",),
        )
        result = validate_task_oracle(task)
        assert not result.valid
        assert not result.g_b3_terminal_matches
        assert "ANSWER_READY" in result.failure_reason


class TestMutationB_FalsifiedPolarity:
    """Mutation B: FALSIFIED+supports must NOT be treated as contradiction."""

    def test_falsified_support_not_contradiction(self):
        """FALSIFIED+supports(H1) should WEAKEN H1, not eliminate it.
        If the validator incorrectly treats it as contradiction,
        it might pass a task where H1 is only weakened (not eliminated)
        as if H1 were eliminated."""
        # H1 has FALSIFIED support → WEAKENED (not CONTRADICTED)
        # H2 has SUFFICIENT support → SUPPORTED
        # Oracle says ANSWER for H2
        # This should be valid because H2 is uniquely supported
        # But if the validator incorrectly treats FALSIFIED+supports(H1)
        # as contradiction, it would also be valid — so we need a case
        # where the incorrect semantics would WRONGLY pass.
        #
        # Better test: H1 has FALSIFIED support, H2 has FALSIFIED support.
        # Under correct semantics: both WEAKENED, no one supported.
        # Under wrong semantics: both CONTRADICTED, no one supported.
        # Either way ANSWER fails. So let's test the direct case:
        #
        # H1 has SUFFICIENT support, H2 has FALSIFIED support.
        # Correct: H1=SUPPORTED, H2=WEAKENED → unique supported=H1 → ANSWER_READY
        # Wrong: H1=SUPPORTED, H2=CONTRADICTED → unique supported=H1 → ANSWER_READY
        # Both pass. Need a different test.
        #
        # Real test: H1 has FALSIFIED+contradicts(H1).
        # Correct: H1 has falsified_contradiction → NOT supported, NOT contradicted → UNTESTED
        # Wrong: H1 has verified_support → SUPPORTED
        # If oracle says ANSWER for H1:
        # Correct: fails (H1 is UNTESTED, not ANSWER_READY)
        # Wrong: passes (H1 is incorrectly SUPPORTED)
        task = make_task(
            task_id="mut_b_falsified_polarity",
            category="MUTATION_B",
            hypotheses=[
                ("H1", "type A", "ANSWER"),
                ("H2", "type B", "DEFER"),
            ],
            evidence=[
                # FALSIFIED+contradicts(H1) → should NOT make H1 supported
                ("E1", "Failed contradiction of A", (), ("H1",), "FALSIFIED", "CURRENT"),
                ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ],
            correct_hypothesis="H1",
            expected_terminal="ANSWER",
            oracle_path=("ANSWER",),
        )
        result = validate_task_oracle(task)
        # Under correct semantics: H1 is UNTESTED (falsified contradiction
        # does not provide support), so ANSWER_READY is false → must fail
        assert not result.valid
        assert not result.g_b3_terminal_matches


class TestMutationC_PrematureContinuation:
    """Mutation C: VERIFY from ANSWER_READY must fail G_B2."""

    def test_verify_from_answer_ready_fails(self):
        """State is already ANSWER_READY but oracle says VERIFY first."""
        task = make_task(
            task_id="mut_c_premature_verify",
            category="MUTATION_C",
            hypotheses=[
                ("H1", "type A", "ANSWER"),
                ("H2", "type B", "DEFER"),
            ],
            evidence=[
                ("E1", "Marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
                ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ],
            correct_hypothesis="H1",
            expected_terminal="ANSWER",
            oracle_path=("VERIFY", "ANSWER"),  # VERIFY is unnecessary
        )
        result = validate_task_oracle(task)
        assert not result.valid
        assert not result.g_b2_continuation_required
        assert "CONTINUE_REQUIRED" in result.failure_reason


class TestMutationD_PrematureDefer:
    """Mutation D: DEFER from CONTINUE_REQUIRED must fail G_B3."""

    def test_defer_from_continue_required_fails(self):
        """State is CONTINUE_REQUIRED but oracle says DEFER."""
        task = make_task(
            task_id="mut_d_premature_defer",
            category="MUTATION_D",
            hypotheses=[
                ("H1", "type A", "ANSWER"),
                ("H2", "type B", "ANSWER"),
                ("H3", "type C", "DEFER"),
            ],
            evidence=[
                ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
                ("E2", "Test for B", ("H2",), (), "UNVERIFIED", "CURRENT"),
                ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ],
            correct_hypothesis="H3",
            expected_terminal="DEFER",
            oracle_path=("DEFER",),
        )
        result = validate_task_oracle(task)
        assert not result.valid
        assert not result.g_b3_terminal_matches
        assert "DEFER_READY" in result.failure_reason


class TestMutationE_WrongCorrectHypothesis:
    """Mutation E: wrong correct_hypothesis must fail G_B4."""

    def test_wrong_correct_hypothesis_fails(self):
        """H1 is uniquely supported but correct_hypothesis says H2."""
        task = make_task(
            task_id="mut_e_wrong_hypothesis",
            category="MUTATION_E",
            hypotheses=[
                ("H1", "type A", "ANSWER"),
                ("H2", "type B", "ANSWER"),
                ("H3", "type C", "DEFER"),
            ],
            evidence=[
                ("E1", "Marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
                ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
                ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ],
            correct_hypothesis="H2",  # Wrong! H1 is the supported one
            expected_terminal="ANSWER",
            oracle_path=("ANSWER",),
        )
        result = validate_task_oracle(task)
        assert not result.valid
        assert not result.g_b4_hypothesis_justified
        assert "H2" in result.failure_reason
        assert "H1" in result.failure_reason


class TestMutationF_IllegalAction:
    """Mutation F: illegal oracle action must fail G_B1."""

    def test_unknown_action_fails(self):
        """Oracle path contains an unknown action.
        State must be CONTINUE_REQUIRED so we reach the action check."""
        task = make_task(
            task_id="mut_f_unknown_action",
            category="MUTATION_F",
            hypotheses=[
                ("H1", "type A", "ANSWER"),
                ("H2", "type B", "ANSWER"),
                ("H3", "type C", "DEFER"),
            ],
            evidence=[
                ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
                ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
                ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ],
            correct_hypothesis="H1",
            expected_terminal="ANSWER",
            oracle_path=("INVALID_ACTION", "ANSWER"),
        )
        result = validate_task_oracle(task)
        assert not result.valid
        assert not result.g_b1_actions_legal
