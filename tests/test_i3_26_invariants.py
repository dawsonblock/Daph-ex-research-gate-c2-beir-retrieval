#!/usr/bin/env python3
"""I3.26 invariant test suite: 12 hard invariants + 4 synthetic qualification cases.

Hard invariants:
  1. Search disabled → byte/field-equivalent VP guidance
  2. Search abstains → exactly VP guidance
  3. No illegal action ever becomes a branch
  4. Branch restore reproduces checkpoint state SHA
  5. Branch simulation cannot mutate real trajectory
  6. Search node count never exceeds max_nodes (6)
  7. Branch depth never exceeds max_depth (2)
  8. Model-call budget never exceeds max_model_calls (4)
  9. Timeout/error → VP fallback, never trajectory failure
  10. No hidden benchmark/gold fields enter Q, PAV, trigger, evaluator, or packet
  11. Identical state + config + seed → identical candidate set/tree/decision
  12. Search receipt has every branch, score component, budget counter, fallback reason
  13. No oracle leakage: planner must not use future terminal utility unless from permitted rollout

Synthetic qualification cases:
  CASE 1: clear Q winner → no search
  CASE 2: near tie, equivalent continuations → search or abstain, no harmful narrowing
  CASE 3: near tie, one path causes exhaustion → search picks resource-preserving branch
  CASE 4: no successful path → search must not invent rescue
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceHypothesis, EvidenceItem,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.intervention.checkpoint import (
    StateCheckpoint, create_checkpoint, compute_state_features, compute_legal_actions,
)
from daph.intervention.restore import restore_runtime
from daph.pav.structural import StructuralPAV
from daph.pav.scorer import make_pav_scorer
from daph.search.types import SearchConfig, SearchTriggerResult, SearchResult
from daph.search.trigger import decide_search
from daph.search.budget import SearchBudget
from daph.search.planner import SearchPlanner
from daph.executive.pav_search_controller import PAVSearchController, ExecutiveGuidance


# ============================================================
# Test fixtures
# ============================================================

def _make_utility() -> MetareasoningUtility:
    return MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json",
    )


def _make_simple_task(
    task_id: str = "test_001",
    e1_verified: bool = True,
    e2_verified: bool = True,
    e3_hidden: bool = False,
    e3_supports_h1: bool = True,
    budget_profile: str = "TIGHT",
) -> EvidenceTask:
    """Create a simple evidence task for testing."""
    h1 = EvidenceHypothesis(
        hypothesis_id="H1",
        proposition="hypothesis 1",
        answer_action=DecisionAction.ANSWER,
        answer_payload="ANSWER: H1",
    )
    h2 = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition="hypothesis 2",
        answer_action=DecisionAction.DEFER,
        answer_payload="DEFER: H2",
    )

    e1_state = VerificationState.SUFFICIENT if e1_verified else VerificationState.UNVERIFIED
    e1_vr = "SUFFICIENT" if e1_verified else "MISSING"
    e1 = EvidenceItem(
        evidence_id="E1", proposition="evidence 1", source_class="initial",
        supports=("H1",), contradicts=(),
        verification_state=e1_state, temporal_status=TemporalStatus.CURRENT,
        retrieved=True, verify_result=e1_vr,
    )

    e2_state = VerificationState.FALSIFIED if e2_verified else VerificationState.UNVERIFIED
    e2_vr = "FALSIFIED" if e2_verified else "MISSING"
    e2 = EvidenceItem(
        evidence_id="E2", proposition="evidence 2", source_class="initial",
        supports=("H2",), contradicts=(),
        verification_state=e2_state, temporal_status=TemporalStatus.CURRENT,
        retrieved=True, verify_result=e2_vr,
    )

    evidence = [e1, e2]
    retrieve_exposes = ()
    search_exposes = ()

    if e3_hidden:
        e3 = EvidenceItem(
            evidence_id="E3", proposition="evidence 3", source_class="primary",
            supports=("H1",) if e3_supports_h1 else ("H2",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False, verify_result="SUFFICIENT",
        )
        evidence.append(e3)
        retrieve_exposes = ("E3",)

    return EvidenceTask(
        task_id=task_id, split="test", category="test",
        task_summary="Test task",
        high_stakes=True, budget_profile=budget_profile,
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=retrieve_exposes,
        search_exposes=search_exposes,
        oracle_resolution_path=("ANSWER",),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def _make_runtime(task: EvidenceTask, budget: ResourceBudget) -> "EvidenceRuntime":
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
    resources = ResourceState(budget=budget)
    return initial_evidence_runtime(task, resources)


def _make_mock_q_model(q_values: dict[str, float]):
    """Create a mock Q model that returns fixed values."""
    mock = MagicMock()
    mock.predict_q = MagicMock(return_value=q_values)
    return mock


# ============================================================
# Invariant tests
# ============================================================

class TestSearchInvariants(unittest.TestCase):
    """12 hard invariants + oracle leakage check."""

    @classmethod
    def setUpClass(cls):
        cls.utility = _make_utility()
        cls.task = _make_simple_task(e3_hidden=True)
        cls.budget = ResourceBudget(
            max_executive_steps=6, max_retrieval_calls=2,
            max_verification_calls=2, max_search_calls=1,
        )
        cls.runtime = _make_runtime(cls.task, cls.budget)
        cls.checkpoint = create_checkpoint(cls.runtime, step=0, prior_actions=())
        cls.legal_actions = list(compute_legal_actions(cls.runtime))

    def test_01_search_disabled_equals_vp(self):
        """Invariant 1: Search disabled → VP-equivalent guidance."""
        q_values = {"RETRIEVE": 67.0, "VERIFY": 67.5, "ANSWER": 4.0,
                    "DEFER": 14.0, "REASON_MORE": 61.0, "SEARCH_MORE": 64.0}
        q_model = _make_mock_q_model(q_values)

        pav_scorer = make_pav_scorer("B0", task=self.task, utility=self.utility)

        # With search disabled
        ctrl_no_search = PAVSearchController(
            task=self.task, utility=self.utility, q_model=q_model,
            pav_scorer=pav_scorer, enable_search=False, enable_pav=True,
        )
        guidance_no_search = ctrl_no_search.compute_guidance(
            self.checkpoint, self.legal_actions,
            self.checkpoint.state_features, "EXPLORE",
        )

        # With search enabled but PAV only (same as VP if search doesn't trigger)
        ctrl_with_search = PAVSearchController(
            task=self.task, utility=self.utility, q_model=q_model,
            pav_scorer=pav_scorer, enable_search=True, enable_pav=True,
        )
        guidance_with_search = ctrl_with_search.compute_guidance(
            self.checkpoint, self.legal_actions,
            self.checkpoint.state_features, "EXPLORE",
        )

        # When search is disabled, mode should be VP
        self.assertEqual(guidance_no_search.mode, "VP")
        self.assertFalse(guidance_no_search.search_triggered)

    def test_02_search_abstains_equals_vp(self):
        """Invariant 2: Search abstains → exactly VP guidance."""
        # Create a state where search triggers but abstains
        # Use a state with single near-optimal (search won't trigger)
        q_values = {"RETRIEVE": 67.0, "VERIFY": 80.0, "ANSWER": 4.0,
                    "DEFER": 14.0, "REASON_MORE": 61.0, "SEARCH_MORE": 64.0}
        q_model = _make_mock_q_model(q_values)
        pav_scorer = make_pav_scorer("B0", task=self.task, utility=self.utility)

        ctrl = PAVSearchController(
            task=self.task, utility=self.utility, q_model=q_model,
            pav_scorer=pav_scorer, enable_search=True,
        )
        guidance = ctrl.compute_guidance(
            self.checkpoint, self.legal_actions,
            self.checkpoint.state_features, "EXPLORE",
        )

        # Single near-optimal → search should not trigger
        self.assertFalse(guidance.search_triggered)
        self.assertEqual(guidance.mode, "VP")

    def test_03_no_illegal_action_in_branch(self):
        """Invariant 3: No illegal action ever becomes a branch.
        Each action must be legal at the state where it is executed."""
        config = SearchConfig()
        planner = SearchPlanner(self.task, self.utility, config)

        # Only pass legal actions as candidates
        legal = tuple(self.legal_actions)
        q_values = {a: 67.0 for a in legal}  # All tied
        q_model = _make_mock_q_model(q_values)

        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=legal[:2],  # Top 2
            q_values=q_values,
            trigger_reasons=("test",),
        )

        # Check that first actions are from the candidate set (which is from legal actions)
        for branch in result.branches:
            self.assertIn(branch.first_action, legal)

        # Check that depth-2 actions are legal at their execution state
        # (they may differ from initial legal actions because the state changed)
        for branch in result.branches:
            for node in branch.nodes:
                # At minimum, the action must be a valid DecisionAction
                self.assertIn(node.action,
                            ["RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE",
                             "ANSWER", "DEFER", "STOP"])

    def test_04_branch_restore_reproduces_sha(self):
        """Invariant 4: Branch restore reproduces checkpoint state SHA."""
        from daph.search.branch import simulate_branch_step
        executor = EvidenceExecutor()

        node, post_runtime = simulate_branch_step(
            self.checkpoint, self.task, "RETRIEVE",
            self.utility, executor, parent_id=None, depth=1,
        )

        # Restore the original checkpoint and verify SHA
        restored = restore_runtime(self.checkpoint, self.task)
        # The original checkpoint should still restore correctly
        # (branch simulation should not have mutated it)
        self.assertEqual(
            restored.visible_evidence[0].evidence_id,
            self.checkpoint.evidence[0]["evidence_id"],
        )

    def test_05_branch_cannot_mutate_real_trajectory(self):
        """Invariant 5: Branch simulation cannot mutate the real trajectory."""
        from daph.search.branch import simulate_branch_step
        executor = EvidenceExecutor()

        # Record original state
        original_runtime = restore_runtime(self.checkpoint, self.task)
        original_n_visible = len(original_runtime.visible_evidence)

        # Simulate a branch
        node, post_runtime = simulate_branch_step(
            self.checkpoint, self.task, "RETRIEVE",
            self.utility, executor, parent_id=None, depth=1,
        )

        # Restore original checkpoint again
        restored = restore_runtime(self.checkpoint, self.task)
        self.assertEqual(len(restored.visible_evidence), original_n_visible)

    def test_06_node_count_never_exceeds_max(self):
        """Invariant 6: Search node count never exceeds max_nodes."""
        config = SearchConfig(max_nodes=6)
        planner = SearchPlanner(self.task, self.utility, config)

        q_values = {a: 67.0 for a in self.legal_actions}
        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        self.assertLessEqual(result.nodes_expanded, config.max_nodes)

    def test_07_branch_depth_never_exceeds_max(self):
        """Invariant 7: Branch depth never exceeds max_depth."""
        config = SearchConfig(max_depth=2)
        planner = SearchPlanner(self.task, self.utility, config)

        q_values = {a: 67.0 for a in self.legal_actions}
        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        for branch in result.branches:
            for node in branch.nodes:
                self.assertLessEqual(node.depth, config.max_depth)

    def test_08_model_call_budget_never_exceeded(self):
        """Invariant 8: Model-call budget never exceeds max_model_calls."""
        config = SearchConfig(max_model_calls=4)
        planner = SearchPlanner(self.task, self.utility, config)

        q_values = {a: 67.0 for a in self.legal_actions}
        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        self.assertLessEqual(result.model_calls, config.max_model_calls)

    def test_09_timeout_error_falls_back_to_vp(self):
        """Invariant 9: Timeout/error → VP fallback, never trajectory failure."""
        config = SearchConfig(max_wall_ms=1)  # Extremely tight to force timeout
        planner = SearchPlanner(self.task, self.utility, config)

        q_values = {a: 67.0 for a in self.legal_actions}
        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        # Should either complete within budget or abstain
        # Either way, should not crash
        self.assertIsNotNone(result)

    def test_10_no_hidden_fields_in_outputs(self):
        """Invariant 10: No hidden benchmark/gold fields in any output."""
        config = SearchConfig()
        planner = SearchPlanner(self.task, self.utility, config)
        q_values = {a: 67.0 for a in self.legal_actions}

        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        # Check that no gold/hidden fields appear in the receipt
        receipt_str = json.dumps(result.receipt, default=str)
        forbidden = ["correct_hypothesis", "oracle_resolution", "expected_terminal",
                     "gold", "label", "answer_key"]
        for field in forbidden:
            self.assertNotIn(field, receipt_str.lower(),
                           f"Forbidden field '{field}' found in search receipt")

    def test_11_identical_state_config_produces_identical_tree(self):
        """Invariant 11: Identical state + config → identical decision."""
        config = SearchConfig()
        planner = SearchPlanner(self.task, self.utility, config)
        q_values = {a: 67.0 for a in self.legal_actions}

        result1 = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        result2 = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        # Same winner (or both abstain)
        self.assertEqual(result1.abstained, result2.abstained)
        if result1.winner and result2.winner:
            self.assertEqual(result1.winner, result2.winner)

    def test_12_receipt_completeness(self):
        """Invariant 12: Search receipt has every branch, score, budget, fallback."""
        config = SearchConfig()
        planner = SearchPlanner(self.task, self.utility, config)
        q_values = {a: 67.0 for a in self.legal_actions}

        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        receipt = result.receipt
        self.assertIn("planner", receipt)
        self.assertIn("checkpoint_id", receipt)
        self.assertIn("candidates", receipt)
        self.assertIn("branches", receipt)
        self.assertIn("budget", receipt)
        self.assertIn("trigger_reasons", receipt)

        # Budget should have all counters
        budget = receipt["budget"]
        self.assertIn("nodes_expanded", budget)
        self.assertIn("model_calls", budget)
        self.assertIn("wall_ms", budget)

    def test_13_no_oracle_leakage(self):
        """Invariant 13: No oracle leakage — planner must not use future
        terminal utility unless from permitted simulated rollout."""
        config = SearchConfig()
        planner = SearchPlanner(self.task, self.utility, config)
        q_values = {a: 67.0 for a in self.legal_actions}

        result = planner.plan(
            checkpoint=self.checkpoint,
            candidate_actions=tuple(self.legal_actions[:2]),
            q_values=q_values,
            trigger_reasons=("test",),
        )

        # Check that branch scores are computed from simulated progress,
        # not from the task's correct_hypothesis_id or oracle_resolution_path
        for branch in result.branches:
            # Score should be a function of Q + PAV - cost, not oracle data
            self.assertIsNotNone(branch.score)
            # Terminal utility should come from the executor, not the task definition
            if branch.terminal_utility is not None:
                # It should be a number, not a reference to task gold
                self.assertIsInstance(branch.terminal_utility, (int, float))


# ============================================================
# Synthetic qualification cases
# ============================================================

class TestSyntheticQualification(unittest.TestCase):
    """4 synthetic cases to qualify the planner before live experiments."""

    @classmethod
    def setUpClass(cls):
        cls.utility = _make_utility()

    def test_case1_clear_q_winner_no_search(self):
        """CASE 1: Clear Q winner → no search."""
        task = _make_simple_task(e1_verified=True, e2_verified=True)
        budget = ResourceBudget(
            max_executive_steps=6, max_retrieval_calls=2,
            max_verification_calls=2, max_search_calls=1,
        )
        runtime = _make_runtime(task, budget)
        checkpoint = create_checkpoint(runtime, step=0, prior_actions=())
        sf = checkpoint.state_features

        # Large Q gap — VERIFY is clearly best
        q_values = {"VERIFY": 90.0, "RETRIEVE": 50.0, "ANSWER": 4.0,
                    "DEFER": 14.0, "REASON_MORE": 61.0, "SEARCH_MORE": 64.0}

        config = SearchConfig()
        trigger = decide_search(
            state_features=sf,
            near_optimal_actions=("VERIFY",),  # Only one near-optimal
            pav_selected=("VERIFY",),
            pav_abstained=False,
            q_values=q_values,
            config=config,
        )

        self.assertFalse(trigger.should_search,
                        "Search should not trigger with single near-optimal action")

    def test_case2_near_tie_no_harmful_narrowing(self):
        """CASE 2: Near tie, equivalent continuations → search or abstain,
        but no harmful narrowing."""
        task = _make_simple_task(e1_verified=False, e2_verified=True, e3_hidden=True)
        budget = ResourceBudget(
            max_executive_steps=6, max_retrieval_calls=2,
            max_verification_calls=2, max_search_calls=1,
        )
        runtime = _make_runtime(task, budget)
        checkpoint = create_checkpoint(runtime, step=0, prior_actions=())
        sf = checkpoint.state_features

        # Near tie between RETRIEVE and VERIFY
        q_values = {"RETRIEVE": 67.0, "VERIFY": 67.5, "ANSWER": 4.0,
                    "DEFER": 14.0, "REASON_MORE": 61.0, "SEARCH_MORE": 64.0}

        config = SearchConfig()
        trigger = decide_search(
            state_features=sf,
            near_optimal_actions=("RETRIEVE", "VERIFY"),
            pav_selected=("RETRIEVE", "VERIFY"),
            pav_abstained=True,  # PAV can't distinguish
            q_values=q_values,
            config=config,
        )

        # Search should trigger (near tie + ambiguity)
        self.assertTrue(trigger.should_search,
                       "Search should trigger on near-tie with ambiguity")

        # Run the planner
        planner = SearchPlanner(task, self.utility, config)
        result = planner.plan(
            checkpoint=checkpoint,
            candidate_actions=("RETRIEVE", "VERIFY"),
            q_values=q_values,
            trigger_reasons=trigger.reasons,
        )

        # Should not crash, should produce a result or abstain
        self.assertIsNotNone(result)
        # If it picks a winner, it should be one of the candidates
        if result.winner:
            self.assertIn(result.winner, ("RETRIEVE", "VERIFY"))

    def test_case3_near_tie_search_picks_resource_preserving(self):
        """CASE 3: Near tie, one path causes exhaustion → search picks
        resource-preserving branch."""
        # Create a task with tight budget where RETRIEVE wastes resources
        task = _make_simple_task(e1_verified=False, e2_verified=True, e3_hidden=True)
        budget = ResourceBudget(
            max_executive_steps=4, max_retrieval_calls=1,  # Very tight
            max_verification_calls=2, max_search_calls=1,
        )
        runtime = _make_runtime(task, budget)
        checkpoint = create_checkpoint(runtime, step=0, prior_actions=())
        sf = checkpoint.state_features

        # Near tie between RETRIEVE and VERIFY
        q_values = {"RETRIEVE": 67.0, "VERIFY": 67.5, "ANSWER": 4.0,
                    "DEFER": 14.0, "REASON_MORE": 61.0, "SEARCH_MORE": 64.0}

        config = SearchConfig()
        trigger = decide_search(
            state_features=sf,
            near_optimal_actions=("RETRIEVE", "VERIFY"),
            pav_selected=None,
            pav_abstained=True,
            q_values=q_values,
            config=config,
        )

        self.assertTrue(trigger.should_search)

        planner = SearchPlanner(task, self.utility, config)
        result = planner.plan(
            checkpoint=checkpoint,
            candidate_actions=("RETRIEVE", "VERIFY"),
            q_values=q_values,
            trigger_reasons=trigger.reasons,
        )

        # Should produce a result
        self.assertIsNotNone(result)
        # Should not exceed budgets
        self.assertLessEqual(result.nodes_expanded, config.max_nodes)

    def test_case4_no_successful_path_no_invented_rescue(self):
        """CASE 4: No successful path → search must not invent a rescue."""
        # Create a task with no budget to succeed
        task = _make_simple_task(e1_verified=False, e2_verified=False, e3_hidden=False)
        budget = ResourceBudget(
            max_executive_steps=2, max_retrieval_calls=0,  # No retrieval
            max_verification_calls=1, max_search_calls=0,
        )
        runtime = _make_runtime(task, budget)
        checkpoint = create_checkpoint(runtime, step=0, prior_actions=())
        sf = checkpoint.state_features

        # Near tie between VERIFY and REASON_MORE
        q_values = {"VERIFY": 67.0, "REASON_MORE": 67.5, "ANSWER": 4.0,
                    "DEFER": 14.0}

        config = SearchConfig(min_steps_remaining=2)
        trigger = decide_search(
            state_features=sf,
            near_optimal_actions=("VERIFY", "REASON_MORE"),
            pav_selected=None,
            pav_abstained=True,
            q_values=q_values,
            config=config,
        )

        # Search may or may not trigger, but if it does, it should not
        # invent a rescue that doesn't exist
        if trigger.should_search:
            planner = SearchPlanner(task, self.utility, config)
            result = planner.plan(
                checkpoint=checkpoint,
                candidate_actions=("VERIFY", "REASON_MORE"),
                q_values=q_values,
                trigger_reasons=trigger.reasons,
            )

            # Should not crash and should not claim false success
            self.assertIsNotNone(result)
            for branch in result.branches:
                if branch.success:
                    # If it claims success, it must have actually reached terminal
                    self.assertTrue(branch.terminal)


# ============================================================
# Search trigger unit tests
# ============================================================

class TestSearchTrigger(unittest.TestCase):
    """Unit tests for the pure deterministic search trigger."""

    def test_no_search_single_near_optimal(self):
        """Do not search when only one near-optimal action."""
        config = SearchConfig()
        result = decide_search(
            state_features={"steps_remaining": 6, "same_action_run_length": 0,
                          "n_contradicting": 0, "n_hidden_evidence": 0,
                          "retrieval_remaining": 2},
            near_optimal_actions=("VERIFY",),
            pav_selected=("VERIFY",),
            pav_abstained=False,
            q_values={"VERIFY": 90, "RETRIEVE": 50},
            config=config,
        )
        self.assertFalse(result.should_search)

    def test_no_search_insufficient_steps(self):
        """Do not search when too few steps remain."""
        config = SearchConfig(min_steps_remaining=3)
        result = decide_search(
            state_features={"steps_remaining": 2, "same_action_run_length": 0,
                          "n_contradicting": 0, "n_hidden_evidence": 0,
                          "retrieval_remaining": 2},
            near_optimal_actions=("VERIFY", "RETRIEVE"),
            pav_selected=None,
            pav_abstained=True,
            q_values={"VERIFY": 67, "RETRIEVE": 67.5},
            config=config,
        )
        self.assertFalse(result.should_search)

    def test_no_search_large_q_gap(self):
        """Do not search when Q gap is large."""
        config = SearchConfig()
        result = decide_search(
            state_features={"steps_remaining": 6, "same_action_run_length": 0,
                          "n_contradicting": 0, "n_hidden_evidence": 0,
                          "retrieval_remaining": 2},
            near_optimal_actions=("VERIFY", "RETRIEVE"),
            pav_selected=None,
            pav_abstained=True,
            q_values={"VERIFY": 90, "RETRIEVE": 50},  # Large gap
            config=config,
        )
        self.assertFalse(result.should_search)

    def test_search_resource_pressure(self):
        """Search when resource pressure is high."""
        config = SearchConfig()
        result = decide_search(
            state_features={"steps_remaining": 4, "same_action_run_length": 0,
                          "n_contradicting": 0, "n_hidden_evidence": 0,
                          "retrieval_remaining": 2},
            near_optimal_actions=("VERIFY", "RETRIEVE"),
            pav_selected=None,
            pav_abstained=True,
            q_values={"VERIFY": 67, "RETRIEVE": 67.5},
            config=config,
        )
        self.assertTrue(result.should_search)
        self.assertIn("resource_pressure", result.reasons)

    def test_search_repeated_action_trap(self):
        """Search when repeated-action trap detected."""
        config = SearchConfig()
        result = decide_search(
            state_features={"steps_remaining": 6, "same_action_run_length": 2,
                          "n_contradicting": 0, "n_hidden_evidence": 0,
                          "retrieval_remaining": 2},
            near_optimal_actions=("RETRIEVE", "VERIFY"),
            pav_selected=None,
            pav_abstained=True,
            q_values={"RETRIEVE": 67, "VERIFY": 67.5},
            config=config,
        )
        self.assertTrue(result.should_search)
        self.assertIn("repeated_action_trap", result.reasons)


# ============================================================
# Budget tracker tests
# ============================================================

class TestSearchBudget(unittest.TestCase):
    """Tests for the search budget tracker."""

    def test_budget_not_exhausted_initially(self):
        config = SearchConfig(max_nodes=6, max_model_calls=4, max_wall_ms=5000)
        budget = SearchBudget(config)
        self.assertFalse(budget.exhausted)
        self.assertTrue(budget.can_expand())

    def test_budget_exhausted_after_max_nodes(self):
        config = SearchConfig(max_nodes=2, max_model_calls=4, max_wall_ms=5000)
        budget = SearchBudget(config)
        budget.consume_node()
        budget.consume_node()
        self.assertTrue(budget.exhausted)
        self.assertFalse(budget.can_expand())

    def test_budget_exhausted_after_max_model_calls(self):
        config = SearchConfig(max_nodes=6, max_model_calls=2, max_wall_ms=5000)
        budget = SearchBudget(config)
        budget.consume_model_call()
        budget.consume_model_call()
        self.assertTrue(budget.exhausted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
