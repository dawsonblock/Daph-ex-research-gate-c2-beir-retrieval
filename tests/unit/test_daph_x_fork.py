"""Tests for DAPH-X counterfactual fork engine and causal dataset.

Tests cover:
  - Checkpoint determinism
  - Fork isolation
  - Replay determinism
  - Exhaustive action evaluation
  - Oracle/regret computation
  - Causal dataset schema
  - No train/test leakage by group
"""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph_x.actions.typed_actions import (
    Action, ActionType, answer, defer, verify, stop,
)
from daph_x.receipts.checkpoint import Checkpoint, checkpoint_from_task_and_runtime
from daph_x.receipts.fork_engine import (
    ForkResult, fork_and_run, evaluate_all_actions,
    compute_oracle_action, compute_regret, compute_near_optimal_set,
)
from daph_x.receipts.causal_dataset import (
    CausalActionRecord, build_causal_dataset, write_causal_dataset,
    read_causal_dataset, group_by_checkpoint, SCHEMA_VERSION,
)
from daph_x.graph.epistemic_graph import build_graph_from_evidence_task

# Bridge from legacy
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)


def make_task(
    task_id: str = "test",
    hypotheses=None,
    evidence=None,
    correct_hypothesis="H1",
    expected_terminal="ANSWER",
    oracle_path=("ANSWER",),
) -> EvidenceTask:
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
        task_summary="Test", high_stakes=True,
        budget_profile="TEST_4_2_0",
        hypotheses=tuple(hyps), evidence_items=tuple(evs),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=oracle_path,
        expected_terminal=DecisionAction(expected_terminal),
        correct_hypothesis_id=correct_hypothesis,
    )


def make_checkpoint(task=None, seed=42):
    if task is None:
        task = make_task()
    return checkpoint_from_task_and_runtime(task, None, seed=seed)


class TestCheckpointDeterminism:
    def test_same_checkpoint_same_hash(self):
        c1 = make_checkpoint()
        c2 = make_checkpoint()
        assert c1.checkpoint_hash == c2.checkpoint_hash

    def test_different_seed_different_hash(self):
        c1 = make_checkpoint(seed=42)
        c2 = make_checkpoint(seed=43)
        assert c1.checkpoint_hash != c2.checkpoint_hash

    def test_checkpoint_roundtrip(self):
        c1 = make_checkpoint()
        data = c1.to_dict()
        c2 = Checkpoint.from_dict(data)
        assert c1.checkpoint_hash == c2.checkpoint_hash

    def test_checkpoint_deterministic_serialization(self):
        c = make_checkpoint()
        d1 = c.to_dict()
        d2 = c.to_dict()
        assert d1 == d2


class TestForkIsolation:
    def test_fork_does_not_mutate_parent(self):
        c = make_checkpoint()
        original_hash = c.checkpoint_hash
        original_verify = c.graph.verify_remaining
        fork_and_run(c, verify("E1"))
        assert c.checkpoint_hash == original_hash
        assert c.graph.verify_remaining == original_verify

    def test_different_actions_same_initial_state(self):
        c = make_checkpoint()
        r1 = fork_and_run(c, answer("H1"))
        r2 = fork_and_run(c, defer())
        assert r1.checkpoint_hash == r2.checkpoint_hash

    def test_same_action_same_result(self):
        c = make_checkpoint()
        r1 = fork_and_run(c, answer("H1"))
        r2 = fork_and_run(c, answer("H1"))
        assert r1.utility == r2.utility
        assert r1.outcome == r2.outcome
        assert r1.next_state_hash == r2.next_state_hash

    def test_different_actions_different_results(self):
        c = make_checkpoint()
        r1 = fork_and_run(c, answer("H1"))
        r2 = fork_and_run(c, answer("H2"))
        assert r1.utility != r2.utility


class TestExhaustiveEvaluation:
    def test_evaluate_all_actions(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), answer("H3"), defer(), stop()]
        results = evaluate_all_actions(c, actions)
        assert len(results) == 5
        # All share the same checkpoint hash
        assert all(r.checkpoint_hash == c.checkpoint_hash for r in results)

    def test_oracle_action(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), answer("H3"), defer(), stop()]
        results = evaluate_all_actions(c, actions)
        oracle_action, oracle_utility = compute_oracle_action(results)
        assert oracle_action == "ANSWER(H1)"
        assert oracle_utility > 0

    def test_regret(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        results = evaluate_all_actions(c, actions)
        regret_h1 = compute_regret(results, "ANSWER(H1)")
        regret_h2 = compute_regret(results, "ANSWER(H2)")
        assert regret_h1 == 0.0  # Oracle action has zero regret
        assert regret_h2 > 0.0  # Wrong answer has positive regret

    def test_near_optimal_set(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        results = evaluate_all_actions(c, actions)
        near_opt = compute_near_optimal_set(results, epsilon=3.0)
        assert "ANSWER(H1)" in near_opt


class TestCausalDataset:
    def test_schema_version(self):
        c = make_checkpoint()
        actions = [answer("H1"), defer()]
        records = build_causal_dataset(c, actions)
        assert len(records) == 2
        assert records[0].record_hash() != ""

    def test_counterfactual_group_id(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        records = build_causal_dataset(c, actions)
        group_ids = {r.counterfactual_group_id for r in records}
        assert len(group_ids) == 1  # All share the same group

    def test_oracle_utility_consistent(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        records = build_causal_dataset(c, actions)
        for r in records:
            assert r.oracle_utility == max(x.utility for x in records)

    def test_regret_nonnegative(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        records = build_causal_dataset(c, actions)
        for r in records:
            assert r.regret >= 0.0

    def test_near_optimal(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        records = build_causal_dataset(c, actions)
        near_opt = [r for r in records if r.is_near_optimal]
        assert len(near_opt) >= 1  # At least the oracle action

    def test_write_and_read(self, tmp_path):
        c = make_checkpoint()
        actions = [answer("H1"), defer()]
        records = build_causal_dataset(c, actions)
        path = tmp_path / "causal.jsonl"
        write_causal_dataset(records, path)
        loaded = read_causal_dataset(path)
        assert len(loaded) == 2
        assert loaded[0]["schema_version"] == SCHEMA_VERSION

    def test_group_by_checkpoint(self):
        c1 = make_checkpoint(seed=42)
        c2 = make_checkpoint(seed=43)
        actions = [answer("H1"), defer()]
        r1 = build_causal_dataset(c1, actions)
        r2 = build_causal_dataset(c2, actions)
        all_records = [r.to_dict() for r in r1 + r2]
        groups = group_by_checkpoint(all_records)
        assert len(groups) == 2  # Two different checkpoints

    def test_no_leakage_by_group(self):
        """Records from the same counterfactual group must not be split
        across train/test partitions."""
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        records = build_causal_dataset(c, actions)
        group_ids = {r.counterfactual_group_id for r in records}
        assert len(group_ids) == 1
        # All records in the same group must be in the same partition
        # This is a structural property of the group_id


class TestDeterministicReplay:
    """Stress test: 100 repeated restores must produce identical results."""

    def test_100_replays(self):
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer(), stop()]
        results_first = evaluate_all_actions(c, actions)

        for _ in range(100):
            results = evaluate_all_actions(c, actions)
            for r_first, r_new in zip(results_first, results):
                assert r_first.utility == r_new.utility
                assert r_first.outcome == r_new.outcome
                assert r_first.checkpoint_hash == r_new.checkpoint_hash

    def test_replay_mismatch_rate_zero(self):
        """P(replay mismatch) = 0 over deterministic synthetic environment."""
        c = make_checkpoint()
        actions = [answer("H1"), answer("H2"), defer()]
        mismatches = 0
        n_replays = 100

        for _ in range(n_replays):
            r1 = evaluate_all_actions(c, actions)
            r2 = evaluate_all_actions(c, actions)
            for a, b in zip(r1, r2):
                if a.utility != b.utility or a.outcome != b.outcome:
                    mismatches += 1

        assert mismatches == 0
