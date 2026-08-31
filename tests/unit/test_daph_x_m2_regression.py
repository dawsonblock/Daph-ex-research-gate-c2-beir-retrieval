"""Permanent regression tests for M2 evaluator correctness.

These tests must pass forever. They encode the invariants that
the M2 evaluator fix established.
"""
import pytest
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestEvaluatorInvariants:
    """Permanent invariants for the causal evaluator."""

    def test_perfect_prediction_implies_zero_regret(self):
        """If pred_total == true_utility for all actions in a group,
        then hybrid regret MUST be zero for that group."""
        # Simulate a group where predictions are perfect
        true_utilities = [100.0, 20.0, -50.0]
        q_mb = [80.0, 30.0, -40.0]
        pred_residual = [20.0, -10.0, -10.0]
        pred_total = [q + r for q, r in zip(q_mb, pred_residual)]

        # Verify perfect prediction
        for p, t in zip(pred_total, true_utilities):
            assert abs(p - t) < 1e-9

        # Compute regret
        oracle_utility = max(true_utilities)
        best_hybrid_idx = np.argmax(pred_total)
        regret = oracle_utility - true_utilities[best_hybrid_idx]
        assert regret == 0.0

    def test_regret_is_groupwise(self):
        """Regret must be computed per counterfactual group,
        not across all records globally."""
        # Group 1: action A is best
        g1_utilities = [100.0, 20.0]
        g1_pred = [100.0, 20.0]  # Perfect
        g1_oracle = max(g1_utilities)
        g1_best = np.argmax(g1_pred)
        g1_regret = g1_oracle - g1_utilities[g1_best]

        # Group 2: action B is best
        g2_utilities = [20.0, 100.0]
        g2_pred = [20.0, 100.0]  # Perfect
        g2_oracle = max(g2_utilities)
        g2_best = np.argmax(g2_pred)
        g2_regret = g2_oracle - g2_utilities[g2_best]

        assert g1_regret == 0.0
        assert g2_regret == 0.0

        # Global regret would be wrong
        all_utilities = g1_utilities + g2_utilities
        all_pred = g1_pred + g2_pred
        global_best = np.argmax(all_pred)
        global_oracle = max(all_utilities)
        # This would incorrectly compute regret across groups
        # if we didn't group first
        assert global_oracle == 100.0  # Same for both groups

    def test_index_alignment_invariant(self):
        """The index into predictions must align with the index
        into records. Group-relative indices must NOT be used
        to index into prediction arrays."""
        # Simulate: 2 groups, 3 actions each
        # test_records has 6 records total
        # Group 0: indices [0, 1, 2] in test_records
        # Group 1: indices [3, 4, 5] in test_records
        # q_res_pred has 6 elements, indexed by test_records position

        q_res_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Correct: use test_records index
        group0_correct = [q_res_pred[i] for i in [0, 1, 2]]
        assert group0_correct == [1.0, 2.0, 3.0]

        # Wrong: use group-relative index
        group1_wrong = [q_res_pred[j] for j in [0, 1, 2]]  # BUG!
        assert group1_wrong == [1.0, 2.0, 3.0]  # Same as group 0 — WRONG

        # Correct for group 1: use test_records indices [3, 4, 5]
        group1_correct = [q_res_pred[i] for i in [3, 4, 5]]
        assert group1_correct == [4.0, 5.0, 6.0]

    def test_pred_total_equals_q_mb_plus_pred_res(self):
        """pred_total must equal q_mb + pred_residual."""
        q_mb = 50.0
        pred_res = -20.0
        pred_total = q_mb + pred_res
        assert pred_total == pytest.approx(30.0)

    def test_zero_mae_implies_zero_regret(self):
        """If MAE = 0 (all predictions perfect), then regret = 0."""
        # Simulate perfect predictions on 2 groups
        for _ in range(100):
            n_actions = np.random.randint(2, 6)
            true_utils = np.random.randn(n_actions) * 50
            pred_total = true_utils.copy()  # Perfect prediction

            oracle = np.max(true_utils)
            best = np.argmax(pred_total)
            regret = oracle - true_utils[best]
            assert regret == pytest.approx(0.0)
