"""Tests for the ANSWER_PROBE_GATE_V1 analysis scripts (GPU-free): the
separation stop-gate and the training/evaluation ladder. Both were validated
against synthetic data with known separation properties before any real
collection data existed -- these tests pin that same validation permanently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analyze_answer_probe_separation import cohens_d, grouped_bootstrap_mean_diff_ci
from scripts.train_answer_probe_gate import fit_logistic_regression, logistic_predict_proba, utility


class TestCohensD:
    def test_identical_groups_give_zero(self):
        assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-9)

    def test_large_separation_gives_large_d(self):
        a = [0.1, 0.15, 0.2, 0.12]
        b = [0.8, 0.85, 0.9, 0.82]
        d = cohens_d(a, b)
        assert d < -3.0  # a's mean is far below b's, large negative effect size

    def test_sign_flips_with_argument_order(self):
        a = [0.1, 0.2, 0.15]
        b = [0.8, 0.9, 0.85]
        assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a), abs=1e-9)


class TestGroupedBootstrapMeanDiffCI:
    def test_large_separation_ci_excludes_zero(self):
        pairs_a = [("f1", 0.1), ("f1", 0.15), ("f2", 0.12), ("f2", 0.18)]
        pairs_b = [("f1", 0.8), ("f1", 0.85), ("f2", 0.82), ("f2", 0.88)]
        lo, hi = grouped_bootstrap_mean_diff_ci(pairs_a, pairs_b)
        assert lo < hi
        assert not (lo <= 0.0 <= hi)  # CI excludes zero

    def test_identical_groups_ci_straddles_zero(self):
        pairs_a = [("f1", 0.5), ("f1", 0.5), ("f2", 0.5), ("f2", 0.5)]
        pairs_b = [("f1", 0.5), ("f1", 0.5), ("f2", 0.5), ("f2", 0.5)]
        lo, hi = grouped_bootstrap_mean_diff_ci(pairs_a, pairs_b)
        assert lo <= 0.0 <= hi


class TestLogisticRegression:
    def test_perfectly_separable_data_achieves_high_train_accuracy(self):
        rng = np.random.RandomState(0)
        X_pos = rng.normal(loc=3.0, scale=0.3, size=(20, 2))
        X_neg = rng.normal(loc=-3.0, scale=0.3, size=(20, 2))
        X = np.vstack([X_pos, X_neg])
        y = np.array([1.0] * 20 + [0.0] * 20)
        w, b, mean, std = fit_logistic_regression(X, y, iterations=1000)
        preds = [1 if logistic_predict_proba(x, w, b, mean, std) >= 0.5 else 0 for x in X]
        accuracy = sum(1 for p, t in zip(preds, y) if p == t) / len(y)
        assert accuracy >= 0.9

    def test_random_labels_give_near_chance_accuracy(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(40, 2))
        y = rng.randint(0, 2, size=40).astype(float)
        w, b, mean, std = fit_logistic_regression(X, y, iterations=500)
        preds = [1 if logistic_predict_proba(x, w, b, mean, std) >= 0.5 else 0 for x in X]
        accuracy = sum(1 for p, t in zip(preds, y) if p == t) / len(y)
        # not a tight bound -- just confirms it doesn't spuriously memorize
        # noise into near-100% train accuracy the way an overfit model might
        assert accuracy <= 0.85


class TestUtility:
    def test_all_accept_uses_q_direct(self):
        records = [{"q_direct": 1, "q_memory": 0}, {"q_direct": 0, "q_memory": 1}]
        assert utility(records, ["ACCEPT", "ACCEPT"]) == pytest.approx(0.5)

    def test_all_escalate_uses_q_memory(self):
        records = [{"q_direct": 1, "q_memory": 0}, {"q_direct": 0, "q_memory": 1}]
        assert utility(records, ["ESCALATE", "ESCALATE"]) == pytest.approx(0.5)

    def test_oracle_style_mixed_decisions_reach_the_max_of_each_task(self):
        records = [{"q_direct": 1, "q_memory": 0}, {"q_direct": 0, "q_memory": 1}]
        assert utility(records, ["ACCEPT", "ESCALATE"]) == pytest.approx(1.0)

    def test_empty_records_returns_zero_not_a_crash(self):
        assert utility([], []) == 0.0
