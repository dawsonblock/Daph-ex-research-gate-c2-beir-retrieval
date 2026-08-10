"""Tests for hrm_adaptive_memory/experiments/eob_v1_dataset.py -- the D0/D2/D3
generator for the Executive Opportunity Benchmark v1.

The central integrity property under test: every D0 answer is independently
re-derivable by reference_solve() from the question text alone (never trusted
from the generator's own internal state), and D2/D3 evidence records never
leak the exact question or accidentally match the correct answer.
"""
from __future__ import annotations

import hrm_adaptive_memory.evaluation  # noqa: F401  (cycle-breaker, see other test files)

import pytest

from hrm_adaptive_memory.experiments.eob_v1_dataset import (
    ReferenceSolverMismatch, _distractor_value, build_d0_tasks, build_d2_tasks,
    build_d3_tasks, reference_solve, select_d0_subset)


class TestReferenceSolver:
    def test_arithmetic(self):
        assert reference_solve("If you compute 12 plus 30, what is the result?") == "42"
        assert reference_solve("If you compute 50 minus 8, what is the result?") == "42"
        assert reference_solve("If you compute 6 times 7, what is the result?") == "42"

    def test_comparison_highest_and_lowest(self):
        q_high = "Alpha has height 10. Beta has height 20. Gamma has height 5. Which entity has the highest height?"
        assert reference_solve(q_high) == "Beta"
        q_low = "Alpha has height 10. Beta has height 20. Gamma has height 5. Which entity has the lowest height?"
        assert reference_solve(q_low) == "Gamma"

    def test_transform_reverse_upper_concat(self):
        assert reference_solve("Reverse the letters in 'cat'.") == "tac"
        assert reference_solve("Convert 'cat' to uppercase.") == "CAT"
        assert reference_solve("Concatenate 'foo' and 'bar', in that order.") == "foobar"

    def test_restatement(self):
        assert reference_solve("Assume X is set to 99. What is the value of X?") == "99"

    def test_unmatched_question_raises(self):
        with pytest.raises(ReferenceSolverMismatch):
            reference_solve("This matches no known template.")


class TestBuildD0Tasks:
    def test_every_task_passes_reference_solver_verification_by_construction(self):
        """build_d0_tasks itself raises on any mismatch -- reaching this line
        at all is already the proof; this test exists so a future refactor
        that accidentally removes the internal check gets caught."""
        tasks = build_d0_tasks(seed=1, tasks_per_family=5)
        assert len(tasks) == 20  # 4 families x 5
        for t in tasks:
            assert reference_solve(t.question) == t.answer

    def test_all_four_families_present(self):
        tasks = build_d0_tasks(seed=1, tasks_per_family=3)
        assert {t.family for t in tasks} == {"arithmetic", "comparison", "transform", "restatement"}

    def test_task_ids_are_unique(self):
        tasks = build_d0_tasks(seed=1, tasks_per_family=10)
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_d0_tasks_have_no_evidence(self):
        for t in build_d0_tasks(seed=1, tasks_per_family=3):
            assert t.evidence == []

    def test_deterministic_given_same_seed(self):
        a = build_d0_tasks(seed=42, tasks_per_family=5)
        b = build_d0_tasks(seed=42, tasks_per_family=5)
        assert [(t.question, t.answer) for t in a] == [(t.question, t.answer) for t in b]


class TestBuildD2Tasks:
    def test_confirming_evidence_never_leaks_the_verbatim_question(self):
        d0 = build_d0_tasks(seed=2, tasks_per_family=50)
        d2 = build_d2_tasks(d0, seed=3)
        for t in d2:
            assert t.question not in t.evidence[0]["content"]

    def test_required_evidence_id_matches_the_single_confirming_record(self):
        d0 = build_d0_tasks(seed=2, tasks_per_family=5)
        d2 = build_d2_tasks(d0, seed=3)
        for t in d2:
            assert len(t.evidence) == 1
            assert t.required_evidence_ids == [t.evidence[0]["evidence_id"]]

    def test_d2_preserves_the_same_question_and_answer_as_its_d0_source(self):
        d0 = build_d0_tasks(seed=2, tasks_per_family=5)
        d2 = build_d2_tasks(d0, seed=3)
        for d0_t, d2_t in zip(d0, d2):
            assert d2_t.question == d0_t.question
            assert d2_t.answer == d0_t.answer


class TestBuildD3Tasks:
    def test_distractor_value_never_equals_the_correct_answer(self):
        """build_d3_tasks itself raises on any collision -- this stress-tests
        across many seeds/family draws to catch a rare-but-possible collision
        (this exact test caught a real bug: the transform distractor could
        randomly regenerate the same last character 1/26 of the time)."""
        for seed in range(20):
            d0 = build_d0_tasks(seed=100 + seed, tasks_per_family=20)
            d3 = build_d3_tasks(d0, seed=200 + seed)  # raises on any collision
            for t in d3:
                dv = _distractor_value(t.family, t.evidence[0]["content"])
                assert dv != t.answer

    def test_distractor_is_not_required_evidence(self):
        """The distractor is a plausible-looking but wrong/irrelevant record
        -- the answer never legitimately depends on it."""
        d0 = build_d0_tasks(seed=2, tasks_per_family=5)
        d3 = build_d3_tasks(d0, seed=3)
        for t in d3:
            assert t.required_evidence_ids == []

    def test_d3_preserves_the_same_question_and_answer_as_its_d0_source(self):
        d0 = build_d0_tasks(seed=2, tasks_per_family=5)
        d3 = build_d3_tasks(d0, seed=3)
        for d0_t, d3_t in zip(d0, d3):
            assert d3_t.question == d0_t.question
            assert d3_t.answer == d0_t.answer


class TestSelectD0Subset:
    def test_returns_exactly_n_tasks(self):
        d0 = build_d0_tasks(seed=1, tasks_per_family=25)  # 100 tasks
        subset = select_d0_subset(d0, seed=10, n=60)
        assert len(subset) == 60

    def test_subset_is_drawn_from_the_original_pool(self):
        d0 = build_d0_tasks(seed=1, tasks_per_family=25)
        subset = select_d0_subset(d0, seed=10, n=60)
        pool_ids = {t.task_id for t in d0}
        assert all(t.task_id in pool_ids for t in subset)

    def test_no_duplicates_in_the_subset(self):
        d0 = build_d0_tasks(seed=1, tasks_per_family=25)
        subset = select_d0_subset(d0, seed=10, n=60)
        ids = [t.task_id for t in subset]
        assert len(ids) == len(set(ids))

    def test_deterministic_given_same_seed(self):
        d0 = build_d0_tasks(seed=1, tasks_per_family=25)
        a = select_d0_subset(d0, seed=10, n=60)
        b = select_d0_subset(d0, seed=10, n=60)
        assert [t.task_id for t in a] == [t.task_id for t in b]

    def test_requesting_more_than_the_pool_raises(self):
        d0 = build_d0_tasks(seed=1, tasks_per_family=25)
        with pytest.raises(ValueError):
            select_d0_subset(d0, seed=10, n=101)
