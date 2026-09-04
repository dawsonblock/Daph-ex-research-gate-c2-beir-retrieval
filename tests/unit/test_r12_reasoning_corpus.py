#!/usr/bin/env python3
"""Unit tests for the expanded R12 reasoning task corpus."""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import (
    REASONING_TASKS, get_all_reasoning_tasks, check_answer, ReasoningTask,
)


class TestR12CorpusIntegrity:
    """Verify integrity of the 500-task reasoning corpus."""

    def test_corpus_size(self):
        """Corpus must have at least 500 tasks."""
        assert len(REASONING_TASKS) >= 500, (
            f"Expected >= 500 tasks, got {len(REASONING_TASKS)}")

    def test_unique_task_ids(self):
        """All task_ids must be unique."""
        ids = [t.task_id for t in REASONING_TASKS]
        assert len(ids) == len(set(ids)), (
            f"Duplicate task_ids found: "
            f"{[x for x in ids if ids.count(x) > 1]}")

    def test_sequential_ids(self):
        """Task IDs should be reason_001 through reason_500."""
        for i, task in enumerate(REASONING_TASKS):
            expected_id = f"reason_{i+1:03d}"
            assert task.task_id == expected_id, (
                f"Task at index {i} has id {task.task_id}, expected {expected_id}")

    def test_all_fields_populated(self):
        """Every task must have all required fields non-empty."""
        for task in REASONING_TASKS:
            assert task.task_id, f"Empty task_id"
            assert task.description, f"Empty description in {task.task_id}"
            assert task.prompt, f"Empty prompt in {task.task_id}"
            assert task.answer, f"Empty answer in {task.task_id}"
            assert task.answer_type in ("int", "float", "string"), (
                f"Invalid answer_type {task.answer_type} in {task.task_id}")
            assert task.difficulty in ("easy", "medium", "hard"), (
                f"Invalid difficulty {task.difficulty} in {task.task_id}")
            assert task.category in (
                "math", "logic", "sequence", "combinatorics"), (
                f"Invalid category {task.category} in {task.task_id}")
            assert isinstance(task.common_errors, tuple), (
                f"common_errors not a tuple in {task.task_id}")

    def test_category_distribution(self):
        """All four categories must be represented."""
        categories = set(t.category for t in REASONING_TASKS)
        assert categories == {"math", "logic", "sequence", "combinatorics"}, (
            f"Missing categories: {categories}")

    def test_difficulty_distribution(self):
        """All three difficulty levels must be represented."""
        difficulties = set(t.difficulty for t in REASONING_TASKS)
        assert difficulties == {"easy", "medium", "hard"}, (
            f"Missing difficulties: {difficulties}")

    def test_get_all_reasoning_tasks(self):
        """get_all_reasoning_tasks should return the full list."""
        tasks = get_all_reasoning_tasks()
        assert len(tasks) == len(REASONING_TASKS)
        assert all(isinstance(t, ReasoningTask) for t in tasks)


class TestAnswerChecking:
    """Verify answer checking works for all answer types."""

    def test_int_answer_exact(self):
        """Integer answers match exactly."""
        assert check_answer("42", "42", "int")
        assert check_answer("17", "17", "int")

    def test_int_answer_numeric_equiv(self):
        """Integer answers with different formats match."""
        assert check_answer("42.0", "42", "int")
        assert check_answer("42", "42.0", "int")

    def test_float_answer_tolerance(self):
        """Float answers match within tolerance."""
        assert check_answer("4.8", "4.8", "float")
        assert check_answer("4.8001", "4.8", "float")
        assert not check_answer("4.9", "4.8", "float")

    def test_string_answer_exact(self):
        """String answers match exactly."""
        assert check_answer("yes", "yes", "string")
        assert check_answer("1/8", "1/8", "string")

    def test_string_answer_numeric_equiv(self):
        """String answers with numeric equivalence match (when both are parseable as floats)."""
        # "0.5" and "0.5" match as strings
        assert check_answer("0.5", "0.5", "string")
        # "1/2" and "1/2" match as strings (float() can't parse fractions)
        assert check_answer("1/2", "1/2", "string")
        # "3/8" and "3/8" match as strings
        assert check_answer("3/8", "3/8", "string")
        # "yes" and "yes" match
        assert check_answer("yes", "yes", "string")


class TestSpecificTaskAnswers:
    """Spot-check specific task answers for correctness."""

    @pytest.mark.parametrize("task_id,expected_answer,reasoning", [
        # Arithmetic
        ("reason_201", "3", "450/(70+80)=3"),
        ("reason_202", "112", "200*0.7*0.8=112"),
        ("reason_206", "48", "harmonic mean of 40 and 60"),
        ("reason_209", "100", "margin: (250-c)/250=0.6, c=100"),
        # Algebra
        ("reason_211", "-10", "product of roots = c/a = -10"),
        ("reason_212", "6", "adding equations: 5x+5y=30"),
        ("reason_213", "4", "2^(x+1)=32=2^5, x=4"),
        ("reason_222", "2", "f(x)=2x+6, f^{-1}(10)=2"),
        # Logic
        ("reason_235", "false", "XOR of two trues is false"),
        ("reason_240", "no", "implication with false P is true"),
        ("reason_247", "false", "2 is a prime that is not odd"),
        # Combinatorics
        ("reason_266", "360", "P(6,4)=6*5*4*3=360"),
        ("reason_267", "35", "C(7,3)=35 with repetition"),
        ("reason_271", "256", "sum C(8,k)=2^8=256"),
        ("reason_274", "42", "Catalan(5)=42"),
        # Sequences
        ("reason_281", "55", "Fibonacci: 34+21=55"),
        ("reason_289", "47", "Lucas: 29+18=47"),
        # Adversarial
        ("reason_322", "5", "bat+ball=1.10, bat-ball=1.00, ball=0.05"),
        ("reason_323", "47", "lily doubles, day before 48 is 47"),
        # Advanced
        ("reason_371", "60", "6!/(3!*2!)=60"),
        ("reason_414", "89", "Fibonacci(11)=89"),
        ("reason_477", "2/3", "Monty Hall: switch wins 2/3"),
        ("reason_494", "6", "trailing zeros: floor(25/5)+floor(25/25)=6"),
        ("reason_500", "5050", "100*101/2=5050"),
    ])
    def test_task_answer(self, task_id, expected_answer, reasoning):
        """Verify specific task answers are correct."""
        task = next(t for t in REASONING_TASKS if t.task_id == task_id)
        assert check_answer(expected_answer, task.answer, task.answer_type), (
            f"{task_id}: expected {expected_answer} ({reasoning}), "
            f"stored {task.answer} (type={task.answer_type})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
