"""Tests for hrm_adaptive_memory/experiments/exec_training_dataset.py -- the
ANSWER_NOW-viable family for Executive v0's training split.

The central property under test: every generated question round-trips
through the REAL, unmodified extract_subject/extract_target_relation with
zero mismatch -- this is the whole point of this module (eliminating the
privileged-parsing ambiguity found in EOB-v1/v2's D0/D2/D3).
"""
from __future__ import annotations

import hrm_adaptive_memory.evaluation  # noqa: F401  (cycle-breaker)

import pytest

from hrm_adaptive_memory.experiments.exec_training_dataset import (
    ParserVerificationError, build_answer_now_tasks, verify_native_parsing)


class TestBuildAnswerNowTasks:
    def test_builds_both_domains(self):
        tasks = build_answer_now_tasks()
        domains = {t.domain for t in tasks}
        assert domains == {"capital", "element"}

    def test_no_duplicate_task_ids(self):
        tasks = build_answer_now_tasks()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_no_duplicate_subjects(self):
        """Duplicate subjects across facts would make identity resolution
        ambiguous in a pooled corpus."""
        tasks = build_answer_now_tasks()
        subjects = [t.subject for t in tasks]
        assert len(subjects) == len(set(subjects))

    def test_every_task_has_zero_evidence_by_construction(self):
        # ExecTrainingTask has no evidence field -- this family is
        # zero-evidence by design, verified at the suite-builder level
        # (required_evidence_ids=[] is set there); here we just confirm the
        # dataclass carries no evidence-adjacent state that could leak in.
        tasks = build_answer_now_tasks()
        for t in tasks:
            assert not hasattr(t, "evidence")

    def test_deterministic(self):
        a = build_answer_now_tasks()
        b = build_answer_now_tasks()
        assert [(t.question, t.answer) for t in a] == [(t.question, t.answer) for t in b]


class TestVerifyNativeParsing:
    def test_the_full_curated_set_passes(self):
        """The actual integrity property this module exists for: every
        hand-curated task round-trips through the REAL parser with zero
        mismatch. Reaching this line without an exception IS the test."""
        tasks = build_answer_now_tasks()
        verify_native_parsing(tasks)  # raises on any mismatch

    def test_a_genuinely_incompatible_question_is_rejected(self):
        """Mutation-style: construct a task whose question does NOT match
        the real parser's patterns (the exact failure mode found in EOB-v1's
        original D0/D2/D3 bypass) and confirm it is rejected, not silently
        accepted."""
        from hrm_adaptive_memory.experiments.exec_training_dataset import ExecTrainingTask
        bad = ExecTrainingTask(
            task_id="bad-1", domain="capital",
            question="If you compute 6 minus 60, what is the capital city for France?",
            answer="Paris", subject="France", metadata={"relation": "capital city"})
        with pytest.raises(ParserVerificationError):
            verify_native_parsing([bad])

    def test_wrong_declared_subject_is_rejected(self):
        from hrm_adaptive_memory.experiments.exec_training_dataset import ExecTrainingTask
        bad = ExecTrainingTask(
            task_id="bad-2", domain="capital",
            question="What is the capital city for France?",
            answer="Paris", subject="Germany",  # wrong on purpose
            metadata={"relation": "capital city"})
        with pytest.raises(ParserVerificationError):
            verify_native_parsing([bad])
