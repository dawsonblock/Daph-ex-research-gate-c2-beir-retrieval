#!/usr/bin/env python3
"""Regression tests for the shared answer verifier.

These tests pin the critical behavior that was broken in C4 evaluator_v1:
HRM control tokens (e.g. ``<|box_end|>``) must be stripped before comparing
answers. They also verify that the C4 verifier and the historical
``context_study.verify_answer`` produce identical results, ensuring a single
source of truth.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hrm_adaptive_memory.evaluation.verifiers import normalize_answer, verify_answer
from hrm_adaptive_memory.experiments.context_study import OracleTask, verify_answer as historical_verify


# --- normalize_answer: control token stripping ---

def test_normalize_strips_box_end():
    assert normalize_answer("THETA-OLIVE<|box_end|>") == "theta-olive"


def test_normalize_strips_multiple_control_tokens():
    assert normalize_answer("<|box_start|>42<|box_end|>") == "42"


def test_normalize_collapses_whitespace():
    assert normalize_answer("  alpha   beta  ") == "alpha beta"


def test_normalize_preserves_hyphens():
    assert normalize_answer("THETA-OLIVE") == "theta-olive"


# --- verify_answer: exact ---

def test_exact_with_control_token():
    score, passed = verify_answer("exact", "THETA-OLIVE", "THETA-OLIVE<|box_end|>")
    assert passed and score == 1.0


def test_exact_case_insensitive():
    score, passed = verify_answer("exact", "Alpha", "alpha")
    assert passed and score == 1.0


def test_exact_mismatch():
    score, passed = verify_answer("exact", "alpha", "beta")
    assert not passed and score == 0.0


# --- verify_answer: numeric ---

def test_numeric_with_control_token():
    score, passed = verify_answer("numeric", "42", "42<|box_end|>")
    assert passed and score == 1.0


def test_numeric_float():
    score, passed = verify_answer("numeric", "3.14", "3.14")
    assert passed and score == 1.0


def test_numeric_last_number_wins():
    score, passed = verify_answer("numeric", "42", "7 and 42")
    assert passed and score == 1.0


def test_numeric_mismatch():
    score, passed = verify_answer("numeric", "42", "7")
    assert not passed and score == 0.0


# --- verify_answer: canonical ---

def test_canonical_with_control_token():
    """The exact bug that broke C4 evaluator_v1: canonical answer + <|box_end|>."""
    score, passed = verify_answer("canonical", "THETA-OLIVE", "THETA-OLIVE<|box_end|>")
    assert passed and score == 1.0


def test_canonical_multi_word():
    score, passed = verify_answer("canonical", "alpha beta", "alpha beta<|box_end|>")
    assert passed and score == 1.0


def test_canonical_must_be_last():
    """Listing the answer among other candidates does not pass."""
    score, passed = verify_answer("canonical", "alpha", "alpha beta")
    assert not passed and score == 0.0


def test_canonical_mismatch():
    score, passed = verify_answer("canonical", "alpha", "beta<|box_end|>")
    assert not passed and score == 0.0


# --- Fallback ---

def test_unknown_verifier_falls_back_to_exact():
    score, passed = verify_answer("unknown_type", "alpha", "alpha<|box_end|>")
    assert passed and score == 1.0


# --- Parity: C4 shared verifier == historical context_study verifier ---

def _make_task(verifier: str, answer: str) -> OracleTask:
    return OracleTask(
        task_id="t1", question="q", answer=answer,
        required_evidence_ids=("e1",), oracle_evidence_ids=("e1",),
        family="f", template_id="tm", source_cluster_id="c", split="dev",
        verifier=verifier,
    )


def test_parity_exact_with_control_token():
    task = _make_task("exact", "THETA-OLIVE")
    output = "THETA-OLIVE<|box_end|>"
    assert historical_verify(task, output) == verify_answer("exact", task.answer, output)


def test_parity_canonical_with_control_token():
    task = _make_task("canonical", "THETA-OLIVE")
    output = "THETA-OLIVE<|box_end|>"
    assert historical_verify(task, output) == verify_answer("canonical", task.answer, output)


def test_parity_numeric():
    task = _make_task("numeric", "42")
    output = "42<|box_end|>"
    assert historical_verify(task, output) == verify_answer("numeric", task.answer, output)


def test_parity_exact_mismatch():
    task = _make_task("exact", "alpha")
    output = "beta<|box_end|>"
    assert historical_verify(task, output) == verify_answer("exact", task.answer, output)


def test_parity_canonical_must_be_last():
    task = _make_task("canonical", "alpha")
    output = "alpha beta"
    assert historical_verify(task, output) == verify_answer("canonical", task.answer, output)


# --- Extended coverage: symbolic, enum, boolean, json_field ---

def test_symbolic_answer_with_control_token():
    """Symbolic answers (e.g. THETA-OLIVE) must survive control token stripping."""
    score, passed = verify_answer("canonical", "THETA-OLIVE", "THETA-OLIVE<|box_end|>")
    assert passed and score == 1.0


def test_enum_answer():
    """Enum answers (e.g. 'accredited', 'provisional') use canonical verifier."""
    score, passed = verify_answer("canonical", "accredited", "accredited<|box_end|>")
    assert passed and score == 1.0


def test_enum_answer_mismatch():
    score, passed = verify_answer("canonical", "accredited", "provisional<|box_end|>")
    assert not passed and score == 0.0


def test_boolean_answer_true():
    """Boolean answers use canonical verifier."""
    score, passed = verify_answer("canonical", "true", "true<|box_end|>")
    assert passed and score == 1.0


def test_boolean_answer_false():
    score, passed = verify_answer("canonical", "false", "false<|box_end|>")
    assert passed and score == 1.0


def test_json_field_answer():
    """JSON field answers (e.g. '{"grade": "accredited"}') use canonical verifier."""
    score, passed = verify_answer("canonical", '{"grade": "accredited"}', '{"grade": "accredited"}<|box_end|>')
    assert passed and score == 1.0


def test_json_field_answer_with_extra_text():
    """JSON field answer must be the last candidate, not embedded in other text."""
    score, passed = verify_answer("canonical", "accredited", "provisional accredited<|box_end|>")
    assert passed and score == 1.0


# --- Case and whitespace differences ---

def test_case_difference_exact():
    score, passed = verify_answer("exact", "Alpha", "ALPHA")
    assert passed and score == 1.0


def test_case_difference_canonical():
    score, passed = verify_answer("canonical", "Theta-Olive", "theta-olive<|box_end|>")
    assert passed and score == 1.0


def test_whitespace_difference_exact():
    score, passed = verify_answer("exact", "alpha  beta", "alpha beta")
    assert passed and score == 1.0


def test_whitespace_with_newlines():
    score, passed = verify_answer("exact", "alpha", "\n  alpha  \n")
    assert passed and score == 1.0


def test_leading_trailing_whitespace_canonical():
    score, passed = verify_answer("canonical", "alpha beta", "  alpha beta  <|box_end|>")
    assert passed and score == 1.0


# --- Extended parity: all verifier types ---

def test_parity_symbolic():
    task = _make_task("canonical", "THETA-OLIVE")
    output = "THETA-OLIVE<|box_end|>"
    assert historical_verify(task, output) == verify_answer("canonical", task.answer, output)


def test_parity_enum():
    task = _make_task("canonical", "accredited")
    output = "accredited<|box_end|>"
    assert historical_verify(task, output) == verify_answer("canonical", task.answer, output)


def test_parity_boolean():
    task = _make_task("canonical", "true")
    output = "true<|box_end|>"
    assert historical_verify(task, output) == verify_answer("canonical", task.answer, output)


def test_parity_case_difference():
    task = _make_task("exact", "Alpha")
    output = "ALPHA<|box_end|>"
    assert historical_verify(task, output) == verify_answer("exact", task.answer, output)


def test_parity_whitespace():
    task = _make_task("exact", "alpha beta")
    output = "alpha   beta<|box_end|>"
    assert historical_verify(task, output) == verify_answer("exact", task.answer, output)


# --- Frozen fixture set: assert historical == shared for all types ---

FROZEN_FIXTURES = [
    ("exact", "THETA-OLIVE", "THETA-OLIVE<|box_end|>"),
    ("exact", "alpha", "alpha"),
    ("exact", "Alpha", "alpha"),
    ("exact", "alpha beta", "alpha   beta"),
    ("exact", "42", "42<|box_end|>"),
    ("exact", "alpha", "beta"),
    ("numeric", "42", "42<|box_end|>"),
    ("numeric", "3.14", "3.14"),
    ("numeric", "42", "7 and 42"),
    ("numeric", "42", "7"),
    ("canonical", "THETA-OLIVE", "THETA-OLIVE<|box_end|>"),
    ("canonical", "alpha beta", "alpha beta<|box_end|>"),
    ("canonical", "alpha", "alpha beta"),
    ("canonical", "alpha", "beta<|box_end|>"),
    ("canonical", "accredited", "accredited<|box_end|>"),
    ("canonical", "true", "true<|box_end|>"),
    ("canonical", "false", "false<|box_end|>"),
]


def test_frozen_fixture_set_historical_equals_shared():
    """For every frozen fixture, historical verifier == shared verifier."""
    for verifier, answer, output in FROZEN_FIXTURES:
        task = _make_task(verifier, answer)
        h = historical_verify(task, output)
        s = verify_answer(verifier, answer, output)
        assert h == s, (
            f"Parity failure: verifier={verifier!r} answer={answer!r} "
            f"output={output!r}: historical={h} shared={s}"
        )
