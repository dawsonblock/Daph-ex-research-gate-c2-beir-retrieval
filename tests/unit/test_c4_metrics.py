"""Tests for the C4 metrics module — the authoritative scoring formulas.

These tests pin the exact quality, oracle-gap capture, and selector-gap
capture formulas so that no other module can silently recreate them
incorrectly.
"""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.c4.metrics import (
    compute_quality,
    evidence_complete,
    oracle_gap_capture,
    selector_gap_capture,
    arm_quality,
)


# --- 1.1  Quality ---

class TestComputeQuality:
    """All four states of the partial-credit quality rule."""

    def test_complete_and_correct(self):
        assert compute_quality(correct=True, evidence_complete=True) == 1.0

    def test_complete_and_incorrect(self):
        assert compute_quality(correct=False, evidence_complete=True) == 0.5

    def test_incomplete_and_correct(self):
        assert compute_quality(correct=True, evidence_complete=False) == 0.25

    def test_incomplete_and_incorrect(self):
        assert compute_quality(correct=False, evidence_complete=False) == 0.0

    @pytest.mark.parametrize("correct,complete,expected", [
        (True, True, 1.0),
        (False, True, 0.5),
        (True, False, 0.25),
        (False, False, 0.0),
    ])
    def test_all_states_parametrized(self, correct, complete, expected):
        assert compute_quality(correct=correct, evidence_complete=complete) == expected


class TestEvidenceComplete:
    def test_all_required_present(self):
        assert evidence_complete(["a", "b"], ["a", "b", "c"]) is True

    def test_missing_one(self):
        assert evidence_complete(["a", "b"], ["a", "c"]) is False

    def test_empty_required(self):
        assert evidence_complete([], ["a"]) is True

    def test_empty_selected_with_required(self):
        assert evidence_complete(["a"], []) is False


# --- 1.2  Oracle-gap capture ---

class TestOracleGapCapture:
    """OGC = (Q(C4_4) - Q(C4_0)) / (Q(C4_6) - Q(C4_0)).

    The numerator MUST use C4_4, not C4_5.  This test uses synthetic
    numbers where C4_5 and C4_4 are intentionally different so the old
    bug cannot silently return.
    """

    def test_standard_case(self):
        qualities = {"C4_0": 0.16, "C4_4": 0.34, "C4_6": 0.95}
        # (0.34 - 0.16) / (0.95 - 0.16) = 0.18 / 0.79
        expected = (0.34 - 0.16) / (0.95 - 0.16)
        assert oracle_gap_capture(qualities) == pytest.approx(expected)

    def test_c4_5_not_in_numerator(self):
        """Regression test: C4_5 must NOT appear in the OGC numerator."""
        qualities = {"C4_0": 0.0, "C4_4": 0.5, "C4_5": 0.9, "C4_6": 1.0}
        # Correct: (0.5 - 0.0) / (1.0 - 0.0) = 0.5
        # Old bug:  (0.9 - 0.0) / (1.0 - 0.0) = 0.9
        assert oracle_gap_capture(qualities) == pytest.approx(0.5)
        assert oracle_gap_capture(qualities) != pytest.approx(0.9)

    def test_zero_gap_returns_none(self):
        qualities = {"C4_0": 0.5, "C4_4": 0.5, "C4_6": 0.5}
        assert oracle_gap_capture(qualities) is None

    def test_missing_arm_returns_none(self):
        assert oracle_gap_capture({"C4_0": 0.1, "C4_4": 0.2}) is None
        assert oracle_gap_capture({"C4_0": 0.1, "C4_6": 0.2}) is None


# --- 1.3  Selector-gap capture ---

class TestSelectorGapCapture:
    """SGC = (Q(C4_4) - Q(C4_3)) / (Q(C4_5) - Q(C4_3))."""

    def test_standard_case(self):
        qualities = {"C4_3": 0.21, "C4_4": 0.34, "C4_5": 0.79}
        # (0.34 - 0.21) / (0.79 - 0.21) = 0.13 / 0.58
        expected = (0.34 - 0.21) / (0.79 - 0.21)
        assert selector_gap_capture(qualities) == pytest.approx(expected)

    def test_zero_gap_returns_none(self):
        qualities = {"C4_3": 0.5, "C4_4": 0.5, "C4_5": 0.5}
        assert selector_gap_capture(qualities) is None

    def test_missing_arm_returns_none(self):
        assert selector_gap_capture({"C4_3": 0.1, "C4_4": 0.2}) is None


# --- 1.4  Arm quality ---

class TestArmQuality:
    def test_mean(self):
        assert arm_quality([1.0, 0.5, 0.0, 0.25]) == pytest.approx(0.4375)

    def test_empty(self):
        assert arm_quality([]) == 0.0

    def test_all_correct(self):
        assert arm_quality([1.0, 1.0, 1.0]) == pytest.approx(1.0)
