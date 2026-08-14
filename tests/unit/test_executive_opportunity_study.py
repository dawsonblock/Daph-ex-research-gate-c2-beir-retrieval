"""Tests for the frozen, pure decision logic in
scripts/run_executive_opportunity_study.py -- the pipeline integration itself
is validated via --dry-run (see configs/gate_executive_opportunity_v1.json),
but E2's heuristic is the one piece of decision logic that must match the
frozen protocol byte-for-byte, so it gets a direct unit test.
"""
from __future__ import annotations

import pytest

from scripts.run_executive_opportunity_study import competition_bucket, e2_heuristic


class TestE2Heuristic:
    """Matches configs/gate_executive_opportunity_v1.json E2_FROZEN_HEURISTIC
    exactly: USE_CERTIFIED_MEMORY iff identity resolved AND >=1 complete path."""

    @pytest.mark.parametrize("status", ["EXACT", "RESOLVED"])
    def test_resolved_identity_with_a_complete_path_uses_memory(self, status):
        assert e2_heuristic(status, 1) == "A1_USE_CERTIFIED_MEMORY"
        assert e2_heuristic(status, 5) == "A1_USE_CERTIFIED_MEMORY"

    @pytest.mark.parametrize("status", ["EXACT", "RESOLVED"])
    def test_resolved_identity_with_zero_complete_paths_answers_now(self, status):
        assert e2_heuristic(status, 0) == "A0_ANSWER_NOW"

    @pytest.mark.parametrize("status", ["AMBIGUOUS", "UNRESOLVED", "NONE", ""])
    def test_unresolved_identity_answers_now_regardless_of_paths(self, status):
        assert e2_heuristic(status, 0) == "A0_ANSWER_NOW"
        assert e2_heuristic(status, 8) == "A0_ANSWER_NOW"  # paths alone are not sufficient


class TestCompetitionBucket:
    def test_boundaries(self):
        assert competition_bucket(0) == "1"
        assert competition_bucket(1) == "1"
        assert competition_bucket(2) == "2-3"
        assert competition_bucket(3) == "2-3"
        assert competition_bucket(4) == "4-6"
        assert competition_bucket(6) == "4-6"
        assert competition_bucket(7) == "7+"
        assert competition_bucket(100) == "7+"
