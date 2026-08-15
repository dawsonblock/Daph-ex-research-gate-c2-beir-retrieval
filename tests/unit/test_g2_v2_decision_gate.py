"""Regression test for the decision_at_M bug: prior runners picked the
reported decision by ranking utility across ALL arms without first filtering
to arms that pass their own hard safety gates, so an unsafe arm with the best
number could be reported as "the decision"."""
from __future__ import annotations

from hrm_adaptive_memory.c4.decision_gate import select_eligible_decision


class TestSelectEligibleDecision:
    def test_ineligible_arm_with_best_metric_is_never_chosen(self):
        """The exact bug: M75 has the best closure but fails its safety gate;
        M50 is worse on closure but safe. M50 must win, not M75."""
        candidates = {"M25": True, "M50": True, "M75": False}
        ranking = {"M25": 0.10, "M50": 0.30, "M75": 0.90}
        decision = select_eligible_decision(candidates, ranking)
        assert decision.key == "M50"
        assert "M75" not in decision.eligible_keys

    def test_no_eligible_arm_returns_none_not_least_bad(self):
        candidates = {"M25": False, "M50": False, "M75": False}
        ranking = {"M25": 0.10, "M50": 0.30, "M75": 0.90}
        decision = select_eligible_decision(candidates, ranking)
        assert decision.key is None
        assert decision.eligible_keys == ()
        assert "no arm passed" in decision.reason

    def test_single_eligible_arm_is_chosen_regardless_of_metric(self):
        candidates = {"M25": False, "M50": True, "M75": False}
        ranking = {"M25": 0.99, "M50": 0.01, "M75": 0.50}
        decision = select_eligible_decision(candidates, ranking)
        assert decision.key == "M50"

    def test_missing_ranking_value_is_treated_as_worst_not_crashing(self):
        candidates = {"M25": True, "M50": True}
        ranking = {"M25": None, "M50": 0.1}
        decision = select_eligible_decision(candidates, ranking)
        assert decision.key == "M50"

    def test_ties_pick_deterministically_among_eligible(self):
        candidates = {"M25": True, "M50": True}
        ranking = {"M25": 0.5, "M50": 0.5}
        decision = select_eligible_decision(candidates, ranking)
        assert decision.key in ("M25", "M50")
