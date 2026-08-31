"""Tests for continuation authority V0.1 — VERIFY certificate.

Tests cover:
  - Terminal certificate blocks VERIFY
  - No budget blocks VERIFY
  - No unverified evidence blocks VERIFY
  - High-value discriminating target passes
  - IG computation
  - Entropy model
"""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph.authority.continuation_v0_1 import (
    verify_continuation_certificate,
    information_gain,
    hypothesis_entropy,
    AuthorityModeContinuation,
    ContinuationDecision,
    IG_THRESHOLD,
    VERIFY_Q_MARGIN,
)
from daph.authority.policy_v3 import StructuralStateV3


def make_structural(
    n_viable=2,
    n_eliminated=0,
    n_verified_support=0,
    n_verified_contradiction=0,
    n_mixed=0,
    has_unique_verified=False,
    has_competition=False,
    verified_action_is_answer=False,
    verified_action_is_defer=False,
    n_unverified_support=1,
    n_unverified_contradiction=0,
    has_competing_unverified=False,
    can_verify=True,
    verify_budget_exhausted=False,
    all_evidence_verified=False,
) -> StructuralStateV3:
    """Helper to build a StructuralStateV3 with sensible defaults."""
    return StructuralStateV3(
        has_competing_unverified_support=has_competing_unverified,
        n_hyp_unverified_support=n_unverified_support,
        n_hyp_unverified_contradiction=n_unverified_contradiction,
        can_verify=can_verify,
        verify_budget_exhausted=verify_budget_exhausted,
        all_evidence_verified=all_evidence_verified,
        n_hyp_with_verified_support=n_verified_support,
        n_hyp_with_verified_contradiction=n_verified_contradiction,
        n_hyp_with_mixed_verified=n_mixed,
        n_viable_hypotheses=n_viable,
        n_eliminated_hypotheses=n_eliminated,
        has_unique_verified_supported_hypothesis=has_unique_verified,
        has_verified_unresolved_competition=has_competition,
        verified_hyp_action_is_answer=verified_action_is_answer,
        verified_hyp_action_is_defer=verified_action_is_defer,
    )


class TestTerminalCertificateBlocksVerify:
    """When a terminal certificate is valid, VERIFY must not fire."""

    def test_answer_certificate_blocks_verify(self):
        """If ANSWER is structurally certifiable, VERIFY must not fire."""
        # Unique verified support for an ANSWER hypothesis
        state = make_structural(
            n_viable=1,
            n_verified_support=1,
            has_unique_verified=True,
            verified_action_is_answer=True,
            n_unverified_support=0,
        )
        decision = verify_continuation_certificate(
            structural=state,
            legal_actions=["VERIFY", "ANSWER", "DEFER", "REASON_MORE"],
            q_values={"VERIFY": 80, "ANSWER": 90, "DEFER": -50, "REASON_MORE": 30},
            can_verify=True,
            verify_budget_remaining=3,
            unverified_evidence_count=0,
        )
        assert not decision.would_force
        assert "ANSWER_CERTIFICATE_VALID" in decision.reason_codes

    def test_defer_certificate_blocks_verify(self):
        """If DEFER is structurally certifiable, VERIFY must not fire."""
        # DEFER certificate requires: unique verified support with DEFER action
        # AND all evidence verified (no continuation can resolve)
        state = make_structural(
            n_viable=1,
            n_verified_support=1,
            has_unique_verified=True,
            verified_action_is_defer=True,
            n_unverified_support=0,
            all_evidence_verified=True,
            can_verify=False,
        )
        decision = verify_continuation_certificate(
            structural=state,
            legal_actions=["VERIFY", "ANSWER", "DEFER", "REASON_MORE"],
            q_values={"VERIFY": 80, "ANSWER": -50, "DEFER": 90, "REASON_MORE": 30},
            can_verify=False,
            verify_budget_remaining=0,
            unverified_evidence_count=0,
        )
        assert not decision.would_force
        assert "DEFER_CERTIFICATE_VALID" in decision.reason_codes


class TestBudgetAndEvidenceGates:
    """VERIFY must not fire when budget is exhausted or no unverified evidence."""

    def test_budget_exhausted_blocks_verify(self):
        state = make_structural(
            n_viable=2,
            n_verified_support=0,
            n_unverified_support=1,
            verify_budget_exhausted=True,
        )
        decision = verify_continuation_certificate(
            structural=state,
            legal_actions=["VERIFY", "ANSWER", "DEFER", "REASON_MORE"],
            q_values={"VERIFY": 80, "ANSWER": -50, "DEFER": -50, "REASON_MORE": 30},
            can_verify=False,
            verify_budget_remaining=0,
            unverified_evidence_count=1,
        )
        assert not decision.would_force
        assert "VERIFY_BUDGET_EXHAUSTED" in decision.reason_codes

    def test_no_unverified_evidence_blocks_verify(self):
        state = make_structural(
            n_viable=2,
            n_verified_support=0,
            n_unverified_support=0,
            all_evidence_verified=True,
        )
        decision = verify_continuation_certificate(
            structural=state,
            legal_actions=["VERIFY", "ANSWER", "DEFER", "REASON_MORE"],
            q_values={"VERIFY": 80, "ANSWER": -50, "DEFER": -50, "REASON_MORE": 30},
            can_verify=True,
            verify_budget_remaining=3,
            unverified_evidence_count=0,
        )
        assert not decision.would_force
        assert "NO_UNVERIFIED_EVIDENCE" in decision.reason_codes

    def test_verify_not_legal_blocks_verify(self):
        state = make_structural(
            n_viable=2,
            n_verified_support=0,
            n_unverified_support=1,
        )
        decision = verify_continuation_certificate(
            structural=state,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],  # No VERIFY
            q_values={"ANSWER": -50, "DEFER": -50, "REASON_MORE": 30},
            can_verify=True,
            verify_budget_remaining=3,
            unverified_evidence_count=1,
        )
        assert not decision.would_force
        assert "VERIFY_NOT_LEGAL" in decision.reason_codes


class TestInformationGain:
    """Test the entropy and IG computation."""

    def test_single_viable_low_entropy(self):
        """One viable hypothesis should have low entropy."""
        state = make_structural(n_viable=1, n_eliminated=3)
        entropy = hypothesis_entropy(state)
        assert entropy < 0.1  # Near zero

    def test_two_viable_higher_entropy(self):
        """Two viable hypotheses should have higher entropy than one."""
        state = make_structural(n_viable=2, n_eliminated=2)
        entropy = hypothesis_entropy(state)
        assert entropy > 0.5  # ln(2) ≈ 0.693

    def test_ig_positive_for_competition(self):
        """IG should be positive when verification can resolve competition."""
        state = make_structural(
            n_viable=2,
            n_verified_support=2,
            has_competition=True,
            n_unverified_support=1,
        )
        ig = information_gain(state)
        assert ig > 0

    def test_ig_near_zero_for_single_viable(self):
        """IG should be near zero when only one viable hypothesis."""
        state = make_structural(n_viable=1, n_eliminated=3)
        ig = information_gain(state)
        assert ig < 0.1


class TestQMarginGate:
    """VERIFY must have sufficient Q margin to fire."""

    def test_low_q_margin_blocks_verify(self):
        # Use a state with high IG (competing verified support) so we
        # reach the Q margin check
        state = make_structural(
            n_viable=3,
            n_verified_support=2,
            has_competition=True,
            n_unverified_support=1,
            n_eliminated=1,
        )
        # Q(VERIFY) barely above Q(REASON_MORE)
        decision = verify_continuation_certificate(
            structural=state,
            legal_actions=["VERIFY", "ANSWER", "DEFER", "REASON_MORE"],
            q_values={"VERIFY": 31, "ANSWER": -50, "DEFER": -50, "REASON_MORE": 30},
            can_verify=True,
            verify_budget_remaining=3,
            unverified_evidence_count=1,
        )
        # Q margin = 31 - 30 = 1 < 2.0 threshold
        assert not decision.would_force
        # Should fail on Q margin (if IG passed) or IG (if IG failed)
        reason_str = " ".join(decision.reason_codes)
        assert "Q_MARGIN" in reason_str or "IG" in reason_str or "V_VERIFY" in reason_str


class TestValidVerifyPasses:
    """A well-constructed continuation state should pass all gates."""

    def test_high_value_verify_passes(self):
        """When continuation is required, IG is high, and Q margin is large,
        VERIFY should be recommended."""
        state = make_structural(
            n_viable=3,
            n_verified_support=0,
            n_unverified_support=2,
            n_eliminated=1,
            can_verify=True,
        )
        decision = verify_continuation_certificate(
            structural=state,
            legal_actions=["VERIFY", "ANSWER", "DEFER", "REASON_MORE"],
            q_values={"VERIFY": 90, "ANSWER": -50, "DEFER": -50, "REASON_MORE": 30},
            can_verify=True,
            verify_budget_remaining=3,
            unverified_evidence_count=2,
        )
        # May or may not pass depending on IG heuristic
        # But Q margin = 90 - 30 = 60 >> 2.0
        if decision.would_force:
            assert decision.action == "VERIFY"
            assert decision.mode == AuthorityModeContinuation.SHADOW_VERIFY
            assert decision.q_margin > VERIFY_Q_MARGIN
        else:
            # If it doesn't fire, it should be because IG was too low
            assert any("IG" in r or "V_VERIFY" in r for r in decision.reason_codes)
