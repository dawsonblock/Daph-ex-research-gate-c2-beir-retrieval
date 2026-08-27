"""Unit and invariant tests for DAPH Authority V2 (A2AD_ASYMMETRIC_HARD_SELECT).

12 hard invariants plus boundary tests at the threshold.
"""
import pytest
from daph.authority import (
    AuthorityMode,
    AuthorityDecision,
    StructuralState,
    decide_authority,
    build_receipt,
    AUTHORITY_THRESHOLD,
)


# ============================================================
# Helper: create a safe DEFER structural state
# ============================================================

def safe_defer_structural():
    return StructuralState(
        has_competing_unverified_support=False,
        n_hyp_unverified_support=1,
        n_hyp_unverified_contradiction=1,
        can_verify=False,
        verify_budget_exhausted=False,
        all_evidence_verified=False,
    )


def unsafe_defer_structural():
    return StructuralState(
        has_competing_unverified_support=True,
        n_hyp_unverified_support=2,
        n_hyp_unverified_contradiction=0,
        can_verify=True,
        verify_budget_exhausted=False,
        all_evidence_verified=False,
    )


def answer_structural():
    return StructuralState(
        has_competing_unverified_support=False,
        n_hyp_unverified_support=0,
        n_hyp_unverified_contradiction=0,
        can_verify=False,
        verify_budget_exhausted=True,
        all_evidence_verified=True,
    )


# ============================================================
# 12 Hard Invariants
# ============================================================

class TestHardInvariants:
    """The 12 hard invariants from the I3.29 plan."""

    def test_1_answer_authority_matches_v1_on_known_answer_cases(self):
        """Invariant 1: ANSWER authority behavior matches frozen V1 on known ANSWER cases."""
        q = {"ANSWER": 100.0, "VERIFY": 90.0, "DEFER": -30.0, "STOP": -30.0}
        legal = ["ANSWER", "VERIFY", "DEFER", "STOP"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=answer_structural(), answer_safety_passed=True)
        assert decision.mode == AuthorityMode.HARD_ANSWER
        assert decision.action == "ANSWER"

    def test_2_defer_cannot_fire_if_competing_support(self):
        """Invariant 2: DEFER cannot hard-fire if has_competing_unverified_support=True."""
        q = {"DEFER": 70.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=unsafe_defer_structural())
        assert decision.mode == AuthorityMode.ADVISORY
        assert "DEFER_COMPETING_UNVERIFIED_SUPPORT" in decision.reason_codes

    def test_3_defer_cannot_fire_if_not_argmax(self):
        """Invariant 3: DEFER cannot fire if DEFER is not Q argmax."""
        q = {"VERIFY": 90.0, "DEFER": 70.0, "STOP": -30.0}
        legal = ["VERIFY", "DEFER", "STOP"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=safe_defer_structural())
        assert decision.mode == AuthorityMode.ADVISORY

    def test_4_defer_cannot_fire_if_gap_below_5(self):
        """Invariant 4: DEFER cannot fire if q_gap < 5."""
        q = {"DEFER": 70.0, "REASON_MORE": 67.0, "STOP": -30.0}
        legal = ["DEFER", "REASON_MORE", "STOP"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=safe_defer_structural())
        assert decision.mode == AuthorityMode.ADVISORY
        assert "DEFER_GAP_TOO_SMALL" in decision.reason_codes

    def test_5_answer_cannot_fire_if_safety_fails(self):
        """Invariant 5: ANSWER cannot fire if ANSWER safety fails."""
        q = {"ANSWER": 100.0, "VERIFY": 90.0, "DEFER": -30.0}
        legal = ["ANSWER", "VERIFY", "DEFER"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=answer_structural(), answer_safety_passed=False)
        assert decision.mode == AuthorityMode.ADVISORY
        assert "ANSWER_SAFETY_FAILED" in decision.reason_codes

    def test_6_advisory_leaves_behavior_unchanged(self):
        """Invariant 6: Advisory mode does not force any action."""
        q = {"DEFER": 70.0, "REASON_MORE": 67.0, "STOP": -30.0}
        legal = ["DEFER", "REASON_MORE", "STOP"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=safe_defer_structural())
        assert decision.mode == AuthorityMode.ADVISORY
        assert decision.action is None

    def test_7_hard_authority_action_must_be_legal(self):
        """Invariant 7: Hard authority action must be in legal_actions."""
        q = {"ANSWER": 100.0, "VERIFY": 90.0}
        legal = ["VERIFY"]  # ANSWER not legal
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=answer_structural())
        # ANSWER is not legal, so it can't be argmax of legal actions
        assert decision.mode == AuthorityMode.ADVISORY

    def test_8_same_state_produces_same_decision(self):
        """Invariant 8: Deterministic — same input produces same output."""
        q = {"DEFER": 70.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        structural = safe_defer_structural()
        d1 = decide_authority(q_values=q, legal_actions=legal, structural=structural)
        d2 = decide_authority(q_values=q, legal_actions=legal, structural=structural)
        assert d1.mode == d2.mode
        assert d1.action == d2.action
        assert d1.reason_codes == d2.reason_codes

    def test_9_no_hidden_evidence_consumed(self):
        """Invariant 9: Authority logic uses only StructuralState fields, not hidden evidence."""
        # The decide_authority function only takes q_values, legal_actions, structural, answer_safety
        # It does not accept or inspect hidden evidence
        # This is enforced by the API signature
        import inspect
        sig = inspect.signature(decide_authority)
        params = list(sig.parameters.keys())
        assert "hidden_evidence" not in params
        assert "evidence" not in params
        assert "verify_result" not in params

    def test_10_no_future_outcome_fields(self):
        """Invariant 10: No future outcome fields enter authority logic."""
        import inspect
        sig = inspect.signature(decide_authority)
        params = list(sig.parameters.keys())
        assert "future_outcome" not in params
        assert "oracle" not in params
        assert "expected_terminal" not in params

    def test_11_receipt_is_complete(self):
        """Invariant 11: Authority receipt contains all required fields."""
        q = {"DEFER": 70.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        structural = safe_defer_structural()
        decision = decide_authority(q_values=q, legal_actions=legal, structural=structural)
        receipt = build_receipt(
            state_features={"n_verified": 0, "n_visible_evidence": 2},
            legal_actions=legal, q_values=q,
            structural=structural, decision=decision)
        required = ["state_sha256", "legal_actions", "q_values", "q_argmax",
                    "q_gap", "structural", "authority", "rule_version", "threshold"]
        for field in required:
            assert field in receipt, f"Missing field: {field}"

    def test_12_error_fails_closed_to_advisory(self):
        """Invariant 12: Any authority-layer error fails closed to ADVISORY."""
        # Pass invalid input that will cause an error
        decision = decide_authority(
            q_values=None,  # type: ignore — will cause error
            legal_actions=["DEFER"],
            structural=safe_defer_structural())
        assert decision.mode == AuthorityMode.ADVISORY
        assert decision.action is None
        assert any("AUTHORITY_ERROR" in code for code in decision.reason_codes)


# ============================================================
# Boundary tests at the threshold
# ============================================================

class TestBoundaryConditions:
    """Boundary tests at the authority threshold."""

    def test_gap_4_999_is_advisory(self):
        """Gap of 4.999 should be ADVISORY (below threshold)."""
        q = {"DEFER": 70.0, "STOP": 65.001, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=safe_defer_structural())
        assert decision.mode == AuthorityMode.ADVISORY
        assert decision.q_gap < AUTHORITY_THRESHOLD

    def test_gap_5_000_is_eligible(self):
        """Gap of exactly 5.000 should be eligible (>= is inclusive)."""
        q = {"DEFER": 70.0, "STOP": 65.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=safe_defer_structural())
        assert decision.mode == AuthorityMode.HARD_DEFER
        assert decision.q_gap >= AUTHORITY_THRESHOLD

    def test_gap_5_001_is_eligible(self):
        """Gap of 5.001 should be eligible (above threshold)."""
        q = {"DEFER": 70.0, "STOP": 64.999, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=safe_defer_structural())
        assert decision.mode == AuthorityMode.HARD_DEFER

    def test_answer_gap_4_999_is_advisory(self):
        """ANSWER gap of 4.999 should be ADVISORY."""
        q = {"ANSWER": 100.0, "VERIFY": 95.001, "DEFER": -30.0}
        legal = ["ANSWER", "VERIFY", "DEFER"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=answer_structural(), answer_safety_passed=True)
        assert decision.mode == AuthorityMode.ADVISORY

    def test_answer_gap_5_000_is_eligible(self):
        """ANSWER gap of exactly 5.000 should be eligible."""
        q = {"ANSWER": 100.0, "VERIFY": 95.0, "DEFER": -30.0}
        legal = ["ANSWER", "VERIFY", "DEFER"]
        decision = decide_authority(
            q_values=q, legal_actions=legal,
            structural=answer_structural(), answer_safety_passed=True)
        assert decision.mode == AuthorityMode.HARD_ANSWER


# ============================================================
# DEFER safety predicate tests
# ============================================================

class TestDeferSafetyPredicate:
    """Tests for the DEFER structural safety predicate."""

    def test_defer_fires_when_verify_unavailable(self):
        """DEFER fires when VERIFY is unavailable and all other conditions pass."""
        s = StructuralState(
            has_competing_unverified_support=False,
            n_hyp_unverified_support=1, n_hyp_unverified_contradiction=1,
            can_verify=False, verify_budget_exhausted=False, all_evidence_verified=False)
        q = {"DEFER": 70.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(q_values=q, legal_actions=legal, structural=s)
        assert decision.mode == AuthorityMode.HARD_DEFER

    def test_defer_fires_when_verify_budget_exhausted(self):
        """DEFER fires when verification budget is exhausted."""
        s = StructuralState(
            has_competing_unverified_support=False,
            n_hyp_unverified_support=1, n_hyp_unverified_contradiction=1,
            can_verify=True, verify_budget_exhausted=True, all_evidence_verified=False)
        q = {"DEFER": 70.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(q_values=q, legal_actions=legal, structural=s)
        assert decision.mode == AuthorityMode.HARD_DEFER

    def test_defer_fires_when_all_evidence_verified(self):
        """DEFER fires when all evidence is already verified."""
        s = StructuralState(
            has_competing_unverified_support=False,
            n_hyp_unverified_support=0, n_hyp_unverified_contradiction=0,
            can_verify=True, verify_budget_exhausted=False, all_evidence_verified=True)
        q = {"DEFER": 70.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(q_values=q, legal_actions=legal, structural=s)
        assert decision.mode == AuthorityMode.HARD_DEFER

    def test_defer_blocked_when_verify_available_and_unverified_evidence(self):
        """DEFER blocked when VERIFY is available and evidence is unverified."""
        s = StructuralState(
            has_competing_unverified_support=False,
            n_hyp_unverified_support=1, n_hyp_unverified_contradiction=1,
            can_verify=True, verify_budget_exhausted=False, all_evidence_verified=False)
        q = {"DEFER": 70.0, "VERIFY": 60.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "VERIFY", "STOP", "ANSWER"]
        decision = decide_authority(q_values=q, legal_actions=legal, structural=s)
        assert decision.mode == AuthorityMode.ADVISORY
        assert "DEFER_SAFETY_FAILED_CONTINUATION_AVAILABLE" in decision.reason_codes

    def test_defer_blocked_when_competing_support_even_with_large_gap(self):
        """DEFER blocked when has_competing_unverified_support=True even with gap=100."""
        s = unsafe_defer_structural()
        q = {"DEFER": 70.0, "STOP": -30.0, "ANSWER": -120.0}
        legal = ["DEFER", "STOP", "ANSWER"]
        decision = decide_authority(q_values=q, legal_actions=legal, structural=s)
        assert decision.mode == AuthorityMode.ADVISORY
        assert "DEFER_COMPETING_UNVERIFIED_SUPPORT" in decision.reason_codes
