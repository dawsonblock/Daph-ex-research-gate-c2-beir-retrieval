"""Tests for V3 authority policy with positive structural certificates."""
import pytest
from daph.authority.policy_v3 import (
    StructuralStateV3, decide_authority_v3,
    answer_structural_certificate, defer_structural_certificate,
    AuthorityMode, FROZEN_RULE_VERSION_V3,
)
from daph.authority.policy import AUTHORITY_THRESHOLD, I2_EPSILON_Q


def make_structural(**kwargs):
    """Create a StructuralStateV3 with sensible defaults."""
    defaults = dict(
        has_competing_unverified_support=False,
        n_hyp_unverified_support=0,
        n_hyp_unverified_contradiction=0,
        can_verify=False,
        verify_budget_exhausted=True,
        all_evidence_verified=False,
        n_hyp_with_verified_support=0,
        n_hyp_with_verified_contradiction=0,
        n_hyp_with_mixed_verified=0,
        n_viable_hypotheses=2,
        n_eliminated_hypotheses=0,
        has_unique_verified_supported_hypothesis=False,
        has_verified_unresolved_competition=False,
        verified_hyp_action_is_answer=False,
        verified_hyp_action_is_defer=False,
    )
    defaults.update(kwargs)
    return StructuralStateV3(**defaults)


class TestAnswerStructuralCertificate:
    """Test the ANSWER positive structural certificate."""

    def test_unique_verified_support_answer_passes(self):
        """ANSWER certificate passes when unique verified support with ANSWER action."""
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_answer=True,
        )
        assert answer_structural_certificate(s) is True

    def test_no_verified_support_fails(self):
        """ANSWER certificate fails when no verified support."""
        s = make_structural(
            n_hyp_with_verified_support=0,
            has_unique_verified_supported_hypothesis=False,
        )
        assert answer_structural_certificate(s) is False

    def test_verified_support_but_defer_action_fails(self):
        """ANSWER certificate fails when verified hypothesis has DEFER action."""
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_answer=False,
            verified_hyp_action_is_defer=True,
        )
        assert answer_structural_certificate(s) is False

    def test_all_evidence_verified_no_contradiction_but_no_unique_support_fails(self):
        """D5 bug fix: all evidence verified, no contradiction, but no unique support → FAILS.

        The legacy clause (all_evidence_verified + no contradiction) was REMOVED
        because it accepted competing verified support (the D5 pattern).
        Now ANSWER requires unique verified support + ANSWER action mapping.
        """
        s = make_structural(
            all_evidence_verified=True,
            n_hyp_with_verified_contradiction=0,
            n_hyp_with_verified_support=2,  # competing support
            has_unique_verified_supported_hypothesis=False,
            has_verified_unresolved_competition=True,
        )
        assert answer_structural_certificate(s) is False

    def test_all_evidence_verified_unique_support_answer_passes(self):
        """All evidence verified, unique support, ANSWER action → passes."""
        s = make_structural(
            all_evidence_verified=True,
            n_hyp_with_verified_contradiction=0,
            n_hyp_with_verified_support=1,
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_answer=True,
            has_verified_unresolved_competition=False,
        )
        assert answer_structural_certificate(s) is True

    def test_competing_verified_support_fails(self):
        """ANSWER certificate fails with competing verified support."""
        s = make_structural(
            has_verified_unresolved_competition=True,
            n_hyp_with_verified_support=2,
        )
        assert answer_structural_certificate(s) is False

    def test_d5_pattern_explicitly_blocked(self):
        """D5 pattern: 2 supported hypotheses, 0 contradictions, all verified → BLOCKED.

        This is the specific case that caused the I3.30 blocking defect.
        The legacy clause accepted this; the fixed certificate must reject it.
        """
        s = make_structural(
            all_evidence_verified=True,
            n_hyp_with_verified_support=2,
            n_hyp_with_verified_contradiction=0,
            has_unique_verified_supported_hypothesis=False,
            has_verified_unresolved_competition=True,
            verified_hyp_action_is_answer=True,  # even if one maps to ANSWER
        )
        assert answer_structural_certificate(s) is False


class TestDeferStructuralCertificate:
    """Test the DEFER positive structural certificate."""

    def test_unique_verified_support_defer_passes(self):
        """DEFER certificate passes when unique verified support with DEFER action."""
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_defer=True,
            has_verified_unresolved_competition=False,
        )
        assert defer_structural_certificate(s) is True

    def test_defer_blocked_when_continuation_could_overturn(self):
        """DEFER certificate fails when verify budget remains and unverified evidence exists.

        Per EPISTEMIC_SEMANTICS_V1.md 6.2, DEFER_READY requires no admissible
        continuation can resolve the state. If verify budget remains AND
        unverified evidence exists, further verification could overturn the
        support or create competition, so the state is CONTINUE_REQUIRED.
        """
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_defer=True,
            has_verified_unresolved_competition=False,
            can_verify=True,
            all_evidence_verified=False,
        )
        assert defer_structural_certificate(s) is False

    def test_defer_passes_when_verify_budget_exhausted(self):
        """DEFER certificate passes when verify budget exhausted, even with unverified evidence."""
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_defer=True,
            has_verified_unresolved_competition=False,
            can_verify=False,
            all_evidence_verified=False,
        )
        assert defer_structural_certificate(s) is True

    def test_elimination_passes(self):
        """DEFER certificate passes when verification eliminated hypotheses."""
        s = make_structural(
            n_eliminated_hypotheses=1,
            n_viable_hypotheses=1,
            has_verified_unresolved_competition=False,
        )
        assert defer_structural_certificate(s) is True

    def test_defer_blocked_with_competing_support(self):
        """DEFER certificate fails when there's unresolved competing support."""
        s = make_structural(
            has_verified_unresolved_competition=True,
            n_hyp_with_verified_support=2,
            has_unique_verified_supported_hypothesis=False,
            n_eliminated_hypotheses=0,
            verify_budget_exhausted=True,
        )
        assert defer_structural_certificate(s) is False

    def test_resource_exhaustion_no_verified_passes(self):
        """Legacy D1: resource exhaustion, no verified evidence, all evidence verified."""
        s = make_structural(
            verify_budget_exhausted=True,
            n_hyp_with_verified_support=0,
            n_hyp_with_verified_contradiction=0,
            all_evidence_verified=True,
        )
        assert defer_structural_certificate(s) is True

    def test_resource_exhaustion_with_unverified_evidence_fails(self):
        """Legacy D1 certificate must NOT fire when unverified evidence remains.

        Without the all_evidence_verified check, this certificate would fire
        on D3 tasks (competing unverified support, no verify budget) and
        force DEFER when CONTINUE is correct.
        """
        s = make_structural(
            verify_budget_exhausted=True,
            n_hyp_with_verified_support=0,
            n_hyp_with_verified_contradiction=0,
            all_evidence_verified=False,
        )
        assert defer_structural_certificate(s) is False

    def test_no_certificate_fails(self):
        """DEFER certificate fails when no positive certificate exists."""
        s = make_structural(
            verify_budget_exhausted=True,
            n_hyp_with_verified_support=1,
            n_hyp_with_verified_contradiction=0,
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_defer=False,
            verified_hyp_action_is_answer=True,
            n_eliminated_hypotheses=0,
            n_viable_hypotheses=2,
        )
        assert defer_structural_certificate(s) is False

    def test_competing_unverified_support_fails_certificate(self):
        """DEFER certificate fails when only unverified competing support."""
        s = make_structural(
            has_competing_unverified_support=True,
            n_hyp_with_verified_support=0,
            n_eliminated_hypotheses=0,
            verify_budget_exhausted=False,
        )
        assert defer_structural_certificate(s) is False


class TestDecideAuthorityV3:
    """Test the full V3 authority decision."""

    def test_answer_authority_with_positive_certificate(self):
        """ANSWER authority fires with positive structural certificate."""
        q = {"ANSWER": 100.0, "DEFER": -30.0, "VERIFY": 50.0, "REASON_MORE": 60.0}
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_answer=True,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "DEFER", "VERIFY", "REASON_MORE"],
            structural=s,
        )
        assert decision.mode == AuthorityMode.HARD_ANSWER
        assert decision.action == "ANSWER"
        assert decision.structural_predicate_passed is True
        assert "POSITIVE_CERTIFICATE" in decision.reason_codes[0]

    def test_answer_authority_blocked_without_certificate(self):
        """ANSWER authority blocked when no positive structural certificate."""
        q = {"ANSWER": 100.0, "DEFER": -30.0, "VERIFY": 50.0}
        s = make_structural(
            has_unique_verified_supported_hypothesis=False,
            verified_hyp_action_is_answer=False,
            all_evidence_verified=False,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "DEFER", "VERIFY"],
            structural=s,
        )
        assert decision.mode == AuthorityMode.ADVISORY
        assert "NO_POSITIVE_STRUCTURAL_CERTIFICATE" in decision.reason_codes[0]

    def test_defer_authority_with_positive_certificate(self):
        """DEFER authority fires with positive structural certificate."""
        q = {"DEFER": 70.0, "ANSWER": -120.0, "VERIFY": 60.0, "REASON_MORE": 65.0}
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_defer=True,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "DEFER", "VERIFY", "REASON_MORE"],
            structural=s,
        )
        assert decision.mode == AuthorityMode.HARD_DEFER
        assert decision.action == "DEFER"
        assert decision.structural_predicate_passed is True

    def test_defer_authority_with_elimination_certificate(self):
        """DEFER authority fires with elimination certificate."""
        q = {"DEFER": 70.0, "ANSWER": -120.0, "REASON_MORE": 60.0}
        s = make_structural(
            n_eliminated_hypotheses=1,
            n_viable_hypotheses=1,
            n_hyp_with_verified_contradiction=1,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            structural=s,
        )
        assert decision.mode == AuthorityMode.HARD_DEFER

    def test_defer_authority_blocked_without_certificate(self):
        """DEFER authority blocked when no positive structural certificate.

        This is the I3.29 D3 false DEFER case: verified support exists but
        the verified hypothesis says ANSWER, not DEFER. No elimination.
        No resource exhaustion (can still verify).
        """
        q = {"DEFER": -27.87, "ANSWER": -131.43, "REASON_MORE": -67.24, "STOP": -56.69}
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_answer=True,
            verified_hyp_action_is_defer=False,
            n_eliminated_hypotheses=0,
            n_viable_hypotheses=3,
            verify_budget_exhausted=True,
            n_hyp_with_verified_support=1,
            n_hyp_with_verified_contradiction=0,
            can_verify=True,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE", "STOP"],
            structural=s,
        )
        # Q argmax is DEFER, but no positive certificate
        assert decision.mode == AuthorityMode.ADVISORY
        assert "NO_POSITIVE_STRUCTURAL_CERTIFICATE" in decision.reason_codes[0]

    def test_answer_authority_blocked_for_d2_false_case(self):
        """ANSWER authority blocked for I3.29 D2 false ANSWER case.

        D2: verified support exists but verified hypothesis says DEFER.
        The ANSWER certificate should NOT pass because verified_hyp_action_is_answer=False.
        """
        q = {"ANSWER": 88.76, "DEFER": 66.41, "REASON_MORE": -46.25, "STOP": -37.02}
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_answer=False,
            verified_hyp_action_is_defer=True,
            all_evidence_verified=False,
            n_hyp_with_verified_contradiction=1,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE", "STOP"],
            structural=s,
        )
        # Q argmax is ANSWER, but no positive certificate (verified hyp says DEFER)
        assert decision.mode == AuthorityMode.ADVISORY
        assert "NO_POSITIVE_STRUCTURAL_CERTIFICATE" in decision.reason_codes[0]

    def test_gap_too_small_remains_advisory(self):
        """Small Q gap remains advisory regardless of certificate."""
        q = {"ANSWER": 100.0, "RETRIEVE": 99.5, "DEFER": -30.0}
        s = make_structural(
            has_unique_verified_supported_hypothesis=True,
            verified_hyp_action_is_answer=True,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "RETRIEVE", "DEFER"],
            structural=s,
        )
        assert decision.mode == AuthorityMode.ADVISORY
        assert "GAP_TOO_SMALL" in decision.reason_codes[0]

    def test_frozen_threshold_unchanged(self):
        """Verify the frozen threshold is still 5.0."""
        assert AUTHORITY_THRESHOLD == 5.0

    def test_frozen_epsilon_unchanged(self):
        """Verify the frozen epsilon is still 3.0."""
        assert I2_EPSILON_Q == 3.0

    def test_rule_version(self):
        """Verify the V3 rule version identifier."""
        assert FROZEN_RULE_VERSION_V3 == "A2AD_V3_POSITIVE_CERTIFICATE"

    def test_fail_closed_on_error(self):
        """Authority fails closed to ADVISORY on error."""
        decision = decide_authority_v3(
            q_values={},
            legal_actions=[],
            structural=make_structural(),
        )
        assert decision.mode == AuthorityMode.ADVISORY

    def test_competing_verified_support_blocks_terminal(self):
        """Both ANSWER and DEFER authority blocked with competing verified support."""
        q = {"ANSWER": 100.0, "DEFER": -30.0, "REASON_MORE": 50.0}
        s = make_structural(
            has_verified_unresolved_competition=True,
            n_hyp_with_verified_support=2,
            has_unique_verified_supported_hypothesis=False,
        )
        decision = decide_authority_v3(
            q_values=q,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            structural=s,
        )
        # ANSWER is argmax but no certificate (competing support)
        assert decision.mode == AuthorityMode.ADVISORY
