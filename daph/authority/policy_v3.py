"""DAPH Adaptive Authority V3 — positive structural certificate authority.

This module implements the V3 authority rule with POSITIVE structural certificates
instead of absence-of-danger checks.

The rule is:
  Force(ANSWER) iff:
    - Q(ANSWER) == argmax(Q) over legal actions
    - Q(ANSWER) - Q(second_best) >= 5.0  (frozen threshold)
    - ANSWER is legal
    - ANSWER is sole near-optimal (within epsilon_q=3.0 of Q_max)
    - PositiveStructuralCertificate(ANSWER):
        has_unique_verified_supported_hypothesis AND verified_hyp_action_is_answer
      OR (legacy D1/D4 support): all_evidence_verified AND n_hyp_with_verified_contradiction == 0

  Force(DEFER) iff:
    - Q(DEFER) == argmax(Q) over legal actions
    - Q(DEFER) - Q(second_best) >= 5.0  (frozen threshold)
    - DEFER is legal
    - DEFER is sole near-optimal
    - PositiveStructuralCertificate(DEFER):
        verified_hyp_action_is_defer
      OR n_eliminated_hypotheses > 0 AND n_viable_hypotheses <= 1
      OR (legacy D1): verify_budget_exhausted AND n_hyp_with_verified_support == 0 AND n_hyp_with_verified_contradiction == 0

  Otherwise: ADVISORY

This replaces the V2 pattern:
  HighQConfidence(a) AND NOT KnownUnsafe(s)
with:
  HighQConfidence(a) AND PositiveStructuralCertificate(a)

The frozen threshold (5.0) and near-optimal epsilon (3.0) are unchanged.
V2 policy is NOT modified — this is a new module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daph.authority.policy import (
    AuthorityMode, AuthorityDecision, StructuralState,
    AUTHORITY_THRESHOLD, I2_EPSILON_Q,
)


@dataclass(frozen=True)
class StructuralStateV3:
    """V3 structural state with post-verification topology features.

    Extends V2 StructuralState with observable verified-evidence topology.
    All features are observable: no verify_result oracle, no hidden evidence,
    no future outcomes.
    """
    # V2 features (retained)
    has_competing_unverified_support: bool
    n_hyp_unverified_support: int
    n_hyp_unverified_contradiction: int
    can_verify: bool
    verify_budget_exhausted: bool
    all_evidence_verified: bool

    # V3 post-verification features
    n_hyp_with_verified_support: int
    n_hyp_with_verified_contradiction: int
    n_hyp_with_mixed_verified: int
    n_viable_hypotheses: int
    n_eliminated_hypotheses: int
    has_unique_verified_supported_hypothesis: bool
    has_verified_unresolved_competition: bool
    verified_hyp_action_is_answer: bool
    verified_hyp_action_is_defer: bool

    def as_dict(self) -> dict:
        return {
            "has_competing_unverified_support": self.has_competing_unverified_support,
            "n_hyp_unverified_support": self.n_hyp_unverified_support,
            "n_hyp_unverified_contradiction": self.n_hyp_unverified_contradiction,
            "can_verify": self.can_verify,
            "verify_budget_exhausted": self.verify_budget_exhausted,
            "all_evidence_verified": self.all_evidence_verified,
            "n_hyp_with_verified_support": self.n_hyp_with_verified_support,
            "n_hyp_with_verified_contradiction": self.n_hyp_with_verified_contradiction,
            "n_hyp_with_mixed_verified": self.n_hyp_with_mixed_verified,
            "n_viable_hypotheses": self.n_viable_hypotheses,
            "n_eliminated_hypotheses": self.n_eliminated_hypotheses,
            "has_unique_verified_supported_hypothesis": self.has_unique_verified_supported_hypothesis,
            "has_verified_unresolved_competition": self.has_verified_unresolved_competition,
            "verified_hyp_action_is_answer": self.verified_hyp_action_is_answer,
            "verified_hyp_action_is_defer": self.verified_hyp_action_is_defer,
        }

    def to_v2(self) -> StructuralState:
        """Convert to V2 StructuralState for backward compatibility."""
        return StructuralState(
            has_competing_unverified_support=self.has_competing_unverified_support,
            n_hyp_unverified_support=self.n_hyp_unverified_support,
            n_hyp_unverified_contradiction=self.n_hyp_unverified_contradiction,
            can_verify=self.can_verify,
            verify_budget_exhausted=self.verify_budget_exhausted,
            all_evidence_verified=self.all_evidence_verified,
        )


def answer_structural_certificate(s: StructuralStateV3) -> bool:
    """Positive structural certificate for ANSWER authority.

    ANSWER is structurally supported when:
    1. There is a uniquely verified-supported hypothesis whose action is ANSWER, OR
    2. All evidence is verified with no contradictions (legacy D4 pattern)
    """
    # Primary certificate: unique verified support with ANSWER action
    if s.has_unique_verified_supported_hypothesis and s.verified_hyp_action_is_answer:
        return True

    # Legacy certificate: all evidence verified, no verified contradiction
    if s.all_evidence_verified and s.n_hyp_with_verified_contradiction == 0:
        return True

    return False


def defer_structural_certificate(s: StructuralStateV3) -> bool:
    """Positive structural certificate for DEFER authority.

    DEFER is structurally supported when:
    1. There is a uniquely verified-supported hypothesis whose action is DEFER, OR
    2. Verification eliminated all but <=1 viable hypothesis (exhaustion), OR
    3. Resource exhaustion with no verified evidence (legacy D1 pattern)
    """
    # Primary certificate: unique verified support with DEFER action
    if s.has_unique_verified_supported_hypothesis and s.verified_hyp_action_is_defer:
        return True

    # Elimination certificate: verification eliminated hypotheses, few viable remain
    if s.n_eliminated_hypotheses > 0 and s.n_viable_hypotheses <= 1:
        return True

    # Legacy D1 certificate: resource exhaustion, no verified evidence
    if (s.verify_budget_exhausted
        and s.n_hyp_with_verified_support == 0
        and s.n_hyp_with_verified_contradiction == 0):
        return True

    return False


def decide_authority_v3(
    *,
    q_values: dict[str, float],
    legal_actions: list[str],
    structural: StructuralStateV3,
) -> AuthorityDecision:
    """V3 authority decision with positive structural certificates.

    Uses the same frozen threshold (5.0) and near-optimal epsilon (3.0) as V2.
    Replaces absence-of-danger with positive structural certificate.
    Fails closed to ADVISORY on any error.
    """
    try:
        # Filter to legal actions only
        legal_q = {a: q_values[a] for a in legal_actions if a in q_values}
        if not legal_q:
            return AuthorityDecision(
                mode=AuthorityMode.ADVISORY,
                action=None,
                reason_codes=["NO_LEGAL_ACTIONS_WITH_Q_VALUES"],
            )

        # Sort by Q value descending
        sorted_q = sorted(legal_q.items(), key=lambda x: -x[1])
        q_argmax = sorted_q[0][0]
        q_max = sorted_q[0][1]
        q_second = sorted_q[1][1] if len(sorted_q) > 1 else q_max
        q_gap = q_max - q_second

        # Near-optimal set: actions within epsilon_q of Q_max
        near_optimal = [a for a, q in legal_q.items() if q >= q_max - I2_EPSILON_Q]

        # ---- ANSWER authority ----
        if q_argmax == "ANSWER" and "ANSWER" in legal_actions:
            if q_gap < AUTHORITY_THRESHOLD:
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["ANSWER_GAP_TOO_SMALL"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                )

            if len(near_optimal) != 1 or near_optimal[0] != "ANSWER":
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["ANSWER_NOT_SOLE_NEAR_OPTIMAL"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                )

            # POSITIVE structural certificate (replaces absence-of-danger)
            if not answer_structural_certificate(structural):
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["ANSWER_NO_POSITIVE_STRUCTURAL_CERTIFICATE"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                    structural_predicate_passed=False,
                )

            # All conditions pass
            return AuthorityDecision(
                mode=AuthorityMode.HARD_ANSWER,
                action="ANSWER",
                reason_codes=["ANSWER_CLEAR_GAP_SOLE_NEAR_OPT_POSITIVE_CERTIFICATE"],
                q_gap=q_gap,
                q_argmax=q_argmax,
                safety_predicate_passed=True,
                structural_predicate_passed=True,
            )

        # ---- DEFER authority ----
        if q_argmax == "DEFER" and "DEFER" in legal_actions:
            if q_gap < AUTHORITY_THRESHOLD:
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["DEFER_GAP_TOO_SMALL"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                )

            if len(near_optimal) != 1 or near_optimal[0] != "DEFER":
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["DEFER_NOT_SOLE_NEAR_OPTIMAL"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                )

            # POSITIVE structural certificate (replaces absence-of-danger)
            if not defer_structural_certificate(structural):
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["DEFER_NO_POSITIVE_STRUCTURAL_CERTIFICATE"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                    structural_predicate_passed=False,
                )

            # All conditions pass
            return AuthorityDecision(
                mode=AuthorityMode.HARD_DEFER,
                action="DEFER",
                reason_codes=["DEFER_CLEAR_GAP_SOLE_NEAR_OPT_POSITIVE_CERTIFICATE"],
                q_gap=q_gap,
                q_argmax=q_argmax,
                safety_predicate_passed=True,
                structural_predicate_passed=True,
            )

        # ---- Otherwise: advisory ----
        return AuthorityDecision(
            mode=AuthorityMode.ADVISORY,
            action=None,
            reason_codes=["ARGMAX_NOT_ANSWER_OR_DEFER"],
            q_gap=q_gap,
            q_argmax=q_argmax,
        )

    except Exception as e:
        return AuthorityDecision(
            mode=AuthorityMode.ADVISORY,
            action=None,
            reason_codes=[f"AUTHORITY_ERROR: {str(e)}"],
        )


# Rule version identifier
FROZEN_RULE_VERSION_V3 = "A2AD_V3_POSITIVE_CERTIFICATE"
