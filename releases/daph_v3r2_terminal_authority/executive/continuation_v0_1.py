"""DAPH Continuation Authority V0.1 — VERIFY certificate.

This module implements the first continuation authority certificate for
DAPH. Unlike terminal authority (ANSWER/DEFER), continuation authority
forces information-gathering actions (VERIFY, SEARCH, RETRIEVE) when
the state is CONTINUE_REQUIRED and the model proposes premature
termination.

Design principles:
1. Start with VERIFY only (explicit target, predictable epistemic transition)
2. SHADOW first — do not allow HARD continuation authority until precision is demonstrated
3. Information-gain formulation — require expected IG > cost
4. Never fire when a terminal certificate is valid
5. Require Q margin — LCB(VERIFY) > UCB(DEFER) + delta

The certificate:
  C_VERIFY(s, e) = 1
  iff:
    1. R(s) = CONTINUE_REQUIRED (no terminal certificate valid)
    2. v(e) = UNVERIFIED (evidence is currently unverified)
    3. IG(e; s) > eta (expected information gain exceeds threshold)
    4. b_verify > 0 (verification budget remains)
    5. C_ANSWER(s) = 0 (no answer certificate valid)
    6. C_DEFER(s) = 0 (no defer certificate valid)
    7. Q(s, VERIFY(e)) - max_{a != VERIFY(e)} Q(s, a) > tau_V (Q margin)

Information gain:
  IG(e; s) = H(H | s) - E[H(H | s, VERIFY(e))]

  where H(H | s) is hypothesis entropy given current state,
  and E[H(H | s, VERIFY(e))] is expected entropy after verification.

Resource-adjusted value:
  V_verify(e; s) = IG(e; s) - lambda_V * c_verify

  Verification authority should require V_verify > 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from daph.authority.policy_v3 import (
    StructuralStateV3,
    answer_structural_certificate,
    defer_structural_certificate,
)


class AuthorityModeContinuation(str, Enum):
    """Continuation authority modes."""
    ADVISORY = "advisory"
    HARD_VERIFY = "hard_verify"
    SHADOW_VERIFY = "shadow_verify"


@dataclass(frozen=True)
class ContinuationDecision:
    """Continuation authority decision."""
    mode: AuthorityModeContinuation
    action: str | None  # "VERIFY" or None
    target_evidence_id: str | None
    would_force: bool
    information_gain: float
    verification_value: float
    reason_codes: list[str]
    q_margin: float | None = None


# ============================================================
# Information gain computation
# ============================================================

def hypothesis_entropy(structural: StructuralStateV3) -> float:
    """Compute hypothesis entropy given structural state.

    Uses a simple probability model:
    - Viable hypotheses share probability equally
    - Eliminated hypotheses have probability ~0
    - Unverified hypotheses have reduced probability

    This is an approximation. A full Bayesian model would use
    evidence likelihoods, but those are not observable in the
    structural state.
    """
    n_viable = max(structural.n_viable_hypotheses, 1)
    n_total = n_viable + structural.n_eliminated_hypotheses

    if n_total == 0:
        return 0.0

    # Simple model: each viable hypothesis has equal probability
    p = 1.0 / n_viable

    # Adjust for verified support
    if structural.n_hyp_with_verified_support > 0:
        # Hypotheses with verified support are more likely
        p_support = 0.7 / structural.n_hyp_with_verified_support
        p_other = 0.3 / max(n_viable - structural.n_hyp_with_verified_support, 1)
        # This is a rough approximation
        probs = []
        for i in range(n_viable):
            if i < structural.n_hyp_with_verified_support:
                probs.append(p_support)
            else:
                probs.append(p_other)
    else:
        probs = [p] * n_viable

    # Normalize
    total = sum(probs)
    if total > 0:
        probs = [pr / total for pr in probs]

    # Entropy
    entropy = -sum(pr * math.log(pr + 1e-10) for pr in probs if pr > 0)
    return entropy


def expected_entropy_after_verify(structural: StructuralStateV3) -> float:
    """Estimate expected entropy after verifying one more piece of evidence.

    If there's a unique supported hypothesis, verification is likely to
    confirm it, reducing entropy. If there's competing support, verification
    may resolve it. If no support, verification may create it.
    """
    n_viable = max(structural.n_viable_hypotheses, 1)

    # If only one viable hypothesis, verification likely confirms it
    if n_viable == 1:
        return 0.1 * hypothesis_entropy(structural)

    # If competing verified support, verification may resolve
    if structural.has_verified_unresolved_competition:
        # Verification could eliminate one competitor
        return 0.5 * hypothesis_entropy(structural)

    # If no verified support, verification may create unique support
    if structural.n_hyp_with_verified_support == 0:
        # Verification could create support for one hypothesis
        return 0.6 * hypothesis_entropy(structural)

    # Default: verification provides moderate information
    return 0.7 * hypothesis_entropy(structural)


def information_gain(structural: StructuralStateV3) -> float:
    """Compute expected information gain from verifying one more evidence item."""
    h_before = hypothesis_entropy(structural)
    h_after = expected_entropy_after_verify(structural)
    return max(0.0, h_before - h_after)


# ============================================================
# VERIFY continuation certificate
# ============================================================

# Thresholds (to be calibrated)
IG_THRESHOLD = 0.3  # Minimum information gain
VERIFY_Q_MARGIN = 2.0  # Minimum Q margin for VERIFY
VERIFY_COST = 1.0  # Cost of one verification
LAMBDA_V = 0.5  # Cost weight


def verify_continuation_certificate(
    structural: StructuralStateV3,
    legal_actions: list[str],
    q_values: dict[str, float],
    can_verify: bool,
    verify_budget_remaining: int,
    unverified_evidence_count: int,
) -> ContinuationDecision:
    """Evaluate VERIFY continuation authority certificate.

    Returns a ContinuationDecision indicating whether VERIFY should be forced.

    The certificate requires:
    1. No terminal certificate is valid (CONTINUE_REQUIRED)
    2. Verification is legal and budget remains
    3. Unverified evidence exists
    4. Information gain exceeds threshold
    5. Q margin for VERIFY exceeds threshold
    """
    reason_codes = []

    # 1. Check terminal certificates
    if answer_structural_certificate(structural):
        reason_codes.append("ANSWER_CERTIFICATE_VALID")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=0.0, verification_value=0.0,
            reason_codes=reason_codes,
        )

    if defer_structural_certificate(structural):
        reason_codes.append("DEFER_CERTIFICATE_VALID")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=0.0, verification_value=0.0,
            reason_codes=reason_codes,
        )

    # 2. Check VERIFY is legal and budget remains
    if "VERIFY" not in legal_actions:
        reason_codes.append("VERIFY_NOT_LEGAL")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=0.0, verification_value=0.0,
            reason_codes=reason_codes,
        )

    if not can_verify or verify_budget_remaining <= 0:
        reason_codes.append("VERIFY_BUDGET_EXHAUSTED")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=0.0, verification_value=0.0,
            reason_codes=reason_codes,
        )

    # 3. Check unverified evidence exists
    if unverified_evidence_count <= 0:
        reason_codes.append("NO_UNVERIFIED_EVIDENCE")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=0.0, verification_value=0.0,
            reason_codes=reason_codes,
        )

    # 4. Compute information gain
    ig = information_gain(structural)
    v_verify = ig - LAMBDA_V * VERIFY_COST

    if ig < IG_THRESHOLD:
        reason_codes.append(f"IG_BELOW_THRESHOLD: {ig:.3f} < {IG_THRESHOLD}")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=ig, verification_value=v_verify,
            reason_codes=reason_codes,
        )

    if v_verify <= 0:
        reason_codes.append(f"V_VERIFY_NEGATIVE: {v_verify:.3f}")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=ig, verification_value=v_verify,
            reason_codes=reason_codes,
        )

    # 5. Check Q margin
    q_verify = q_values.get("VERIFY", 0.0)
    q_other = {a: v for a, v in q_values.items() if a != "VERIFY"}
    q_max_other = max(q_other.values()) if q_other else q_verify
    q_margin = q_verify - q_max_other

    if q_margin < VERIFY_Q_MARGIN:
        reason_codes.append(f"Q_MARGIN_BELOW_THRESHOLD: {q_margin:.3f} < {VERIFY_Q_MARGIN}")
        return ContinuationDecision(
            mode=AuthorityModeContinuation.ADVISORY,
            action=None, target_evidence_id=None,
            would_force=False, information_gain=ig, verification_value=v_verify,
            reason_codes=reason_codes, q_margin=q_margin,
        )

    # All conditions pass — would force VERIFY
    reason_codes.append("VERIFY_CLEAR_IG_Q_MARGIN_CONTINUATION_REQUIRED")

    return ContinuationDecision(
        mode=AuthorityModeContinuation.SHADOW_VERIFY,  # Start in SHADOW
        action="VERIFY",
        target_evidence_id=None,  # Caller selects target
        would_force=True,
        information_gain=ig,
        verification_value=v_verify,
        reason_codes=reason_codes,
        q_margin=q_margin,
    )


# ============================================================
# Frozen rule version
# ============================================================

FROZEN_RULE_VERSION_CONTINUATION = "A2CV_V0.1_VERIFY_CERTIFICATE_SHADOW"
