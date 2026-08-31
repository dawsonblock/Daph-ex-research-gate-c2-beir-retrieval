"""DAPH Adaptive Authority V2 — pure deterministic authority decision.

This module implements the frozen A2AD_ASYMMETRIC_HARD_SELECT authority rule
for DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V2.

The rule is:
  Force(ANSWER) iff:
    - Q(ANSWER) == argmax(Q) over legal actions
    - Q(ANSWER) - Q(second_best) >= 5.0
    - ANSWER is legal
    - ANSWER is sole near-optimal (within epsilon_q=3.0 of Q_max)
    - AnswerSafety(state) passes

  Force(DEFER) iff:
    - Q(DEFER) == argmax(Q) over legal actions
    - Q(DEFER) - Q(second_best) >= 5.0
    - DEFER is legal
    - DEFER is sole near-optimal
    - has_competing_unverified_support == false
    - DeferSafety(state) passes

  Otherwise: ADVISORY

Any authority-layer error fails closed to ADVISORY.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AuthorityMode(str, Enum):
    HARD_ANSWER = "HARD_ANSWER"
    HARD_DEFER = "HARD_DEFER"
    ADVISORY = "ADVISORY"


@dataclass(frozen=True)
class AuthorityDecision:
    """The result of an authority decision."""
    mode: AuthorityMode
    action: str | None  # "ANSWER", "DEFER", or None for advisory
    reason_codes: list[str] = field(default_factory=list)
    q_gap: float = 0.0
    q_argmax: str = ""
    safety_predicate_passed: bool = False
    structural_predicate_passed: bool = False

    def as_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "q_gap": round(self.q_gap, 4),
            "q_argmax": self.q_argmax,
            "safety_predicate_passed": self.safety_predicate_passed,
            "structural_predicate_passed": self.structural_predicate_passed,
        }


@dataclass(frozen=True)
class StructuralState:
    """Observable structural features used by the DEFER safety predicate."""
    has_competing_unverified_support: bool
    n_hyp_unverified_support: int
    n_hyp_unverified_contradiction: int
    can_verify: bool
    verify_budget_exhausted: bool
    all_evidence_verified: bool

    def as_dict(self) -> dict:
        return {
            "has_competing_unverified_support": self.has_competing_unverified_support,
            "n_hyp_unverified_support": self.n_hyp_unverified_support,
            "n_hyp_unverified_contradiction": self.n_hyp_unverified_contradiction,
            "can_verify": self.can_verify,
            "verify_budget_exhausted": self.verify_budget_exhausted,
            "all_evidence_verified": self.all_evidence_verified,
        }


# Frozen constants
AUTHORITY_THRESHOLD = 5.0
I2_EPSILON_Q = 3.0
FROZEN_RULE_VERSION = "A2AD_V2"


def decide_authority(
    *,
    q_values: dict[str, float],
    legal_actions: list[str],
    structural: StructuralState,
    answer_safety_passed: bool = True,
) -> AuthorityDecision:
    """Pure deterministic authority decision.

    Args:
        q_values: Q(s, a) for each action
        legal_actions: list of legal action names
        structural: observable structural state for DEFER safety predicate
        answer_safety_passed: whether ANSWER safety predicate passes

    Returns:
        AuthorityDecision with mode, action, and reason codes

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
            if q_gap >= AUTHORITY_THRESHOLD:
                if len(near_optimal) == 1 and near_optimal[0] == "ANSWER":
                    if answer_safety_passed:
                        return AuthorityDecision(
                            mode=AuthorityMode.HARD_ANSWER,
                            action="ANSWER",
                            reason_codes=["ANSWER_CLEAR_GAP_SOLE_NEAR_OPT_SAFETY_PASS"],
                            q_gap=q_gap,
                            q_argmax=q_argmax,
                            safety_predicate_passed=True,
                        )
                    else:
                        return AuthorityDecision(
                            mode=AuthorityMode.ADVISORY,
                            action=None,
                            reason_codes=["ANSWER_SAFETY_FAILED"],
                            q_gap=q_gap,
                            q_argmax=q_argmax,
                            safety_predicate_passed=False,
                        )
                else:
                    return AuthorityDecision(
                        mode=AuthorityMode.ADVISORY,
                        action=None,
                        reason_codes=["ANSWER_NOT_SOLE_NEAR_OPTIMAL"],
                        q_gap=q_gap,
                        q_argmax=q_argmax,
                    )
            else:
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["ANSWER_GAP_TOO_SMALL"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
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

            # Structural safety predicate
            if structural.has_competing_unverified_support:
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["DEFER_COMPETING_UNVERIFIED_SUPPORT"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                    structural_predicate_passed=False,
                )

            # DEFER safety: at least one of the continuation-dominance conditions
            defer_safety = (
                not structural.can_verify  # VERIFY unavailable
                or structural.verify_budget_exhausted  # verification budget exhausted
                or structural.all_evidence_verified  # verification already complete
            )

            if not defer_safety:
                return AuthorityDecision(
                    mode=AuthorityMode.ADVISORY,
                    action=None,
                    reason_codes=["DEFER_SAFETY_FAILED_CONTINUATION_AVAILABLE"],
                    q_gap=q_gap,
                    q_argmax=q_argmax,
                    structural_predicate_passed=False,
                )

            # All conditions pass
            return AuthorityDecision(
                mode=AuthorityMode.HARD_DEFER,
                action="DEFER",
                reason_codes=["DEFER_CLEAR_GAP_SOLE_NEAR_OPT_NO_COMPETING_SAFETY_PASS"],
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
        # Fail closed to ADVISORY
        return AuthorityDecision(
            mode=AuthorityMode.ADVISORY,
            action=None,
            reason_codes=[f"AUTHORITY_ERROR: {str(e)}"],
        )


def build_receipt(
    *,
    state_features: dict,
    legal_actions: list[str],
    q_values: dict[str, float],
    structural: StructuralState,
    decision: AuthorityDecision,
) -> dict:
    """Build an authority receipt for provenance and analysis."""
    state_sha = hashlib.sha256(
        json.dumps(state_features, sort_keys=True).encode()
    ).hexdigest()

    return {
        "state_sha256": state_sha,
        "legal_actions": list(legal_actions),
        "q_values": {a: round(q, 4) for a, q in q_values.items()},
        "q_argmax": decision.q_argmax,
        "q_gap": round(decision.q_gap, 4),
        "structural": structural.as_dict(),
        "authority": decision.as_dict(),
        "rule_version": FROZEN_RULE_VERSION,
        "threshold": AUTHORITY_THRESHOLD,
        "i2_epsilon_q": I2_EPSILON_Q,
    }
