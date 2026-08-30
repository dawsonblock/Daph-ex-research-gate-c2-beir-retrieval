"""DAPH I3.30R3 — Authority Isolation Module.

This module implements the core authority-isolation experiment infrastructure.

The key invariant is:

    # Everything above this line must be identical.
    decision = evaluate_authority(state)
    if arm == V3_HARD and decision.should_force:
        action = decision.forced_action
    else:
        action = llm_action

V3_SHADOW and V3_HARD must execute the same code up to the final override
decision. One function. One decision object. One conditional.

Components:
    - ArmMode: enum for the three experimental arms (V1, V3_SHADOW, V3_HARD)
    - AuthorityDecisionV3: normalized decision object with force_applied /
      action_changed fields separated from certificate_passed
    - AuthorityEffect: enum for event-level causal classification
    - evaluate_v3_authority: unified V3 authority evaluation (shared by
      V3_SHADOW and V3_HARD)
    - classify_authority_effect: classify each authority event as
      RESCUE / BREAK / BENEFICIAL_NONRESCUE / HARMFUL_NONBREAK / NEUTRAL
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from daph.authority.policy import (
    AuthorityMode,
    AuthorityDecision,
    AUTHORITY_THRESHOLD,
    I2_EPSILON_Q,
)
from daph.authority.policy_v3 import (
    StructuralStateV3,
    decide_authority_v3,
    answer_structural_certificate,
    defer_structural_certificate,
)


# ============================================================
# Arm mode — the treatment variable
# ============================================================

class ArmMode(str, Enum):
    """The experimental arm. This is the ONLY treatment variable.

    V1:          Q_V1 + V1 epsilon + V1 ANSWER-only hard authority
    V3_SHADOW:   Q_V3R2 + V3 epsilon + V3 certificates evaluated/logged,
                 but hard override NEVER applied
    V3_HARD:     Q_V3R2 + V3 epsilon + V3 certificates + hard override applied

    The critical invariant: V3_SHADOW and V3_HARD execute identical code
    up to the final override decision.
    """
    V1 = "v1"
    V3_SHADOW = "v3_shadow"
    V3_HARD = "v3_hard"


# ============================================================
# Normalized authority decision — separates certificate from force
# ============================================================

@dataclass(frozen=True)
class AuthorityDecisionV3:
    """Normalized V3 authority decision with separated concerns.

    Key distinctions:
    - certificate_passed: did the structural certificate pass?
    - would_force: would the hard path force an action? (certificate + Q gap + sole near-opt)
    - force_applied: was the hard override actually applied? (would_force AND arm == V3_HARD)
    - action_changed: did the forced action differ from the LLM proposal?

    These are NOT the same:
        certificate_passed = yes
        force_applied = yes
        llm_action = ANSWER
        action_changed = no
    → This is NOT a causal intervention.
    """
    # Arm context
    arm: str

    # Q analysis
    q_values: dict[str, float]
    q_argmax: str
    q_second_best: str | None
    q_gap: float

    # Epsilon / near-optimal set
    epsilon_set: tuple[str, ...]

    # Certificate evaluation (always computed for V3 arms)
    certificate_evaluated: bool
    certificate_passed: bool
    certificate_type: str | None
    certificate_components: dict[str, Any]

    # Authority decision
    authority_mode: str  # "A0_advisory", "A2AD_hard_ANSWER", "A2AD_hard_DEFER"
    would_force: bool
    forced_action: str | None

    # LLM and executed action (filled after LLM call)
    llm_proposed_action: str | None = None
    executed_action: str | None = None
    force_applied: bool = False
    action_changed: bool = False

    # Structural state (for receipts)
    structural_state: dict[str, Any] | None = None

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "q_values": {k: round(v, 4) for k, v in self.q_values.items()},
            "q_argmax": self.q_argmax,
            "q_second_best": self.q_second_best,
            "q_gap": round(self.q_gap, 4),
            "epsilon_set": list(self.epsilon_set),
            "certificate_evaluated": self.certificate_evaluated,
            "certificate_passed": self.certificate_passed,
            "certificate_type": self.certificate_type,
            "certificate_components": self.certificate_components,
            "authority_mode": self.authority_mode,
            "would_force": self.would_force,
            "forced_action": self.forced_action,
            "llm_proposed_action": self.llm_proposed_action,
            "executed_action": self.executed_action,
            "force_applied": self.force_applied,
            "action_changed": self.action_changed,
        }


# ============================================================
# Authority effect classification
# ============================================================

class AuthorityEffect(str, Enum):
    """Event-level causal classification of an authority intervention."""
    RESCUE = "rescue"                # forced succeeds, shadow fails
    BREAK = "break"                  # forced fails, shadow succeeds
    BENEFICIAL_NONRESCUE = "beneficial_nonrescue"  # both succeed, forced higher utility
    HARMFUL_NONBREAK = "harmful_nonbreak"          # both succeed, forced lower utility
    NEUTRAL = "neutral"              # same outcome and effectively same utility


def classify_authority_effect(
    forced_success: bool,
    shadow_success: bool,
    forced_utility: float,
    shadow_utility: float,
    utility_tolerance: float = 0.01,
) -> AuthorityEffect:
    """Classify an authority event by comparing forced vs shadow outcomes.

    Args:
        forced_success: did the forced trajectory succeed?
        shadow_success: did the shadow trajectory succeed?
        forced_utility: utility of the forced trajectory
        shadow_utility: utility of the shadow trajectory
        utility_tolerance: threshold below which utility difference is "same"

    Returns:
        AuthorityEffect classification
    """
    if forced_success and not shadow_success:
        return AuthorityEffect.RESCUE
    if not forced_success and shadow_success:
        return AuthorityEffect.BREAK
    if forced_success and shadow_success:
        if forced_utility > shadow_utility + utility_tolerance:
            return AuthorityEffect.BENEFICIAL_NONRESCUE
        if forced_utility < shadow_utility - utility_tolerance:
            return AuthorityEffect.HARMFUL_NONBREAK
        return AuthorityEffect.NEUTRAL
    # Both failed
    if forced_utility > shadow_utility + utility_tolerance:
        return AuthorityEffect.BENEFICIAL_NONRESCUE
    if forced_utility < shadow_utility - utility_tolerance:
        return AuthorityEffect.HARMFUL_NONBREAK
    return AuthorityEffect.NEUTRAL


# ============================================================
# Unified V3 authority evaluation
# ============================================================

def _classify_certificate_type(structural: StructuralStateV3, action: str) -> dict:
    """Classify the positive structural certificate type and components."""
    if action == "ANSWER":
        if (structural.has_unique_verified_supported_hypothesis
                and structural.verified_hyp_action_is_answer
                and not structural.has_verified_unresolved_competition):
            return {
                "certificate_type": "unique_verified_support_answer",
                "components": {
                    "has_unique_verified_supported_hypothesis": True,
                    "verified_hyp_action_is_answer": True,
                    "has_verified_unresolved_competition": False,
                },
            }
    elif action == "DEFER":
        if (structural.has_unique_verified_supported_hypothesis
                and structural.verified_hyp_action_is_defer):
            if not (structural.can_verify and not structural.all_evidence_verified):
                return {
                    "certificate_type": "unique_verified_support_defer",
                    "components": {
                        "has_unique_verified_supported_hypothesis": True,
                        "verified_hyp_action_is_defer": True,
                        "continuation_admissible": False,
                    },
                }
        if structural.n_eliminated_hypotheses > 0 and structural.n_viable_hypotheses <= 1:
            return {
                "certificate_type": "elimination",
                "components": {
                    "n_eliminated_hypotheses": structural.n_eliminated_hypotheses,
                    "n_viable_hypotheses": structural.n_viable_hypotheses,
                },
            }
        if (structural.verify_budget_exhausted
                and structural.n_hyp_with_verified_support == 0
                and structural.n_hyp_with_verified_contradiction == 0
                and structural.all_evidence_verified):
            return {
                "certificate_type": "resource_exhaustion_no_verified",
                "components": {
                    "verify_budget_exhausted": True,
                    "n_hyp_with_verified_support": 0,
                    "n_hyp_with_verified_contradiction": 0,
                    "all_evidence_verified": True,
                },
            }
    return {
        "certificate_type": "NONE",
        "components": {},
    }


def evaluate_v3_authority(
    *,
    q_values: dict[str, float],
    legal_actions: list[str],
    structural: StructuralStateV3,
) -> AuthorityDecisionV3:
    """Unified V3 authority evaluation — shared by V3_SHADOW and V3_HARD.

    This function computes the certificate and would_force decision.
    It does NOT know which arm it is running under.
    The caller decides whether to apply the force based on the arm.

    Returns an AuthorityDecisionV3 with:
    - certificate_evaluated = True
    - certificate_passed = whether the certificate passed
    - would_force = whether the hard path would force
    - forced_action = the action that would be forced (or None)
    - force_applied = False (caller sets this based on arm)
    - llm_proposed_action = None (caller fills after LLM call)
    - executed_action = None (caller fills after decision)
    """
    # Filter to legal actions only
    legal_q = {a: q_values[a] for a in legal_actions if a in q_values}
    if not legal_q:
        return AuthorityDecisionV3(
            arm="",  # caller fills
            q_values=q_values,
            q_argmax="",
            q_second_best=None,
            q_gap=0.0,
            epsilon_set=(),
            certificate_evaluated=True,
            certificate_passed=False,
            certificate_type="NONE",
            certificate_components={},
            authority_mode="A0_advisory",
            would_force=False,
            forced_action=None,
            structural_state=structural.as_dict(),
        )

    # Sort by Q value descending
    sorted_q = sorted(legal_q.items(), key=lambda x: -x[1])
    q_argmax = sorted_q[0][0]
    q_max = sorted_q[0][1]
    q_second = sorted_q[1][1] if len(sorted_q) > 1 else q_max
    q_gap = q_max - q_second

    # Near-optimal set
    near_optimal = tuple(
        a for a, q in legal_q.items() if q >= q_max - I2_EPSILON_Q
    )

    # Evaluate V3 authority decision (reuses the frozen decide_authority_v3)
    decision = decide_authority_v3(
        q_values=q_values,
        legal_actions=legal_actions,
        structural=structural,
    )

    # Determine certificate info
    cert_info = {"certificate_type": "NONE", "components": {}}
    if decision.mode == AuthorityMode.HARD_ANSWER:
        cert_info = _classify_certificate_type(structural, "ANSWER")
        authority_mode_str = "A2AD_hard_ANSWER"
        would_force = True
        forced_action = "ANSWER"
    elif decision.mode == AuthorityMode.HARD_DEFER:
        cert_info = _classify_certificate_type(structural, "DEFER")
        authority_mode_str = "A2AD_hard_DEFER"
        would_force = True
        forced_action = "DEFER"
    else:
        authority_mode_str = "A0_advisory"
        would_force = False
        forced_action = None
        # Still classify what certificate would have been for logging
        if q_argmax in ("ANSWER", "DEFER"):
            cert_info = _classify_certificate_type(structural, q_argmax)

    return AuthorityDecisionV3(
        arm="",  # caller fills
        q_values=legal_q,
        q_argmax=q_argmax,
        q_second_best=sorted_q[1][0] if len(sorted_q) > 1 else None,
        q_gap=q_gap,
        epsilon_set=near_optimal,
        certificate_evaluated=True,
        certificate_passed=(decision.mode in (AuthorityMode.HARD_ANSWER, AuthorityMode.HARD_DEFER)),
        certificate_type=cert_info["certificate_type"],
        certificate_components=cert_info["components"],
        authority_mode=authority_mode_str,
        would_force=would_force,
        forced_action=forced_action,
        structural_state=structural.as_dict(),
    )


# ============================================================
# Apply authority — the single conditional
# ============================================================

def apply_authority(
    decision: AuthorityDecisionV3,
    arm: ArmMode,
    llm_action: str,
) -> tuple[str, AuthorityDecisionV3]:
    """Apply authority based on arm mode. This is the ONLY place the treatment varies.

    The invariant:
        # Everything above this line must be identical.
        if arm == V3_HARD and decision.would_force:
            action = decision.forced_action
        else:
            action = llm_action

    Returns (executed_action, updated_decision).
    """
    force_applied = False
    executed_action = llm_action

    if arm == ArmMode.V3_HARD and decision.would_force and decision.forced_action:
        executed_action = decision.forced_action
        force_applied = True

    action_changed = (executed_action != llm_action)

    updated = AuthorityDecisionV3(
        arm=arm.value,
        q_values=decision.q_values,
        q_argmax=decision.q_argmax,
        q_second_best=decision.q_second_best,
        q_gap=decision.q_gap,
        epsilon_set=decision.epsilon_set,
        certificate_evaluated=decision.certificate_evaluated,
        certificate_passed=decision.certificate_passed,
        certificate_type=decision.certificate_type,
        certificate_components=decision.certificate_components,
        authority_mode=decision.authority_mode,
        would_force=decision.would_force,
        forced_action=decision.forced_action,
        llm_proposed_action=llm_action,
        executed_action=executed_action,
        force_applied=force_applied,
        action_changed=action_changed,
        structural_state=decision.structural_state,
    )

    return executed_action, updated


# ============================================================
# State hashing
# ============================================================

def state_sha(state_features: dict, structural_state: dict | None = None) -> str:
    """Compute a deterministic SHA for a decision state."""
    payload = {"sf": state_features}
    if structural_state:
        payload["struct"] = structural_state
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


# ============================================================
# Receipt builder
# ============================================================

def build_normalized_receipt(
    *,
    task_id: str,
    arm: str,
    step: int,
    state_features: dict,
    decision: AuthorityDecisionV3,
    legal_actions: list[str],
    epistemically_admissible: list[str] | None = None,
    resource_state: dict | None = None,
) -> dict:
    """Build a normalized per-step receipt.

    Every decision step emits one receipt with the full decision context.
    """
    return {
        "task_id": task_id,
        "arm": arm,
        "step": step,
        "state_sha": state_sha(state_features, decision.structural_state),
        "legal_actions": sorted(legal_actions),
        "epistemically_admissible_actions": sorted(epistemically_admissible) if epistemically_admissible else [],
        "q_values": {k: round(v, 4) for k, v in decision.q_values.items()},
        "q_argmax": decision.q_argmax,
        "q_second_best": decision.q_second_best,
        "q_gap": round(decision.q_gap, 4),
        "epsilon_set": list(decision.epsilon_set),
        "certificate_evaluated": decision.certificate_evaluated,
        "certificate_passed": decision.certificate_passed,
        "certificate_type": decision.certificate_type,
        "certificate_components": decision.certificate_components,
        "authority_mode": decision.authority_mode,
        "would_force": decision.would_force,
        "forced_action": decision.forced_action,
        "llm_proposed_action": decision.llm_proposed_action,
        "executed_action": decision.executed_action,
        "force_applied": decision.force_applied,
        "action_changed": decision.action_changed,
        "structural_state": decision.structural_state,
        "resource_state": resource_state,
    }
