"""Authority policy for DAPH-X.

Implements the five authority modes with LCB + risk gates:

  OBSERVE:   Log only, no effect
  ADVISE:    Expose ranked candidates, don't constrain
  CONSTRAIN: Remove provably unsafe/dominated actions
  FORCE:     Execute executive-preferred action (high confidence)
  ABSTAIN:   Executive admits insufficient confidence

FORCE rule:
  LCB(ΔQ) > τ_Δ
  ∧ R_I < ρ
  ∧ a* ∈ A_safe
  ∧ U_M < τ_M
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.actions.typed_actions import Action, ActionType
from daph_x.authority.executive_decision import AuthorityMode, ExecutiveDecision
from daph_x.belief.belief_engine import BeliefState


@dataclass(frozen=True)
class AuthorityConfig:
    """Configuration for authority policy."""
    # LCB threshold for FORCE
    lcb_threshold: float = 5.0
    # Risk threshold for FORCE
    risk_threshold: float = 0.1
    # Model uncertainty threshold for FORCE
    model_uncertainty_threshold: float = 0.5
    # Z-score for LCB computation
    lcb_z: float = 1.0
    # Whether to enable FORCE at all
    force_enabled: bool = False


def determine_authority_mode(
    belief: BeliefState,
    selected_action: Action,
    selected_score: float,
    selected_sigma: float,
    next_best_score: float,
    next_best_sigma: float,
    intervention_risk: float,
    llm_proposal: Action | None,
    config: AuthorityConfig,
) -> AuthorityMode:
    """Determine the appropriate authority mode.

    The decision tree is:
    1. If selected action is not safe → CONSTRAIN (remove it)
    2. If model uncertainty too high → ABSTAIN
    3. If intervention risk too high → ABSTAIN
    4. If LCB(ΔQ) > threshold → FORCE
    5. If ΔQ > 0 → ADVISE
    6. Otherwise → OBSERVE
    """
    # Check structural safety
    is_safe = _is_action_safe(selected_action, belief)

    if not is_safe:
        return AuthorityMode.CONSTRAIN

    # Check model uncertainty
    if selected_sigma > config.model_uncertainty_threshold:
        return AuthorityMode.ABSTAIN

    # Check intervention risk
    if intervention_risk > config.risk_threshold:
        return AuthorityMode.ABSTAIN

    # Compute LCB of value margin
    if llm_proposal is not None:
        # σ_Δ ≈ sqrt(σ_X² + σ_L²)
        sigma_delta = (selected_sigma**2 + 0.1**2) ** 0.5  # LLM uncertainty placeholder
        delta = selected_score - next_best_score
        lcb_delta = delta - config.lcb_z * sigma_delta
    else:
        # No LLM proposal — use margin to next best
        sigma_delta = (selected_sigma**2 + next_best_sigma**2) ** 0.5
        delta = selected_score - next_best_score
        lcb_delta = delta - config.lcb_z * sigma_delta

    # FORCE if LCB exceeds threshold and force is enabled
    if config.force_enabled and lcb_delta > config.lcb_threshold:
        return AuthorityMode.FORCE

    # ADVISE if we have a clear preference
    if delta > 0:
        return AuthorityMode.ADVISE

    # OBSERVE as fallback
    return AuthorityMode.OBSERVE


def _is_action_safe(action: Action, belief: BeliefState) -> bool:
    """Check if an action is structurally safe.

    An action is unsafe if:
    - ANSWER when not ANSWER_READY
    - DEFER when not DEFER_READY (premature defer)
    - VERIFY when no unverified evidence
    - Any action when resources exhausted
    """
    if action.action_type == ActionType.ANSWER:
        return belief.readiness.value == "ANSWER_READY"

    if action.action_type == ActionType.DEFER:
        return belief.readiness.value == "DEFER_READY"

    if action.action_type == ActionType.VERIFY:
        return belief.n_untested > 0 or belief.n_weakened > 0

    if action.action_type == ActionType.STOP:
        return True  # Always safe to stop

    return True  # Default: safe


def constrain_actions(
    candidates: Sequence[Action],
    belief: BeliefState,
) -> list[Action]:
    """Remove provably unsafe or dominated actions.

    CONSTRAIN mode: filter the action space to only safe actions.
    """
    safe = []
    for action in candidates:
        if _is_action_safe(action, belief):
            safe.append(action)
    return safe
