"""Executive decision contract for DAPH-X.

The executive receives a canonical state and returns an ExecutiveDecision.
This is the single interface that all DAPH-X components use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from daph_x.actions.typed_actions import Action


class AuthorityMode(str, Enum):
    """How much authority the executive exercises."""
    OBSERVE = "OBSERVE"      # Record only, no effect
    ADVISE = "ADVISE"        # Tell LLM preferred actions, don't remove alternatives
    CONSTRAIN = "CONSTRAIN"  # Remove dominated/illegal/unsafe actions
    FORCE = "FORCE"          # Execute executive-preferred action
    ABSTAIN = "ABSTAIN"      # Executive admits insufficient confidence


@dataclass(frozen=True)
class ExecutiveDecision:
    """The result of executive deliberation.

    This is the single output contract for the DAPH-X executive.
    Every decision includes:
      - The selected action
      - The authority mode
      - Expected value and uncertainty
      - Risk assessment
      - Structural certificate status
      - Justification
      - Provenance (hash of state, candidates, scores)
    """
    # The selected action
    selected_action: Action

    # How much authority is exercised
    authority_mode: AuthorityMode

    # Value estimates
    expected_value: float
    value_margin: float          # V(selected) - V(next_best)
    value_lcb: float             # Lower confidence bound of value margin

    # Uncertainty decomposition
    model_uncertainty: float     # U_M(s,a) — Q/value model uncertainty
    epistemic_uncertainty: float # U_E(s) — task/epistemic uncertainty
    transition_uncertainty: float # U_T(s,a) — world model uncertainty

    # Risk
    intervention_risk: float     # R_I(s,a) — P(force makes things worse)

    # Structural constraints
    structural_certificate: bool  # True if a hard structural certificate applies
    certificate_type: str | None  # e.g. "unique_verified_support_answer"

    # Justification
    rationale: str

    # Provenance
    state_hash: str = ""
    candidate_count: int = 0
    action_scores: Mapping[str, float] = field(default_factory=dict)
    llm_proposal: str | None = None
    llm_proposal_score: float | None = None
    intervention_advantage: float | None = None  # V(selected) - V(llm_proposal)

    def to_dict(self) -> dict:
        return {
            "selected_action": str(self.selected_action),
            "authority_mode": self.authority_mode.value,
            "expected_value": self.expected_value,
            "value_margin": self.value_margin,
            "value_lcb": self.value_lcb,
            "model_uncertainty": self.model_uncertainty,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "transition_uncertainty": self.transition_uncertainty,
            "intervention_risk": self.intervention_risk,
            "structural_certificate": self.structural_certificate,
            "certificate_type": self.certificate_type,
            "rationale": self.rationale,
            "state_hash": self.state_hash,
            "candidate_count": self.candidate_count,
            "llm_proposal": self.llm_proposal,
            "llm_proposal_score": self.llm_proposal_score,
            "intervention_advantage": self.intervention_advantage,
        }
