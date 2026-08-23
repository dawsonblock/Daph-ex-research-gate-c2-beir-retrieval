"""DAPH I3.4 — Compact structural feature vector for phase-aware control.

Every feature is:
  - controller-visible (available before action)
  - non-oracular (no future information)
  - reconstructable from current observable state only

No future effects. No hidden task information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PhaseFeatures:
    """Compact structural feature vector z_t.

    Used as input to the phase classifier and action-value estimator.
    """

    # Phase (assigned by classifier, included for convenience)
    phase: str

    # Hypothesis structure
    n_live: int
    n_eliminated: int
    n_total: int

    # Evidence structure
    n_visible: int
    n_hidden: int
    n_verified: int
    n_supporting: int
    n_contradicting: int

    # Resource state
    retrieval_remaining: int
    search_remaining: int
    verify_remaining: int
    steps_remaining: int

    # Affordances (derived from resources + legal targets)
    can_retrieve: bool
    can_search: bool
    can_verify: bool

    # Decision state (exposed label)
    decision_state: str

    # T2 flag
    t2: bool

    # Step index within trajectory
    step: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "n_live": self.n_live,
            "n_eliminated": self.n_eliminated,
            "n_total": self.n_total,
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "n_verified": self.n_verified,
            "n_supporting": self.n_supporting,
            "n_contradicting": self.n_contradicting,
            "retrieval_remaining": self.retrieval_remaining,
            "search_remaining": self.search_remaining,
            "verify_remaining": self.verify_remaining,
            "steps_remaining": self.steps_remaining,
            "can_retrieve": self.can_retrieve,
            "can_search": self.can_search,
            "can_verify": self.can_verify,
            "decision_state": self.decision_state,
            "t2": self.t2,
            "step": self.step,
        }

    def as_vector(self) -> list[float]:
        """Numeric feature vector for ML models.

        Order is fixed and documented. Boolean fields are 0.0/1.0.
        """
        return [
            float(self.n_live),
            float(self.n_eliminated),
            float(self.n_total),
            float(self.n_visible),
            float(self.n_hidden),
            float(self.n_verified),
            float(self.n_supporting),
            float(self.n_contradicting),
            float(self.retrieval_remaining),
            float(self.search_remaining),
            float(self.verify_remaining),
            float(self.steps_remaining),
            1.0 if self.can_retrieve else 0.0,
            1.0 if self.can_search else 0.0,
            1.0 if self.can_verify else 0.0,
            1.0 if self.t2 else 0.0,
            float(self.step),
        ]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _FEATURE_NAMES


_FEATURE_NAMES = (
    "n_live",
    "n_eliminated",
    "n_total",
    "n_visible",
    "n_hidden",
    "n_verified",
    "n_supporting",
    "n_contradicting",
    "retrieval_remaining",
    "search_remaining",
    "verify_remaining",
    "steps_remaining",
    "can_retrieve",
    "can_search",
    "can_verify",
    "t2",
    "step",
)


def features_from_receipt(
    receipt: Mapping[str, Any],
    *,
    phase: str = "",
) -> PhaseFeatures:
    """Build PhaseFeatures from a mechanism receipt.

    Receipts contain: n_live_hypotheses, n_eliminated_hypotheses,
    legal_actions, decision_state_exposed, t2, step, and resource info
    derivable from legal_actions.
    """
    n_live = int(receipt.get("n_live_hypotheses", 0))
    n_eliminated = int(receipt.get("n_eliminated_hypotheses", 0))
    n_total = n_live + n_eliminated

    legal_actions = receipt.get("legal_actions", [])
    can_retrieve = "RETRIEVE" in legal_actions
    can_search = "SEARCH_MORE" in legal_actions
    can_verify = "VERIFY" in legal_actions

    decision_state = receipt.get("decision_state_exposed", "UNKNOWN")
    t2 = bool(receipt.get("t2", False))
    step = int(receipt.get("step", 0))

    # Resource counts are not directly in receipts, but we can infer
    # remaining counts from the legal_actions and the budget defaults.
    # For the transition dataset, we'll supplement from the trajectory
    # record where available. For now, use affordance booleans as
    # proxies and set remaining to 0 if not available.
    retrieval_remaining = 1 if can_retrieve else 0
    search_remaining = 1 if can_search else 0
    verify_remaining = 1 if can_verify else 0
    steps_remaining = 12 - step  # default budget max_executive_steps=12

    # Evidence counts are not in receipts directly. They will be
    # supplemented from the snapshot when available. For now, use
    # verified/supporting/contradicting from the receipt if present.
    n_visible = int(receipt.get("n_visible_evidence", 0))
    n_hidden = int(receipt.get("n_hidden_evidence", 0))
    n_verified = int(receipt.get("n_verified", 0))
    n_supporting = int(receipt.get("n_supporting", 0))
    n_contradicting = int(receipt.get("n_contradicting", 0))

    return PhaseFeatures(
        phase=phase,
        n_live=n_live,
        n_eliminated=n_eliminated,
        n_total=n_total,
        n_visible=n_visible,
        n_hidden=n_hidden,
        n_verified=n_verified,
        n_supporting=n_supporting,
        n_contradicting=n_contradicting,
        retrieval_remaining=retrieval_remaining,
        search_remaining=search_remaining,
        verify_remaining=verify_remaining,
        steps_remaining=steps_remaining,
        can_retrieve=can_retrieve,
        can_search=can_search,
        can_verify=can_verify,
        decision_state=decision_state,
        t2=t2,
        step=step,
    )
