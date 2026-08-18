"""Feature extractor for selective governor intervention gate.

Extracts strictly controller-visible features from ControllerObservation,
remaining_steps, prior_actions, and prior_outcomes.

Never accesses latent oracle values, topology labels, evaluator metadata,
or future outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from hrm_adaptive_memory.executive.governor.chain_progress import extract_chain_progress
from hrm_adaptive_memory.executive.governor.resources import normalize_resources
from hrm_adaptive_memory.executive.governor.state import build_governor_state
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation

FEATURE_SCHEMA = "DAPH_V2B_I3_5_2_INTERVENTION_FEATURES_V1"
FEATURE_VERSION = 1

# Fixed list of numerical / categorical feature column names for tabular models
FEATURE_NAMES = [
    "remaining_steps",
    "prior_action_count",
    "repeated_no_gain",
    "has_cognitive_state",
    "evidence_count",
    "verified_count",
    "conflict_count",
    "retrieval_budget_remaining",
    "verification_budget_remaining",
    "search_budget_remaining",
    "reasoning_budget_remaining",
    "chain_started",
    "chain_completed",
    "chain_length",
    "chain_stage",
    # One-hot / categorical mappings
    "verif_SUFFICIENT",
    "verif_MISSING",
    "verif_FALSIFIED",
    "verif_NONE",
    "temporal_CURRENT",
    "temporal_STALE",
    "temporal_UNKNOWN",
    "temporal_NONE",
    "last_act_RETRIEVE",
    "last_act_VERIFY",
    "last_act_SEARCH_MORE",
    "last_act_REASON_MORE",
    "last_act_NONE",
]


@dataclass(frozen=True)
class InterventionFeatures:
    """Typed container for controller-visible intervention features."""
    remaining_steps: int
    prior_action_count: int
    last_action: str | None
    last_outcome: str | None
    repeated_no_gain: bool
    has_cognitive_state: bool
    evidence_count: int
    verified_count: int
    verification_state: str
    temporal_status: str
    conflict_count: int
    reasoning_depth: int
    retrieval_budget_remaining: int
    verification_budget_remaining: int
    search_budget_remaining: int
    reasoning_budget_remaining: int
    chain_started: bool
    chain_completed: bool
    chain_length: int
    chain_stage: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "remaining_steps": self.remaining_steps,
            "prior_action_count": self.prior_action_count,
            "last_action": self.last_action,
            "last_outcome": self.last_outcome,
            "repeated_no_gain": self.repeated_no_gain,
            "has_cognitive_state": self.has_cognitive_state,
            "evidence_count": self.evidence_count,
            "verified_count": self.verified_count,
            "verification_state": self.verification_state,
            "temporal_status": self.temporal_status,
            "conflict_count": self.conflict_count,
            "reasoning_depth": self.reasoning_depth,
            "retrieval_budget_remaining": self.retrieval_budget_remaining,
            "verification_budget_remaining": self.verification_budget_remaining,
            "search_budget_remaining": self.search_budget_remaining,
            "reasoning_budget_remaining": self.reasoning_budget_remaining,
            "chain_started": self.chain_started,
            "chain_completed": self.chain_completed,
            "chain_length": self.chain_length,
            "chain_stage": self.chain_stage,
        }

    def to_numeric_vector(self) -> list[float]:
        """Convert features into a fixed-length numeric vector."""
        d = self.as_dict()
        vec: list[float] = [
            float(d["remaining_steps"]),
            float(d["prior_action_count"]),
            1.0 if d["repeated_no_gain"] else 0.0,
            1.0 if d["has_cognitive_state"] else 0.0,
            float(d["evidence_count"]),
            float(d["verified_count"]),
            float(d["conflict_count"]),
            float(d["retrieval_budget_remaining"]),
            float(d["verification_budget_remaining"]),
            float(d["search_budget_remaining"]),
            float(d["reasoning_budget_remaining"]),
            1.0 if d["chain_started"] else 0.0,
            1.0 if d["chain_completed"] else 0.0,
            float(d["chain_length"]),
            float(d["chain_stage"]),
            # Verification one-hot
            1.0 if d["verification_state"] == "SUFFICIENT" else 0.0,
            1.0 if d["verification_state"] == "MISSING" else 0.0,
            1.0 if d["verification_state"] == "FALSIFIED" else 0.0,
            1.0 if d["verification_state"] in ("NONE", "UNKNOWN") else 0.0,
            # Temporal one-hot
            1.0 if d["temporal_status"] == "CURRENT" else 0.0,
            1.0 if d["temporal_status"] == "STALE" else 0.0,
            1.0 if d["temporal_status"] == "UNKNOWN" else 0.0,
            1.0 if d["temporal_status"] in ("NONE", "") else 0.0,
            # Last action one-hot
            1.0 if d["last_action"] == "RETRIEVE" else 0.0,
            1.0 if d["last_action"] == "VERIFY" else 0.0,
            1.0 if d["last_action"] == "SEARCH_MORE" else 0.0,
            1.0 if d["last_action"] == "REASON_MORE" else 0.0,
            1.0 if d["last_action"] is None else 0.0,
        ]
        return vec


def extract_features(
    observation: ControllerObservation,
    *,
    remaining_steps: int,
    prior_actions: tuple[str, ...],
    prior_outcomes: tuple[str, ...],
) -> InterventionFeatures:
    """Extract controller-visible intervention features from observation and history."""
    gov_state = build_governor_state(
        observation=observation,
        remaining_steps=remaining_steps,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )
    cs = observation.cognitive_state
    norm_res = normalize_resources(observation.resource_state)

    evidence_count = len(cs.relevant_memories) if cs and cs.relevant_memories else 0

    verif_state_str = "NONE"
    verified_count = 0
    if cs and cs.verification_states:
        for v in cs.verification_states:
            st = v.state.value if hasattr(v.state, "value") else str(v.state)
            if st == "SUFFICIENT":
                verified_count += 1
            verif_state_str = st

    temporal_str = "NONE"
    if cs and cs.temporal_status:
        temporal_str = (
            cs.temporal_status.value
            if hasattr(cs.temporal_status, "value")
            else str(cs.temporal_status)
        )

    conflict_count = len(cs.unresolved_conflicts) if cs and cs.unresolved_conflicts else 0

    cp = gov_state.chain_progress
    chain_stage = cp.stages_completed
    chain_started = cp.is_started
    chain_completed = cp.is_complete
    chain_length = len(cp.stage_outcomes)

    return InterventionFeatures(
        remaining_steps=remaining_steps,
        prior_action_count=len(prior_actions),
        last_action=prior_actions[-1] if prior_actions else None,
        last_outcome=prior_outcomes[-1] if prior_outcomes else None,
        repeated_no_gain=gov_state.repeated_no_gain,
        has_cognitive_state=gov_state.has_cognitive_state,
        evidence_count=evidence_count,
        verified_count=verified_count,
        verification_state=verif_state_str,
        temporal_status=temporal_str,
        conflict_count=conflict_count,
        reasoning_depth=chain_stage,
        retrieval_budget_remaining=norm_res.retrieval_remaining,
        verification_budget_remaining=norm_res.verification_remaining,
        search_budget_remaining=norm_res.search_remaining,
        reasoning_budget_remaining=norm_res.reasoning_tokens_remaining,
        chain_started=chain_started,
        chain_completed=chain_completed,
        chain_length=chain_length,
        chain_stage=chain_stage,
    )
