"""Governor assessor: the central API that builds the decision frame.

class GeneralGovernor:
    def assess(observation) -> GovernorDecisionFrame

The frame is given to DeepSeek, which makes the final choice.
The governor does NOT choose the action — it constructs the decision frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from hrm_adaptive_memory.executive.governor.state import (
    GovernorState, build_governor_state)
from hrm_adaptive_memory.executive.governor.bottlenecks import (
    DecisionBottleneck, detect_bottlenecks)
from hrm_adaptive_memory.executive.governor.transition_model import predict_outcome
from hrm_adaptive_memory.executive.governor.candidate_features import (
    CandidateActionAssessment, assess_candidate)
from hrm_adaptive_memory.executive.governor.action_semantics import (
    get_action_semantics, FROZEN_ACTION_SEMANTICS)
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation


ASSESSOR_SCHEMA = "DAPH_V2B_I3_5_GOVERNOR_ASSESSOR_V1"
ASSESSOR_VERSION = 1


@dataclass(frozen=True)
class GovernorDecisionFrame:
    """The complete decision frame produced by the governor.

    This is given to DeepSeek as part of the model packet.
    DeepSeek chooses from the candidates — the governor does NOT choose.
    """
    active_bottlenecks: tuple[DecisionBottleneck, ...]
    candidates: tuple[CandidateActionAssessment, ...]
    governor_top_action: str  # for diagnostic only, not forced on the model
    governor_reason_code: str
    chain_progress: dict | None = None  # chain tracking info for the model

    def as_dict(self) -> dict:
        return {
            "active_bottlenecks": [b.as_dict() for b in self.active_bottlenecks],
            "candidates": [c.as_dict() for c in self.candidates],
            "governor_top_action": self.governor_top_action,
            "governor_reason_code": self.governor_reason_code,
            "chain_progress": self.chain_progress,
        }

    def as_model_packet(self) -> dict:
        """Compact representation for the model prompt."""
        packet = {
            "current_bottlenecks": [
                {"kind": b.kind, "severity": b.severity}
                for b in self.active_bottlenecks
            ],
            "candidate_actions": [
                {
                    "action": c.action,
                    "targets_blocker": c.targets_current_blocker,
                    "expected_information_change": c.information_value,
                    "expected_task_progress": c.expected_progress,
                    "resource_cost": c.resource_cost,
                    "repeat_penalty": c.repeat_penalty,
                    "option_preservation": c.option_preservation,
                    "policy_risk": c.policy_risk,
                    "creates_external_information": c.creates_external_information,
                    "only_transforms_existing": c.only_transforms_existing,
                    "recently_failed": c.recently_failed,
                    "terminates_under_uncertainty": c.terminates_under_uncertainty,
                }
                for c in self.candidates
            ],
        }
        if self.chain_progress:
            packet["chain_progress"] = self.chain_progress
        return packet


class GeneralGovernor:
    """The general governor: model-based executive layer.

    assess(observation) → GovernorDecisionFrame

    The governor:
    1. Builds GovernorState from ControllerObservation
    2. Detects current bottlenecks
    3. Predicts outcomes for each legal action
    4. Scores each candidate with topology-invariant features
    5. Ranks candidates and builds the decision frame
    """

    def __init__(self, max_steps: int = 25):
        self._max_steps = max_steps

    def assess(
        self,
        observation: ControllerObservation,
        remaining_steps: int | None = None,
        prior_actions: tuple[str, ...] | None = None,
        prior_outcomes: tuple[str, ...] | None = None,
    ) -> GovernorDecisionFrame:
        """Assess the current state and build a decision frame."""
        if remaining_steps is None:
            remaining_steps = self._max_steps - len(observation.executed_actions)

        state = build_governor_state(
            observation=observation,
            remaining_steps=remaining_steps,
            prior_actions=prior_actions,
            prior_outcomes=prior_outcomes,
        )

        # Detect bottlenecks
        bottlenecks = detect_bottlenecks(state)

        # Assess each legal action
        candidates: list[CandidateActionAssessment] = []
        for action in state.legal_actions:
            outcome = predict_outcome(state, action, bottlenecks)
            assessment = assess_candidate(state, action, outcome, bottlenecks)
            candidates.append(assessment)

        # Rank by score (descending)
        candidates.sort(key=lambda c: c.score(), reverse=True)

        # Governor top action (diagnostic only — not forced on model)
        top_action = candidates[0].action if candidates else "DEFER"
        reason_code = self._build_reason_code(candidates, bottlenecks)

        # Build chain progress info for the model packet
        chain = state.chain_progress
        chain_info = None
        if chain.is_started or chain.is_poisoned or chain.total_steps > 0:
            chain_info = chain.as_dict()

        return GovernorDecisionFrame(
            active_bottlenecks=bottlenecks,
            candidates=tuple(candidates),
            governor_top_action=top_action,
            governor_reason_code=reason_code,
            chain_progress=chain_info,
        )

    def _build_reason_code(
        self,
        candidates: list[CandidateActionAssessment],
        bottlenecks: tuple[DecisionBottleneck, ...],
    ) -> str:
        """Build a compact reason code for the governor's top recommendation."""
        if not candidates:
            return "NO_CANDIDATES"
        top = candidates[0]
        active = bottlenecks[0] if bottlenecks else None
        if active and active.kind == "READY_TO_ANSWER":
            return "READY_TO_ANSWER"
        if active and active.kind == "CHAIN_DISCOVERY":
            return "CHAIN_DISCOVERY_NEEDED"
        if active and active.kind == "CHAIN_INCOMPLETE":
            return "CHAIN_CONTINUATION_NEEDED"
        if active and active.kind == "IRREDUCIBLE_UNCERTAINTY":
            return "TASK_UNSOLVABLE_DEFER"
        if top.recently_failed:
            return "AVOID_REPEATED_FAILURE"
        if top.targets_current_blocker:
            return f"TARGETS_{active.kind}" if active else "TARGETS_BLOCKER"
        if top.creates_external_information:
            return "ADDS_NEW_INFORMATION"
        return "BEST_AVAILABLE_OPTION"
