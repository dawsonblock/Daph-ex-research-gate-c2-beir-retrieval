"""Deterministic controller conditions for the V2B-I2 controlled experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import CognitiveStateSnapshot, VerificationState

from .actions import ActionProposal


@dataclass(frozen=True)
class ControlObservation:
    """The control condition deliberately excludes cognitive-state fields."""

    task_id: str
    task_summary: str
    executive_steps_used: int
    executive_steps_remaining: int


class ControlController(Protocol):
    controller_id: str

    def choose(self, observation: ControlObservation) -> ActionProposal: ...


class StateAwareController(Protocol):
    controller_id: str

    def choose(self, state: CognitiveStateSnapshot) -> ActionProposal: ...


class FixedBaselineController:
    """Fixed retrieve → reason-more → answer baseline with no cognitive inputs."""

    controller_id = "v2b_i2_fixed_baseline_v1"
    _ACTIONS = (DecisionAction.RETRIEVE, DecisionAction.REASON_MORE, DecisionAction.ANSWER)

    def choose(self, observation: ControlObservation) -> ActionProposal:
        index = min(observation.executive_steps_used, len(self._ACTIONS) - 1)
        return ActionProposal(self._ACTIONS[index], "FIXED_BASELINE")


class DeterministicCognitiveStateController:
    """Rule-based full-state condition; this is not a learned executive."""

    controller_id = "v2b_i2_deterministic_cognitive_state_v1"

    @staticmethod
    def _has(state: CognitiveStateSnapshot, predicate: str) -> bool:
        return any(fact.predicate == predicate and fact.args == (state.task_id,)
                   for fact in state.policy_facts)

    @staticmethod
    def _attempted(state: CognitiveStateSnapshot, action: DecisionAction) -> bool:
        return any(decision.selected_action == action.value for decision in state.prior_decisions)

    def choose(self, state: CognitiveStateSnapshot) -> ActionProposal:
        if self._has(state, "unresolved_conflict"):
            return ActionProposal(DecisionAction.DEFER, "UNRESOLVED_CONFLICT")
        if (self._has(state, "reasoning_required")
                and not self._attempted(state, DecisionAction.REASON_MORE)):
            return ActionProposal(DecisionAction.REASON_MORE, "REASONING_REQUIRED")
        verification = state.verification_states[0].state
        if verification in {VerificationState.UNVERIFIED, VerificationState.STALE}:
            return ActionProposal(DecisionAction.VERIFY, "VERIFY_REQUIRED")
        if verification is VerificationState.SUFFICIENT and not self._has(state, "stale"):
            return ActionProposal(DecisionAction.ANSWER, "EVIDENCE_SUFFICIENT")
        if verification in {VerificationState.MISSING, VerificationState.FALSIFIED}:
            if not self._attempted(state, DecisionAction.RETRIEVE):
                return ActionProposal(DecisionAction.RETRIEVE, "RETRIEVE_REQUIRED")
            if not self._attempted(state, DecisionAction.SEARCH_MORE):
                return ActionProposal(DecisionAction.SEARCH_MORE, "SEARCH_REQUIRED")
        return ActionProposal(DecisionAction.DEFER, "INSUFFICIENT_EVIDENCE")
