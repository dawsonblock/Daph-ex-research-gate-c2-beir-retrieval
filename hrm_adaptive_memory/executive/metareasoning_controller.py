"""Matched controller used in both V2B-I3 masking conditions."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import CognitiveStateSnapshot, TemporalStatus, VerificationState

from .actions import ActionProposal


@dataclass(frozen=True)
class ControllerObservation:
    """Same controller contract in both conditions; only `cognitive_state` is masked."""

    task_id: str
    task_summary: str
    resource_state: dict[str, int]
    allowed_actions: tuple[DecisionAction, ...]
    executed_actions: tuple[DecisionAction, ...]
    rejected_actions: tuple[DecisionAction, ...]
    cognitive_state: CognitiveStateSnapshot | None


@dataclass(frozen=True)
class ObservationMask:
    """Frozen controller-visible fields; resource state and action history are always basic inputs."""

    verification: bool = False
    provenance: bool = False
    temporal: bool = False
    conflicts: bool = False
    prior_outcomes: bool = False
    composition: bool = False

    def sha256(self) -> str:
        encoded = json.dumps({
            "verification": self.verification, "provenance": self.provenance,
            "temporal": self.temporal, "conflicts": self.conflicts,
            "prior_outcomes": self.prior_outcomes, "composition": self.composition,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


STATE_BLIND_MASK = ObservationMask()
STATE_AWARE_MASK = ObservationMask(
    verification=True, provenance=True, temporal=True, conflicts=True,
    prior_outcomes=True, composition=True)


def load_observation_masks(path: str | Path) -> Mapping[str, ObservationMask]:
    """Load frozen I3 ablation masks; unknown fields fail closed."""
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema") != "DAPH_V2B_I3_OBSERVATION_MASKS_V1"
            or payload.get("status") != "FROZEN_FOR_DEVELOPMENT"):
        raise ValueError("I3 observation masks must be frozen for development")
    raw_masks = payload.get("masks")
    if not isinstance(raw_masks, Mapping) or not raw_masks:
        raise ValueError("I3 observation masks must be nonempty")
    fields = {"verification", "provenance", "temporal", "conflicts", "prior_outcomes", "composition"}
    masks: dict[str, ObservationMask] = {}
    for name, raw in raw_masks.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError("I3 masks require a condition name and exactly the frozen fields")
        if not all(isinstance(value, bool) for value in raw.values()):
            raise ValueError("I3 observation mask values must be booleans")
        masks[name] = ObservationMask(**dict(raw))
    if masks.get("STATE_BLIND_CONTROLLER") != STATE_BLIND_MASK:
        raise ValueError("I3 state-blind mask must hide every cognitive field")
    if masks.get("STATE_AWARE_CONTROLLER") != STATE_AWARE_MASK:
        raise ValueError("I3 state-aware mask must expose every frozen cognitive field")
    return masks


def apply_observation_mask(snapshot: CognitiveStateSnapshot, mask: ObservationMask) -> CognitiveStateSnapshot | None:
    """Return the exact bounded cognitive projection supplied to a controller."""
    if mask == STATE_BLIND_MASK:
        return None
    return replace(
        snapshot,
        relevant_memories=snapshot.relevant_memories if mask.verification else (),
        verification_states=snapshot.verification_states if mask.verification else (),
        provenance_summaries=snapshot.provenance_summaries if mask.provenance else (),
        temporal_status=snapshot.temporal_status if mask.temporal else TemporalStatus.UNKNOWN,
        unresolved_conflicts=snapshot.unresolved_conflicts if mask.conflicts else (),
        prior_decisions=snapshot.prior_decisions if mask.prior_outcomes else (),
        prior_outcomes=snapshot.prior_outcomes if mask.prior_outcomes else (),
        observation_signals=snapshot.observation_signals if mask.composition else (),
    )


class MatchedMetareasoningController:
    """One deterministic architecture whose cognitive-state input is masked by the runner.

    This controller is not a learned executive and is only a protocol fixture.
    Its code path, action vocabulary, and parameters are identical under both
    conditions; the runner controls whether the same snapshot argument is
    present or masked. The controller has no condition-specific parameters.
    """

    algorithm_id = "v2b_i3_matched_metareasoning_controller_v1"

    controller_id = algorithm_id

    @staticmethod
    def _executed(observation: ControllerObservation, action: DecisionAction) -> bool:
        return action in observation.executed_actions

    @staticmethod
    def _rejected(observation: ControllerObservation, action: DecisionAction) -> bool:
        return action in observation.rejected_actions

    @staticmethod
    def _available(observation: ControllerObservation, action: DecisionAction) -> bool:
        if action not in observation.allowed_actions:
            return False
        resources = observation.resource_state
        if action is DecisionAction.RETRIEVE:
            return resources["retrieval_calls_remaining"] > 0
        if action is DecisionAction.VERIFY:
            return resources["verification_calls_remaining"] > 0
        if action is DecisionAction.SEARCH_MORE:
            return resources["search_calls_remaining"] > 0
        if action is DecisionAction.REASON_MORE:
            return resources["reasoning_tokens_remaining"] >= 128
        return resources["executive_steps_remaining"] > 0

    def _state_aware_choice(self, observation: ControllerObservation,
                            state: CognitiveStateSnapshot) -> ActionProposal:
        if "NO_FINAL_ASSERTION_REQUESTED" in observation.task_summary:
            return ActionProposal(DecisionAction.STOP, "TASK_REQUESTS_INTERNAL_STOP")
        verification = state.verification_states[0].state if state.verification_states else None
        composition_incomplete = "COMPOSITION_INCOMPLETE" in state.observation_signals
        if state.unresolved_conflicts:
            return ActionProposal(DecisionAction.DEFER, "OBSERVED_UNRESOLVED_CONFLICT")
        if (verification in {VerificationState.UNVERIFIED, VerificationState.STALE}
                or state.temporal_status is TemporalStatus.STALE):
            if self._available(observation, DecisionAction.VERIFY):
                return ActionProposal(DecisionAction.VERIFY, "OBSERVED_VERIFICATION_GAP")
            return ActionProposal(DecisionAction.DEFER, "VERIFICATION_BUDGET_UNAVAILABLE")
        if composition_incomplete and not self._executed(observation, DecisionAction.REASON_MORE):
            return ActionProposal(DecisionAction.REASON_MORE, "OBSERVED_COMPOSITION_INCOMPLETE")
        if (verification is VerificationState.SUFFICIENT
                and state.temporal_status is TemporalStatus.CURRENT
                and not composition_incomplete):
            return ActionProposal(DecisionAction.ANSWER, "OBSERVED_CURRENT_SUPPORTED_EVIDENCE")
        if verification in {VerificationState.MISSING, VerificationState.FALSIFIED}:
            if (not self._executed(observation, DecisionAction.RETRIEVE)
                    and self._available(observation, DecisionAction.RETRIEVE)
                    and not self._rejected(observation, DecisionAction.RETRIEVE)):
                return ActionProposal(DecisionAction.RETRIEVE, "OBSERVED_EVIDENCE_GAP")
            if (not self._executed(observation, DecisionAction.VERIFY)
                    and self._available(observation, DecisionAction.VERIFY)
                    and not self._rejected(observation, DecisionAction.VERIFY)):
                return ActionProposal(DecisionAction.VERIFY, "OBSERVED_RETRIEVAL_DID_NOT_RESOLVE")
            if (not self._executed(observation, DecisionAction.SEARCH_MORE)
                    and self._available(observation, DecisionAction.SEARCH_MORE)
                    and not self._rejected(observation, DecisionAction.SEARCH_MORE)):
                return ActionProposal(DecisionAction.SEARCH_MORE, "OBSERVED_EVIDENCE_STILL_MISSING")
        return ActionProposal(DecisionAction.DEFER, "OBSERVED_INSUFFICIENT_EVIDENCE")

    def _state_blind_choice(self, observation: ControllerObservation) -> ActionProposal:
        """Fixed fallback path, with policy/resource rejection feedback for replanning."""
        if "NO_FINAL_ASSERTION_REQUESTED" in observation.task_summary:
            return ActionProposal(DecisionAction.STOP, "TASK_REQUESTS_INTERNAL_STOP")
        if (not self._executed(observation, DecisionAction.RETRIEVE)
                and not self._rejected(observation, DecisionAction.RETRIEVE)
                and self._available(observation, DecisionAction.RETRIEVE)):
            return ActionProposal(DecisionAction.RETRIEVE, "MASKED_STATE_FALLBACK_RETRIEVE")
        if (not self._executed(observation, DecisionAction.VERIFY)
                and not self._rejected(observation, DecisionAction.VERIFY)
                and self._available(observation, DecisionAction.VERIFY)):
            return ActionProposal(DecisionAction.VERIFY, "MASKED_STATE_FALLBACK_VERIFY")
        if (not self._executed(observation, DecisionAction.REASON_MORE)
                and self._available(observation, DecisionAction.REASON_MORE)):
            return ActionProposal(DecisionAction.REASON_MORE, "MASKED_STATE_FALLBACK_REASON")
        if not self._rejected(observation, DecisionAction.ANSWER):
            return ActionProposal(DecisionAction.ANSWER, "MASKED_STATE_FALLBACK_ANSWER")
        return ActionProposal(DecisionAction.DEFER, "MASKED_STATE_NO_LEGAL_PROGRESS")

    def choose(self, observation: ControllerObservation) -> ActionProposal:
        if observation.cognitive_state is not None:
            return self._state_aware_choice(observation, observation.cognitive_state)
        return self._state_blind_choice(observation)
