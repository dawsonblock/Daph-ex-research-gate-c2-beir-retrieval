"""Governor state: controller-visible state for the governor layer.

The governor sees only what the controller sees under the current condition.
For STATE_BLIND, it cannot access cognitive_state fields.
For STATE_AWARE, it can access the full CognitiveStateSnapshot.

This is critical: the governor must not leak evaluator information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.executive.governor.resources import (
    GovernorResourceState, normalize_resources)
from hrm_adaptive_memory.executive.governor.chain_progress import (
    ChainProgress, extract_chain_progress, untried_composable_actions)


GOVERNOR_STATE_SCHEMA = "DAPH_V2B_I3_5_GOVERNOR_STATE_V1"
GOVERNOR_STATE_VERSION = 1


@dataclass(frozen=True)
class GovernorState:
    """Controller-visible state for governor assessment.

    Fields:
        observation: The full ControllerObservation (already masked).
        legal_actions: Actions allowed by policy and resources.
        remaining_steps: Steps remaining before max-step termination.
        prior_actions: Ordered tuple of previously executed actions.
        prior_outcomes: Ordered tuple of outcome codes from prior actions.
        prior_action_results: Map from action name to count of times executed.
        last_action: The most recent action, or None if no actions yet.
        last_outcome: The most recent outcome code, or None.
        repeated_no_gain: Whether the last action produced no gain and was repeated.
        resources: Typed GovernorResourceState (normalized from runtime dict).
    """
    observation: ControllerObservation
    legal_actions: tuple[str, ...]
    remaining_steps: int
    prior_actions: tuple[str, ...]
    prior_outcomes: tuple[str, ...]
    prior_action_results: dict[str, int]
    last_action: str | None
    last_outcome: str | None
    repeated_no_gain: bool
    resources: GovernorResourceState

    @property
    def task_id(self) -> str:
        return self.observation.task_id

    @property
    def has_cognitive_state(self) -> bool:
        """Whether the governor can see DAPH cognitive state (aware condition)."""
        return self.observation.cognitive_state is not None

    def action_count(self, action: str) -> int:
        """How many times an action has been executed."""
        return self.prior_action_results.get(action, 0)

    def action_was_executed(self, action: str) -> bool:
        """Whether an action has been executed at least once."""
        return self.action_count(action) > 0

    def outcome_sequence_unchanged(self, action: str) -> bool:
        """Whether the last two outcomes for this action were identical.

        This is a true no-gain signal: the action was repeated and produced
        the same outcome code, meaning observable state likely didn't change.
        """
        if self.action_count(action) < 2:
            return False
        # Find the last two outcomes for this action
        outcomes_for_action = []
        for i, a in enumerate(self.prior_actions):
            if a == action and i < len(self.prior_outcomes):
                outcomes_for_action.append(self.prior_outcomes[i])
        if len(outcomes_for_action) >= 2:
            return outcomes_for_action[-1] == outcomes_for_action[-2]
        return False

    @property
    def chain_progress(self) -> ChainProgress:
        """Chain progression from V2_STAGE_N outcomes in prior_outcomes.

        Extracted from the cognitive_state's prior_outcomes (aware condition)
        or from the governor's own prior_outcomes tracking.
        """
        # Try cognitive_state first (aware condition)
        cs = self.observation.cognitive_state
        if cs is not None and cs.prior_outcomes:
            return extract_chain_progress(
                tuple(cs.prior_outcomes), self.prior_actions)
        # Fall back to governor's own tracking
        return extract_chain_progress(
            self.prior_outcomes, self.prior_actions)

    def untried_composable(self) -> tuple[str, ...]:
        """Composable actions not yet tried, for chain discovery."""
        return untried_composable_actions(self.prior_actions, self.legal_actions)


def build_governor_state(
    observation: ControllerObservation,
    remaining_steps: int,
    prior_actions: tuple[str, ...] | None = None,
    prior_outcomes: tuple[str, ...] | None = None,
) -> GovernorState:
    """Construct a GovernorState from a ControllerObservation and history."""
    prior_actions = prior_actions or tuple(
        a.value if hasattr(a, "value") else str(a)
        for a in observation.executed_actions)
    prior_outcomes = prior_outcomes or ()

    # Count action executions
    action_counts: dict[str, int] = {}
    for a in prior_actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    # Legal actions from observation
    legal = tuple(
        a.value if hasattr(a, "value") else str(a)
        for a in observation.allowed_actions)

    # Detect repeated no-gain using outcome-based detection.
    # A true no-gain is: same action repeated AND same outcome code,
    # meaning observable state likely didn't change.
    # Merely repeating an action is NOT no-gain — multi-step topologies
    # legitimately require repeated SEARCH_MORE or REASON_MORE.
    repeated_no_gain = False
    if len(prior_actions) >= 2 and len(prior_outcomes) >= 2:
        if (prior_actions[-1] == prior_actions[-2]
                and prior_outcomes[-1] == prior_outcomes[-2]):
            repeated_no_gain = True

    # Normalize resources from the real runtime dict
    resources = normalize_resources(observation.resource_state)

    last_action = prior_actions[-1] if prior_actions else None
    last_outcome = prior_outcomes[-1] if prior_outcomes else None

    return GovernorState(
        observation=observation,
        legal_actions=legal,
        remaining_steps=remaining_steps,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
        prior_action_results=action_counts,
        last_action=last_action,
        last_outcome=last_outcome,
        repeated_no_gain=repeated_no_gain,
        resources=resources,
    )
