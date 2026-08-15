"""Chain progression tracking for the governor.

Extracts V2_STAGE_N outcomes from prior_outcomes to track composition chain
progress. This lets the governor know:
- Whether a composition chain has been started
- How many stages have been completed
- Which actions advanced the chain (and which didn't)
- Whether the chain is complete

This is critical for DEPTH_4_PLUS tasks where the model must execute a
specific sequence of actions in order. Without chain tracking, the governor
cannot distinguish "chain not started" from "chain complete."
"""
from __future__ import annotations

import re
from dataclasses import dataclass


CHAIN_PROGRESS_SCHEMA = "DAPH_V2B_I3_5_CHAIN_PROGRESS_V1"
CHAIN_PROGRESS_VERSION = 1

# Pattern for V2 stage outcomes
_STAGE_PATTERN = re.compile(r"^V2_STAGE_(\d+)$")

# Actions that can compose (advance the chain)
COMPOSABLE_ACTIONS = ("RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE")


@dataclass(frozen=True)
class ChainProgress:
    """Tracks composition chain progression from prior outcomes.

    Fields:
        stages_completed: Number of V2_STAGE_N outcomes seen (0 = not started).
        stage_outcomes: Tuple of V2_STAGE_N outcome strings in order.
        actions_that_advanced: Tuple of actions that appended a V2_STAGE_N.
        actions_that_failed: Tuple of actions that were tried but didn't advance.
        is_started: Whether any stage has been completed.
        is_complete: Whether the chain appears complete (heuristic: >= 3 stages
                     and no CONTROL_POISONED in recent outcomes).
        is_poisoned: Whether CONTROL_POISONED appears in prior outcomes.
        total_steps: Total number of prior actions executed.
    """
    stages_completed: int
    stage_outcomes: tuple[str, ...]
    actions_that_advanced: tuple[str, ...]
    actions_that_failed: tuple[str, ...]
    is_started: bool
    is_complete: bool
    is_poisoned: bool
    total_steps: int

    @property
    def needs_discovery(self) -> bool:
        """Whether the governor should enter chain discovery mode.

        True when the chain hasn't started and the task isn't poisoned,
        meaning the model needs to try composable actions to find the
        right starting action.
        """
        return not self.is_started and not self.is_poisoned and not self.is_complete

    @property
    def needs_continuation(self) -> bool:
        """Whether the chain is partially complete and needs more steps."""
        return self.is_started and not self.is_complete and not self.is_poisoned

    def as_dict(self) -> dict:
        return {
            "stages_completed": self.stages_completed,
            "stage_outcomes": list(self.stage_outcomes),
            "actions_that_advanced": list(self.actions_that_advanced),
            "actions_that_failed": list(self.actions_that_failed),
            "is_started": self.is_started,
            "is_complete": self.is_complete,
            "is_poisoned": self.is_poisoned,
            "needs_discovery": self.needs_discovery,
            "needs_continuation": self.needs_continuation,
        }


def extract_chain_progress(
    prior_outcomes: tuple[str, ...],
    prior_actions: tuple[str, ...],
) -> ChainProgress:
    """Extract chain progress from prior outcomes and actions.

    Args:
        prior_outcomes: Ordered tuple of outcome codes from executed actions.
        prior_actions: Ordered tuple of action names executed.

    Returns:
        ChainProgress describing the current chain state.
    """
    stage_outcomes: list[str] = []
    actions_advanced: list[str] = []
    actions_failed: list[str] = []
    is_poisoned = False
    total_steps = len(prior_actions)

    # Track which actions advanced the chain and which didn't
    for i, outcome in enumerate(prior_outcomes):
        if outcome == "CONTROL_POISONED":
            is_poisoned = True
            continue

        match = _STAGE_PATTERN.match(outcome)
        if match:
            stage_outcomes.append(outcome)
            # The action that produced this outcome advanced the chain
            if i < len(prior_actions):
                action = prior_actions[i]
                if action not in actions_advanced:
                    actions_advanced.append(action)
        elif i < len(prior_actions):
            action = prior_actions[i]
            # Action was composable but didn't advance the chain
            if action in COMPOSABLE_ACTIONS and action not in actions_advanced:
                if action not in actions_failed:
                    actions_failed.append(action)

    stages_completed = len(stage_outcomes)
    is_started = stages_completed > 0

    # Chain is complete when we see composition_complete signal
    # Heuristic: >= 3 stages and not poisoned
    # The exact completion threshold depends on the chain length,
    # but 3 is the minimum for dev (length 3) and we can't know
    # the exact target from the governor's perspective.
    # A more reliable signal is that the last action's outcome was
    # TASK_SUCCESS or that verification_state became SUFFICIENT.
    # For now, we use >= 3 stages + not poisoned as a heuristic.
    is_complete = stages_completed >= 3 and not is_poisoned

    return ChainProgress(
        stages_completed=stages_completed,
        stage_outcomes=tuple(stage_outcomes),
        actions_that_advanced=tuple(actions_advanced),
        actions_that_failed=tuple(actions_failed),
        is_started=is_started,
        is_complete=is_complete,
        is_poisoned=is_poisoned,
        total_steps=total_steps,
    )


def untried_composable_actions(
    prior_actions: tuple[str, ...],
    legal_actions: tuple[str, ...],
) -> tuple[str, ...]:
    """Return composable actions that haven't been tried yet.

    Used in chain discovery mode: when the chain hasn't started,
    the governor should recommend trying untried composable actions
    to discover which one starts the chain.
    """
    tried = set(prior_actions)
    return tuple(
        a for a in COMPOSABLE_ACTIONS
        if a not in tried and a in legal_actions
    )
