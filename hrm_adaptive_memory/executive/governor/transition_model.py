"""Transition predictor: predict bounded outcomes for candidate actions.

Version 1 is deterministic and symbolic — not learned.
For each action, it predicts:
- what state channels might change
- whether the action can resolve the current bottleneck
- whether the action may repeat a failed strategy
- whether the action is terminal

This is the key component that makes the governor topology-aware:
instead of UNVERIFIED → VERIFY, it says:
"VERIFY may change verification status, but if VERIFY was already tried
and didn't resolve the bottleneck, it may repeat a failed strategy."
"""
from __future__ import annotations

from dataclasses import dataclass
from hrm_adaptive_memory.executive.governor.state import GovernorState
from hrm_adaptive_memory.executive.governor.action_semantics import (
    ActionSemantics, get_action_semantics)
from hrm_adaptive_memory.executive.governor.bottlenecks import (
    DecisionBottleneck, NONE, LOW, MEDIUM, HIGH)


TRANSITION_SCHEMA = "DAPH_V2B_I3_5_TRANSITION_MODEL_V1"
TRANSITION_VERSION = 1


@dataclass(frozen=True)
class PredictedActionOutcome:
    """Bounded prediction of what an action will do.

    All predictions are categorical (NONE/LOW/MEDIUM/HIGH) or boolean.
    No numeric probabilities or values — those would be fake calibration.
    """
    action: str
    possible_changes: tuple[str, ...]
    information_channels_affected: tuple[str, ...]
    expected_resource_cost: tuple[str, ...]
    can_resolve_current_blocker: bool
    may_repeat_failed_strategy: bool
    targets_active_bottleneck: bool
    terminal: bool
    adds_new_information: bool
    only_transforms_existing: bool

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "possible_changes": list(self.possible_changes),
            "information_channels_affected": list(self.information_channels_affected),
            "expected_resource_cost": list(self.expected_resource_cost),
            "can_resolve_current_blocker": self.can_resolve_current_blocker,
            "may_repeat_failed_strategy": self.may_repeat_failed_strategy,
            "targets_active_bottleneck": self.targets_active_bottleneck,
            "terminal": self.terminal,
            "adds_new_information": self.adds_new_information,
            "only_transforms_existing": self.only_transforms_existing,
        }


def predict_outcome(
    state: GovernorState,
    action: str,
    bottlenecks: tuple[DecisionBottleneck, ...],
) -> PredictedActionOutcome:
    """Predict the bounded outcome of executing `action` in `state`.

    Uses action semantics + bottleneck analysis + action history.
    """
    semantics = get_action_semantics(action)
    active_bottleneck = bottlenecks[0] if bottlenecks else None

    # Determine possible state changes
    changes: list[str] = []
    if semantics.can_change_verification:
        changes.append("verification_status")
    if semantics.can_add_evidence:
        changes.append("evidence_count")
    if semantics.can_reduce_conflict:
        changes.append("conflict_status")
    if semantics.can_change_reasoning_state:
        changes.append("composition_status")
    if semantics.can_change_temporal_status:
        changes.append("temporal_status")

    # Information channels affected
    channels: list[str] = []
    if semantics.can_add_evidence:
        channels.append("external_evidence")
    if semantics.can_change_verification:
        channels.append("verification")
    if semantics.can_reduce_conflict:
        channels.append("conflict_resolution")
    if semantics.can_change_reasoning_state:
        channels.append("internal_reasoning")

    # Resource cost
    cost = list(semantics.cost_channels)

    # Can resolve current blocker?
    can_resolve = False
    targets_bottleneck = False
    if active_bottleneck and active_bottleneck.kind != "READY_TO_ANSWER":
        if action in active_bottleneck.targetable_by:
            targets_bottleneck = True
            # Can resolve only if not already tried and failed
            if not _action_already_tried_without_gain(state, action):
                can_resolve = True
        # Chain discovery: action targets chain bottleneck
        if active_bottleneck.kind in ("CHAIN_DISCOVERY", "CHAIN_INCOMPLETE"):
            # Check if this action already failed to advance the chain
            chain = state.chain_progress
            if action in chain.actions_that_failed:
                can_resolve = False
                targets_bottleneck = False
            elif action in chain.actions_that_advanced:
                # This action advanced before — might advance again
                can_resolve = True
                targets_bottleneck = True

    # May repeat failed strategy?
    may_repeat = _action_already_tried_without_gain(state, action)

    # Chain-aware: action was tried but didn't advance the chain
    chain = state.chain_progress
    if action in chain.actions_that_failed and chain.is_started:
        may_repeat = True

    # Terminal?
    terminal = semantics.is_terminal

    # Adds new information?
    adds_info = semantics.external_information

    # Only transforms existing?
    only_transforms = (not semantics.external_information
                       and not semantics.internal_compute
                       and not semantics.is_terminal)

    return PredictedActionOutcome(
        action=action,
        possible_changes=tuple(changes),
        information_channels_affected=tuple(channels),
        expected_resource_cost=tuple(cost),
        can_resolve_current_blocker=can_resolve,
        may_repeat_failed_strategy=may_repeat,
        targets_active_bottleneck=targets_bottleneck,
        terminal=terminal,
        adds_new_information=adds_info,
        only_transforms_existing=only_transforms,
    )


def _action_already_tried_without_gain(state: GovernorState, action: str) -> bool:
    """Whether an action was already tried without observable gain.

    Uses outcome-based detection rather than mere repetition:
    - Action tried 2+ times with the SAME outcome code → true no-gain
    - Action tried 2+ times with DIFFERENT outcomes → not no-gain (state may be changing)
    - Action tried once as the last action → not no-gain (we haven't seen if it helped yet)

    Multi-step topologies legitimately require repeated SEARCH_MORE or REASON_MORE,
    so mere repetition is NOT sufficient to declare no-gain.
    """
    count = state.action_count(action)
    if count < 2:
        return False
    # Check if the last two executions of this action produced the same outcome
    return state.outcome_sequence_unchanged(action)
