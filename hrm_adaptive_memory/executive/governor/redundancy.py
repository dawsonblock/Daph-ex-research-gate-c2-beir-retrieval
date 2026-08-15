"""Redundancy detection: diminishing-return penalties for repeated actions.

This is a genuine metareasoning primitive. If an action was already tried
and didn't resolve the bottleneck, repeating it has diminishing value.

D(a) = f(number of previous attempts, recent gain, same-state recurrence)

The penalty is categorical: NONE, LOW, MEDIUM, HIGH.
"""
from __future__ import annotations

from hrm_adaptive_memory.executive.governor.state import GovernorState
from hrm_adaptive_memory.executive.governor.bottlenecks import NONE, LOW, MEDIUM, HIGH


REDUNDANCY_SCHEMA = "DAPH_V2B_I3_5_REDUNDANCY_V1"
REDUNDANCY_VERSION = 1


def compute_redundancy(state: GovernorState, action: str) -> str:
    """Compute the redundancy penalty for an action.

    Returns: NONE, LOW, MEDIUM, or HIGH

    Rules:
    - Action never tried: NONE
    - Action tried once, not the last action: LOW (might help on different aspect)
    - Action tried once, was the last action: MEDIUM (just tried, didn't resolve)
    - Action tried 2+ times: HIGH (clearly not working)
    - Action tried 2+ times with same outcome: HIGH (strongly penalize)
    """
    count = state.action_count(action)
    if count == 0:
        return NONE

    if count >= 2:
        return HIGH

    # count == 1
    if state.last_action == action:
        # Was just tried and we're still here
        return MEDIUM

    # Tried once but not the last action — might still help
    return LOW


def compute_redundancy_detail(
    state: GovernorState,
    action: str,
) -> dict:
    """Detailed redundancy information for the governor frame."""
    count = state.action_count(action)
    penalty = compute_redundancy(state, action)

    return {
        "action": action,
        "prior_attempts": count,
        "redundancy_penalty": penalty,
        "was_last_action": state.last_action == action,
        "repeated_no_gain": state.repeated_no_gain and state.last_action == action,
    }
