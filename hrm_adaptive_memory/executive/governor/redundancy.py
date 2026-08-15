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

    Rules (outcome-based, not merely repetition-based):
    - Action never tried: NONE
    - Action tried once: LOW (might help on a different aspect)
    - Action tried 2+ times with SAME outcome: HIGH (true no-gain)
    - Action tried 2+ times with DIFFERENT outcomes: MEDIUM (state may be changing)

    Multi-step topologies legitimately require repeated SEARCH_MORE or REASON_MORE.
    Mere repetition is NOT sufficient for HIGH penalty — the outcome must be unchanged.
    """
    count = state.action_count(action)
    if count == 0:
        return NONE

    if count >= 2:
        # Check if outcomes were identical (true no-gain)
        if state.outcome_sequence_unchanged(action):
            return HIGH
        # Different outcomes — state may be changing, but still repeated
        return MEDIUM

    # count == 1 — tried once, might still help
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
