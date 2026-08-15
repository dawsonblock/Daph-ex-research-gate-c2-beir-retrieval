"""Option value: future action-set preservation.

A good governor recognizes irreversible or resource-consuming decisions.
ANSWER/DEFER/STOP terminate → future options = 0.
VERIFY/RETRIEVE/SEARCH_MORE keep most options available.

O(a) = future action-set preservation
"""
from __future__ import annotations

from hrm_adaptive_memory.executive.governor.action_semantics import get_action_semantics
from hrm_adaptive_memory.executive.governor.state import GovernorState
from hrm_adaptive_memory.executive.governor.bottlenecks import NONE, LOW, MEDIUM, HIGH


OPTION_VALUE_SCHEMA = "DAPH_V2B_I3_5_OPTION_VALUE_V1"
OPTION_VALUE_VERSION = 1


def estimate_option_value(state: GovernorState, action: str) -> str:
    """Estimate how well an action preserves future options.

    Returns: NONE, LOW, MEDIUM, or HIGH

    Terminal actions (ANSWER, DEFER, STOP) → NONE
    Actions that consume scarce resources → reduced
    Actions that don't terminate → HIGH (default)
    """
    semantics = get_action_semantics(action)

    if semantics.is_terminal:
        return NONE

    # Check if the action consumes the last remaining resource of its type
    resources = state.resource_state
    cost_channels = semantics.cost_channels

    # If any cost channel has only 1 remaining, option value drops
    scarce = False
    for channel in cost_channels:
        if channel == "steps":
            continue
        remaining = resources.get(channel, 0)
        if remaining <= 1:
            scarce = True
            break

    if scarce:
        return MEDIUM

    # If remaining steps are low, option value drops
    if state.remaining_steps <= 3:
        return MEDIUM

    return HIGH


def preserves_future_options(action: str) -> bool:
    """Whether an action preserves future options (non-terminal)."""
    semantics = get_action_semantics(action)
    return not semantics.is_terminal
