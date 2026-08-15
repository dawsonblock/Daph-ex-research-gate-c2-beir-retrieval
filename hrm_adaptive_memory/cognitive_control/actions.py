"""The frozen, constrained action vocabulary for the first V2B experiment."""
from __future__ import annotations

from typing import Iterable

from .core import DecisionAction


V2B_ACTION_SCHEMA = "DAPH_V2B_EXECUTIVE_ACTIONS_V1"
V2B_ACTIONS = (
    DecisionAction.ANSWER,
    DecisionAction.RETRIEVE,
    DecisionAction.VERIFY,
    DecisionAction.SEARCH_MORE,
    DecisionAction.REASON_MORE,
    DecisionAction.DEFER,
    DecisionAction.STOP,
)


def validate_v2b_action(value: DecisionAction | str) -> DecisionAction:
    try:
        action = value if isinstance(value, DecisionAction) else DecisionAction(value)
    except ValueError as error:
        raise ValueError("unknown V2B executive action") from error
    if action not in V2B_ACTIONS:
        raise ValueError(f"action is outside the frozen V2B action space: {action.value}")
    return action


def validate_v2b_actions(values: Iterable[DecisionAction | str]) -> tuple[DecisionAction, ...]:
    return tuple(validate_v2b_action(value) for value in values)
