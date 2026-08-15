"""Action semantics: frozen semantic contracts for each executive action.

These describe WHAT an action does in terms of information channels,
not WHEN to use it. This is critical for topology generalization:
the model should reason from action consequences, not state→action rules.

Example failure pattern without this:
    UNVERIFIED → VERIFY  (rigid rule, fails on novel topologies)

With action semantics:
    VERIFY may change verification status, costs 1 verification unit,
    does not add new evidence, targets current blocker IF blocker is verification.
    If VERIFY already tried and didn't resolve → redundancy penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from hrm_adaptive_memory.executive.actions import DecisionAction


ACTION_SEMANTICS_SCHEMA = "DAPH_V2B_I3_5_ACTION_SEMANTICS_V1"
ACTION_SEMANTICS_VERSION = 1


@dataclass(frozen=True)
class ActionSemantics:
    """Frozen semantic contract for an executive action.

    Fields describe what the action CAN do, not what it WILL do.
    The governor uses these to predict outcomes and score candidates.
    """
    action: str
    cost_channels: tuple[str, ...]
    can_add_evidence: bool
    can_change_verification: bool
    can_reduce_conflict: bool
    can_change_reasoning_state: bool
    can_change_temporal_status: bool
    can_terminate: bool
    external_information: bool
    internal_compute: bool
    is_terminal: bool

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "cost_channels": list(self.cost_channels),
            "can_add_evidence": self.can_add_evidence,
            "can_change_verification": self.can_change_verification,
            "can_reduce_conflict": self.can_reduce_conflict,
            "can_change_reasoning_state": self.can_change_reasoning_state,
            "can_change_temporal_status": self.can_change_temporal_status,
            "can_terminate": self.can_terminate,
            "external_information": self.external_information,
            "internal_compute": self.internal_compute,
            "is_terminal": self.is_terminal,
        }


# Frozen semantic contracts for all executive actions
FROZEN_ACTION_SEMANTICS: dict[str, ActionSemantics] = {
    "ANSWER": ActionSemantics(
        action="ANSWER",
        cost_channels=("steps",),
        can_add_evidence=False,
        can_change_verification=False,
        can_reduce_conflict=False,
        can_change_reasoning_state=False,
        can_change_temporal_status=False,
        can_terminate=True,
        external_information=False,
        internal_compute=False,
        is_terminal=True,
    ),
    "RETRIEVE": ActionSemantics(
        action="RETRIEVE",
        cost_channels=("retrieval", "steps"),
        can_add_evidence=True,
        can_change_verification=True,
        can_reduce_conflict=False,
        can_change_reasoning_state=False,
        can_change_temporal_status=True,
        can_terminate=False,
        external_information=True,
        internal_compute=False,
        is_terminal=False,
    ),
    "VERIFY": ActionSemantics(
        action="VERIFY",
        cost_channels=("verification", "steps"),
        can_add_evidence=False,
        can_change_verification=True,
        can_reduce_conflict=True,
        can_change_reasoning_state=False,
        can_change_temporal_status=True,
        can_terminate=False,
        external_information=True,
        internal_compute=False,
        is_terminal=False,
    ),
    "VERIFY_ALTERNATE_SOURCE": ActionSemantics(
        action="VERIFY_ALTERNATE_SOURCE",
        cost_channels=("verification", "steps"),
        can_add_evidence=True,
        can_change_verification=True,
        can_reduce_conflict=True,
        can_change_reasoning_state=False,
        can_change_temporal_status=True,
        can_terminate=False,
        external_information=True,
        internal_compute=False,
        is_terminal=False,
    ),
    "SEARCH_MORE": ActionSemantics(
        action="SEARCH_MORE",
        cost_channels=("search", "steps"),
        can_add_evidence=True,
        can_change_verification=True,
        can_reduce_conflict=True,
        can_change_reasoning_state=False,
        can_change_temporal_status=True,
        can_terminate=False,
        external_information=True,
        internal_compute=False,
        is_terminal=False,
    ),
    "REASON_MORE": ActionSemantics(
        action="REASON_MORE",
        cost_channels=("reasoning", "steps"),
        can_add_evidence=False,
        can_change_verification=False,
        can_reduce_conflict=False,
        can_change_reasoning_state=True,
        can_change_temporal_status=False,
        can_terminate=False,
        external_information=False,
        internal_compute=True,
        is_terminal=False,
    ),
    "SPAWN_SPECIALIST": ActionSemantics(
        action="SPAWN_SPECIALIST",
        cost_channels=("specialist", "steps"),
        can_add_evidence=True,
        can_change_verification=True,
        can_reduce_conflict=True,
        can_change_reasoning_state=True,
        can_change_temporal_status=True,
        can_terminate=False,
        external_information=True,
        internal_compute=True,
        is_terminal=False,
    ),
    "SWITCH_STRATEGY": ActionSemantics(
        action="SWITCH_STRATEGY",
        cost_channels=("steps",),
        can_add_evidence=False,
        can_change_verification=False,
        can_reduce_conflict=False,
        can_change_reasoning_state=True,
        can_change_temporal_status=False,
        can_terminate=False,
        external_information=False,
        internal_compute=True,
        is_terminal=False,
    ),
    "ABANDON_STRATEGY": ActionSemantics(
        action="ABANDON_STRATEGY",
        cost_channels=("steps",),
        can_add_evidence=False,
        can_change_verification=False,
        can_reduce_conflict=False,
        can_change_reasoning_state=True,
        can_change_temporal_status=False,
        can_terminate=False,
        external_information=False,
        internal_compute=True,
        is_terminal=False,
    ),
    "DEFER": ActionSemantics(
        action="DEFER",
        cost_channels=("steps",),
        can_add_evidence=False,
        can_change_verification=False,
        can_reduce_conflict=False,
        can_change_reasoning_state=False,
        can_change_temporal_status=False,
        can_terminate=True,
        external_information=False,
        internal_compute=False,
        is_terminal=True,
    ),
    "STOP": ActionSemantics(
        action="STOP",
        cost_channels=("steps",),
        can_add_evidence=False,
        can_change_verification=False,
        can_reduce_conflict=False,
        can_change_reasoning_state=False,
        can_change_temporal_status=False,
        can_terminate=True,
        external_information=False,
        internal_compute=False,
        is_terminal=True,
    ),
}


def get_action_semantics(action: str) -> ActionSemantics:
    """Get the frozen semantics for an action."""
    if action not in FROZEN_ACTION_SEMANTICS:
        raise ValueError(f"Unknown action: {action}")
    return FROZEN_ACTION_SEMANTICS[action]


def all_action_semantics() -> dict[str, ActionSemantics]:
    """Get all frozen action semantics."""
    return dict(FROZEN_ACTION_SEMANTICS)
