"""Schema-constrained action proposals for the first V2B executive experiment."""
from __future__ import annotations

from dataclasses import dataclass

from hrm_adaptive_memory.cognitive_control.actions import validate_v2b_action
from hrm_adaptive_memory.cognitive_control.core import DecisionAction


@dataclass(frozen=True)
class ActionProposal:
    action: DecisionAction
    reason_code: str
    target_id: str | None = None

    def __post_init__(self) -> None:
        validate_v2b_action(self.action)
        if (not self.reason_code or self.reason_code != self.reason_code.upper()
                or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                       for character in self.reason_code)):
            raise ValueError("action proposals require an uppercase structured reason_code")
