"""Shared controller protocol for V2B-I3 executive experiments.

Both the deterministic protocol fixture (`MatchedMetareasoningController`)
and the I3.4 pinned-model controller (`PinnedModelController`) implement this
protocol.  The loop and runner depend on this interface, never on a concrete
controller class, so the same execution pathway works under both controllers
without condition-specific branching.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .actions import ActionProposal
from .metareasoning_controller import ControllerObservation


@runtime_checkable
class ControllerProtocol(Protocol):
    """Minimal contract every V2B-I3 controller must satisfy.

    Implementations must be condition-agnostic: the same code path runs under
    every observation mask.  Condition identity is evaluator-only metadata and
    must never appear in controller-visible input or controller code.
    """

    controller_id: str
    algorithm_id: str

    def choose(self, observation: ControllerObservation) -> ActionProposal: ...
