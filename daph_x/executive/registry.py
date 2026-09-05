"""Operator registry — central catalog of all cognitive operators.

Manages operator profiles, admissibility filtering, and admission gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from daph_x.executive.budget import BudgetEnvelope
from daph_x.operators.external.base import (
    CognitiveOperator,
    CostVector,
    OperatorSpec,
    StateMode,
)
from daph_x.operators.types import RuntimeState


class OperatorStatus(str, Enum):
    """Lifecycle status of an operator in the registry."""
    EXPERIMENTAL = "EXPERIMENTAL"   # Not yet qualified
    ROUTABLE = "ROUTABLE"           # Passed admission gates; eligible for routing
    RETIRED = "RETIRED"             # No longer in active consideration
    QUARANTINED = "QUARANTINED"     # Suspended due to detected issue


@dataclass
class RegistryEntry:
    """A registered operator with its current status and metadata."""
    operator: CognitiveOperator
    status: OperatorStatus = OperatorStatus.EXPERIMENTAL
    admission_gates_passed: set[str] = field(default_factory=set)
    notes: str = ""

    @property
    def spec(self) -> OperatorSpec:
        return self.operator.spec

    @property
    def operator_id(self) -> str:
        return self.spec.operator_id

    @property
    def is_routable(self) -> bool:
        return self.status == OperatorStatus.ROUTABLE


class OperatorRegistry:
    """Central registry of cognitive operators.

    Operators must be registered before they can be used in tournaments
    or routing. Admission gates (G0-G4 per R14_PROTOCOL.md §7) must be
    passed before an operator becomes ROUTABLE.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def register(self, operator: CognitiveOperator, notes: str = "") -> RegistryEntry:
        """Register a new operator. Starts as EXPERIMENTAL."""
        op_id = operator.spec.operator_id
        if op_id in self._entries:
            raise ValueError(f"Operator already registered: {op_id}")
        entry = RegistryEntry(operator=operator, notes=notes)
        self._entries[op_id] = entry
        return entry

    def get(self, operator_id: str) -> RegistryEntry | None:
        return self._entries.get(operator_id)

    def get_operator(self, operator_id: str) -> CognitiveOperator | None:
        entry = self._entries.get(operator_id)
        return entry.operator if entry else None

    def all_entries(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def routable_entries(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.is_routable]

    def experimental_entries(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.status == OperatorStatus.EXPERIMENTAL]

    def retired_entries(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.status == OperatorStatus.RETIRED]

    def mark_routable(self, operator_id: str, gate: str = "") -> None:
        entry = self._entries.get(operator_id)
        if entry is None:
            raise KeyError(f"Unknown operator: {operator_id}")
        if gate:
            entry.admission_gates_passed.add(gate)
        entry.status = OperatorStatus.ROUTABLE

    def mark_retired(self, operator_id: str, reason: str = "") -> None:
        entry = self._entries.get(operator_id)
        if entry is None:
            raise KeyError(f"Unknown operator: {operator_id}")
        entry.status = OperatorStatus.RETIRED
        if reason:
            entry.notes = reason

    def mark_quarantined(self, operator_id: str, reason: str = "") -> None:
        entry = self._entries.get(operator_id)
        if entry is None:
            raise KeyError(f"Unknown operator: {operator_id}")
        entry.status = OperatorStatus.QUARANTINED
        if reason:
            entry.notes = reason

    def pass_gate(self, operator_id: str, gate: str) -> None:
        """Record that an operator has passed a specific admission gate."""
        entry = self._entries.get(operator_id)
        if entry is None:
            raise KeyError(f"Unknown operator: {operator_id}")
        entry.admission_gates_passed.add(gate)

    def admissible_operators(
        self,
        state: RuntimeState,
        capabilities: set[str] | Sequence[str] | None = None,
        budget: BudgetEnvelope | None = None,
        only_routable: bool = True,
    ) -> list[CognitiveOperator]:
        """Return all operators admissible given state, capabilities, and budget."""
        result = []
        for entry in self._entries.values():
            if only_routable and not entry.is_routable:
                continue
            op = entry.operator
            if op.is_admissible(state, capabilities=capabilities, budget=budget):
                result.append(op)
        return result

    def admissible_specs(
        self,
        state: RuntimeState,
        capabilities: set[str] | Sequence[str] | None = None,
        budget: BudgetEnvelope | None = None,
        only_routable: bool = True,
    ) -> list[OperatorSpec]:
        """Return specs of all admissible operators."""
        ops = self.admissible_operators(
            state, capabilities=capabilities, budget=budget, only_routable=only_routable
        )
        return [op.spec for op in ops]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, operator_id: str) -> bool:
        return operator_id in self._entries
