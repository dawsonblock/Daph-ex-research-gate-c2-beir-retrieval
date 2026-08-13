"""Bounded structured cognitive state exposed to the V2B-I2 controller."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .datalog import DatalogFact


class VerificationState(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    MISSING = "MISSING"
    UNVERIFIED = "UNVERIFIED"
    FALSIFIED = "FALSIFIED"
    STALE = "STALE"


class TemporalStatus(str, Enum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MemorySummary:
    memory_id: str
    relevance_score: float
    verification_state: VerificationState
    source_lineage_count: int
    evidence_count: int
    conflict_state: str
    temporal_status: TemporalStatus


@dataclass(frozen=True)
class VerificationSummary:
    target_id: str
    state: VerificationState
    evidence_count: int
    last_verified: str | None


@dataclass(frozen=True)
class ConflictSummary:
    conflict_id: str
    relation: str
    source_lineage_count: int
    status: str


@dataclass(frozen=True)
class DecisionSummary:
    decision_id: str
    selected_action: str
    reason_code: str
    outcome: str | None


@dataclass(frozen=True)
class CognitiveStateSnapshot:
    """A deliberately bounded executive view, never an unfiltered event dump."""

    task_id: str
    task_summary: str
    relevant_memories: tuple[MemorySummary, ...]
    verification_states: tuple[VerificationSummary, ...]
    provenance_summaries: tuple[str, ...]
    temporal_status: TemporalStatus
    unresolved_conflicts: tuple[ConflictSummary, ...]
    prior_decisions: tuple[DecisionSummary, ...]
    prior_outcomes: tuple[str, ...]
    resource_state: Mapping[str, int | float]
    policy_facts: tuple[DatalogFact, ...]
    observation_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_summary:
            raise ValueError("cognitive snapshots require task identity and a summary")
        for field in (self.relevant_memories, self.verification_states,
                      self.provenance_summaries, self.unresolved_conflicts,
                      self.prior_decisions, self.prior_outcomes, self.policy_facts,
                      self.observation_signals):
            if len(field) > 16:
                raise ValueError("cognitive snapshots expose at most 16 items per category")
