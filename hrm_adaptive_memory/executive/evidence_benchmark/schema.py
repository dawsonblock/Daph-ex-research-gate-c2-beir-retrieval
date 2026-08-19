"""Evidence-bearing task and runtime schema for I3.7.

Core types:
  - EvidenceItem: proposition-level evidence with hypothesis relationships
  - EvidenceTask: benchmark task with hidden evidence items and resolution paths
  - EvidenceRuntime: runtime tracking available/retrieved/verified evidence
  - EvidenceSnapshot: cognitive state with proposition-level evidence

All evidence is controller-visible once retrieved. Hidden evidence is
not in the snapshot until RETRIEVE or SEARCH_MORE exposes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)


EVIDENCE_SCHEMA = "DAPH_V2B_I3_7_EVIDENCE_BENCHMARK_V1"
EVIDENCE_VERSION = 1


@dataclass(frozen=True)
class EvidenceItem:
    """A single proposition-level evidence item.

    Attributes:
        evidence_id: unique identifier (e.g. "E1")
        proposition: the claim this evidence makes
        source_class: category of source (e.g. "primary", "secondary", "search")
        supports: hypothesis IDs this evidence supports
        contradicts: hypothesis IDs this evidence contradicts
        verification_state: UNVERIFIED, SUFFICIENT, FALSIFIED, STALE, MISSING
        temporal_status: CURRENT, STALE, UNKNOWN
        retrieved: whether this item has been exposed to the controller
        verify_result: what VERIFY produces (SUFFICIENT or FALSIFIED), or None
    """
    evidence_id: str
    proposition: str
    source_class: str
    supports: tuple[str, ...]
    contradicts: tuple[str, ...]
    verification_state: VerificationState
    temporal_status: TemporalStatus
    retrieved: bool = False
    verify_result: str | None = None  # "SUFFICIENT" or "FALSIFIED" or None

    def as_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "proposition": self.proposition,
            "source_class": self.source_class,
            "supports": list(self.supports),
            "contradicts": list(self.contradicts),
            "verification_state": self.verification_state.value,
            "temporal_status": self.temporal_status.value,
            "retrieved": self.retrieved,
            "verify_result": self.verify_result,
        }


@dataclass(frozen=True)
class EvidenceHypothesis:
    """A hypothesis with its expected answer action."""
    hypothesis_id: str
    proposition: str
    answer_action: DecisionAction
    answer_payload: str

    def as_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "proposition": self.proposition,
            "answer_action": self.answer_action.value,
            "answer_payload": self.answer_payload,
        }


@dataclass(frozen=True)
class EvidenceRelation:
    """How evidence relates to hypotheses — for snapshot construction."""
    evidence_id: str
    hypothesis_id: str
    relation: str  # "supports" or "contradicts"


@dataclass(frozen=True)
class EvidenceTask:
    """A benchmark task with proposition-level evidence.

    The task defines:
      - competing hypotheses (H1, H2, ...)
      - hidden evidence items that can be retrieved/searched/verified
      - which evidence operations expose which items
      - the oracle resolution path (evaluation-side only, never controller-visible)
      - the expected terminal action
    """
    task_id: str
    split: str
    category: str
    task_summary: str
    high_stakes: bool
    budget_profile: str
    hypotheses: tuple[EvidenceHypothesis, ...]
    evidence_items: tuple[EvidenceItem, ...]
    # Which evidence IDs are exposed by each action
    retrieve_exposes: tuple[str, ...]      # evidence IDs exposed by RETRIEVE
    search_exposes: tuple[str, ...]        # evidence IDs exposed by SEARCH_MORE
    # Oracle resolution path (evaluation-side only)
    oracle_resolution_path: tuple[str, ...]  # e.g. ("RETRIEVE:E1", "VERIFY:E1", "ANSWER")
    expected_terminal: DecisionAction
    # The hypothesis that is correct (evaluation-side only)
    correct_hypothesis_id: str

    def __post_init__(self) -> None:
        if not self.task_id or self.task_id != self.task_id.lower():
            raise ValueError("evidence tasks require lowercase ids")
        if not self.hypotheses or len(self.hypotheses) < 2:
            raise ValueError("evidence tasks require at least 2 hypotheses")
        if not self.evidence_items:
            raise ValueError("evidence tasks require at least 1 evidence item")
        if self.expected_terminal not in {DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}:
            raise ValueError("evidence tasks require a terminal action")
        # Verify all evidence IDs referenced in supports/contradicts map to hypotheses
        h_ids = {h.hypothesis_id for h in self.hypotheses}
        for ev in self.evidence_items:
            for h_id in ev.supports + ev.contradicts:
                if h_id not in h_ids:
                    raise ValueError(
                        f"evidence {ev.evidence_id} references unknown hypothesis {h_id}")

    @property
    def initial_evidence(self) -> tuple[EvidenceItem, ...]:
        """Evidence items that are visible at the start (retrieved=True)."""
        return tuple(e for e in self.evidence_items if e.retrieved)

    @property
    def hidden_evidence(self) -> tuple[EvidenceItem, ...]:
        """Evidence items that must be retrieved/searched to become visible."""
        return tuple(e for e in self.evidence_items if not e.retrieved)

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "category": self.category,
            "task_summary": self.task_summary,
            "high_stakes": self.high_stakes,
            "budget_profile": self.budget_profile,
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "evidence_items": [e.as_dict() for e in self.evidence_items],
            "retrieve_exposes": list(self.retrieve_exposes),
            "search_exposes": list(self.search_exposes),
            "oracle_resolution_path": list(self.oracle_resolution_path),
            "expected_terminal": self.expected_terminal.value,
            "correct_hypothesis_id": self.correct_hypothesis_id,
        }


@dataclass(frozen=True)
class EvidenceRuntime:
    """Runtime state for evidence-bearing tasks.

    Tracks which evidence items have been retrieved, their verification
    states, and the overall task state.
    """
    task: EvidenceTask
    resources: 'ResourceState'
    # Evidence items with current states (may differ from initial after VERIFY)
    evidence: tuple[EvidenceItem, ...]
    retrieved_evidence_ids: tuple[str, ...]
    verified_evidence_ids: tuple[str, ...]
    searched: bool = False
    reasoning_complete: bool = False

    @property
    def visible_evidence(self) -> tuple[EvidenceItem, ...]:
        """Evidence items currently visible to the controller."""
        return tuple(e for e in self.evidence if e.retrieved)

    @property
    def hidden_evidence(self) -> tuple[EvidenceItem, ...]:
        """Evidence items not yet exposed to the controller."""
        return tuple(e for e in self.evidence if not e.retrieved)

    @property
    def verified_evidence(self) -> tuple[EvidenceItem, ...]:
        """Evidence items that have been verified."""
        return tuple(e for e in self.evidence
                     if e.verification_state in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED))

    @property
    def supporting_evidence(self) -> tuple[EvidenceItem, ...]:
        """Verified evidence that supports some hypothesis."""
        return tuple(e for e in self.verified_evidence
                     if e.verification_state is VerificationState.SUFFICIENT and e.supports)

    @property
    def contradicting_evidence(self) -> tuple[EvidenceItem, ...]:
        """Verified evidence that contradicts some hypothesis."""
        return tuple(e for e in self.verified_evidence
                     if e.verification_state is VerificationState.FALSIFIED and e.supports)


@dataclass(frozen=True)
class EvidenceActionExecution:
    """Result of executing an action on an evidence-bearing task."""
    action: DecisionAction
    runtime: EvidenceRuntime
    terminal: bool
    task_success: bool | None
    outcome_code: str
    evidence_exposed: tuple[str, ...]  # evidence IDs newly exposed
    evidence_verified: tuple[str, ...]  # evidence IDs newly verified


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Controller-visible cognitive state with proposition-level evidence.

    This extends the aggregate CognitiveStateSnapshot with actual
    proposition-level evidence items, their hypothesis relationships,
    and verification states.
    """
    task_id: str
    task_summary: str
    visible_evidence: tuple[EvidenceItem, ...]
    hidden_evidence_count: int
    hypotheses: tuple[EvidenceHypothesis, ...]
    verified_count: int
    supporting_count: int
    contradicting_count: int
    searched: bool
    reasoning_complete: bool
    resource_state: Mapping[str, int | float]
    prior_actions: tuple[str, ...]
    prior_outcomes: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_summary": self.task_summary,
            "visible_evidence": [e.as_dict() for e in self.visible_evidence],
            "hidden_evidence_count": self.hidden_evidence_count,
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "verified_count": self.verified_count,
            "supporting_count": self.supporting_count,
            "contradicting_count": self.contradicting_count,
            "searched": self.searched,
            "reasoning_complete": self.reasoning_complete,
            "resource_state": dict(self.resource_state),
            "prior_actions": list(self.prior_actions),
            "prior_outcomes": list(self.prior_outcomes),
        }


def initial_evidence_runtime(
    task: EvidenceTask,
    resources: 'ResourceState',
) -> EvidenceRuntime:
    """Create the initial runtime for an evidence-bearing task."""
    return EvidenceRuntime(
        task=task,
        resources=resources,
        evidence=task.evidence_items,
        retrieved_evidence_ids=tuple(e.evidence_id for e in task.evidence_items if e.retrieved),
        verified_evidence_ids=(),
    )
