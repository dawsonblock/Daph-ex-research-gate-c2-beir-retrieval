"""R13 Cognitive Operator Framework.

Defines the standard interface for all cognitive actions in R13.
Every operator must implement this protocol so that the tournament,
router, and evaluation infrastructure can treat them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import time


@dataclass
class CostEstimate:
    """Pre-execution cost estimate for an operator."""
    tokens: int = 0
    latency_ms: float = 0.0
    model_calls: int = 0
    gpu_seconds: float = 0.0
    # Normalized composite cost (filled by cost accounting)
    normalized: float = 0.0


@dataclass
class CostRecord:
    """Actual recorded cost of an operator execution."""
    tokens: int = 0
    latency_ms: float = 0.0
    model_calls: int = 0
    gpu_seconds: float = 0.0
    normalized: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "model_calls": self.model_calls,
            "gpu_seconds": self.gpu_seconds,
            "normalized": self.normalized,
        }


@dataclass
class Observation:
    """Standardized observation returned by every operator.

    All operators must return this structure so that downstream
    state updates, evaluation, and attribution work uniformly.
    """
    # The candidate answer produced by this operator (if any)
    candidate_answer: str = ""
    # The reasoning trace produced (if any)
    reasoning_trace: str = ""
    # Self-reported confidence (0-100)
    confidence: float = 0.0
    # Verification score (0-1)
    verification_score: float = 0.0
    # Structured evidence from verification/critique
    evidence: dict = field(default_factory=dict)
    # Whether the operator produced a usable result
    success: bool = True
    # Failure reason if success=False
    failure_reason: str = ""
    # Operator metadata
    operator_name: str = ""
    # Actual cost
    cost: CostRecord = field(default_factory=CostRecord)
    # Raw model responses (for audit)
    raw_responses: list = field(default_factory=list)
    # Additional operator-specific data
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_answer": self.candidate_answer,
            "reasoning_trace": self.reasoning_trace[:500],  # Truncate for storage
            "confidence": self.confidence,
            "verification_score": self.verification_score,
            "evidence": self.evidence,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "operator_name": self.operator_name,
            "cost": self.cost.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class CheckpointState:
    """Immutable checkpoint state from which all operators start.

    Frozen from R12 data. Contains everything an operator needs
    to observe the current epistemic state and execute.
    """
    task_id: str
    task_prompt: str
    correct_answer: str
    answer_type: str
    difficulty: str
    category: str

    # Current candidates (up to checkpoint K)
    candidates: list = field(default_factory=list)  # list of dicts

    # Current checkpoint K
    k: int = 0

    # Pre-computed state features
    features: dict = field(default_factory=dict)

    # MaxCal pick at this checkpoint
    maxcal_answer: str = ""
    maxcal_correct: bool = False
    maxcal_confidence: float = 0.0

    # Previous state (for trajectory features)
    prev_state: Any = None

    def serialize(self) -> dict:
        """Serialize to a dict for immutability/audit."""
        return {
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "correct_answer": self.correct_answer,
            "answer_type": self.answer_type,
            "difficulty": self.difficulty,
            "category": self.category,
            "candidates": self.candidates,
            "k": self.k,
            "features": self.features,
            "maxcal_answer": self.maxcal_answer,
            "maxcal_correct": self.maxcal_correct,
            "maxcal_confidence": self.maxcal_confidence,
        }


@runtime_checkable
class CognitiveOperator(Protocol):
    """Standard interface for all cognitive operators in R13.

    Every operator must implement:
    - name: unique identifier
    - is_admissible: whether the operator can run from this state
    - estimate_cost: pre-execution cost estimate
    - execute: run the operator and return an Observation
    """

    name: str

    def is_admissible(self, state: CheckpointState) -> bool:
        """Check if this operator can be executed from the given state."""
        ...

    def estimate_cost(self, state: CheckpointState, budget: float = 1.0) -> CostEstimate:
        """Estimate the cost of executing this operator."""
        ...

    def execute(self, state: CheckpointState, budget: float = 1.0) -> Observation:
        """Execute the operator from the given state.

        Returns an Observation with the result and actual cost.
        """
        ...


def compute_normalized_cost(
    cost: CostRecord,
    w_tokens: float = 1.0,
    w_latency: float = 0.001,  # ms → seconds scale
    w_calls: float = 10.0,     # each call is expensive
    w_gpu: float = 1.0,
) -> float:
    """Compute normalized composite cost.

    Weights are chosen so that each component contributes roughly
    equally for typical operator executions. Raw metrics are preserved
    separately for independent reporting.
    """
    return (
        w_tokens * cost.tokens
        + w_latency * cost.latency_ms
        + w_calls * cost.model_calls
        + w_gpu * cost.gpu_seconds
    ) / 1000.0  # Scale to ~O(1) range
