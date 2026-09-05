"""R13 core types: strictly separate runtime state from oracle/evaluation information.

This is the most important architectural boundary in R13.
RuntimeState contains only what the executive can observe before choosing an action.
EvaluationLabels contains ground truth and is never available to operators, the router,
or the authority gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple
import hashlib
import json


@dataclass(frozen=True)
class Candidate:
    """A single candidate answer (immutable)."""
    candidate_id: str
    answer: str
    reasoning_trace: str
    temperature: float
    seed: int
    generation_index: int
    metadata: Mapping[str, Any] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrajectoryPoint:
    """Pre-computed per-prefix observable statistics."""
    k: int
    top_answer: str
    p_top1: float
    p_top2: float
    margin: float
    entropy: float
    n_unique: int


@dataclass(frozen=True)
class RuntimeState:
    """Observable runtime state. Contains no oracle/ground-truth information.

    Critical invariant: RuntimeState ∩ oracle information = ∅.
    No field named 'correct_answer', 'is_correct', 'maxcal_correct', etc.
    """
    checkpoint_id: str
    task_id: str
    task_prompt: str
    answer_type: str
    category: str
    difficulty: str
    candidates: Tuple[Candidate, ...]
    trajectory: Tuple[TrajectoryPoint, ...]
    k: int
    current_answer: str
    observable_features: Mapping[str, float]
    state_hash: str

    def canonical_bytes(self) -> bytes:
        """Serialize to canonical bytes for hashing."""
        data = {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "answer_type": self.answer_type,
            "category": self.category,
            "difficulty": self.difficulty,
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "answer": c.answer,
                    "reasoning_trace": c.reasoning_trace,
                    "temperature": c.temperature,
                    "seed": c.seed,
                    "generation_index": c.generation_index,
                    "metadata": dict(c.metadata),
                }
                for c in self.candidates
            ],
            "trajectory": [
                {
                    "k": t.k,
                    "top_answer": t.top_answer,
                    "p_top1": t.p_top1,
                    "p_top2": t.p_top2,
                    "margin": t.margin,
                    "entropy": t.entropy,
                    "n_unique": t.n_unique,
                }
                for t in self.trajectory
            ],
            "k": self.k,
            "current_answer": self.current_answer,
            "observable_features": dict(self.observable_features),
        }
        return json.dumps(data, sort_keys=True, default=str).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class EvaluationLabels:
    """Ground-truth/evaluation labels. Never imported by operators, authority, value."""
    task_id: str
    correct_answer: str
    answer_type: str


@dataclass(frozen=True)
class Observation:
    """Result of executing a cognitive operator."""
    operator_id: str
    operator_version: str
    candidate_answer: str
    reasoning_trace: str
    confidence: float
    verification_score: float
    evidence: Mapping[str, Any]
    success: bool
    failure_reason: str
    cost: Mapping[str, Any]
    metadata: Mapping[str, Any]
