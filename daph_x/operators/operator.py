"""Base v2 operator interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from daph_x.operators.types import RuntimeState, Observation
from daph_x.backends.llama_cpp_backend import CognitiveBackend


@dataclass(frozen=True)
class CostEstimate:
    tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    wall_ms: float = 0.0
    gpu_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "completion_tokens": self.completion_tokens,
            "model_calls": self.model_calls,
            "wall_ms": self.wall_ms,
            "gpu_ms": self.gpu_ms,
        }


@runtime_checkable
class CognitiveOperatorV2(Protocol):
    operator_id: str
    operator_version: str

    def is_admissible(self, state: RuntimeState) -> bool:
        ...

    def estimate_cost(self, state: RuntimeState) -> CostEstimate:
        ...

    def execute(self, state: RuntimeState, backend: CognitiveBackend) -> Observation:
        ...
