"""Search budget tracker."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from daph.search.types import SearchConfig


@dataclass
class SearchBudget:
    """Tracks resource consumption during search.

    Hard limits enforced:
    - max_nodes: total nodes expanded
    - max_model_calls: total LLM calls (for rollout policy)
    - max_wall_ms: wall time
    """

    config: SearchConfig
    nodes_expanded: int = 0
    model_calls: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def wall_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    @property
    def exhausted(self) -> bool:
        return (
            self.nodes_expanded >= self.config.max_nodes
            or self.model_calls >= self.config.max_model_calls
            or self.wall_ms >= self.config.max_wall_ms
        )

    def can_expand(self) -> bool:
        return not self.exhausted

    def consume_node(self) -> None:
        self.nodes_expanded += 1

    def consume_model_call(self) -> None:
        self.model_calls += 1

    def as_dict(self) -> dict:
        return {
            "nodes_expanded": self.nodes_expanded,
            "model_calls": self.model_calls,
            "wall_ms": round(self.wall_ms, 2),
            "max_nodes": self.config.max_nodes,
            "max_model_calls": self.config.max_model_calls,
            "max_wall_ms": self.config.max_wall_ms,
            "exhausted": self.exhausted,
        }
