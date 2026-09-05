"""Budget envelope and resource constraint definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BudgetEnvelope:
    """Resource budget envelope bounding operator execution."""
    max_tokens: int | None = None
    max_calls: int | None = None
    max_wall_ms: float | None = None
    max_gpu_ms: float | None = None
    max_cost_usd: float | None = None
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical

    def is_exceeded_by(
        self,
        tokens: int | None = None,
        calls: int | None = None,
        wall_ms: float | None = None,
        gpu_ms: float | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        if self.max_tokens is not None and tokens is not None and tokens > self.max_tokens:
            return True
        if self.max_calls is not None and calls is not None and calls > self.max_calls:
            return True
        if self.max_wall_ms is not None and wall_ms is not None and wall_ms > self.max_wall_ms:
            return True
        if self.max_gpu_ms is not None and gpu_ms is not None and gpu_ms > self.max_gpu_ms:
            return True
        if self.max_cost_usd is not None and cost_usd is not None and cost_usd > self.max_cost_usd:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_calls": self.max_calls,
            "max_wall_ms": self.max_wall_ms,
            "max_gpu_ms": self.max_gpu_ms,
            "max_cost_usd": self.max_cost_usd,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BudgetEnvelope:
        return cls(
            max_tokens=data.get("max_tokens"),
            max_calls=data.get("max_calls"),
            max_wall_ms=data.get("max_wall_ms"),
            max_gpu_ms=data.get("max_gpu_ms"),
            max_cost_usd=data.get("max_cost_usd"),
            priority=int(data.get("priority", 0)),
        )
