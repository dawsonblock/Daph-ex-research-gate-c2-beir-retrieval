"""STOP operator — return current MaxCal answer with zero cost."""
from __future__ import annotations

from daph_x.operators.base import (
    CognitiveOperator, CheckpointState, CostEstimate, CostRecord, Observation,
)


class StopOperator:
    """STOP: Return the current MaxCal answer without additional reasoning.

    This is the zero-cost baseline action. It always succeeds and
    produces no new candidates.
    """

    name = "STOP"

    def is_admissible(self, state: CheckpointState) -> bool:
        return True  # STOP is always admissible

    def estimate_cost(self, state: CheckpointState, budget: float = 1.0) -> CostEstimate:
        return CostEstimate(tokens=0, latency_ms=0.0, model_calls=0, gpu_seconds=0.0)

    def execute(self, state: CheckpointState, budget: float = 1.0) -> Observation:
        return Observation(
            candidate_answer=state.maxcal_answer,
            reasoning_trace="",
            confidence=state.maxcal_confidence,
            verification_score=0.0,
            evidence={},
            success=True,
            operator_name=self.name,
            cost=CostRecord(tokens=0, latency_ms=0.0, model_calls=0, gpu_seconds=0.0),
            metadata={"stop": True, "k": state.k},
        )
