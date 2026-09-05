"""STOP v2 — return current selected answer with zero cost."""
from __future__ import annotations

from daph_x.operators.operator import CognitiveOperatorV2, CostEstimate
from daph_x.operators.types import RuntimeState, Observation
from daph_x.backends.llama_cpp_backend import CognitiveBackend


class StopV2:
    operator_id = "STOP"
    operator_version = "2"

    def is_admissible(self, state: RuntimeState) -> bool:
        return True

    def estimate_cost(self, state: RuntimeState) -> CostEstimate:
        return CostEstimate()

    def execute(self, state: RuntimeState, backend: CognitiveBackend) -> Observation:
        return Observation(
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            candidate_answer=state.current_answer,
            reasoning_trace="",
            confidence=0.0,
            verification_score=0.0,
            evidence={},
            success=True,
            failure_reason="",
            cost=CostEstimate().to_dict(),
            metadata={"k": state.k},
        )
