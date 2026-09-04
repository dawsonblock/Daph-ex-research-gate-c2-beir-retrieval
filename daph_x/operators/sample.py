"""SAMPLE_STANDARD operator — generate additional candidates using R12 procedure."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.operators.base import (
    CognitiveOperator, CheckpointState, CostEstimate, CostRecord, Observation,
    compute_normalized_cost,
)
from daph_x.coding.reasoning_tasks import check_answer

# Temperature schedule for additional candidates (continues from R12 schedule)
EXTENDED_TEMP_SCHEDULE = [1.0, 1.2, 1.0, 1.2, 0.8, 0.9, 1.0, 1.2]


def extract_answer(text: str) -> str:
    """Extract answer from 'Answer: <answer>' format."""
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


class SampleStandardOperator:
    """SAMPLE_STANDARD: Generate 2 additional candidates using the R12 procedure.

    Uses the same temperature schedule and prompt as R12.
    This is the R12 GENERATE(+2) action, wrapped as an operator.
    """

    name = "SAMPLE_STANDARD"
    n_new_candidates = 2

    def __init__(self, model=None):
        self._model = model

    def _get_model(self):
        if self._model is None:
            from daph_x.coding.model_interface import CodingModelInterface
            self._model = CodingModelInterface(
                model_path="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                n_gpu_layers=-1, seed=42,
            )
        return self._model

    def is_admissible(self, state: CheckpointState) -> bool:
        return state.k < 12  # Don't exceed R12 max

    def estimate_cost(self, state: CheckpointState, budget: float = 1.0) -> CostEstimate:
        # 2 generation calls, ~256 tokens each
        return CostEstimate(
            tokens=512,
            latency_ms=20000,  # ~10s per call
            model_calls=2,
            gpu_seconds=20.0,
        )

    def execute(self, state: CheckpointState, budget: float = 1.0) -> Observation:
        model = self._get_model()
        prompt = f"{state.task_prompt}\n\nThink step by step. At the very end, write 'Answer: <your answer>' on a new line."

        new_candidates = []
        total_tokens = 0
        total_latency = 0.0
        raw_responses = []

        for i in range(self.n_new_candidates):
            temp = EXTENDED_TEMP_SCHEDULE[i % len(EXTENDED_TEMP_SCHEDULE)]
            seed = 42 + state.k * 137 + i * 31

            t0 = time.monotonic()
            response = model.generate_raw(
                prompt=prompt, temperature=temp,
                max_tokens=256, seed=seed,
            )
            latency = (time.monotonic() - t0) * 1000
            total_latency += latency

            answer = extract_answer(response)
            is_correct = check_answer(answer, state.correct_answer, state.answer_type)
            tokens = max(1, len(response) // 4)
            total_tokens += tokens

            new_candidates.append({
                "answer": answer,
                "is_correct": is_correct,
                "response": response,
                "temperature": temp,
                "seed": seed,
            })
            raw_responses.append(response[:200])

        cost = CostRecord(
            tokens=total_tokens,
            latency_ms=total_latency,
            model_calls=self.n_new_candidates,
            gpu_seconds=total_latency / 1000,
        )
        cost.normalized = compute_normalized_cost(cost)

        return Observation(
            candidate_answer=new_candidates[0]["answer"] if new_candidates else "",
            reasoning_trace=new_candidates[0]["response"][:500] if new_candidates else "",
            confidence=0.0,
            verification_score=0.0,
            evidence={"new_candidates": new_candidates},
            success=True,
            operator_name=self.name,
            cost=cost,
            raw_responses=raw_responses,
            metadata={"n_new": self.n_new_candidates, "k_before": state.k},
        )
