"""SAMPLE_DIVERSE operator — generate candidates with deliberately different strategies."""
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

# Diversity strategy prompts — each forces a different reasoning approach
DIVERSITY_STRATEGIES = [
    "Solve this problem by working backwards from the desired answer.",
    "Solve this problem using a formal step-by-step derivation. Show each algebraic step explicitly.",
    "Solve this problem using a different approach than you would normally use. Try substitution or a concrete example first.",
    "Solve this problem by first identifying what type of problem this is, then applying the standard method for that type.",
]


def extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


class SampleDiverseOperator:
    """SAMPLE_DIVERSE: Generate 2 candidates using controlled strategy prompts.

    Diversity comes from deliberately different reasoning framings,
    not random temperature perturbation. Each candidate uses a
    different strategy prompt to encourage genuinely different
    reasoning paths.
    """

    name = "SAMPLE_DIVERSE"
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
        return state.k < 12

    def estimate_cost(self, state: CheckpointState, budget: float = 1.0) -> CostEstimate:
        return CostEstimate(
            tokens=512,
            latency_ms=20000,
            model_calls=2,
            gpu_seconds=20.0,
        )

    def execute(self, state: CheckpointState, budget: float = 1.0) -> Observation:
        model = self._get_model()

        new_candidates = []
        total_tokens = 0
        total_latency = 0.0
        raw_responses = []

        for i in range(self.n_new_candidates):
            strategy = DIVERSITY_STRATEGIES[i % len(DIVERSITY_STRATEGIES)]
            prompt = (
                f"{strategy}\n\n"
                f"Problem: {state.task_prompt}\n\n"
                f"Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
            )

            # Use moderate temperature for diversity
            temp = 0.7 + i * 0.1
            seed = 42 + state.k * 211 + i * 53

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
                "strategy": strategy[:50],
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
            metadata={
                "n_new": self.n_new_candidates,
                "k_before": state.k,
                "strategies_used": [DIVERSITY_STRATEGIES[i % len(DIVERSITY_STRATEGIES)][:50]
                                    for i in range(self.n_new_candidates)],
            },
        )
