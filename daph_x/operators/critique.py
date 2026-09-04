"""CRITIQUE_RETRY operator — inspect current answer, identify error, generate correction."""
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


def extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def extract_critique(text: str) -> dict:
    """Extract structured critique from model response."""
    critique = {
        "suspected_error": "",
        "error_type": "",
        "correction": "",
    }
    # Try to parse structured fields
    error_match = re.search(r"(?:Suspected error|Error)[:\s]*(.+?)(?:\n\n|\nError type|\nCorrection|$)",
                            text, re.IGNORECASE | re.DOTALL)
    if error_match:
        critique["suspected_error"] = error_match.group(1).strip()[:200]

    type_match = re.search(r"Error type[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if type_match:
        critique["error_type"] = type_match.group(1).strip()[:100]

    corr_match = re.search(r"Correction[:\s]*(.+?)(?:\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if corr_match:
        critique["correction"] = corr_match.group(1).strip()[:200]

    return critique


class CritiqueRetryOperator:
    """CRITIQUE_RETRY: Inspect current answer, identify suspected error, retry.

    Two-phase operator:
    1. Critique phase: Show the model its current best answer and reasoning,
       ask it to identify potential errors.
    2. Retry phase: Ask the model to solve the problem again, avoiding the
       identified error.

    Produces structured diagnostic information about the suspected error.
    """

    name = "CRITIQUE_RETRY"
    n_model_calls = 2  # critique + retry

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
        # Need at least one candidate to critique
        return len(state.candidates) > 0 and state.k < 12

    def estimate_cost(self, state: CheckpointState, budget: float = 1.0) -> CostEstimate:
        # Critique call (~200 tokens) + retry call (~256 tokens)
        return CostEstimate(
            tokens=456,
            latency_ms=20000,
            model_calls=2,
            gpu_seconds=20.0,
        )

    def execute(self, state: CheckpointState, budget: float = 1.0) -> Observation:
        model = self._get_model()

        # Get the current best candidate (MaxCal pick)
        best_cand = max(state.candidates, key=lambda c: c.get("confidence", 50))
        current_answer = best_cand.get("answer", state.maxcal_answer)
        current_trace = best_cand.get("response", "")[:500]

        total_tokens = 0
        total_latency = 0.0
        raw_responses = []

        # Phase 1: Critique
        critique_prompt = (
            f"Problem: {state.task_prompt}\n\n"
            f"A student gave this answer: {current_answer}\n\n"
            f"Their reasoning was:\n{current_trace}\n\n"
            f"Identify any errors in this reasoning. If there are errors, describe:\n"
            f"- Suspected error: <what went wrong>\n"
            f"- Error type: <arithmetic, conceptual, logical, calculation, etc.>\n"
            f"- Correction: <how to fix it>\n\n"
            f"If the answer is correct, say 'No errors found.'"
        )

        t0 = time.monotonic()
        critique_response = model.generate_raw(
            prompt=critique_prompt, temperature=0.3,
            max_tokens=200, seed=42 + state.k * 311,
        )
        critique_latency = (time.monotonic() - t0) * 1000
        total_latency += critique_latency
        critique_tokens = max(1, len(critique_response) // 4)
        total_tokens += critique_tokens
        raw_responses.append(critique_response[:200])

        critique = extract_critique(critique_response)
        has_error = "no errors found" not in critique_response.lower()

        # Phase 2: Retry with critique awareness
        if has_error:
            retry_prompt = (
                f"Problem: {state.task_prompt}\n\n"
                f"A previous attempt gave the answer '{current_answer}' but contained this error:\n"
                f"{critique.get('suspected_error', 'Unknown error')}\n\n"
                f"Solve the problem again, carefully avoiding this error.\n"
                f"Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
            )
        else:
            # If no error found, just re-solve
            retry_prompt = (
                f"Problem: {state.task_prompt}\n\n"
                f"Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
            )

        t0 = time.monotonic()
        retry_response = model.generate_raw(
            prompt=retry_prompt, temperature=0.5,
            max_tokens=256, seed=42 + state.k * 311 + 1,
        )
        retry_latency = (time.monotonic() - t0) * 1000
        total_latency += retry_latency
        retry_tokens = max(1, len(retry_response) // 4)
        total_tokens += retry_tokens
        raw_responses.append(retry_response[:200])

        retry_answer = extract_answer(retry_response)
        retry_correct = check_answer(retry_answer, state.correct_answer, state.answer_type)

        cost = CostRecord(
            tokens=total_tokens,
            latency_ms=total_latency,
            model_calls=2,
            gpu_seconds=total_latency / 1000,
        )
        cost.normalized = compute_normalized_cost(cost)

        return Observation(
            candidate_answer=retry_answer,
            reasoning_trace=retry_response[:500],
            confidence=0.0,
            verification_score=0.0,
            evidence={
                "critique": critique,
                "has_error": has_error,
                "new_candidate": {
                    "answer": retry_answer,
                    "is_correct": retry_correct,
                    "response": retry_response,
                    "temperature": 0.5,
                },
            },
            success=True,
            operator_name=self.name,
            cost=cost,
            raw_responses=raw_responses,
            metadata={
                "k_before": state.k,
                "critique_found_error": has_error,
                "error_type": critique.get("error_type", ""),
            },
        )
