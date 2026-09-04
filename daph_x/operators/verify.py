"""VERIFY_TARGETED operator — construct targeted checks against current hypothesis."""
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


def extract_verification(text: str) -> dict:
    """Extract structured verification from model response."""
    result = {
        "checks_passed": 0,
        "checks_failed": 0,
        "evidence_for": "",
        "evidence_against": "",
        "verdict": "",
    }

    # Count pass/fail
    passes = re.findall(r"(?:PASS|passed|correct|valid|confirmed)", text, re.IGNORECASE)
    fails = re.findall(r"(?:FAIL|failed|incorrect|invalid|contradicted)", text, re.IGNORECASE)
    result["checks_passed"] = len(passes)
    result["checks_failed"] = len(fails)

    # Extract verdict
    verdict_match = re.search(r"Verdict[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).strip()[:100]

    return result


class VerifyTargetedOperator:
    """VERIFY_TARGETED: Construct targeted checks against current hypothesis.

    Two-phase operator:
    1. Verification phase: Ask the model to verify the current leading answer
       by constructing specific checks (substitution, boundary cases, etc.).
    2. Correction phase: If verification fails, generate a corrected answer.

    Produces structured evidence (checks passed/failed, verdict).
    """

    name = "VERIFY_TARGETED"
    n_model_calls = 2  # verify + optional correction

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
        return len(state.candidates) > 0 and state.k < 12

    def estimate_cost(self, state: CheckpointState, budget: float = 1.0) -> CostEstimate:
        return CostEstimate(
            tokens=456,
            latency_ms=20000,
            model_calls=2,
            gpu_seconds=20.0,
        )

    def execute(self, state: CheckpointState, budget: float = 1.0) -> Observation:
        model = self._get_model()

        # Get the current leading answer
        current_answer = state.maxcal_answer
        best_cand = max(state.candidates, key=lambda c: c.get("confidence", 50))
        current_trace = best_cand.get("response", "")[:400]

        total_tokens = 0
        total_latency = 0.0
        raw_responses = []

        # Phase 1: Targeted verification
        verify_prompt = (
            f"Problem: {state.task_prompt}\n\n"
            f"Proposed answer: {current_answer}\n\n"
            f"Reasoning:\n{current_trace}\n\n"
            f"Verify this answer by:\n"
            f"1. Substituting the answer back into the original problem\n"
            f"2. Checking the calculation step by step\n"
            f"3. Testing a simple case if applicable\n\n"
            f"Report:\n"
            f"- Check 1: PASS/FAIL — <explanation>\n"
            f"- Check 2: PASS/FAIL — <explanation>\n"
            f"- Verdict: CORRECT or INCORRECT"
        )

        t0 = time.monotonic()
        verify_response = model.generate_raw(
            prompt=verify_prompt, temperature=0.2,
            max_tokens=200, seed=42 + state.k * 411,
        )
        verify_latency = (time.monotonic() - t0) * 1000
        total_latency += verify_latency
        verify_tokens = max(1, len(verify_response) // 4)
        total_tokens += verify_tokens
        raw_responses.append(verify_response[:200])

        verification = extract_verification(verify_response)
        is_verified = "CORRECT" in verification.get("verdict", "").upper() or \
                      verification["checks_passed"] > verification["checks_failed"]

        # Phase 2: If verification fails, generate correction
        if not is_verified:
            correction_prompt = (
                f"Problem: {state.task_prompt}\n\n"
                f"The answer '{current_answer}' failed verification:\n"
                f"{verify_response[:300]}\n\n"
                f"Solve the problem correctly.\n"
                f"Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
            )

            t0 = time.monotonic()
            correction_response = model.generate_raw(
                prompt=correction_prompt, temperature=0.4,
                max_tokens=256, seed=42 + state.k * 411 + 1,
            )
            correction_latency = (time.monotonic() - t0) * 1000
            total_latency += correction_latency
            correction_tokens = max(1, len(correction_response) // 4)
            total_tokens += correction_tokens
            raw_responses.append(correction_response[:200])

            correction_answer = extract_answer(correction_response)
            correction_correct = check_answer(correction_answer, state.correct_answer, state.answer_type)

            new_candidate = {
                "answer": correction_answer,
                "is_correct": correction_correct,
                "response": correction_response,
                "temperature": 0.4,
            }
            final_answer = correction_answer
        else:
            # Verification passed — keep current answer
            new_candidate = None
            final_answer = current_answer

        cost = CostRecord(
            tokens=total_tokens,
            latency_ms=total_latency,
            model_calls=2 if not is_verified else 1,
            gpu_seconds=total_latency / 1000,
        )
        cost.normalized = compute_normalized_cost(cost)

        return Observation(
            candidate_answer=final_answer,
            reasoning_trace=verify_response[:500],
            confidence=0.0,
            verification_score=1.0 if is_verified else 0.0,
            evidence={
                "verification": verification,
                "verified": is_verified,
                "new_candidate": new_candidate,
            },
            success=True,
            operator_name=self.name,
            cost=cost,
            raw_responses=raw_responses,
            metadata={
                "k_before": state.k,
                "verification_passed": is_verified,
                "checks_passed": verification["checks_passed"],
                "checks_failed": verification["checks_failed"],
            },
        )
