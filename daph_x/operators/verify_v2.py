"""VERIFY_TARGETED v2 — strict enum verdicts, task-family rubrics, conditional correction."""
from __future__ import annotations

import re
from daph_x.operators.operator import CognitiveOperatorV2, CostEstimate
from daph_x.operators.types import RuntimeState, Observation
from daph_x.operators.prompts import (
    VERIFY_PROMPT_ID, VERIFY_PROMPT_TEMPLATE,
    VERIFY_CORRECT_PROMPT_ID, VERIFY_CORRECT_PROMPT_TEMPLATE,
    VERIFY_INCORRECT_PROMPT_ID, VERIFY_INCORRECT_PROMPT_TEMPLATE,
)
from daph_x.operators.verdicts import parse_verification_verdict, VerificationVerdict
from daph_x.backends.llama_cpp_backend import CognitiveBackend, GenerationRequest


def _extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


class VerifyTargetedV2:
    operator_id = "VERIFY_TARGETED"
    operator_version = "2"

    def is_admissible(self, state: RuntimeState) -> bool:
        return len(state.candidates) > 0 and state.k + 1 <= 12

    def estimate_cost(self, state: RuntimeState) -> CostEstimate:
        return CostEstimate(
            tokens=len(state.task_prompt) // 4 + 2 * 100,
            completion_tokens=2 * 256,
            model_calls=2,
            wall_ms=2 * 10000,
        )

    def _get_leader_trace(self, state: RuntimeState) -> tuple:
        for c in state.candidates:
            if c.answer == state.current_answer:
                return c.answer, c.reasoning_trace
        c = state.candidates[0]
        return c.answer, c.reasoning_trace

    def _derive_seed(self, state: RuntimeState, step: int, replicate_id: int) -> int:
        s = f"{state.state_hash}|{self.operator_id}|{self.operator_version}|{replicate_id}|{step}"
        return int(__import__("hashlib").sha256(s.encode()).hexdigest()[:8], 16)

    def execute(self, state: RuntimeState, backend: CognitiveBackend, replicate_id: int = 42) -> Observation:
        current_answer, current_trace = self._get_leader_trace(state)

        # Phase 1: structured verification
        verify_prompt = VERIFY_PROMPT_TEMPLATE.format(
            task_prompt=state.task_prompt,
            current_answer=current_answer,
            current_trace=current_trace[:400],
        )

        req = GenerationRequest(
            prompt=verify_prompt,
            temperature=0.2,
            max_tokens=300,
            seed=self._derive_seed(state, 0, replicate_id),
        )
        verify_result = backend.generate(req)

        verify_raw = verify_result.text
        verify_data = {}
        try:
            import json
            match = re.search(r"\{[\s\S]*\}", verify_raw)
            if match:
                verify_data = json.loads(match.group(0))
        except Exception:
            pass

        verdict = parse_verification_verdict(verify_raw)
        checks = verify_data.get("checks", [])
        confidence = float(verify_data.get("confidence", 0.0) or 0.0)
        correction_needed = verify_data.get("correction_needed", verdict == VerificationVerdict.INCORRECT)
        correction_suggestion = str(verify_data.get("correction_suggestion", ""))

        total_tokens = verify_result.prompt_tokens + verify_result.completion_tokens
        total_completion_tokens = verify_result.completion_tokens
        total_wall_ms = verify_result.latency_ms

        # Phase 2: conditional correction
        if verdict is VerificationVerdict.INCORRECT or correction_needed:
            correction_prompt = VERIFY_INCORRECT_PROMPT_TEMPLATE.format(
                task_prompt=state.task_prompt,
                current_answer=current_answer,
                verification_result=correction_suggestion or verify_raw[:300],
            )
            req2 = GenerationRequest(
                prompt=correction_prompt,
                temperature=0.4,
                max_tokens=256,
                seed=self._derive_seed(state, 1, replicate_id),
            )
            correction_result = backend.generate(req2)

            total_tokens += correction_result.prompt_tokens + correction_result.completion_tokens
            total_completion_tokens += correction_result.completion_tokens
            total_wall_ms += correction_result.latency_ms

            final_answer = _extract_answer(correction_result.text)
            final_trace = correction_result.text
            performed_correction = True
        else:
            final_answer = current_answer
            final_trace = current_trace
            performed_correction = False

        return Observation(
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            candidate_answer=final_answer,
            reasoning_trace=final_trace[:500],
            confidence=confidence,
            verification_score=1.0 if verdict is VerificationVerdict.CORRECT else 0.0,
            evidence={
                "verification_verdict": verdict.value,
                "verification_data": verify_data,
                "checks": checks,
                "correction_needed": correction_needed,
                "correction_suggestion": correction_suggestion,
                "performed_correction": performed_correction,
            },
            success=True,
            failure_reason="",
            cost=CostEstimate(
                tokens=total_tokens,
                completion_tokens=total_completion_tokens,
                model_calls=2 if performed_correction else 1,
                wall_ms=total_wall_ms,
            ).to_dict(),
            metadata={
                "k_before": state.k,
                "replicate_id": replicate_id,
                "prompt_id": VERIFY_PROMPT_ID,
                "prompt_hash": __import__("daph_x.operators.prompts", fromlist=["prompt_hash"]).prompt_hash(VERIFY_PROMPT_ID),
            },
        )
