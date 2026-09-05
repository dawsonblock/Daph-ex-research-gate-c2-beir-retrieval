"""CRITIQUE_RETRY v2 — structured JSON critique; no forced regen when NO_ERROR."""
from __future__ import annotations

import re
from daph_x.operators.operator import CognitiveOperatorV2, CostEstimate
from daph_x.operators.types import RuntimeState, Observation
from daph_x.operators.prompts import CRITIQUE_PROMPT_ID, CRITIQUE_PROMPT_TEMPLATE, CRITIQUE_RETRY_PROMPT_ID, CRITIQUE_RETRY_PROMPT_TEMPLATE
from daph_x.operators.verdicts import parse_critique_verdict, CritiqueVerdict
from daph_x.backends.llama_cpp_backend import CognitiveBackend, GenerationRequest


def _extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


class CritiqueRetryV2:
    operator_id = "CRITIQUE_RETRY"
    operator_version = "2"

    def is_admissible(self, state: RuntimeState) -> bool:
        return len(state.candidates) > 0 and state.k + 1 <= 12

    def estimate_cost(self, state: RuntimeState) -> CostEstimate:
        # Worst case: critique + retry
        return CostEstimate(
            tokens=len(state.task_prompt) // 4 + 2 * 100,
            completion_tokens=2 * 256,
            model_calls=2,
            wall_ms=2 * 10000,
        )

    def _get_leader_trace(self, state: RuntimeState) -> tuple:
        """Get a representative trace for the current leading answer."""
        # Pick the first candidate whose answer equals the current (majority) answer
        for c in state.candidates:
            if c.answer == state.current_answer:
                return c.answer, c.reasoning_trace
        # Fallback to first candidate
        c = state.candidates[0]
        return c.answer, c.reasoning_trace

    def _derive_seed(self, state: RuntimeState, step: int, replicate_id: int) -> int:
        s = f"{state.state_hash}|{self.operator_id}|{self.operator_version}|{replicate_id}|{step}"
        return int(__import__("hashlib").sha256(s.encode()).hexdigest()[:8], 16)

    def execute(self, state: RuntimeState, backend: CognitiveBackend, replicate_id: int = 42) -> Observation:
        current_answer, current_trace = self._get_leader_trace(state)

        # Phase 1: structured critique
        critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(
            task_prompt=state.task_prompt,
            current_answer=current_answer,
            current_trace=current_trace[:500],
        )

        req = GenerationRequest(
            prompt=critique_prompt,
            temperature=0.3,
            max_tokens=200,
            seed=self._derive_seed(state, 0, replicate_id),
        )
        critique_result = backend.generate(req)

        critique_raw = critique_result.text
        critique_data = {}
        try:
            import json
            match = re.search(r"\{[\s\S]*\}", critique_raw)
            if match:
                critique_data = json.loads(match.group(0))
        except Exception:
            pass

        verdict = parse_critique_verdict(critique_raw)
        failure_mode = str(critique_data.get("failure_mode", "OTHER"))
        first_error = str(critique_data.get("first_error", ""))
        explanation = str(critique_data.get("explanation", ""))
        repair_instruction = str(critique_data.get("repair_instruction", ""))
        confidence = float(critique_data.get("confidence", 0.0) or 0.0)

        total_tokens = critique_result.prompt_tokens + critique_result.completion_tokens
        total_completion_tokens = critique_result.completion_tokens
        total_wall_ms = critique_result.latency_ms

        # Phase 2: retry only if verdict is ERROR
        if verdict is CritiqueVerdict.ERROR:
            retry_prompt = CRITIQUE_RETRY_PROMPT_TEMPLATE.format(
                task_prompt=state.task_prompt,
                current_answer=current_answer,
                error_description=first_error or explanation,
                repair_instruction=repair_instruction or "be careful and avoid the identified error",
            )
            req2 = GenerationRequest(
                prompt=retry_prompt,
                temperature=0.5,
                max_tokens=256,
                seed=self._derive_seed(state, 1, replicate_id),
            )
            retry_result = backend.generate(req2)

            total_tokens += retry_result.prompt_tokens + retry_result.completion_tokens
            total_completion_tokens += retry_result.completion_tokens
            total_wall_ms += retry_result.latency_ms

            final_answer = _extract_answer(retry_result.text)
            final_trace = retry_result.text
            performed_retry = True
        else:
            final_answer = current_answer
            final_trace = current_trace
            performed_retry = False

        return Observation(
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            candidate_answer=final_answer,
            reasoning_trace=final_trace[:500],
            confidence=confidence,
            verification_score=0.0,
            evidence={
                "critique_verdict": verdict.value,
                "critique_data": critique_data,
                "failure_mode": failure_mode,
                "first_error": first_error,
                "explanation": explanation,
                "repair_instruction": repair_instruction,
                "performed_retry": performed_retry,
            },
            success=True,
            failure_reason="",
            cost=CostEstimate(
                tokens=total_tokens,
                completion_tokens=total_completion_tokens,
                model_calls=2 if performed_retry else 1,
                wall_ms=total_wall_ms,
            ).to_dict(),
            metadata={
                "k_before": state.k,
                "replicate_id": replicate_id,
                "prompt_id": CRITIQUE_PROMPT_ID,
                "prompt_hash": __import__("daph_x.operators.prompts", fromlist=["prompt_hash"]).prompt_hash(CRITIQUE_PROMPT_ID),
            },
        )
