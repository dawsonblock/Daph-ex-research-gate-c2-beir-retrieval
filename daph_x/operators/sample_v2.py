"""SAMPLE_STANDARD v2 — exact R12 continuation, two more candidates."""
from __future__ import annotations

import re
from daph_x.operators.operator import CognitiveOperatorV2, CostEstimate
from daph_x.operators.types import RuntimeState, Observation
from daph_x.operators.prompts import BASE_PROMPT_TEMPLATE
from daph_x.backends.llama_cpp_backend import CognitiveBackend, GenerationRequest

# Frozen R12 temperature schedule
R12_TEMPERATURES = (0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.2)


def _extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _derive_seed(checkpoint_hash: str, operator_id: str, operator_version: str, replicate_id: int, generation_index: int) -> int:
    s = f"{checkpoint_hash}|{operator_id}|{operator_version}|{replicate_id}|{generation_index}"
    return int.from_bytes(
        (int(__import__("hashlib").sha256(s.encode()).hexdigest()[:8], 16) % (2**31)).to_bytes(4, "big"),
        "big",
    )


class SampleStandardV2:
    operator_id = "SAMPLE_STANDARD"
    operator_version = "2"
    n_new = 2

    def is_admissible(self, state: RuntimeState) -> bool:
        return state.k + self.n_new <= 12

    def estimate_cost(self, state: RuntimeState) -> CostEstimate:
        return CostEstimate(
            tokens=2 * len(state.task_prompt) // 4,
            completion_tokens=2 * 256,
            model_calls=2,
            wall_ms=2 * 10000,
        )

    def execute(self, state: RuntimeState, backend: CognitiveBackend, replicate_id: int = 42) -> Observation:
        new_candidates = []
        raw_responses = []
        total_tokens = 0
        total_completion_tokens = 0
        total_wall_ms = 0.0
        prompt = BASE_PROMPT_TEMPLATE.format(task_prompt=state.task_prompt)

        for i in range(self.n_new):
            next_idx = state.k + i
            temperature = R12_TEMPERATURES[next_idx]
            seed = _derive_seed(state.state_hash, self.operator_id, self.operator_version, replicate_id, i)

            req = GenerationRequest(
                prompt=prompt,
                temperature=temperature,
                max_tokens=256,
                seed=seed,
            )
            result = backend.generate(req)

            total_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_wall_ms += result.latency_ms

            answer = _extract_answer(result.text)
            new_candidates.append({
                "answer": answer,
                "response": result.text,
                "temperature": temperature,
                "seed": seed,
            })
            raw_responses.append(result.text[:200])

        return Observation(
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            candidate_answer=new_candidates[0]["answer"] if new_candidates else "",
            reasoning_trace=new_candidates[0]["response"][:500] if new_candidates else "",
            confidence=0.0,
            verification_score=0.0,
            evidence={"new_candidates": new_candidates},
            success=True,
            failure_reason="",
            cost=CostEstimate(
                tokens=total_tokens,
                completion_tokens=total_completion_tokens,
                model_calls=self.n_new,
                wall_ms=total_wall_ms,
            ).to_dict(),
            metadata={
                "n_new": self.n_new,
                "k_before": state.k,
                "temperatures": [R12_TEMPERATURES[state.k + i] for i in range(self.n_new)],
                "replicate_id": replicate_id,
                "raw_responses": raw_responses,
            },
        )
