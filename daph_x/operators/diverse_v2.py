"""SAMPLE_DIVERSE v2 — two distinct strategies per checkpoint, frozen registry."""
from __future__ import annotations

import re
from daph_x.operators.operator import CognitiveOperatorV2, CostEstimate
from daph_x.operators.types import RuntimeState, Observation
from daph_x.operators.prompts import DIVERSE_STRATEGIES, DIVERSE_PROMPT_SET_ID
from daph_x.backends.llama_cpp_backend import CognitiveBackend, GenerationRequest


def _extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _hash_to_int(checkpoint_hash: str, operator_version: str, replicate_id: int) -> int:
    s = f"{checkpoint_hash}|SAMPLE_DIVERSE|{operator_version}|{replicate_id}"
    return int(__import__("hashlib").sha256(s.encode()).hexdigest()[:8], 16)


class SampleDiverseV2:
    operator_id = "SAMPLE_DIVERSE"
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

    def _choose_strategies(self, state: RuntimeState, replicate_id: int) -> list:
        strategy_ids = sorted(DIVERSE_STRATEGIES.keys())
        offset = _hash_to_int(state.state_hash, self.operator_version, replicate_id) % len(strategy_ids)
        return [
            strategy_ids[(offset + i) % len(strategy_ids)]
            for i in range(self.n_new)
        ]

    def execute(self, state: RuntimeState, backend: CognitiveBackend, replicate_id: int = 42) -> Observation:
        chosen_strategies = self._choose_strategies(state, replicate_id)
        new_candidates = []
        raw_responses = []
        total_tokens = 0
        total_completion_tokens = 0
        total_wall_ms = 0.0

        for i, strategy_id in enumerate(chosen_strategies):
            template = DIVERSE_STRATEGIES[strategy_id]
            prompt = template.format(task_prompt=state.task_prompt)

            seed = _hash_to_int(state.state_hash, self.operator_version, replicate_id) + i
            seed = (seed % (2**31 - 1)) + 1  # positive 32-bit

            req = GenerationRequest(
                prompt=prompt,
                temperature=0.7 + i * 0.1,
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
                "strategy_id": strategy_id,
                "temperature": 0.7 + i * 0.1,
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
                "strategy_ids": chosen_strategies,
                "replicate_id": replicate_id,
                "raw_responses": raw_responses,
            },
        )
