"""Model interface for DAPH-X coding experiment.

Uses llama-cpp-python to load Qwen2.5-7B-Instruct-Q4_K_M and generate
candidate code solutions. Generates multiple candidates per task using
different temperatures and prompt variations to create genuine
disagreement opportunities.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daph_x.coding.tasks import CodingTask


@dataclass(frozen=True)
class ModelCallResult:
    """Result of a single model generation call."""
    task_id: str
    candidate_id: str
    solution_code: str
    temperature: float
    prompt_variant: str
    latency_ms: float
    prompt_hash: str
    raw_output: str
    error: str | None = None


class CodingModelInterface:
    """Interface to Qwen2.5-7B-Instruct for code generation.

    Generates multiple candidate solutions per task using:
      - Different temperatures (0.0, 0.3, 0.7, 1.0)
      - Different prompt variants (standard, with hints, with constraints)

    This creates genuine disagreement opportunities for DAPH-X to evaluate.
    """

    MODEL_NAME = "Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        seed: int = 42,
        model_name: str | None = None,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self._llm = None
        self._call_counter = 0
        # Derive model name from path if not specified
        if model_name:
            self.model_name = model_name
        else:
            import os
            basename = os.path.basename(model_path)
            self.model_name = basename.replace(".gguf", "")

    def _get_llm(self):
        """Lazily load the model on first use."""
        if self._llm is None:
            from llama_cpp import Llama
            print(f"Loading model from {self.model_path}...")
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
            print("Model loaded.")
        return self._llm

    def generate_candidates(
        self,
        task: CodingTask,
        n_candidates: int = 4,
        max_tokens: int = 512,
    ) -> list[ModelCallResult]:
        """Generate multiple candidate solutions for a task.

        Uses different temperatures and prompt variants to create
        genuine diversity in the candidate set.

        Args:
            task: The coding task
            n_candidates: Number of candidates to generate
            max_tokens: Maximum tokens per generation
        """
        # Temperature schedule: spread across deterministic → creative
        temperatures = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 0.1]
        # Prompt variants
        prompt_variants = [
            "standard",
            "with_edge_case_hint",
            "with_complexity_constraint",
            "with_error_handling_hint",
            "standard",
            "with_edge_case_hint",
            "with_complexity_constraint",
            "with_error_handling_hint",
        ]

        results = []
        for i in range(n_candidates):
            temp = temperatures[i % len(temperatures)]
            variant = prompt_variants[i % len(prompt_variants)]

            prompt = self._build_prompt(task, variant)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

            try:
                llm = self._get_llm()
                start = time.monotonic()

                result = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert Python programmer. Write correct, efficient code. Return ONLY the function implementation.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=0.95,
                    seed=self.seed + i,
                )

                latency_ms = (time.monotonic() - start) * 1000
                raw_output = result["choices"][0]["message"]["content"] or ""
                solution_code = self._extract_code(raw_output)

                self._call_counter += 1
                candidate_id = f"{task.task_id}_c{i}"

                results.append(ModelCallResult(
                    task_id=task.task_id,
                    candidate_id=candidate_id,
                    solution_code=solution_code,
                    temperature=temp,
                    prompt_variant=variant,
                    latency_ms=latency_ms,
                    prompt_hash=prompt_hash,
                    raw_output=raw_output,
                ))

            except Exception as e:
                candidate_id = f"{task.task_id}_c{i}"
                results.append(ModelCallResult(
                    task_id=task.task_id,
                    candidate_id=candidate_id,
                    solution_code="",
                    temperature=temp,
                    prompt_variant=variant,
                    latency_ms=0,
                    prompt_hash=prompt_hash,
                    raw_output="",
                    error=str(e),
                ))

        return results

    def _build_prompt(self, task: CodingTask, variant: str) -> str:
        """Build a prompt for the model based on the variant."""
        base = f"""Implement the following Python function. Return ONLY the function code, no explanation.

Function signature: def {task.signature}
Description: {task.description}
Difficulty: {task.difficulty}

{task.imports}

def {task.signature}:
    \"\"\"{task.docstring}\"\"\"
"""

        if variant == "standard":
            return base + "\nImplement this function correctly. Handle all edge cases."

        elif variant == "with_edge_case_hint":
            return base + f"""

Important: Make sure to handle these edge cases:
- Empty input
- Boundary conditions
- Single-element cases

Think carefully about edge cases before writing the implementation."""

        elif variant == "with_complexity_constraint":
            return base + "\nImplement this efficiently. Aim for optimal time and space complexity."

        elif variant == "with_error_handling_hint":
            return base + "\nHandle potential errors gracefully. The function should not crash on any valid input."

        return base

    def _extract_code(self, raw_output: str) -> str:
        """Extract Python code from the model output."""
        code = raw_output.strip()

        # Strip markdown code blocks
        if "```python" in code:
            start = code.find("```python") + len("```python")
            end = code.find("```", start)
            if end > start:
                code = code[start:end].strip()
        elif "```" in code:
            start = code.find("```") + 3
            end = code.find("```", start)
            if end > start:
                code = code[start:end].strip()

        return code

    @property
    def model_hash(self) -> str:
        """Hash of the model identity for provenance."""
        return hashlib.sha256(
            f"{self.model_name}:{self.model_path}".encode()
        ).hexdigest()[:16]
