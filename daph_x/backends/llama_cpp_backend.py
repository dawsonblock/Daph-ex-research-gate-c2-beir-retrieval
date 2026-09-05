"""Backend abstraction for cognitive operator execution.

Library code receives a backend object through dependency injection.
No operator should instantiate its own model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import os
import time
import hashlib


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    temperature: float
    max_tokens: int
    seed: int
    system_prompt: str = ""
    logprobs: bool = False


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str
    seed: int
    model_id: str
    model_sha256: str
    request_hash: str
    response_hash: str
    error_code: str | None = None
    error_message: str | None = None


def hash_request(request: GenerationRequest) -> str:
    s = f"{request.prompt}|{request.temperature}|{request.max_tokens}|{request.seed}|{request.system_prompt}|{request.logprobs}"
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@runtime_checkable
class CognitiveBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    @property
    def model_id(self) -> str:
        ...

    @property
    def model_sha256(self) -> str:
        ...


class LlamaCppBackend:
    """Minimal llama.cpp backend using existing CodingModelInterface."""

    def __init__(self, model_path: str | None = None, n_gpu_layers: int = -1, seed: int = 42):
        if model_path is None:
            model_path = os.environ.get(
                "DAPH_X_MODEL_PATH",
                "/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            )
        self._model_path = model_path
        self._n_gpu_layers = n_gpu_layers
        self._seed = seed
        self._model = None
        self._model_sha256 = sha256_file(model_path)
        self._model_id = os.path.basename(model_path)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def _get_model(self):
        if self._model is None:
            from daph_x.coding.model_interface import CodingModelInterface
            self._model = CodingModelInterface(
                model_path=self._model_path,
                n_gpu_layers=self._n_gpu_layers,
                seed=self._seed,
            )
        return self._model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.monotonic()
        try:
            text = self._get_model().generate_raw(
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                seed=request.seed,
            )
            finish_reason = "stop"
            error_code = None
            error_message = None
        except Exception as e:
            text = ""
            finish_reason = "error"
            error_code = type(e).__name__
            error_message = str(e)

        latency_ms = (time.monotonic() - t0) * 1000

        # Token counts: use actual character count estimate until real tokenizer exposed
        prompt_tokens = len(request.prompt) // 4
        completion_tokens = len(text) // 4
        total_tokens = prompt_tokens + completion_tokens

        request_hash = hash_request(request)
        response_hash = hashlib.sha256(text.encode()).hexdigest()

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            seed=request.seed,
            model_id=self._model_id,
            model_sha256=self._model_sha256,
            request_hash=request_hash,
            response_hash=response_hash,
            error_code=error_code,
            error_message=error_message,
        )
