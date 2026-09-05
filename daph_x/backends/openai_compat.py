"""OpenAI-compatible external backend for R14 external operators.

Supports any endpoint exposing POST /v1/chat/completions:
  - ThinkBooster service
  - OptiLLM proxy
  - llama-server / llama.cpp
  - Ollama
  - vLLM OpenAI server
  - remote providers

This is Lane A (black-box) per R14_PROTOCOL.md §5.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    import requests
except ImportError:
    requests = None


@dataclass(frozen=True)
class ChatMessage:
    role: str          # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ExternalGenerationRequest:
    messages: tuple[ChatMessage, ...]
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 42
    stop: tuple[str, ...] = field(default_factory=tuple)
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        if self.stop:
            payload["stop"] = list(self.stop)
        payload.update(self.extra_params)
        return payload

    def request_hash(self) -> str:
        s = json.dumps({
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
            "extra_params": self.extra_params,
        }, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()


@dataclass(frozen=True)
class ExternalGenerationResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    finish_reason: str
    seed: int
    model_id: str
    request_hash: str
    response_hash: str
    raw_response: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error_code is None


class OpenAICompatibleBackend:
    """Lane A backend: calls any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        provider_name: str = "openai_compatible",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout_s = timeout_s
        self.provider_name = provider_name

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def model_sha256(self) -> str:
        # External backends do not expose model file hashes.
        # Use a stable identifier of (provider, model) for provenance.
        return hashlib.sha256(f"{self.provider_name}:{self.model}".encode()).hexdigest()

    @property
    def capabilities(self) -> set[str]:
        return {"openai_compatible"}

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        if requests is None:
            raise ImportError("requests package required for OpenAICompatibleBackend")

        url = f"{self.base_url}/v1/chat/completions"
        payload = request.to_payload()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        t0 = time.monotonic()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
            latency_ms = (time.monotonic() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as e:
            return ExternalGenerationResult(
                text="",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                finish_reason="timeout",
                seed=request.seed,
                model_id=self.model,
                request_hash=request.request_hash(),
                response_hash="",
                error_code="TIMEOUT",
                error_message=str(e),
            )
        except requests.exceptions.RequestException as e:
            return ExternalGenerationResult(
                text="",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                finish_reason="error",
                seed=request.seed,
                model_id=self.model,
                request_hash=request.request_hash(),
                response_hash="",
                error_code="REQUEST_ERROR",
                error_message=str(e),
            )
        except Exception as e:
            return ExternalGenerationResult(
                text="",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                finish_reason="error",
                seed=request.seed,
                model_id=self.model,
                request_hash=request.request_hash(),
                response_hash="",
                error_code=type(e).__name__,
                error_message=str(e),
            )

        # Parse OpenAI-format response
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
        except (KeyError, IndexError, TypeError) as e:
            return ExternalGenerationResult(
                text="",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_ms=latency_ms,
                finish_reason="parse_error",
                seed=request.seed,
                model_id=self.model,
                request_hash=request.request_hash(),
                response_hash="",
                raw_response=data,
                error_code="PARSE_ERROR",
                error_message=f"Failed to parse response: {e}",
            )

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        response_hash = hashlib.sha256(text.encode()).hexdigest()

        return ExternalGenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            seed=request.seed,
            model_id=self.model,
            request_hash=request.request_hash(),
            response_hash=response_hash,
            raw_response=data,
        )
