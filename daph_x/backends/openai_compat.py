"""OpenAI-compatible external backend for R14 external operators.

Supports any endpoint exposing POST {base_url}/chat/completions:
  - ThinkBooster service (base_url includes strategy/scorer path)
  - OptiLLM proxy (base_url = http://localhost:8000/v1)
  - llama-server / llama.cpp (base_url = http://localhost:8080/v1)
  - Ollama, vLLM OpenAI server, remote providers

URL semantics: base_url is exactly what the OpenAI SDK means by base URL.
The backend appends /chat/completions to base_url, nothing more.

This is Lane A (black-box) per R14_PROTOCOL.md §5.
"""
from __future__ import annotations

import copy
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


@dataclass(frozen=True)
class ServiceIdentity:
    """Identity and provenance of an external service endpoint."""
    provider_name: str
    base_url: str
    model: str
    provider_commit_sha: str | None = None
    provider_version: str | None = None
    service_config_hash: str | None = None
    base_model_id: str | None = None
    base_model_revision: str | None = None
    scorer_model_id: str | None = None
    scorer_model_revision: str | None = None
    strategy_config_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "base_url": self.base_url,
            "model": self.model,
            "provider_commit_sha": self.provider_commit_sha,
            "provider_version": self.provider_version,
            "service_config_hash": self.service_config_hash,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "scorer_model_id": self.scorer_model_id,
            "scorer_model_revision": self.scorer_model_revision,
            "strategy_config_hash": self.strategy_config_hash,
        }


class OpenAICompatibleBackend:
    """Lane A backend: calls any OpenAI-compatible /chat/completions endpoint.

    base_url is exactly what the OpenAI SDK means by base URL.
    The chat completions endpoint is: {base_url}/chat/completions

    For standard OpenAI: base_url = "https://api.openai.com/v1"
    For OptiLLM:         base_url = "http://localhost:8000/v1"
    For llama-server:    base_url = "http://localhost:8080/v1"
    For ThinkBooster:    base_url = "http://localhost:8001/v1/beam_search/prm"

    This backend is immutable. Use with_base_url() to create a derived
    backend pointing at a different endpoint (e.g. ThinkBooster strategy path).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        provider_name: str = "openai_compatible",
        capabilities: set[str] | None = None,
        service_identity: ServiceIdentity | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self._timeout_s = timeout_s
        self._provider_name = provider_name
        self._capabilities = capabilities if capabilities is not None else {"openai_compatible"}
        self._service_identity = service_identity or ServiceIdentity(
            provider_name=provider_name,
            base_url=self._base_url,
            model=model,
        )

    # --- Immutable accessors ---

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def model_sha256(self) -> str:
        return hashlib.sha256(f"{self._provider_name}:{self._model}".encode()).hexdigest()

    @property
    def capabilities(self) -> set[str]:
        return set(self._capabilities)

    @property
    def service_identity(self) -> ServiceIdentity:
        return self._service_identity

    @property
    def chat_completions_url(self) -> str:
        """Full URL for chat completions endpoint."""
        return f"{self._base_url}/chat/completions"

    # --- Derivation ---

    def with_base_url(self, base_url: str) -> OpenAICompatibleBackend:
        """Create a new backend with a different base_url, preserving all other config."""
        return OpenAICompatibleBackend(
            base_url=base_url,
            model=self._model,
            api_key=self._api_key,
            timeout_s=self._timeout_s,
            provider_name=self._provider_name,
            capabilities=self._capabilities,
            service_identity=ServiceIdentity(
                provider_name=self._service_identity.provider_name,
                base_url=base_url.rstrip("/"),
                model=self._service_identity.model,
                provider_commit_sha=self._service_identity.provider_commit_sha,
                provider_version=self._service_identity.provider_version,
                service_config_hash=self._service_identity.service_config_hash,
                base_model_id=self._service_identity.base_model_id,
                base_model_revision=self._service_identity.base_model_revision,
                scorer_model_id=self._service_identity.scorer_model_id,
                scorer_model_revision=self._service_identity.scorer_model_revision,
                strategy_config_hash=self._service_identity.strategy_config_hash,
            ),
        )

    def with_capability(self, capability: str) -> OpenAICompatibleBackend:
        """Create a new backend with an additional capability."""
        new_caps = self._capabilities | {capability}
        return OpenAICompatibleBackend(
            base_url=self._base_url,
            model=self._model,
            api_key=self._api_key,
            timeout_s=self._timeout_s,
            provider_name=self._provider_name,
            capabilities=new_caps,
            service_identity=self._service_identity,
        )

    # --- Generation ---

    def generate(
        self,
        request: ExternalGenerationRequest,
        base_url_override: str | None = None,
    ) -> ExternalGenerationResult:
        """Execute a chat completions request.

        Args:
            request: The generation request.
            base_url_override: Optional override for base_url. Use this for
                thread-safe routing to different endpoints (e.g. ThinkBooster
                strategy paths) without mutating shared backend state.
        """
        if requests is None:
            raise ImportError("requests package required for OpenAICompatibleBackend")

        url = f"{(base_url_override or self._base_url).rstrip('/')}/chat/completions"
        payload = request.to_payload()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        t0 = time.monotonic()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout_s)
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
                model_id=self._model,
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
                model_id=self._model,
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
                model_id=self._model,
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
                model_id=self._model,
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
            model_id=self._model,
            request_hash=request.request_hash(),
            response_hash=response_hash,
            raw_response=data,
        )
