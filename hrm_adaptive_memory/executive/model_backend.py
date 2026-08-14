"""Model backend interface and DeepSeek API implementation for I3.4.

The ``ModelBackend`` protocol abstracts the pinned model call so the
controller is provider-agnostic.  The ``DeepSeekBackend`` implementation calls
the OpenAI-compatible DeepSeek API.  The API key is loaded from the
``DEEPSEEK_API_KEY`` environment variable and is never persisted in the
repository or in controller identity.

For development and testing, ``StubBackend`` returns a deterministic
JSON response without any network call.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelCallResult:
    """Raw output and metadata from one model invocation."""

    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_ms: int
    model_name: str
    system_fingerprint: str | None
    finish_reason: str | None


@runtime_checkable
class ModelBackend(Protocol):
    """Provider-agnostic interface for the pinned model."""

    model_name: str

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int) -> ModelCallResult: ...


@dataclass
class DeepSeekBackend:
    """DeepSeek API backend (OpenAI-compatible).

    The API key is read from the ``DEEPSEEK_API_KEY`` environment variable.
    It is never stored in the backend instance, in controller identity, or in
    any repository file.
    """

    model_name: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com/v1"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    _api_key: str | None = field(default=None, repr=False)

    def _get_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable is not set; "
                "cannot call the DeepSeek API")
        return key

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int) -> ModelCallResult:
        import urllib.error
        import urllib.request

        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._get_api_key()}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            start = time.monotonic()
            try:
                with urllib.request.urlopen(
                        request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read())
                latency = int((time.monotonic() - start) * 1000)
                choice = body["choices"][0]
                usage = body.get("usage", {})
                return ModelCallResult(
                    raw_output=choice["message"]["content"] or "",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                    latency_ms=latency,
                    model_name=body.get("model", self.model_name),
                    system_fingerprint=body.get("system_fingerprint"),
                    finish_reason=choice.get("finish_reason"),
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    OSError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                continue
        # All retries exhausted; raise so the controller can fail closed.
        raise RuntimeError(f"DeepSeek API failed after {self.max_retries} retries: {last_error}")


@dataclass
class StubBackend:
    """Deterministic backend for development and testing (no network)."""

    model_name: str = "stub-deterministic-v1"
    _responses: tuple[str, ...] = (
        '{"action": "RETRIEVE", "reason_code": "STUB_INITIAL_RETRIEVE", "target_id": null}',
        '{"action": "VERIFY", "reason_code": "STUB_VERIFY", "target_id": null}',
        '{"action": "ANSWER", "reason_code": "STUB_ANSWER", "target_id": null}',
    )
    _call_index: int = field(default=0, repr=False)

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int) -> ModelCallResult:
        response = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        return ModelCallResult(
            raw_output=response,
            prompt_tokens=len(system_prompt) // 4 + len(user_prompt) // 4,
            completion_tokens=len(response) // 4,
            reasoning_tokens=0,
            latency_ms=0,
            model_name=self.model_name,
            system_fingerprint=None,
            finish_reason="stop",
        )
