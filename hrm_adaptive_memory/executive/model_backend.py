"""Model backend interface and DeepSeek API implementation for I3.4.1.

The ``ModelBackend`` protocol abstracts the pinned model call so the
controller is provider-agnostic.  The ``DeepSeekBackend`` implementation calls
the OpenAI-compatible DeepSeek API.  The API key is loaded from the
``DEEPSEEK_API_KEY`` environment variable and is never persisted in the
repository or in controller identity.

I3.4.1 wiring:
- The backend now accepts a ``FrozenGenerationConfig`` and uses it to build
  the actual API request, including ``thinking`` and ``response_format``.
- The backend now uses ``FROZEN_RETRY_POLICY`` to decide which HTTP status
  codes and exceptions are retryable.
- The backend now emits a ``CallReceipt`` for every attempt (including
  retries and failures), collected in ``call_receipts``.
- A ``strict_json`` flag selects between the permissive development decoder
  extractor and strict whole-response JSON parsing for scientific runs.

For development and testing, ``StubBackend`` returns a deterministic
JSON response without any network call.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .i3_4_generation_config import FrozenGenerationConfig, FROZEN_CONFIG
from .i3_4_retry_policy import (
    FROZEN_RETRY_POLICY, CallReceipt, RetryPolicy, make_call_receipt)


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


def _build_request_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_mode: str,
    response_format: str,
) -> bytes:
    """Build the DeepSeek API request payload from the frozen generation config.

    This is the single source of truth for what is sent to the API.  It
    explicitly includes ``thinking`` and ``response_format`` so the frozen
    generation config is authoritative over execution.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Explicitly disable thinking.  DeepSeek V4 defaults to thinking enabled,
    # so this must be sent explicitly to control the hidden reasoning budget.
    if thinking_mode == "disabled":
        payload["thinking"] = {"type": "disabled"}
    elif thinking_mode == "enabled":
        payload["thinking"] = {"type": "enabled"}
    # Explicitly request JSON object output mode for strict schema enforcement.
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return json.dumps(payload).encode()


@dataclass
class DeepSeekBackend:
    """DeepSeek API backend (OpenAI-compatible).

    The API key is read from the ``DEEPSEEK_API_KEY`` environment variable.
    It is never stored in the backend instance, in controller identity, or in
    any repository file.

    I3.4.1: The backend is now wired to the frozen generation config and
    frozen retry policy.  Every attempt emits a CallReceipt.
    """

    model_name: str = FROZEN_CONFIG.model
    base_url: str = "https://api.deepseek.com/v1"
    config: FrozenGenerationConfig = field(default_factory=lambda: FROZEN_CONFIG)
    retry_policy: RetryPolicy = field(default_factory=lambda: FROZEN_RETRY_POLICY)
    # Metadata for call receipts (set by the experiment runner)
    experiment_id: str = ""
    pair_id: str = ""
    task_id: str = ""
    condition: str = ""
    # Accumulated receipts (one per attempt, including retries and failures)
    call_receipts: list[CallReceipt] = field(default_factory=list, repr=False)
    _api_key: str | None = field(default=None, repr=False)
    _call_counter: int = field(default=0, repr=False, init=False)

    def _get_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable is not set; "
                "cannot call the DeepSeek API")
        return key

    def _request_sha256(self, payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int) -> ModelCallResult:
        """Generate a model response using the frozen generation config.

        The actual API request includes ``thinking`` and ``response_format``
        from the frozen config.  Retries are governed by the frozen retry
        policy.  Every attempt emits a CallReceipt.
        """
        import urllib.error
        import urllib.request

        payload = _build_request_payload(
            model=self.config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_mode=self.config.thinking_mode,
            response_format=self.config.response_format,
        )
        request_sha = self._request_sha256(payload)
        prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()
        packet_sha = hashlib.sha256(user_prompt.encode()).hexdigest()
        gen_config_sha = self.config.sha256()

        last_error: Exception | None = None
        max_retries = self.retry_policy.max_retries

        for attempt in range(max_retries):
            call_id = f"{self.experiment_id}:call_{self._call_counter}"
            self._call_counter += 1

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
                        request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read())
                latency = int((time.monotonic() - start) * 1000)
                choice = body["choices"][0]
                usage = body.get("usage", {})
                raw_output = choice["message"]["content"] or ""
                result = ModelCallResult(
                    raw_output=raw_output,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                    latency_ms=latency,
                    model_name=body.get("model", self.config.model),
                    system_fingerprint=body.get("system_fingerprint"),
                    finish_reason=choice.get("finish_reason"),
                )
                # Emit success receipt.
                receipt = make_call_receipt(
                    call_id=call_id, pair_id=self.pair_id,
                    attempt_index=attempt, task_id=self.task_id,
                    condition=self.condition,
                    request_sha256=request_sha, packet_sha256=packet_sha,
                    prompt_sha256=prompt_sha,
                    generation_config_sha256=gen_config_sha,
                    result_class="success",
                    http_status=200,
                    reported_model=result.model_name,
                    system_fingerprint=result.system_fingerprint,
                    latency_ms=result.latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    reasoning_tokens=result.reasoning_tokens,
                    finish_reason=result.finish_reason,
                    raw_output=raw_output,
                )
                self.call_receipts.append(receipt)
                return result

            except urllib.error.HTTPError as exc:
                latency = int((time.monotonic() - start) * 1000)
                status = exc.code
                error_msg = str(exc)
                last_error = exc
                # Emit error receipt.
                receipt = make_call_receipt(
                    call_id=call_id, pair_id=self.pair_id,
                    attempt_index=attempt, task_id=self.task_id,
                    condition=self.condition,
                    request_sha256=request_sha, packet_sha256=packet_sha,
                    prompt_sha256=prompt_sha,
                    generation_config_sha256=gen_config_sha,
                    result_class="http_error",
                    http_status=status,
                    latency_ms=latency,
                    error_message=error_msg,
                )
                self.call_receipts.append(receipt)
                # Consult the frozen retry policy.
                if not self.retry_policy.should_retry_http(status):
                    # Non-retryable HTTP error: fail immediately.
                    raise RuntimeError(
                        f"DeepSeek API returned HTTP {status} (non-retryable): {exc}") from exc
                if attempt < max_retries - 1:
                    time.sleep(self.retry_policy.retry_backoff_seconds * (attempt + 1))
                continue

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                latency = int((time.monotonic() - start) * 1000)
                error_msg = str(exc)
                last_error = exc
                # Determine result class for the receipt.
                if isinstance(exc, TimeoutError):
                    result_class = "timeout"
                else:
                    result_class = "connection_error"
                # Emit error receipt.
                receipt = make_call_receipt(
                    call_id=call_id, pair_id=self.pair_id,
                    attempt_index=attempt, task_id=self.task_id,
                    condition=self.condition,
                    request_sha256=request_sha, packet_sha256=packet_sha,
                    prompt_sha256=prompt_sha,
                    generation_config_sha256=gen_config_sha,
                    result_class=result_class,
                    latency_ms=latency,
                    error_message=error_msg,
                )
                self.call_receipts.append(receipt)
                # Consult the frozen retry policy.
                if not self.retry_policy.should_retry_exception(exc):
                    raise RuntimeError(
                        f"DeepSeek API failed (non-retryable {type(exc).__name__}): {exc}") from exc
                if attempt < max_retries - 1:
                    time.sleep(self.retry_policy.retry_backoff_seconds * (attempt + 1))
                continue

        # All retries exhausted.
        raise RuntimeError(
            f"DeepSeek API failed after {max_retries} retries: {last_error}")


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


@dataclass
class LocalLlamaBackend:
    """Local llama.cpp server backend (OpenAI-compatible).

    Calls a local llama.cpp server started via ``llama serve``.
    No API key required.  The server URL defaults to
    ``http://127.0.0.1:8080/v1``.

    This backend is designed for reproducible local inference using
    GGUF-quantized models such as LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M.

    Frozen configuration for scientific runs should record:
      - model repository and quantization
      - GGUF SHA-256
      - llama.cpp version (system_fingerprint)
      - context size, temperature, top_p, top_k, repeat_penalty, seed
      - threads, GPU layers
    """

    model_name: str = "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M"
    base_url: str = "http://127.0.0.1:8080/v1"
    timeout_seconds: int = 120
    # Metadata for call receipts (set by the experiment runner)
    experiment_id: str = ""
    pair_id: str = ""
    task_id: str = ""
    condition: str = ""
    _call_counter: int = field(default=0, repr=False, init=False)

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int) -> ModelCallResult:
        """Generate a model response via the local llama.cpp server."""
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

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
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
            raw_output = choice["message"]["content"] or ""
            timings = body.get("timings", {})
            result = ModelCallResult(
                raw_output=raw_output,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                latency_ms=latency,
                model_name=body.get("model", self.model_name),
                system_fingerprint=body.get("system_fingerprint"),
                finish_reason=choice.get("finish_reason"),
            )
            self._call_counter += 1
            return result

        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Local llama server returned HTTP {exc.code}: {exc}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Local llama server connection failed: {exc}") from exc
