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
    """Raw output and metadata from one model invocation.

    For backends that normalize output (e.g. LocalLlamaBackend),
    ``raw_output`` contains the normalized output that the decoder
    will consume, while ``provider_raw_output`` preserves the
    exact bytes returned by the provider for provenance.
    When no normalization is applied, both fields are identical.

    LOCAL_POLICY_V2 provenance fields (populated by LocalLlamaBackend):
      - ``json_schema_sha256``: SHA-256 of the JSON schema sent to the
        provider for constrained generation.
      - ``system_prompt_sha256``: SHA-256 of the full system prompt
        (including any action-semantics adapter suffix).
      - ``user_packet_sha256``: SHA-256 of the user prompt (evidence
        packet).
      - ``request_sha256``: SHA-256 of the full request payload.
    """

    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_ms: int
    model_name: str
    system_fingerprint: str | None
    finish_reason: str | None
    provider_raw_output: str = ""
    # LOCAL_POLICY_V2 provenance (optional; populated by LocalLlamaBackend)
    json_schema_sha256: str = ""
    system_prompt_sha256: str = ""
    user_packet_sha256: str = ""
    request_sha256: str = ""

    @property
    def provider_raw_sha256(self) -> str:
        return hashlib.sha256(self.provider_raw_output.encode()).hexdigest()

    @property
    def normalized_sha256(self) -> str:
        return hashlib.sha256(self.raw_output.encode()).hexdigest()

    @property
    def normalization_applied(self) -> bool:
        return self.provider_raw_output != self.raw_output


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
                    provider_raw_output=raw_output,
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
            provider_raw_output=response,
        )


@dataclass
class LocalLlamaBackend:
    """Local llama.cpp server backend (OpenAI-compatible).

    Calls a local llama.cpp server started via ``llama serve``.
    No API key required.  The server URL defaults to
    ``http://127.0.0.1:8080/v1``.

    This backend is designed for reproducible local inference using
    GGUF-quantized models such as LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M.

    Includes retry logic with exponential backoff for connection errors,
    which is essential when multiple workers hit the server concurrently.

    Frozen configuration for scientific runs should record:
      - model repository and quantization
      - GGUF SHA-256
      - llama.cpp version (system_fingerprint)
      - context size, temperature, top_p, top_k, repeat_penalty, seed
      - threads, GPU layers

    R2-DEV-V2: The backend now accepts an optional ``allowed_actions``
    parameter in ``generate()``.  When provided, the JSON schema sent
    to the provider is built dynamically from the allowed action set,
    physically preventing the model from generating gated actions.
    When not provided, the full seven-action vocabulary is used
    (matching the frozen R13 static schema).

    NOTE: The server-based ``response_format`` with ``json_schema`` type
    is NOT reliably enforced by all llama.cpp server versions.  For
    strict schema-constrained generation, use ``R2DirectLlamaBackend``
    which uses ``LlamaGrammar`` directly.
    """

    model_name: str = "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M"
    base_url: str = "http://127.0.0.1:8080/v1"
    timeout_seconds: int = 300
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    # Metadata for call receipts (set by the experiment runner)
    experiment_id: str = ""
    pair_id: str = ""
    task_id: str = ""
    condition: str = ""
    _call_counter: int = field(default=0, repr=False, init=False)

    # NOTE: LOCAL_POLICY_V2 removes the previous _normalize_output method.
    # The llama.cpp JSON-schema constraint (strict=True, with the
    # ^[A-Z][A-Z0-9_]*$ pattern on reason_code) guarantees structural
    # validity at the provider level.  Provider output is passed directly
    # to the strict decoder with no semantic normalization.  If the
    # provider emits malformed output despite the schema, the decoder
    # rejects it and the runtime fails closed — no repair is performed.

    # R2-DEV-V2: Full seven-action vocabulary for default schema.
    _FULL_ACTION_VOCAB = frozenset({
        "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
        "REASON_MORE", "DEFER", "STOP",
    })

    def _build_action_schema(self, allowed_actions: frozenset[str] | None) -> dict:
        """Build the JSON schema for constrained generation.

        When ``allowed_actions`` is provided, the action enum is restricted
        to exactly those actions.  When None, the full seven-action
        vocabulary is used (matching the frozen R13 static schema).
        """
        actions = allowed_actions if allowed_actions is not None else self._FULL_ACTION_VOCAB
        # Use canonical R13 order for the enum
        canonical_order = (
            "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
            "REASON_MORE", "DEFER", "STOP",
        )
        enum = [a for a in canonical_order if a in actions]
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": enum,
                },
                "reason_code": {
                    "type": "string",
                    "pattern": "^[A-Z][A-Z0-9_]*$",
                },
                "target_id": {"type": ["string", "null"]},
            },
            "required": ["action", "reason_code", "target_id"],
            "additionalProperties": False,
        }

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int,
                 allowed_actions: frozenset[str] | None = None) -> ModelCallResult:
        """Generate a model response via the local llama.cpp server.

        Retries on connection errors with exponential backoff.
        The llama.cpp server has a limited number of slots (default 4),
        so concurrent requests may be rejected temporarily.

        R2-DEV-V2: When ``allowed_actions`` is provided, the JSON schema
        sent to the provider restricts the action enum to exactly those
        actions, physically preventing generation of gated actions.

        WARNING: The server-based response_format with json_schema type
        may not be reliably enforced by all llama.cpp server versions.
        For strict schema-constrained generation, use R2DirectLlamaBackend.
        """
        import urllib.error
        import urllib.request

        action_schema = self._build_action_schema(allowed_actions)
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1.0,
            "top_k": 40,
            "repeat_penalty": 1.0,
            "seed": 42,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "action_proposal",
                    "strict": True,
                    "schema": action_schema,
                },
            },
        }).encode()

        # LOCAL_POLICY_V2 provenance hashes
        schema_sha = hashlib.sha256(
            json.dumps(action_schema, sort_keys=True).encode()).hexdigest()
        prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()
        packet_sha = hashlib.sha256(user_prompt.encode()).hexdigest()
        request_sha = hashlib.sha256(payload).hexdigest()

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
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
                provider_raw = choice["message"]["content"] or ""
                # LOCAL_POLICY_V2: no normalization.  The JSON-schema
                # constraint guarantees structural validity at the provider
                # level.  Provider output is passed directly to the strict
                # decoder.
                raw_output = provider_raw
                result = ModelCallResult(
                    raw_output=raw_output,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                    latency_ms=latency,
                    model_name=body.get("model", self.model_name),
                    system_fingerprint=body.get("system_fingerprint"),
                    finish_reason=choice.get("finish_reason"),
                    provider_raw_output=provider_raw,
                    json_schema_sha256=schema_sha,
                    system_prompt_sha256=prompt_sha,
                    user_packet_sha256=packet_sha,
                    request_sha256=request_sha,
                )
                self._call_counter += 1
                return result

            except urllib.error.HTTPError as exc:
                latency = int((time.monotonic() - start) * 1000)
                last_error = exc
                # HTTP 503 = server busy, retry
                if exc.code == 503 and attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_seconds * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"Local llama server returned HTTP {exc.code}: {exc}") from exc

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                latency = int((time.monotonic() - start) * 1000)
                last_error = exc
                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_seconds * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"Local llama server connection failed after "
                    f"{self.max_retries} retries: {exc}") from exc

        raise RuntimeError(
            f"Local llama server failed after {self.max_retries} retries: {last_error}")


@dataclass
class R2DirectLlamaBackend:
    """Direct llama-cpp-python backend with LlamaGrammar enforcement.

    Uses ``LlamaGrammar.from_json_schema()`` to enforce the action schema
    at generation time, physically preventing the model from generating
    actions outside the allowed set.

    Unlike ``LocalLlamaBackend`` (which uses a server and relies on
    ``response_format`` that may not be enforced), this backend calls
    ``llama-cpp-python`` directly and uses ``LlamaGrammar`` which is
    reliably enforced by the underlying llama.cpp engine.

    R2-DEV-V2: This is the canonical backend for strict schema-constrained
    generation.  The ``allowed_actions`` parameter is used to build a
    dynamic JSON schema, which is converted to a GBNF grammar via
    ``LlamaGrammar.from_json_schema()`` and passed to
    ``create_chat_completion()``.
    """

    model_name: str = "gemma-3-12b-it-qat-q4_0"
    model_path: str = ""
    n_gpu_layers: int = -1  # -1 = offload all to GPU
    n_ctx: int = 4096
    # Metadata for call receipts
    experiment_id: str = ""
    pair_id: str = ""
    task_id: str = ""
    condition: str = ""
    _call_counter: int = field(default=0, repr=False, init=False)
    _llm: Any = field(default=None, repr=False, init=False)

    # Full seven-action vocabulary for default schema.
    _FULL_ACTION_VOCAB = frozenset({
        "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
        "REASON_MORE", "DEFER", "STOP",
    })

    def _build_action_schema(self, allowed_actions: frozenset[str] | None) -> dict:
        """Build the JSON schema for constrained generation."""
        actions = allowed_actions if allowed_actions is not None else self._FULL_ACTION_VOCAB
        canonical_order = (
            "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
            "REASON_MORE", "DEFER", "STOP",
        )
        enum = [a for a in canonical_order if a in actions]
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": enum,
                },
                "reason_code": {
                    "type": "string",
                    "pattern": "^[A-Z][A-Z0-9_]*$",
                },
                "target_id": {"type": ["string", "null"]},
            },
            "required": ["action", "reason_code", "target_id"],
            "additionalProperties": False,
        }

    def _get_llm(self):
        """Lazily load the model on first use."""
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        return self._llm

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int,
                 allowed_actions: frozenset[str] | None = None) -> ModelCallResult:
        """Generate a model response with LlamaGrammar enforcement.

        The JSON schema is built from ``allowed_actions`` and converted to
        a GBNF grammar via ``LlamaGrammar.from_json_schema()``.  This
        physically prevents the model from generating actions outside the
        allowed set at the token level.
        """
        from llama_cpp import LlamaGrammar

        action_schema = self._build_action_schema(allowed_actions)
        schema_sha = hashlib.sha256(
            json.dumps(action_schema, sort_keys=True).encode()).hexdigest()
        prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()
        packet_sha = hashlib.sha256(user_prompt.encode()).hexdigest()

        # Build grammar from JSON schema
        grammar = LlamaGrammar.from_json_schema(json.dumps(action_schema))

        llm = self._get_llm()
        start = time.monotonic()

        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=40,
            repeat_penalty=1.0,
            seed=42,
            grammar=grammar,
        )

        latency = int((time.monotonic() - start) * 1000)
        choice = result["choices"][0]
        usage = result.get("usage", {})
        raw_output = choice["message"]["content"] or ""

        # Build request SHA for provenance
        request_payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "grammar_schema": action_schema,
        }
        request_sha = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True).encode()).hexdigest()

        self._call_counter += 1
        return ModelCallResult(
            raw_output=raw_output,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=0,
            latency_ms=latency,
            model_name=self.model_name,
            system_fingerprint=None,
            finish_reason=choice.get("finish_reason"),
            provider_raw_output=raw_output,
            json_schema_sha256=schema_sha,
            system_prompt_sha256=prompt_sha,
            user_packet_sha256=packet_sha,
            request_sha256=request_sha,
        )
