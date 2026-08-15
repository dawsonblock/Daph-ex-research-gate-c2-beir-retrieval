"""Frozen retry policy and call-receipt audit trail for I3.4.1.

The retry matrix is explicit: only transport-like failures are retried.
Invalid model output, HTTP 400, authentication errors, and schema failures
are NOT retried.  Every backend attempt receives its own append-only call
receipt.

Schema identity: ``DAPH_V2B_I3_4_RETRY_POLICY_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

RETRY_POLICY_SCHEMA = "DAPH_V2B_I3_4_RETRY_POLICY_V1"
RETRY_POLICY_VERSION = 1

# Retryable HTTP status codes (transport-like failures).
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

# Retryable exception type names.
RETRYABLE_EXCEPTIONS = frozenset({
    "TimeoutError", "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "ConnectionRefusedError", "OSError",
    "URLError", "HTTPError",
})

# Non-retryable HTTP status codes.
NON_RETRYABLE_HTTP_STATUS = frozenset({400, 401, 403, 404, 422})


@dataclass(frozen=True)
class RetryPolicy:
    """Explicit retry matrix for the hosted-model backend."""

    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    retryable_http_status: frozenset[int] = field(default_factory=lambda: RETRYABLE_HTTP_STATUS)
    retryable_exceptions: frozenset[str] = field(default_factory=lambda: RETRYABLE_EXCEPTIONS)
    non_retryable_http_status: frozenset[int] = field(default_factory=lambda: NON_RETRYABLE_HTTP_STATUS)
    policy_id: str = "v2b_i3_4_retry_policy_v1"

    def should_retry_http(self, status: int) -> bool:
        """Return True if an HTTP error with this status should be retried."""
        if status in self.non_retryable_http_status:
            return False
        return status in self.retryable_http_status

    def should_retry_exception(self, exc: Exception) -> bool:
        """Return True if this exception type should be retried."""
        return type(exc).__name__ in self.retryable_exceptions

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RETRY_POLICY_SCHEMA,
            "schema_version": RETRY_POLICY_VERSION,
            "policy_id": self.policy_id,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "retryable_http_status": sorted(self.retryable_http_status),
            "retryable_exceptions": sorted(self.retryable_exceptions),
            "non_retryable_http_status": sorted(self.non_retryable_http_status),
        }


FROZEN_RETRY_POLICY = RetryPolicy()


@dataclass(frozen=True)
class CallReceipt:
    """Append-only receipt for one backend API attempt.

    Never stores the API key.  Each attempt (including retries within a
    single generate() call) gets its own receipt.
    """

    call_id: str               # unique within the experiment
    pair_id: str               # counterbalanced pair identifier
    attempt_index: int         # 0 for first attempt, 1 for first retry, etc.
    task_id_hash: str          # SHA-256 of task_id (not the raw task_id)
    condition: str             # evaluator metadata, not sent to model
    request_sha256: str        # hash of the full request payload
    packet_sha256: str         # hash of the input packet
    prompt_sha256: str         # hash of the system prompt
    generation_config_sha256: str
    timestamp_utc: str         # ISO 8601 with microseconds
    result_class: str          # "success", "http_error", "timeout", "connection_error", "auth_error", "parse_error"
    http_status: int | None
    reported_model: str | None
    system_fingerprint: str | None
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str | None
    raw_output: str | None
    raw_output_sha256: str | None
    error_message: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "pair_id": self.pair_id,
            "attempt_index": self.attempt_index,
            "task_id_hash": self.task_id_hash,
            "condition": self.condition,
            "request_sha256": self.request_sha256,
            "packet_sha256": self.packet_sha256,
            "prompt_sha256": self.prompt_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "timestamp_utc": self.timestamp_utc,
            "result_class": self.result_class,
            "http_status": self.http_status,
            "reported_model": self.reported_model,
            "system_fingerprint": self.system_fingerprint,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "finish_reason": self.finish_reason,
            "raw_output": self.raw_output,
            "raw_output_sha256": self.raw_output_sha256,
            "error_message": self.error_message,
        }


def make_call_receipt(
    *,
    call_id: str,
    pair_id: str,
    attempt_index: int,
    task_id: str,
    condition: str,
    request_sha256: str,
    packet_sha256: str,
    prompt_sha256: str,
    generation_config_sha256: str,
    result_class: str,
    http_status: int | None = None,
    reported_model: str | None = None,
    system_fingerprint: str | None = None,
    latency_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    finish_reason: str | None = None,
    raw_output: str | None = None,
    error_message: str | None = None,
) -> CallReceipt:
    """Create a call receipt with a hashed task_id and raw_output SHA-256."""
    task_id_hash = hashlib.sha256(task_id.encode()).hexdigest()
    raw_output_sha = (hashlib.sha256(raw_output.encode()).hexdigest()
                      if raw_output is not None else None)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    return CallReceipt(
        call_id=call_id, pair_id=pair_id, attempt_index=attempt_index,
        task_id_hash=task_id_hash, condition=condition,
        request_sha256=request_sha256, packet_sha256=packet_sha256,
        prompt_sha256=prompt_sha256,
        generation_config_sha256=generation_config_sha256,
        timestamp_utc=timestamp, result_class=result_class,
        http_status=http_status, reported_model=reported_model,
        system_fingerprint=system_fingerprint, latency_ms=latency_ms,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens, finish_reason=finish_reason,
        raw_output=raw_output, raw_output_sha256=raw_output_sha,
        error_message=error_message,
    )


def retry_policy_sha256() -> str:
    """Canonical SHA-256 of the frozen retry policy."""
    return hashlib.sha256(
        json.dumps(FROZEN_RETRY_POLICY.as_dict(), sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()
