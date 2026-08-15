"""Tests for the I3.4.1 wired backend integration.

These tests verify that:
- DeepSeekBackend builds requests that include thinking and response_format
  from the frozen generation config (B01).
- DeepSeekBackend uses the frozen retry policy for HTTP and exception
  handling (B02).
- DeepSeekBackend emits a CallReceipt for every attempt (B03).
- The strict decoder mode rejects prose-wrapped JSON (B04).
- The identity policy binds the real generation config hash (B06).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hrm_adaptive_memory.executive.i3_4_generation_config import FROZEN_CONFIG
from hrm_adaptive_memory.executive.i3_4_retry_policy import FROZEN_RETRY_POLICY
from hrm_adaptive_memory.executive.i3_4_model_identity_policy import (
    FROZEN_IDENTITY_POLICY)
from hrm_adaptive_memory.executive.model_backend import (
    DeepSeekBackend, StubBackend, _build_request_payload)
from hrm_adaptive_memory.executive.model_decoder import decode_output


# --- B01: Generation config wired into request payload ---


def test_request_payload_includes_thinking_disabled():
    """The API request must explicitly disable thinking."""
    payload = _build_request_payload(
        model="deepseek-v4-flash",
        system_prompt="system", user_prompt="user",
        temperature=0.0, max_tokens=2048,
        thinking_mode="disabled",
        response_format="json_object",
    )
    parsed = json.loads(payload)
    assert parsed["thinking"] == {"type": "disabled"}


def test_request_payload_includes_response_format_json():
    """The API request must include response_format json_object."""
    payload = _build_request_payload(
        model="deepseek-v4-flash",
        system_prompt="system", user_prompt="user",
        temperature=0.0, max_tokens=2048,
        thinking_mode="disabled",
        response_format="json_object",
    )
    parsed = json.loads(payload)
    assert parsed["response_format"] == {"type": "json_object"}


def test_request_payload_includes_model_and_temperature():
    payload = _build_request_payload(
        model="deepseek-v4-flash",
        system_prompt="s", user_prompt="u",
        temperature=0.0, max_tokens=2048,
        thinking_mode="disabled",
        response_format="json_object",
    )
    parsed = json.loads(payload)
    assert parsed["model"] == "deepseek-v4-flash"
    assert parsed["temperature"] == 0.0
    assert parsed["max_tokens"] == 2048


def test_request_payload_includes_messages():
    payload = _build_request_payload(
        model="deepseek-v4-flash",
        system_prompt="system prompt", user_prompt="user prompt",
        temperature=0.0, max_tokens=2048,
        thinking_mode="disabled",
        response_format="json_object",
    )
    parsed = json.loads(payload)
    assert len(parsed["messages"]) == 2
    assert parsed["messages"][0]["role"] == "system"
    assert parsed["messages"][0]["content"] == "system prompt"
    assert parsed["messages"][1]["role"] == "user"
    assert parsed["messages"][1]["content"] == "user prompt"


def test_backend_uses_frozen_config_model():
    """DeepSeekBackend must use the model from the frozen config."""
    backend = DeepSeekBackend()
    assert backend.model_name == FROZEN_CONFIG.model
    assert backend.config.thinking_mode == "disabled"
    assert backend.config.response_format == "json_object"


# --- B02: Retry policy wired into backend ---


def test_backend_uses_frozen_retry_policy():
    """DeepSeekBackend must use the frozen retry policy by default."""
    backend = DeepSeekBackend()
    assert backend.retry_policy is FROZEN_RETRY_POLICY
    assert backend.retry_policy.max_retries == 3


def test_backend_does_not_retry_http_400():
    """HTTP 400 must not be retried (non-retryable per frozen policy)."""
    backend = DeepSeekBackend()
    assert not backend.retry_policy.should_retry_http(400)


def test_backend_does_not_retry_http_401():
    """HTTP 401 must not be retried."""
    backend = DeepSeekBackend()
    assert not backend.retry_policy.should_retry_http(401)


# --- B03: Call receipts emitted by backend ---


def test_backend_emits_receipt_on_success():
    """DeepSeekBackend must emit a CallReceipt on successful API calls."""
    backend = DeepSeekBackend(_api_key="test-key")
    # Mock the urllib response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": '{"action":"ANSWER","reason_code":"SUFFICIENT","target_id":null}'},
                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "deepseek-v4-flash",
        "system_fingerprint": "fp_test_123",
    }).encode()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = backend.generate(
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_tokens=2048)

    assert len(backend.call_receipts) == 1
    receipt = backend.call_receipts[0]
    assert receipt.result_class == "success"
    assert receipt.http_status == 200
    assert receipt.system_fingerprint == "fp_test_123"
    assert receipt.reported_model == "deepseek-v4-flash"
    assert receipt.raw_output is not None
    assert receipt.raw_output_sha256 is not None
    # Task ID should be hashed, not stored raw
    assert "test" not in receipt.task_id_hash or len(receipt.task_id_hash) == 64


def test_backend_emits_receipt_on_http_error():
    """DeepSeekBackend must emit a CallReceipt on HTTP errors."""
    import urllib.error
    backend = DeepSeekBackend(_api_key="test-key")

    with patch("urllib.request.urlopen",
               side_effect=urllib.error.HTTPError(
                   url="http://test", code=400,
                   msg="Bad Request", hdrs={}, fp=None)):
        with pytest.raises(RuntimeError, match="non-retryable"):
            backend.generate(
                system_prompt="s", user_prompt="u",
                temperature=0.0, max_tokens=2048)

    # Should have one receipt for the failed attempt
    assert len(backend.call_receipts) == 1
    receipt = backend.call_receipts[0]
    assert receipt.result_class == "http_error"
    assert receipt.http_status == 400


def test_backend_emits_receipt_on_timeout():
    """DeepSeekBackend must emit a CallReceipt on timeout."""
    backend = DeepSeekBackend(_api_key="test-key")

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(RuntimeError):  # Will retry then fail
            backend.generate(
                system_prompt="s", user_prompt="u",
                temperature=0.0, max_tokens=2048)

    # Should have receipts for each retry attempt
    assert len(backend.call_receipts) == FROZEN_RETRY_POLICY.max_retries
    for receipt in backend.call_receipts:
        assert receipt.result_class == "timeout"


def test_backend_receipts_never_contain_api_key():
    """Call receipts must never contain the API key."""
    backend = DeepSeekBackend(_api_key="secret-key-12345")
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": '{"action":"ANSWER","reason_code":"SUFFICIENT","target_id":null}'},
                      "finish_reason": "stop"}],
        "usage": {},
        "model": "deepseek-v4-flash",
        "system_fingerprint": "fp_test",
    }).encode()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        backend.generate(
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_tokens=2048)

    for receipt in backend.call_receipts:
        receipt_str = json.dumps(receipt.as_dict())
        assert "secret-key-12345" not in receipt_str


# --- B04: Strict decoder mode ---


def test_strict_decoder_accepts_pure_json():
    """Strict mode should accept a pure JSON response."""
    raw = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    result = decode_output(raw, strict=True)
    assert result.valid
    assert result.proposal is not None
    assert result.proposal.action.value == "ANSWER"


def test_strict_decoder_rejects_prose_wrapped_json():
    """Strict mode should reject JSON embedded in prose."""
    raw = 'I think the answer is clear. {"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null} Done.'
    result = decode_output(raw, strict=True)
    assert not result.valid
    assert result.rejection_code == "STRICT_MODE_NOT_PURE_JSON"


def test_development_decoder_accepts_prose_wrapped_json():
    """Development mode (strict=False) should still accept prose-wrapped JSON."""
    raw = 'I think the answer is clear. {"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null} Done.'
    result = decode_output(raw, strict=False)
    assert result.valid
    assert result.proposal is not None
    assert result.proposal.action.value == "ANSWER"


# --- B06: Identity policy binds real generation config hash ---


def test_identity_policy_generation_config_hash_is_bound():
    """The identity policy must bind the actual generation config hash."""
    assert FROZEN_IDENTITY_POLICY.generation_config_sha256 != ""
    assert FROZEN_IDENTITY_POLICY.generation_config_sha256 == FROZEN_CONFIG.sha256()


# --- Controller uses strict mode by default ---


def test_controller_defaults_to_strict_json():
    """The PinnedModelController should default to strict_json=True."""
    from hrm_adaptive_memory.executive.pinned_model_controller import (
        PinnedModelController)
    controller = PinnedModelController()
    assert controller.strict_json is True
