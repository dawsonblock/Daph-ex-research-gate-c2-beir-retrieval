"""Tests for the I3.4.1 paired experiment runner."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hrm_adaptive_memory.executive.i3_4_paired_runner import (
    PairedExperimentRunner, PairedTaskResult, RUNNER_SCHEMA)
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.metareasoning_controller import (
    ControllerObservation)
from hrm_adaptive_memory.cognitive_control.core import DecisionAction


def _make_observation(task_id: str = "test_task") -> ControllerObservation:
    return ControllerObservation(
        task_id=task_id, task_summary="Test task",
        resource_state={"executive_steps_used": 0, "executive_steps_remaining": 10},
        allowed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY,
                         DecisionAction.ANSWER, DecisionAction.DEFER),
        executed_actions=(),
        rejected_actions=(),
        cognitive_state=None)


def _mock_api_response(content: str, fingerprint: str = "fp_test_123"):
    """Create a mock urllib response."""
    import json
    mock = MagicMock()
    mock.read.return_value = json.dumps({
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "deepseek-v4-flash",
        "system_fingerprint": fingerprint,
    }).encode()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_runner_schema_is_frozen():
    assert RUNNER_SCHEMA == "DAPH_V2B_I3_4_PAIRED_RUNNER_V1"


def test_runner_uses_deepseek_backend():
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)
    assert runner.backend is backend
    assert runner.strict_json is True


def test_runner_pair_produces_two_receipts():
    """Each pair must produce exactly two call receipts."""
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)

    valid_json = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    with patch("urllib.request.urlopen",
               return_value=_mock_api_response(valid_json)):
        result = runner.run_pair(
            task_id="test_task_001",
            blind_observation=_make_observation("test_task_001"),
            aware_observation=_make_observation("test_task_001"))

    assert len(runner.backend.call_receipts) == 2
    assert result.task_id == "test_task_001"
    assert result.blind_result.condition == "BLIND"
    assert result.aware_result.condition == "AWARE"


def test_runner_pair_identity_valid_with_matching_fingerprints():
    """A pair with matching fingerprints should be identity-valid."""
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)

    valid_json = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    with patch("urllib.request.urlopen",
               return_value=_mock_api_response(valid_json, fingerprint="fp_match")):
        result = runner.run_pair(
            task_id="test_task_002",
            blind_observation=_make_observation("test_task_002"),
            aware_observation=_make_observation("test_task_002"))

    assert result.identity_valid is True
    assert result.fingerprint_record.fingerprint_match is True


def test_runner_pair_identity_invalid_with_mismatched_fingerprints():
    """A pair with mismatched fingerprints should be identity-invalid."""
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)

    valid_json = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    responses = [
        _mock_api_response(valid_json, fingerprint="fp_A"),
        _mock_api_response(valid_json, fingerprint="fp_B"),
    ]
    with patch("urllib.request.urlopen", side_effect=responses):
        result = runner.run_pair(
            task_id="test_task_003",
            blind_observation=_make_observation("test_task_003"),
            aware_observation=_make_observation("test_task_003"))

    assert result.identity_valid is False
    assert result.fingerprint_record.pair_valid is False


def test_runner_pair_identity_invalid_with_missing_fingerprint():
    """A pair with missing fingerprint should be identity-invalid (require_fingerprint=True)."""
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)

    valid_json = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    # No system_fingerprint in the response
    mock = MagicMock()
    import json
    mock.read.return_value = json.dumps({
        "choices": [{"message": {"content": valid_json}, "finish_reason": "stop"}],
        "usage": {},
        "model": "deepseek-v4-flash",
    }).encode()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock):
        result = runner.run_pair(
            task_id="test_task_004",
            blind_observation=_make_observation("test_task_004"),
            aware_observation=_make_observation("test_task_004"))

    assert result.identity_valid is False


def test_runner_scheduler_determines_order():
    """The scheduler should determine BLIND→AWARE or AWARE→BLIND."""
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)

    valid_json = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    with patch("urllib.request.urlopen",
               return_value=_mock_api_response(valid_json)):
        result = runner.run_pair(
            task_id="test_task_005",
            blind_observation=_make_observation("test_task_005"),
            aware_observation=_make_observation("test_task_005"))

    # The schedule must specify which condition goes first
    assert result.schedule.first_condition in ("STATE_BLIND_CONTROLLER", "STATE_AWARE_CONTROLLER")
    assert result.schedule.second_condition in ("STATE_BLIND_CONTROLLER", "STATE_AWARE_CONTROLLER")
    assert result.schedule.first_condition != result.schedule.second_condition


def test_runner_strict_json_rejects_prose():
    """The runner should use strict JSON mode by default."""
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)

    prose_json = 'I think the answer is clear. {"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null} Done.'
    with patch("urllib.request.urlopen",
               return_value=_mock_api_response(prose_json)):
        result = runner.run_pair(
            task_id="test_task_006",
            blind_observation=_make_observation("test_task_006"),
            aware_observation=_make_observation("test_task_006"))

    # Both calls should fail closed because strict mode rejects prose
    for call_result in [result.blind_result, result.aware_result]:
        assert call_result.decoder_outcome is not None
        assert not call_result.decoder_outcome.valid
        assert call_result.decoder_outcome.rejection_code == "STRICT_MODE_NOT_PURE_JSON"
        assert call_result.proposal.action.value == "DEFER"


def test_runner_summary():
    """The runner summary should report correct counts."""
    backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=backend)

    valid_json = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    with patch("urllib.request.urlopen",
               return_value=_mock_api_response(valid_json)):
        runner.run_pair(
            task_id="test_task_007",
            blind_observation=_make_observation("test_task_007"),
            aware_observation=_make_observation("test_task_007"))

    summary = runner.runner_summary()
    assert summary["schema"] == RUNNER_SCHEMA
    assert summary["pairs_completed"] == 1
    assert summary["total_receipts"] == 2
    assert summary["strict_json"] is True


def test_runner_receipts_never_contain_api_key():
    """Call receipts must never contain the API key."""
    backend = DeepSeekBackend(_api_key="secret-key-abc123")
    runner = PairedExperimentRunner(backend=backend)

    valid_json = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    with patch("urllib.request.urlopen",
               return_value=_mock_api_response(valid_json)):
        runner.run_pair(
            task_id="test_task_008",
            blind_observation=_make_observation("test_task_008"),
            aware_observation=_make_observation("test_task_008"))

    for receipt in runner.all_receipts():
        receipt_str = str(receipt.as_dict())
        assert "secret-key-abc123" not in receipt_str
