"""Tests for I3.4.1 generation config, retry policy, and pair scheduler."""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.executive.i3_4_generation_config import (
    FROZEN_CONFIG, GENERATION_CONFIG_SCHEMA, config_sha256, save_config)
from hrm_adaptive_memory.executive.i3_4_retry_policy import (
    FROZEN_RETRY_POLICY, RETRY_POLICY_SCHEMA, CallReceipt, make_call_receipt,
    retry_policy_sha256)
from hrm_adaptive_memory.executive.i3_4_pair_scheduler import (
    AWARE_FIRST, BLIND_FIRST, SCHEDULER_SCHEMA, build_pair_schedule,
    check_pair_fingerprints, compute_pair_hash, is_blind_first)


# --- Generation config ---

def test_generation_config_schema():
    assert GENERATION_CONFIG_SCHEMA == "DAPH_V2B_I3_4_FROZEN_GENERATION_CONFIG_V1"


def test_frozen_config_thinking_disabled():
    assert FROZEN_CONFIG.thinking_mode == "disabled"
    assert FROZEN_CONFIG.reasoning_effort is None


def test_frozen_config_json_response_format():
    assert FROZEN_CONFIG.response_format == "json_object"


def test_frozen_config_has_sha256():
    h = config_sha256()
    assert len(h) == 64


def test_frozen_config_is_immutable():
    with pytest.raises(Exception):
        FROZEN_CONFIG.model = "other"  # type: ignore


# --- Retry policy ---

def test_retry_policy_schema():
    assert RETRY_POLICY_SCHEMA == "DAPH_V2B_I3_4_RETRY_POLICY_V1"


def test_retry_policy_retries_429():
    assert FROZEN_RETRY_POLICY.should_retry_http(429) is True


def test_retry_policy_retries_500():
    assert FROZEN_RETRY_POLICY.should_retry_http(500) is True


def test_retry_policy_does_not_retry_400():
    assert FROZEN_RETRY_POLICY.should_retry_http(400) is False


def test_retry_policy_does_not_retry_401():
    assert FROZEN_RETRY_POLICY.should_retry_http(401) is False


def test_retry_policy_retries_timeout():
    assert FROZEN_RETRY_POLICY.should_retry_exception(TimeoutError()) is True


def test_retry_policy_retries_connection_error():
    assert FROZEN_RETRY_POLICY.should_retry_exception(ConnectionError()) is True


def test_retry_policy_has_sha256():
    h = retry_policy_sha256()
    assert len(h) == 64


def test_call_receipt_hashes_task_id():
    receipt = make_call_receipt(
        call_id="c1", pair_id="p1", attempt_index=0,
        task_id="secret-task-1", condition="STATE_BLIND_CONTROLLER",
        request_sha256="abc", packet_sha256="def",
        prompt_sha256="ghi", generation_config_sha256="jkl",
        result_class="success", raw_output='{"action":"ANSWER"}')
    # task_id should be hashed, not stored raw
    assert "secret-task-1" not in receipt.task_id_hash
    assert len(receipt.task_id_hash) == 64
    # raw_output should have a SHA-256
    assert receipt.raw_output_sha256 is not None
    assert len(receipt.raw_output_sha256) == 64


def test_call_receipt_never_contains_api_key():
    receipt = make_call_receipt(
        call_id="c1", pair_id="p1", attempt_index=0,
        task_id="t1", condition="STATE_BLIND_CONTROLLER",
        request_sha256="abc", packet_sha256="def",
        prompt_sha256="ghi", generation_config_sha256="jkl",
        result_class="success")
    d = receipt.as_dict()
    assert "api_key" not in str(d).lower()
    assert "key" not in str(d).lower() or "task_id_hash" in str(d)


# --- Pair scheduler ---

def test_scheduler_schema():
    assert SCHEDULER_SCHEMA == "DAPH_V2B_I3_4_PAIR_SCHEDULER_V1"


def test_pair_hash_is_deterministic():
    h1 = compute_pair_hash("exp1", "task1")
    h2 = compute_pair_hash("exp1", "task1")
    assert h1 == h2


def test_pair_hash_differs_by_experiment():
    h1 = compute_pair_hash("exp1", "task1")
    h2 = compute_pair_hash("exp2", "task1")
    assert h1 != h2


def test_is_blind_first_is_deterministic():
    b1 = is_blind_first("exp1", "task1")
    b2 = is_blind_first("exp1", "task1")
    assert b1 == b2


def test_build_pair_schedule_counterbalanced():
    """The schedule should have a mix of BLIND_FIRST and AWARE_FIRST."""
    task_ids = [f"task_{i:04d}" for i in range(100)]
    schedules = build_pair_schedule(
        experiment_id="exp1", task_ids=task_ids)
    assert len(schedules) == 100
    blind_first_count = sum(1 for s in schedules if s.pair_order == BLIND_FIRST)
    aware_first_count = sum(1 for s in schedules if s.pair_order == AWARE_FIRST)
    # With 100 tasks, we expect roughly 50/50 split
    assert blind_first_count > 0
    assert aware_first_count > 0
    # Each schedule should have first != second
    for s in schedules:
        assert s.first_condition != s.second_condition


def test_build_pair_schedule_deterministic():
    task_ids = [f"task_{i}" for i in range(10)]
    s1 = build_pair_schedule(experiment_id="exp1", task_ids=task_ids)
    s2 = build_pair_schedule(experiment_id="exp1", task_ids=task_ids)
    for a, b in zip(s1, s2):
        assert a.pair_id == b.pair_id
        assert a.pair_order == b.pair_order


def test_check_pair_fingerprints_match():
    record = check_pair_fingerprints(
        pair_id="p1",
        first_call_fingerprint="fp_abc",
        second_call_fingerprint="fp_abc")
    assert record.fingerprint_match is True
    assert record.pair_valid is True


def test_check_pair_fingerprints_mismatch():
    record = check_pair_fingerprints(
        pair_id="p1",
        first_call_fingerprint="fp_abc",
        second_call_fingerprint="fp_xyz")
    assert record.fingerprint_match is False
    assert record.pair_valid is False


def test_check_pair_fingerprints_none_is_valid():
    """If either fingerprint is None, pair is valid (no evidence of drift)."""
    record = check_pair_fingerprints(
        pair_id="p1",
        first_call_fingerprint=None,
        second_call_fingerprint="fp_abc")
    assert record.pair_valid is True
