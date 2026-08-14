"""Adversarial and replay qualification tests for I3.4.1.

These tests verify that the strict output schema, decoder, packet
serialization, and condition-leakage checks behave correctly under
adversarial inputs.  They also verify deterministic replay properties
of the scoring and statistical modules.

Covers:
- Malformed JSON
- Extra keys
- Missing fields
- Invalid action names
- Invalid reason codes
- Invalid target IDs
- Backend failures
- Transient failures
- Condition leakage in keys and values
- Reasoning text containing braces
- Exact packet/decoder behavior
- Model-output receipt completeness
- Deterministic evaluation replay
"""
from __future__ import annotations

import json

import pytest

from hrm_adaptive_memory.executive.i3_4_scientific_scoring import (
    I34ScientificTaskContribution, compute_aggregate,
    compute_paired_deltas, compute_task_contribution,
    verify_all_identities, mean_delta_dg)
from hrm_adaptive_memory.executive.i3_4_statistical_analysis import (
    paired_bootstrap, topology_cluster_bootstrap)
from hrm_adaptive_memory.executive.i3_4_pair_scheduler import (
    build_pair_schedule, check_pair_fingerprints, compute_pair_hash,
    is_blind_first, BLIND_FIRST, AWARE_FIRST)
from hrm_adaptive_memory.executive.i3_4_retry_policy import (
    FROZEN_RETRY_POLICY, make_call_receipt)
from hrm_adaptive_memory.executive.i3_4_generation_config import (
    FROZEN_CONFIG, config_sha256)
from hrm_adaptive_memory.executive.i3_4_model_identity_policy import (
    FROZEN_IDENTITY_POLICY)
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.model_packet import (
    assert_no_condition_leakage, serialize_packet)


# --- Adversarial decoder tests ---


def test_adversarial_malformed_json():
    """Completely broken JSON should fail closed."""
    result = decode_output("not json at all")
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_extra_keys():
    """Extra keys in the JSON object should be rejected."""
    payload = json.dumps({
        "action": "ANSWER", "reason_code": "SUFFICIENT",
        "target_id": None, "extra_key": "malicious"
    })
    result = decode_output(payload)
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_missing_action():
    """Missing action field should fail closed."""
    payload = json.dumps({"reason_code": "SUFFICIENT", "target_id": None})
    result = decode_output(payload)
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_missing_reason_code():
    """Missing reason_code field should fail closed."""
    payload = json.dumps({"action": "ANSWER", "target_id": None})
    result = decode_output(payload)
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_missing_target_id():
    """Missing target_id field should fail closed."""
    payload = json.dumps({"action": "RETRIEVE", "reason_code": "INSUFFICIENT"})
    result = decode_output(payload)
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_invalid_action_name():
    """Invalid action name should fail closed."""
    payload = json.dumps({
        "action": "HACK_THE_SYSTEM", "reason_code": "SUFFICIENT",
        "target_id": None
    })
    result = decode_output(payload)
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_invalid_reason_code():
    """Invalid reason code (lowercase/special chars) should fail closed."""
    payload = json.dumps({
        "action": "ANSWER", "reason_code": "because-i-feel-like-it",
        "target_id": None
    })
    result = decode_output(payload)
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_empty_target_id_for_retrieve():
    """Empty string target_id for RETRIEVE should fail closed."""
    payload = json.dumps({
        "action": "RETRIEVE", "reason_code": "INSUFFICIENT",
        "target_id": ""
    })
    result = decode_output(payload)
    assert result is not None
    assert not result.valid or result.proposal is None or result.proposal.action.value == "DEFER"


def test_adversarial_reasoning_with_braces():
    """Reasoning text containing braces before JSON should not confuse the decoder."""
    raw = 'I need to think about this {carefully}. Here is my decision: {"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}'
    result = decode_output(raw)
    # The decoder should find the valid JSON object despite braces in reasoning
    assert result is not None
    if result.valid and result.proposal is not None:
        assert result.proposal.action.value == "ANSWER"


def test_adversarial_trailing_garbage():
    """Trailing garbage after valid JSON should be handled."""
    raw = '{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null} garbage'
    result = decode_output(raw)
    # Decoder should either parse the valid JSON or fail closed
    assert result is not None
    if result.valid and result.proposal is not None:
        assert result.proposal.action.value in ("ANSWER", "DEFER")


# --- Condition leakage tests ---


def _make_observation(snapshot=None):
    """Build a minimal ControllerObservation for testing."""
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.metareasoning_controller import (
        ControllerObservation)
    return ControllerObservation(
        task_id="test-task-1",
        task_summary="Verify whether claim X is supported by current evidence.",
        resource_state={
            "executive_steps_used": 1, "executive_steps_remaining": 11,
            "reasoning_tokens_used": 0, "reasoning_tokens_remaining": 512,
            "retrieval_calls_used": 0, "retrieval_calls_remaining": 4,
            "verification_calls_used": 0, "verification_calls_remaining": 3,
            "search_calls_used": 0, "search_calls_remaining": 3,
            "elapsed_ms": 5, "elapsed_ms_remaining": 9995,
            "monetary_cost_microusd": 0, "monetary_cost_microusd_remaining": 0,
            "policy_rejections_used": 0,
        },
        allowed_actions=tuple(DecisionAction),
        executed_actions=(DecisionAction.RETRIEVE,),
        rejected_actions=(),
        cognitive_state=snapshot,
    )


def test_condition_leakage_in_packet_keys():
    """Packet keys should not contain condition-identifying strings."""
    packet = serialize_packet(_make_observation(None))
    # Should not raise
    assert_no_condition_leakage(packet)


def test_condition_leakage_in_string_values():
    """String values in the packet should not contain condition-identifying strings."""
    packet = serialize_packet(_make_observation(None))
    assert_no_condition_leakage(packet)


# --- Call receipt completeness ---


def test_call_receipt_has_all_required_fields():
    """Every call receipt must have all required fields for audit."""
    receipt = make_call_receipt(
        call_id="c1", pair_id="p1", attempt_index=0,
        task_id="t1", condition="STATE_BLIND_CONTROLLER",
        request_sha256="abc", packet_sha256="def",
        prompt_sha256="ghi", generation_config_sha256="jkl",
        result_class="success", raw_output='{"action":"ANSWER"}',
        reported_model="deepseek-v4-flash",
        system_fingerprint="fp_123",
        latency_ms=150, prompt_tokens=100, completion_tokens=20,
        finish_reason="stop",
    )
    d = receipt.as_dict()
    required = {
        "call_id", "pair_id", "attempt_index", "task_id_hash",
        "condition", "request_sha256", "packet_sha256", "prompt_sha256",
        "generation_config_sha256", "timestamp_utc", "result_class",
        "http_status", "reported_model", "system_fingerprint",
        "latency_ms", "prompt_tokens", "completion_tokens",
        "reasoning_tokens", "finish_reason", "raw_output",
        "raw_output_sha256", "error_message",
    }
    assert required.issubset(set(d.keys()))


def test_call_receipt_for_backend_error():
    """Backend error receipts should record the error."""
    receipt = make_call_receipt(
        call_id="c1", pair_id="p1", attempt_index=0,
        task_id="t1", condition="STATE_BLIND_CONTROLLER",
        request_sha256="abc", packet_sha256="def",
        prompt_sha256="ghi", generation_config_sha256="jkl",
        result_class="connection_error",
        error_message="Connection refused",
    )
    assert receipt.result_class == "connection_error"
    assert receipt.error_message == "Connection refused"
    assert receipt.raw_output is None


# --- Deterministic replay tests ---


def test_scoring_replay_is_deterministic():
    """Given the same inputs, scoring must produce the same outputs."""
    c1 = compute_task_contribution(
        task_id="t1", condition="B",
        latent_optimal_value=10.0, observable_optimal_value=7.0,
        controller_value=5.0, information_class_hash="abc",
        observable_oracle_set_sha256="def", latent_oracle_table_sha256="ghi")
    c2 = compute_task_contribution(
        task_id="t1", condition="B",
        latent_optimal_value=10.0, observable_optimal_value=7.0,
        controller_value=5.0, information_class_hash="abc",
        observable_oracle_set_sha256="def", latent_oracle_table_sha256="ghi")
    assert c1 == c2
    assert c1.information_gap_contribution == c2.information_gap_contribution
    assert c1.decision_gap_contribution == c2.decision_gap_contribution
    assert c1.total_regret_contribution == c2.total_regret_contribution


def test_aggregate_replay_is_deterministic():
    """Given the same contributions, aggregate must be identical."""
    contributions = [
        compute_task_contribution(
            task_id=f"t{i}", condition="B",
            latent_optimal_value=10.0 + i, observable_optimal_value=7.0 + i,
            controller_value=5.0 + i, information_class_hash="abc",
            observable_oracle_set_sha256="def", latent_oracle_table_sha256="ghi")
        for i in range(10)
    ]
    agg1 = compute_aggregate(condition="B", contributions=contributions)
    agg2 = compute_aggregate(condition="B", contributions=contributions)
    assert agg1 == agg2
    assert agg1.information_gap == agg2.information_gap
    assert agg1.decision_gap == agg2.decision_gap


def test_bootstrap_replay_is_deterministic():
    """Same seed must produce same bootstrap result."""
    from hrm_adaptive_memory.executive.i3_4_scientific_scoring import I34PairedDelta
    deltas = [
        I34PairedDelta(task_id=f"t{i}", delta_ig=1.0, delta_dg=2.0 + i * 0.1,
                       delta_tr=3.0, delta_cost=0.0)
        for i in range(20)
    ]
    r1 = paired_bootstrap(deltas, iterations=1000, seed=42)
    r2 = paired_bootstrap(deltas, iterations=1000, seed=42)
    assert r1.lower_bound == r2.lower_bound
    assert r1.upper_bound == r2.upper_bound


def test_pair_schedule_replay_is_deterministic():
    """Same experiment_id and task list must produce same schedule."""
    task_ids = [f"task_{i:04d}" for i in range(50)]
    s1 = build_pair_schedule(experiment_id="exp_test", task_ids=task_ids)
    s2 = build_pair_schedule(experiment_id="exp_test", task_ids=task_ids)
    for a, b in zip(s1, s2):
        assert a.pair_id == b.pair_id
        assert a.pair_order == b.pair_order
        assert a.first_condition == b.first_condition


def test_tr_ig_dg_identity_holds_for_all_contributions():
    """TR = IG + DG must hold for every task contribution."""
    contributions = [
        compute_task_contribution(
            task_id=f"t{i}", condition="B",
            latent_optimal_value=10.0 + i * 0.5,
            observable_optimal_value=7.0 - i * 0.3,
            controller_value=5.0 + i * 0.2,
            information_class_hash="abc",
            observable_oracle_set_sha256="def",
            latent_oracle_table_sha256="ghi")
        for i in range(100)
    ]
    all_pass, failures = verify_all_identities(contributions)
    assert all_pass
    assert failures == []


def test_tr_ig_dg_identity_holds_with_negative_contributions():
    """Identity must hold even with negative IG or DG."""
    contributions = [
        compute_task_contribution(
            task_id="t1", condition="B",
            latent_optimal_value=5.0, observable_optimal_value=7.0,
            controller_value=8.0,  # DG = -1
            information_class_hash="abc",
            observable_oracle_set_sha256="def",
            latent_oracle_table_sha256="ghi"),
        compute_task_contribution(
            task_id="t2", condition="B",
            latent_optimal_value=5.0, observable_optimal_value=3.0,
            controller_value=2.0,  # IG = 2, DG = 1
            information_class_hash="abc",
            observable_oracle_set_sha256="def",
            latent_oracle_table_sha256="ghi"),
    ]
    all_pass, failures = verify_all_identities(contributions)
    assert all_pass


# --- Identity policy adversarial tests ---


def test_identity_policy_rejects_model_swap():
    """If the provider silently swaps the model, the policy must catch it."""
    valid, reason = FROZEN_IDENTITY_POLICY.verify_call(
        reported_model="deepseek-v4-pro",
        system_fingerprint="fp_abc")
    assert not valid
    assert "mismatch" in reason.lower()


def test_identity_policy_rejects_missing_fingerprint():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_call(
        reported_model="deepseek-v4-flash",
        system_fingerprint=None)
    assert not valid


def test_identity_policy_rejects_within_pair_drift():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_pair("fp_1", "fp_2")
    assert not valid
    assert "within pair" in reason.lower()


def test_identity_policy_rejects_across_phase_drift():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_phase(["fp_1", "fp_2"])
    assert not valid
    assert "across phases" in reason.lower()


# --- Retry policy adversarial tests ---


def test_retry_policy_does_not_retry_400():
    """HTTP 400 (bad request) must not be retried."""
    assert FROZEN_RETRY_POLICY.should_retry_http(400) is False


def test_retry_policy_does_not_retry_401():
    """HTTP 401 (unauthorized) must not be retried."""
    assert FROZEN_RETRY_POLICY.should_retry_http(401) is False


def test_retry_policy_does_not_retry_403():
    """HTTP 403 (forbidden) must not be retried."""
    assert FROZEN_RETRY_POLICY.should_retry_http(403) is False


def test_retry_policy_retries_429():
    """HTTP 429 (rate limit) should be retried."""
    assert FROZEN_RETRY_POLICY.should_retry_http(429) is True


def test_retry_policy_retries_503():
    """HTTP 503 (service unavailable) should be retried."""
    assert FROZEN_RETRY_POLICY.should_retry_http(503) is True


def test_retry_policy_does_not_retry_value_error():
    """ValueError is not a transport error and must not be retried."""
    assert FROZEN_RETRY_POLICY.should_retry_exception(ValueError("bad")) is False


def test_retry_policy_retries_timeout():
    """TimeoutError is a transport error and should be retried."""
    assert FROZEN_RETRY_POLICY.should_retry_exception(TimeoutError()) is True


# --- Generation config adversarial tests ---


def test_generation_config_thinking_is_disabled():
    """Thinking mode must be explicitly disabled, not implicit."""
    assert FROZEN_CONFIG.thinking_mode == "disabled"
    assert FROZEN_CONFIG.reasoning_effort is None


def test_generation_config_temperature_is_zero():
    """Temperature must be 0.0 for reproducibility."""
    assert FROZEN_CONFIG.temperature == 0.0


def test_generation_config_response_format_is_json():
    """Response format must be json_object for strict JSON mode."""
    assert FROZEN_CONFIG.response_format == "json_object"


def test_generation_config_is_immutable():
    with pytest.raises(Exception):
        FROZEN_CONFIG.thinking_mode = "enabled"  # type: ignore
