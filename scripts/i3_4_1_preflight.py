"""I3.4.1 development preflight check.

Verifies that all frozen controls are wired into the execution path
WITHOUT making a real DeepSeek API call.  This is the final gate before
a development-only DeepSeek repeatability run.

Checks:
1. FrozenGenerationConfig is bound to the backend.
2. Request payload includes thinking disabled and JSON mode.
3. Frozen retry policy is used by the backend.
4. CallReceipts are emitted for every attempt.
5. Strict JSON decoder rejects prose.
6. Identity policy binds the real generation config hash.
7. Fingerprint semantics are consistent (missing = invalid when required).
8. Observable oracle views exist and differ across splits.
9. Full experiment identity exists and binds all components.
10. Paired runner produces two receipts per pair.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def run_preflight() -> int:
    """Run all preflight checks. Returns 0 if all pass, 1 otherwise."""
    root = Path(__file__).resolve().parents[1]
    all_pass = True

    print("I3.4.1 Development Preflight Check")
    print("=" * 50)

    # 1. FrozenGenerationConfig bound to backend
    print("\n1. Generation config wiring")
    from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
    from hrm_adaptive_memory.executive.i3_4_generation_config import FROZEN_CONFIG
    backend = DeepSeekBackend()
    all_pass &= check(
        "Backend uses frozen config model",
        backend.model_name == FROZEN_CONFIG.model,
        f"backend={backend.model_name}, config={FROZEN_CONFIG.model}")
    all_pass &= check(
        "Backend uses frozen config thinking_mode",
        backend.config.thinking_mode == "disabled",
        f"thinking_mode={backend.config.thinking_mode}")
    all_pass &= check(
        "Backend uses frozen config response_format",
        backend.config.response_format == "json_object",
        f"response_format={backend.config.response_format}")

    # 2. Request payload includes thinking and response_format
    print("\n2. Request payload construction")
    from hrm_adaptive_memory.executive.model_backend import _build_request_payload
    payload = _build_request_payload(
        model="deepseek-v4-flash", system_prompt="s", user_prompt="u",
        temperature=0.0, max_tokens=2048,
        thinking_mode="disabled", response_format="json_object")
    parsed = json.loads(payload)
    all_pass &= check(
        "Payload includes thinking disabled",
        parsed.get("thinking") == {"type": "disabled"})
    all_pass &= check(
        "Payload includes response_format json_object",
        parsed.get("response_format") == {"type": "json_object"})

    # 3. Frozen retry policy used by backend
    print("\n3. Retry policy wiring")
    from hrm_adaptive_memory.executive.i3_4_retry_policy import FROZEN_RETRY_POLICY
    all_pass &= check(
        "Backend uses frozen retry policy",
        backend.retry_policy is FROZEN_RETRY_POLICY)
    all_pass &= check(
        "HTTP 400 is non-retryable",
        not backend.retry_policy.should_retry_http(400))
    all_pass &= check(
        "HTTP 429 is retryable",
        backend.retry_policy.should_retry_http(429))

    # 4. CallReceipts emitted
    print("\n4. Call receipt integration")
    from unittest.mock import MagicMock, patch
    backend_rc = DeepSeekBackend(_api_key="test-key")
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
        backend_rc.generate(system_prompt="s", user_prompt="u",
                            temperature=0.0, max_tokens=2048)
    all_pass &= check(
        "Success produces one receipt",
        len(backend_rc.call_receipts) == 1,
        f"receipts={len(backend_rc.call_receipts)}")
    all_pass &= check(
        "Receipt has correct result_class",
        backend_rc.call_receipts[0].result_class == "success")
    all_pass &= check(
        "Receipt has generation_config_sha256",
        backend_rc.call_receipts[0].generation_config_sha256 == FROZEN_CONFIG.sha256())

    # 5. Strict JSON decoder
    print("\n5. Strict JSON decoder")
    from hrm_adaptive_memory.executive.model_decoder import decode_output
    pure = decode_output('{"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null}', strict=True)
    all_pass &= check("Pure JSON accepted in strict mode", pure.valid)
    prose = decode_output('I think. {"action": "ANSWER", "reason_code": "SUFFICIENT", "target_id": null} Done.', strict=True)
    all_pass &= check("Prose-wrapped JSON rejected in strict mode", not prose.valid)

    # 6. Identity policy binds real generation config hash
    print("\n6. Identity policy binding")
    from hrm_adaptive_memory.executive.i3_4_model_identity_policy import FROZEN_IDENTITY_POLICY
    all_pass &= check(
        "Identity policy has non-empty generation_config_sha256",
        FROZEN_IDENTITY_POLICY.generation_config_sha256 != "")
    all_pass &= check(
        "Identity policy hash matches frozen config",
        FROZEN_IDENTITY_POLICY.generation_config_sha256 == FROZEN_CONFIG.sha256())

    # 7. Fingerprint semantics
    print("\n7. Fingerprint semantics")
    valid, _ = FROZEN_IDENTITY_POLICY.verify_call("deepseek-v4-flash", None)
    all_pass &= check(
        "Missing fingerprint rejected (require_fingerprint=True)",
        not valid)
    valid, _ = FROZEN_IDENTITY_POLICY.verify_call("deepseek-v4-flash", "fp_abc")
    all_pass &= check(
        "Present fingerprint accepted",
        valid)
    pair_valid, _ = FROZEN_IDENTITY_POLICY.verify_pair(None, "fp_abc")
    all_pass &= check(
        "Pair with missing fingerprint rejected",
        not pair_valid)

    # 8. Observable oracle views
    print("\n8. Observable oracle views")
    views_path = root / "experiments/v2b_i3_4/oracle_views/v2b_i3_4_observable_oracle_views_v1.json"
    if views_path.exists():
        views_data = json.loads(views_path.read_text())
        all_pass &= check(
            "Oracle views file exists",
            True, f"{len(views_data['views'])} views")
        # Check that values differ across splits
        aware_values = {}
        for v in views_data["views"]:
            if v["condition"] == "STATE_AWARE_CONTROLLER":
                aware_values[v["split_name"]] = v["observable_optimal_value"]
        if len(aware_values) >= 2:
            values = list(aware_values.values())
            all_pass &= check(
                "Observable values differ across splits",
                len(set(round(v, 6) for v in values)) > 1,
                f"values={aware_values}")
    else:
        all_pass &= check("Oracle views file exists", False, "file not found")

    # 9. Full experiment identity
    print("\n9. Experiment identity")
    identity_path = root / "experiments/v2b_i3_4/manifests/v2b_i3_4_experiment_identity_v1.json"
    if identity_path.exists():
        identity_data = json.loads(identity_path.read_text())
        all_pass &= check("Experiment identity file exists", True)
        required = ["benchmark", "scientific_criteria", "evaluation_subset",
                    "observable_oracle_views", "controller", "provider_model_policy",
                    "generation_config", "retry_policy", "paired_scheduler",
                    "statistical_implementation", "runtime_environment"]
        missing = [k for k in required if k not in identity_data]
        all_pass &= check(
            "All components bound",
            len(missing) == 0,
            f"missing={missing}" if missing else "all present")
        all_pass &= check(
            "Identity has SHA-256",
            "identity_sha256" in identity_data and len(identity_data["identity_sha256"]) == 64)
    else:
        all_pass &= check("Experiment identity file exists", False, "file not found")

    # 10. Paired runner
    print("\n10. Paired runner integration")
    from hrm_adaptive_memory.executive.i3_4_paired_runner import PairedExperimentRunner
    from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction

    def make_obs(task_id: str) -> ControllerObservation:
        return ControllerObservation(
            task_id=task_id, task_summary="Test",
            resource_state={"executive_steps_used": 0, "executive_steps_remaining": 10},
            allowed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY,
                             DecisionAction.ANSWER, DecisionAction.DEFER),
            executed_actions=(), rejected_actions=(), cognitive_state=None)

    runner_backend = DeepSeekBackend(_api_key="test-key")
    runner = PairedExperimentRunner(backend=runner_backend)
    with patch("urllib.request.urlopen", return_value=mock_response):
        result = runner.run_pair(
            task_id="preflight_test",
            blind_observation=make_obs("preflight_test"),
            aware_observation=make_obs("preflight_test"))
    all_pass &= check(
        "Pair produces two receipts",
        len(runner_backend.call_receipts) == 2,
        f"receipts={len(runner_backend.call_receipts)}")
    all_pass &= check(
        "Pair has schedule",
        result.schedule.first_condition in ("STATE_BLIND_CONTROLLER", "STATE_AWARE_CONTROLLER"))
    all_pass &= check(
        "Pair has identity check",
        hasattr(result, "identity_valid"))

    # Summary
    print("\n" + "=" * 50)
    if all_pass:
        print("PREFLIGHT: ALL CHECKS PASSED")
        print("Ready for development-only DeepSeek repeatability run.")
        return 0
    else:
        print("PREFLIGHT: FAILURES DETECTED")
        print("Do NOT run DeepSeek until all checks pass.")
        return 1


if __name__ == "__main__":
    sys.exit(run_preflight())
