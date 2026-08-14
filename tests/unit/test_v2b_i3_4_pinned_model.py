"""I3.4 pinned-model controller: condition leakage, decoder, replay, identity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, ConflictSummary, DecisionSummary, MemorySummary,
    TemporalStatus, VerificationState, VerificationSummary)
from hrm_adaptive_memory.executive.actions import ActionProposal
from hrm_adaptive_memory.executive.controller_identity import (
    IDENTITY_SCHEMA, build_identity, load_identity, save_identity)
from hrm_adaptive_memory.executive.controller_protocol import ControllerProtocol
from hrm_adaptive_memory.executive.metareasoning_controller import (
    ControllerObservation, MatchedMetareasoningController, ObservationMask,
    STATE_AWARE_MASK, STATE_BLIND_MASK, apply_observation_mask)
from hrm_adaptive_memory.executive.model_backend import StubBackend
from hrm_adaptive_memory.executive.model_decoder import (
    DecoderOutcome, decode_output, OUTPUT_SCHEMA, VALID_ACTION_NAMES)
from hrm_adaptive_memory.executive.model_packet import (
    PACKET_SCHEMA, assert_no_condition_leakage, packet_sha256, serialize_packet)
from hrm_adaptive_memory.executive.model_prompt import (
    PROMPT_ID, SYSTEM_PROMPT, prompt_sha256)
from hrm_adaptive_memory.executive.pinned_model_controller import (
    FAIL_CLOSED_PROPOSAL, PinnedModelController)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    build_observable_snapshot, initial_i3_runtime)
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    MetareasoningBenchmark, load_metareasoning_benchmark)
from hrm_adaptive_memory.executive.metareasoning_loop import (
    STATE_AWARE, STATE_BLIND, V2BMetareasoningExperiment)
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState

ROOT = Path(__file__).parents[2]
BENCHMARK_PATH = ROOT / "experiments/v2b/benchmark/v2b_i3_benchmark_manifest_v1.json"
POLICY_PATH = ROOT / "configs/v2b_i3_policy_v1.json"
MASKS_PATH = ROOT / "configs/v2b_i3_observation_masks_v1.json"


def _benchmark():
    return load_metareasoning_benchmark(BENCHMARK_PATH)


def _make_snapshot() -> CognitiveStateSnapshot:
    return CognitiveStateSnapshot(
        task_id="test-task-1",
        task_summary="Verify whether claim X is supported by current evidence.",
        relevant_memories=(MemorySummary(
            "mem-1", 0.9, VerificationState.UNVERIFIED, 2, 2,
            "NONE", TemporalStatus.CURRENT),),
        verification_states=(VerificationSummary(
            "ver-1", VerificationState.UNVERIFIED, 2, None),),
        provenance_summaries=("lineage_count=2",),
        temporal_status=TemporalStatus.CURRENT,
        unresolved_conflicts=(),
        prior_decisions=(DecisionSummary("dec-1", "RETRIEVE", "INITIAL", "RETRIEVE_OK"),),
        prior_outcomes=("RETRIEVE_OK",),
        resource_state={"executive_steps_remaining": 10},
        policy_facts=(),
        observation_signals=("COMPOSITION_INCOMPLETE",),
    )


def _make_observation(snapshot: CognitiveStateSnapshot | None = None) -> ControllerObservation:
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


# --- Packet schema and condition leakage ---


def test_packet_schema_is_frozen():
    packet = serialize_packet(_make_observation(_make_snapshot()))
    assert packet["schema"] == PACKET_SCHEMA


def test_packet_structure_identical_blind_and_aware():
    blind = serialize_packet(_make_observation(None))
    aware = serialize_packet(_make_observation(_make_snapshot()))
    assert set(blind.keys()) == set(aware.keys())
    assert set(blind["cognitive_state"].keys()) == set(aware["cognitive_state"].keys())


def test_packet_blind_uses_canonical_nulls():
    packet = serialize_packet(_make_observation(None))
    cs = packet["cognitive_state"]
    assert cs["verification_states"] == []
    assert cs["provenance_summaries"] == []
    assert cs["temporal_status"] == "UNKNOWN"
    assert cs["unresolved_conflicts"] == []
    assert cs["prior_decisions"] == []
    assert cs["prior_outcomes"] == []
    assert cs["observation_signals"] == []


def test_packet_aware_preserves_state_values():
    packet = serialize_packet(_make_observation(_make_snapshot()))
    cs = packet["cognitive_state"]
    assert cs["temporal_status"] == "CURRENT"
    assert len(cs["verification_states"]) == 1
    assert cs["verification_states"][0]["state"] == "UNVERIFIED"
    assert cs["observation_signals"] == ["COMPOSITION_INCOMPLETE"]


def test_packet_has_no_condition_identity():
    packet = serialize_packet(_make_observation(_make_snapshot()))
    # Must not contain any condition name, mask, or evaluator metadata.
    assert_no_condition_leakage(packet)


def test_packet_does_not_leak_forbidden_keys_in_any_condition():
    for snapshot in (None, _make_snapshot()):
        packet = serialize_packet(_make_observation(snapshot))
        assert_no_condition_leakage(packet)


def test_packet_sha256_is_deterministic():
    obs = _make_observation(_make_snapshot())
    h1 = packet_sha256(serialize_packet(obs))
    h2 = packet_sha256(serialize_packet(obs))
    assert h1 == h2


def test_packet_sha256_differs_blind_vs_aware():
    blind = packet_sha256(serialize_packet(_make_observation(None)))
    aware = packet_sha256(serialize_packet(_make_observation(_make_snapshot())))
    assert blind != aware


# --- Output decoder ---


@pytest.mark.parametrize("action_name", sorted(VALID_ACTION_NAMES))
def test_decoder_accepts_all_seven_actions(action_name):
    raw = json.dumps({"action": action_name, "reason_code": "TEST_REASON", "target_id": None})
    outcome = decode_output(raw)
    assert outcome.valid
    assert outcome.proposal is not None
    assert outcome.proposal.action.value == action_name


def test_decoder_rejects_unknown_action():
    raw = json.dumps({"action": "FLY", "reason_code": "TEST", "target_id": None})
    outcome = decode_output(raw)
    assert not outcome.valid
    assert outcome.rejection_code == "UNKNOWN_ACTION"


def test_decoder_rejects_missing_action():
    raw = json.dumps({"reason_code": "TEST", "target_id": None})
    outcome = decode_output(raw)
    assert not outcome.valid
    assert outcome.rejection_code == "MISSING_ACTION"


def test_decoder_rejects_missing_reason_code():
    raw = json.dumps({"action": "ANSWER", "target_id": None})
    outcome = decode_output(raw)
    assert not outcome.valid
    assert outcome.rejection_code == "MISSING_REASON_CODE"


def test_decoder_rejects_invalid_reason_code():
    raw = json.dumps({"action": "ANSWER", "reason_code": "lowercase", "target_id": None})
    outcome = decode_output(raw)
    assert not outcome.valid
    assert outcome.rejection_code == "INVALID_REASON_CODE"


def test_decoder_rejects_extra_keys():
    raw = json.dumps({"action": "ANSWER", "reason_code": "TEST", "target_id": None, "extra": 1})
    outcome = decode_output(raw)
    assert not outcome.valid
    assert outcome.rejection_code == "EXTRA_KEYS"


def test_decoder_rejects_malformed_json():
    outcome = decode_output("not json at all")
    assert not outcome.valid
    assert outcome.rejection_code == "NO_JSON_FOUND"


def test_decoder_rejects_empty_output():
    outcome = decode_output("")
    assert not outcome.valid
    assert outcome.rejection_code == "EMPTY_OUTPUT"


def test_decoder_extracts_json_from_surrounding_text():
    raw = 'Let me think.\n{"action": "STOP", "reason_code": "DONE", "target_id": null}\nDone.'
    outcome = decode_output(raw)
    assert outcome.valid
    assert outcome.proposal.action is DecisionAction.STOP


def test_decoder_accepts_string_target_id():
    raw = json.dumps({"action": "VERIFY", "reason_code": "TARGETED", "target_id": "ver-1"})
    outcome = decode_output(raw)
    assert outcome.valid
    assert outcome.proposal.target_id == "ver-1"


def test_decoder_rejects_non_string_target_id():
    raw = json.dumps({"action": "VERIFY", "reason_code": "TEST", "target_id": 123})
    outcome = decode_output(raw)
    assert not outcome.valid
    assert outcome.rejection_code == "INVALID_TARGET_ID"


def test_decoder_rejects_missing_target_id():
    raw = json.dumps({"action": "ANSWER", "reason_code": "TEST"})
    outcome = decode_output(raw)
    assert not outcome.valid
    assert outcome.rejection_code == "MISSING_TARGET_ID"


def test_decoder_strips_whitespace_from_target_id():
    raw = json.dumps({"action": "VERIFY", "reason_code": "TEST", "target_id": "  ver-1  "})
    outcome = decode_output(raw)
    assert outcome.valid
    assert outcome.proposal.target_id == "ver-1"


def test_decoder_extracts_json_after_reasoning_with_braces():
    """The model may emit reasoning text containing braces before the JSON.
    The decoder must extract the first balanced JSON object, not be confused
    by brace characters inside string literals."""
    raw = (
        'I need to think about this {carefully}. '
        'Here is my decision: '
        '{"action": "ANSWER", "reason_code": "REASONED_ANSWER", "target_id": null}'
    )
    outcome = decode_output(raw)
    assert outcome.valid
    assert outcome.proposal.action is DecisionAction.ANSWER


def test_decoder_rejects_trailing_garbage_after_json():
    """If the model emits valid JSON followed by more text, the decoder
    extracts the first JSON object and ignores the trailing text."""
    raw = (
        '{"action": "STOP", "reason_code": "DONE", "target_id": null}'
        ' and some trailing commentary'
    )
    outcome = decode_output(raw)
    assert outcome.valid
    assert outcome.proposal.action is DecisionAction.STOP


def test_packet_rejects_condition_identity_in_string_values():
    """Condition identity substrings must not appear in any string value."""
    from hrm_adaptive_memory.executive.model_packet import (
        assert_no_condition_leakage, serialize_packet)
    from hrm_adaptive_memory.executive.metareasoning_controller import (
        ControllerObservation)
    from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget

    obs = ControllerObservation(
        task_id="STATE_AWARE_CONTROLLER-task-1",  # condition leak in task_id
        task_summary="test",
        resource_state=ResourceState(ResourceBudget()).as_dict(),
        allowed_actions=(),
        executed_actions=(),
        rejected_actions=(),
        policy_feedback=(),
        cognitive_state=None,
    )
    packet = serialize_packet(obs)
    with pytest.raises(ValueError, match="condition identity"):
        assert_no_condition_leakage(packet)


# --- PinnedModelController ---


def test_controller_satisfies_protocol():
    controller = PinnedModelController(backend=StubBackend())
    assert isinstance(controller, ControllerProtocol)


def test_controller_returns_valid_proposal_with_stub():
    controller = PinnedModelController(backend=StubBackend())
    proposal = controller.choose(_make_observation(None))
    assert isinstance(proposal, ActionProposal)
    assert proposal.action in tuple(DecisionAction)


def test_controller_fail_closed_on_malformed_output():
    from hrm_adaptive_memory.executive.model_backend import ModelCallResult

    class BadBackend:
        model_name = "bad-stub"
        def generate(self, *, system_prompt, user_prompt, temperature, max_tokens):
            return ModelCallResult(
                raw_output="I cannot decide", prompt_tokens=10, completion_tokens=5,
                reasoning_tokens=0, latency_ms=1, model_name="bad-stub",
                system_fingerprint=None, finish_reason="stop")

    controller = PinnedModelController(backend=BadBackend())
    proposal = controller.choose(_make_observation(None))
    assert proposal is FAIL_CLOSED_PROPOSAL
    assert proposal.action is DecisionAction.DEFER
    assert proposal.reason_code == "MODEL_OUTPUT_INVALID"
    assert controller.last_decoder_outcome is not None
    assert not controller.last_decoder_outcome.valid


def test_controller_fail_closed_on_backend_error():
    """API/network errors must not crash the loop; controller returns DEFER."""
    from hrm_adaptive_memory.executive.pinned_model_controller import BACKEND_ERROR_PROPOSAL

    class ErrorBackend:
        model_name = "error-stub"
        def generate(self, *, system_prompt, user_prompt, temperature, max_tokens):
            raise ConnectionError("simulated network failure")

    controller = PinnedModelController(backend=ErrorBackend())
    proposal = controller.choose(_make_observation(None))
    assert proposal is BACKEND_ERROR_PROPOSAL
    assert proposal.action is DecisionAction.DEFER
    assert proposal.reason_code == "MODEL_BACKEND_ERROR"
    assert controller.last_backend_error is not None
    assert "network failure" in controller.last_backend_error
    assert controller.last_call_result is None
    assert controller.last_decoder_outcome is None


def test_controller_has_no_condition_branching():
    """The controller code must not reference condition names or masks."""
    import inspect
    import re
    from hrm_adaptive_memory.executive import pinned_model_controller as pmc
    source = inspect.getsource(pmc)
    # Literal substrings that must never appear in the controller source.
    forbidden_literals = {
        "STATE_BLIND", "STATE_AWARE", "NO_VERIFICATION", "NO_PROVENANCE",
        "NO_TEMPORAL", "NO_CONFLICT", "NO_HISTORY", "cognitive_state is not None",
    }
    for term in forbidden_literals:
        if term in source:
            pytest.fail(f"controller source contains forbidden term: {term}")
    # Regex patterns that would indicate condition-specific branching.
    forbidden_patterns = [
        r"if\s+.*aware", r"if\s+.*blind", r"if\s+.*cognitive_state",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, source, re.IGNORECASE):
            pytest.fail(f"controller source matches forbidden pattern: {pattern}")


def test_controller_same_code_path_blind_and_aware():
    """Both conditions must exercise the same choose() method without branching."""
    controller = PinnedModelController(backend=StubBackend())
    blind_proposal = controller.choose(_make_observation(None))
    aware_proposal = controller.choose(_make_observation(_make_snapshot()))
    # Both must be valid ActionProposal objects from the same code path.
    assert isinstance(blind_proposal, ActionProposal)
    assert isinstance(aware_proposal, ActionProposal)


def test_controller_tracks_development_metrics():
    controller = PinnedModelController(backend=StubBackend())
    controller.choose(_make_observation(None))
    metrics = controller.development_metrics()
    assert metrics["call_count"] == 1
    assert metrics["last_packet_sha256"] is not None
    assert metrics["last_model_name"] == "stub-deterministic-v1"


# --- Loop integration ---


def _small_benchmark(n: int = 2) -> "MetareasoningBenchmark":
    """Create a benchmark with only the first *n* tasks for fast testing."""
    import dataclasses
    full = _benchmark()
    return dataclasses.replace(full, tasks=full.tasks[:n])


def test_loop_accepts_pinned_model_controller():
    benchmark = _small_benchmark(2)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = PinnedModelController(backend=StubBackend())
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_BLIND, controller=controller, store_root=tmpdir)
    assert run.condition == STATE_BLIND
    assert run.controller_id == controller.controller_id
    assert len(run.tasks) == 2


def test_loop_accepts_matched_deterministic_controller():
    """The protocol change must not break the existing deterministic fixture."""
    benchmark = _small_benchmark(2)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = MatchedMetareasoningController()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_AWARE, controller=controller, store_root=tmpdir)
    assert run.condition == STATE_AWARE
    assert run.controller_id == controller.controller_id


# --- Development metrics ---


def test_model_metrics_captured_in_traces():
    """PinnedModelController traces must carry model-specific metrics."""
    benchmark = _small_benchmark(2)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = PinnedModelController(backend=StubBackend())
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_BLIND, controller=controller, store_root=tmpdir)
    model_traces = [t for task in run.tasks for t in task.traces if t.model_valid is not None]
    assert len(model_traces) > 0
    for trace in model_traces:
        assert trace.model_valid is True
        assert trace.model_packet_sha256 is not None
        assert trace.model_latency_ms is not None
        assert trace.model_prompt_tokens is not None


def test_deterministic_controller_traces_have_null_model_metrics():
    """MatchedMetareasoningController traces must have None model metrics."""
    benchmark = _small_benchmark(2)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = MatchedMetareasoningController()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_AWARE, controller=controller, store_root=tmpdir)
    for task in run.tasks:
        for trace in task.traces:
            assert trace.model_valid is None
            assert trace.model_latency_ms is None
            assert trace.model_packet_sha256 is None


def test_condition_run_metrics_include_model_fields():
    """I3ConditionRun.metrics must include model-specific aggregations."""
    benchmark = _small_benchmark(2)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = PinnedModelController(backend=StubBackend())
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_BLIND, controller=controller, store_root=tmpdir)
    m = run.metrics
    assert "model_call_count" in m
    assert "model_valid_action_rate" in m
    assert "model_malformed_output_rate" in m
    assert "model_mean_latency_ms" in m
    assert "model_total_prompt_tokens" in m
    assert "model_total_completion_tokens" in m
    assert "model_total_reasoning_tokens" in m
    assert "action_distribution" in m
    assert "terminal_outcome_distribution" in m
    assert "mean_steps_per_task" in m
    assert m["model_call_count"] > 0
    assert m["model_valid_action_rate"] == 1.0  # StubBackend always returns valid JSON


def test_condition_run_metrics_include_action_distribution():
    """Action distribution must cover all seven actions."""
    benchmark = _small_benchmark(2)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = PinnedModelController(backend=StubBackend())
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_BLIND, controller=controller, store_root=tmpdir)
    dist = run.metrics["action_distribution"]
    expected = {"ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"}
    assert set(dist.keys()) == expected


def test_deterministic_controller_metrics_have_zero_model_calls():
    """Deterministic controller runs must report zero model calls."""
    benchmark = _small_benchmark(2)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = MatchedMetareasoningController()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_AWARE, controller=controller, store_root=tmpdir)
    assert run.metrics["model_call_count"] == 0
    assert run.metrics["model_valid_action_rate"] == 0.0


def test_malformed_output_tracked_in_metrics():
    """Malformed model output must be tracked in development metrics."""
    from hrm_adaptive_memory.executive.model_backend import ModelCallResult

    class BadBackend:
        model_name = "bad-stub"
        def generate(self, *, system_prompt, user_prompt, temperature, max_tokens):
            return ModelCallResult(
                raw_output="I cannot decide", prompt_tokens=10, completion_tokens=5,
                reasoning_tokens=0, latency_ms=1, model_name="bad-stub",
                system_fingerprint=None, finish_reason="stop")

    benchmark = _small_benchmark(1)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = PinnedModelController(backend=BadBackend())
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_BLIND, controller=controller, store_root=tmpdir)
    m = run.metrics
    assert m["model_malformed_output_rate"] > 0.0
    # All model calls should be malformed.
    assert m["model_valid_action_rate"] == 0.0


def test_backend_error_tracked_in_condition_metrics():
    """Backend errors must be counted in model_backend_error_count, not
    silently dropped by the model_traces filter."""
    class ErrorBackend:
        model_name = "error-stub"
        def generate(self, *, system_prompt, user_prompt, temperature, max_tokens):
            raise ConnectionError("simulated network failure")

    benchmark = _small_benchmark(1)
    policy = load_frozen_policy(POLICY_PATH)
    experiment = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    controller = PinnedModelController(backend=ErrorBackend())
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        run = experiment.run_condition(
            condition=STATE_BLIND, controller=controller, store_root=tmpdir)
    m = run.metrics
    # Backend error traces must be counted as model calls.
    assert m["model_call_count"] > 0
    assert m["model_backend_error_count"] > 0
    # No valid or malformed decoder outcomes (backend never returned output).
    assert m["model_valid_action_rate"] == 0.0
    assert m["model_malformed_output_rate"] == 0.0


# --- Scientific criteria ---


SCIENTIFIC_CRITERIA_PATH = ROOT / "experiments/v2b_i3_4/configs/v2b_i3_4_scientific_criteria_v1.json"


def test_scientific_criteria_is_frozen():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    assert raw["schema"] == "DAPH_V2B_I3_4_SCIENTIFIC_CRITERIA_V1"
    assert raw["status"] == "FROZEN_BEFORE_HELD_OUT_EVALUATION"


def test_scientific_criteria_defines_primary_hypothesis():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    hyp = raw["primary_hypothesis"]
    assert hyp["metric"] == "trajectory_regret"
    assert hyp["direction"] == "aware < blind"


def test_scientific_criteria_defines_distinct_claims():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    claims = raw["distinct_claims"]
    assert "information_without_exploitation" in claims
    assert "executive_exploitation" in claims
    assert "control_efficiency" in claims


def test_scientific_criteria_defines_paired_statistics():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    plan = raw["statistical_plan"]
    assert plan["primary_test"].startswith("Paired")
    assert plan["bootstrap"]["resampling_unit"] == "task"
    assert plan["bootstrap"]["iterations"] >= 1000
    assert plan["topology_grouping"]["variable"] == "topology_depth_band"


def test_scientific_criteria_defines_evaluation_order():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    order = raw["evaluation_order"]
    assert order["phase_1"] == "held_out_instance"
    assert order["phase_2"] == "held_out_surface"
    assert order["phase_3"] == "held_out_structure_last"


def test_scientific_criteria_records_structural_limitations():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    comp = raw["structural_split_limitations"]["held_out_structure_composition"]
    assert comp["task_count"] == 150
    assert comp["by_difficulty_band"]["HARD"] == 0
    assert comp["by_difficulty_band"]["TIE"] == 0
    assert "DEPTH_1" in comp["by_topology_depth_band"]
    assert "DEPTH_4_PLUS" in comp["by_topology_depth_band"]
    restrictions = raw["structural_split_limitations"]["claim_restrictions"]
    assert any("DEPTH_1" in r and "DEPTH_4_PLUS" in r for r in restrictions)


def test_scientific_criteria_ablations_after_main():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    assert raw["ablation_policy"]["order"] == "after_main_conditions"


def test_scientific_criteria_references_frozen_benchmark():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    refs = raw["frozen_references"]
    assert refs["benchmark_identity"] == "v2b_i3_3_2_scientific_split_v1"
    assert refs["qualification_status"] == "QUALIFIED_FROZEN_BENCHMARK"


def test_scientific_criteria_has_prohibition_clause():
    raw = json.loads(SCIENTIFIC_CRITERIA_PATH.read_text())
    assert "prohibition" in raw
    assert "held-out" in raw["prohibition"].lower() or "held out" in raw["prohibition"].lower()


# --- Scientific Criteria V2 ---


SCIENTIFIC_CRITERIA_V2_PATH = ROOT / "experiments/v2b_i3_4/configs/v2b_i3_4_scientific_criteria_v2.json"


def test_criteria_v2_is_frozen():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    assert raw["schema"] == "DAPH_V2B_I3_4_SCIENTIFIC_CRITERIA_V2"
    assert raw["status"] == "FROZEN_BEFORE_HELD_OUT_EVALUATION"


def test_criteria_v2_supersedes_v1():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    assert "supersedes" in raw
    assert raw["supersedes"]["v1_status"] == "SUPERSEDED_BEFORE_HELD_OUT_EVALUATION"


def test_criteria_v2_primary_hypothesis_uses_dg():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    hyp = raw["primary_hypothesis"]
    assert hyp["metric"] == "decision_gap"
    assert hyp["direction"] == "aware < blind"
    assert hyp["improvement"] == "ΔDG = DG_blind - DG_aware > 0"


def test_criteria_v2_defines_correct_decomposition():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    defs = raw["definitions"]
    assert "IG_M_s" in defs
    assert "DG_M_s" in defs
    assert "TR_M_s" in defs
    assert "identity" in defs
    assert "no_clamping" in defs


def test_criteria_v2_prohibits_tr_substitution():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    assert "prohibition_on_substitution" in raw
    assert "TR" in raw["prohibition_on_substitution"]
    assert "DG" in raw["prohibition_on_substitution"]


def test_criteria_v2_defines_topology_cluster_bootstrap():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    tcb = raw["statistical_plan"]["topology_cluster_bootstrap"]
    assert tcb["cluster_variable"] == "transition_topology_sha256"
    assert tcb["resampling_unit"] == "topology_cluster"
    assert "held_out_structure" in tcb["applies_to"]


def test_criteria_v2_primary_success_criterion():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    crit = raw["statistical_plan"]["primary_success_criterion"]
    assert "LCB_95" in crit["condition"]
    assert crit["condition"].endswith("> 0")


def test_criteria_v2_defines_validity_gates():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    gates = raw["validity_gates"]
    assert "G17" in gates
    assert "TR = IG + DG" in gates["G17"]
    assert "G18" in gates
    assert "G19" in gates
    assert "G23" in gates
    assert "VOID" in gates["void_vs_fail"]
    assert "FAIL" in gates["void_vs_fail"]


def test_criteria_v2_defines_four_distinct_claims():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    claims = raw["distinct_claims"]
    assert "representation_advantage" in claims
    assert "information_without_exploitation" in claims
    assert "executive_exploitation" in claims
    assert "control_efficiency" in claims


def test_criteria_v2_references_scoring_module():
    raw = json.loads(SCIENTIFIC_CRITERIA_V2_PATH.read_text())
    refs = raw["frozen_references"]
    assert refs["scoring_module"] == "hrm_adaptive_memory.executive.i3_4_scientific_scoring"
    assert refs["scoring_schema"] == "DAPH_V2B_I3_4_SCIENTIFIC_SCORING_V1"


# --- System prompt ---


def test_system_prompt_is_frozen():
    assert PROMPT_ID == "DAPH_V2B_I3_4_SYSTEM_PROMPT_V1"
    assert len(SYSTEM_PROMPT) > 100


def test_system_prompt_has_no_condition_identity():
    forbidden = {"STATE_BLIND", "STATE_AWARE", "NO_VERIFICATION", "NO_PROVENANCE",
                 "NO_TEMPORAL", "NO_CONFLICT", "NO_HISTORY"}
    for term in forbidden:
        assert term not in SYSTEM_PROMPT, f"prompt leaks condition name: {term}"


def test_system_prompt_has_no_benchmark_heuristics():
    forbidden = {"i3_", "benchmark", "oracle", "expected_terminal", "latent_state",
                 "difficulty", "topology", "DEPTH_1", "DEPTH_4"}
    for term in forbidden:
        assert term not in SYSTEM_PROMPT, f"prompt leaks benchmark term: {term}"


def test_prompt_sha256_is_stable():
    h1 = prompt_sha256()
    h2 = prompt_sha256()
    assert h1 == h2
    assert len(h1) == 64


# --- Controller identity ---


def test_identity_binds_all_components():
    import platform
    import sys
    identity = build_identity(
        model_name="deepseek-v4-flash", model_provider="deepseek",
        model_revision=None, system_fingerprint="test-fingerprint",
        temperature=0.0, max_tokens=256,
        policy_path="configs/v2b_i3_policy_v1.json",
        policy_sha256="a" * 64,
        utility_path="configs/v2b_i3_1_utility_v1.json",
        utility_sha256="b" * 64,
        observation_masks_path="configs/v2b_i3_observation_masks_v1.json",
        observation_masks_sha256="c" * 64,
        benchmark_manifest_path="experiments/v2b/benchmark/v2b_i3_benchmark_manifest_v1.json",
        benchmark_manifest_sha256="d" * 64,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )
    d = identity.to_dict()
    assert d["schema"] == IDENTITY_SCHEMA
    assert d["model"]["name"] == "deepseek-v4-flash"
    assert d["system_prompt"]["prompt_id"] == PROMPT_ID
    assert d["input_schema"]["schema"] == PACKET_SCHEMA
    assert d["output_schema"]["schema"] == OUTPUT_SCHEMA
    assert d["generation_settings"]["temperature"] == 0.0
    assert "model_backend" in d
    assert d["model_backend"]["deepseek_class"] == "DeepSeekBackend"
    assert "source_sha256" in d["model_backend"]


def test_identity_save_and_load_roundtrip(tmp_path):
    import platform
    import sys
    identity = build_identity(
        model_name="deepseek-v4-flash", model_provider="deepseek",
        model_revision=None, system_fingerprint=None,
        temperature=0.0, max_tokens=256,
        policy_path="configs/v2b_i3_policy_v1.json", policy_sha256="a" * 64,
        utility_path="configs/v2b_i3_1_utility_v1.json", utility_sha256="b" * 64,
        observation_masks_path="configs/v2b_i3_observation_masks_v1.json",
        observation_masks_sha256="c" * 64,
        benchmark_manifest_path="experiments/v2b/benchmark/v2b_i3_benchmark_manifest_v1.json",
        benchmark_manifest_sha256="d" * 64,
        python_version=sys.version.split()[0], platform=platform.platform(),
    )
    path = tmp_path / "controller_identity.json"
    save_identity(identity, path)
    loaded = load_identity(path)
    assert loaded["schema"] == IDENTITY_SCHEMA
    assert loaded["identity_sha256"] == identity.sha256()


def test_identity_rejects_tampered_hash(tmp_path):
    import platform
    import sys
    identity = build_identity(
        model_name="deepseek-v4-flash", model_provider="deepseek",
        model_revision=None, system_fingerprint=None,
        temperature=0.0, max_tokens=256,
        policy_path="configs/v2b_i3_policy_v1.json", policy_sha256="a" * 64,
        utility_path="configs/v2b_i3_1_utility_v1.json", utility_sha256="b" * 64,
        observation_masks_path="configs/v2b_i3_observation_masks_v1.json",
        observation_masks_sha256="c" * 64,
        benchmark_manifest_path="experiments/v2b/benchmark/v2b_i3_benchmark_manifest_v1.json",
        benchmark_manifest_sha256="d" * 64,
        python_version=sys.version.split()[0], platform=platform.platform(),
    )
    path = tmp_path / "controller_identity.json"
    save_identity(identity, path)
    # Tamper with the stored hash.
    raw = json.loads(path.read_text())
    raw["identity_sha256"] = "0" * 64
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_identity(path)
