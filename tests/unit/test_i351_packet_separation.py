"""Tests for I3.5.1 packet separation — BASE vs GOVERNOR packets."""
import pytest
from hrm_adaptive_memory.executive.i3_5_1.packet_builder import (
    build_base_packet, build_governor_packet,
    BASE_PACKET_SCHEMA, GOVERNOR_PACKET_SCHEMA,
    FORBIDDEN_KEYS, assert_no_evaluator_leakage,
    packet_sha256, packet_json,
)
from hrm_adaptive_memory.executive.i3_5_1.conditions import (
    ConditionID, ExperimentalCondition, ObservationMode,
)
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.state import CognitiveStateSnapshot


def _make_observation(cognitive_state=None) -> ControllerObservation:
    """Build a minimal observation for testing."""
    return ControllerObservation(
        task_id="test_task",
        task_summary="Test task",
        resource_state={"retrieval": 5, "verification": 5, "search": 5, "reasoning": 5, "time_ms": 10000},
        allowed_actions=tuple(V2B_ACTIONS),
        executed_actions=(),
        rejected_actions=(),
        cognitive_state=cognitive_state,
    )


class TestPacketSchemas:
    def test_base_and_governor_schemas_differ(self):
        assert BASE_PACKET_SCHEMA != GOVERNOR_PACKET_SCHEMA

    def test_base_packet_has_no_governor_key(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        assert "governor" not in packet
        assert packet["schema"] == BASE_PACKET_SCHEMA

    def test_governor_packet_has_governor_key(self):
        from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
        obs = _make_observation()
        governor = GeneralGovernor()
        frame = governor.assess(
            observation=obs,
            remaining_steps=8,
            prior_actions=(),
            prior_outcomes=(),
        )
        packet = build_governor_packet(obs, frame)
        assert "governor" in packet
        assert packet["schema"] == GOVERNOR_PACKET_SCHEMA

    def test_base_packet_never_has_governor_null(self):
        """No-governor means no governor structure at all, not 'governor': null."""
        obs = _make_observation()
        packet = build_base_packet(obs)
        assert "governor" not in packet
        # Recursive check
        def _check_no_governor(obj):
            if isinstance(obj, dict):
                assert "governor" not in obj, f"Found 'governor' key in BASE packet"
                for v in obj.values():
                    _check_no_governor(v)
            elif isinstance(obj, list):
                for item in obj:
                    _check_no_governor(item)
        _check_no_governor(packet)


class TestForbiddenKeys:
    def test_forbidden_keys_include_evaluator_terms(self):
        assert "condition" in FORBIDDEN_KEYS
        assert "experiment_arm" in FORBIDDEN_KEYS
        assert "governor_enabled" in FORBIDDEN_KEYS
        assert "oracle" in FORBIDDEN_KEYS
        assert "latent" in FORBIDDEN_KEYS
        assert "topology_id" in FORBIDDEN_KEYS
        assert "task_success" in FORBIDDEN_KEYS
        assert "gold" in FORBIDDEN_KEYS

    def test_base_packet_no_evaluator_leakage(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        # Should not raise
        assert_no_evaluator_leakage(packet)

    def test_governor_packet_no_evaluator_leakage(self):
        from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
        obs = _make_observation()
        governor = GeneralGovernor()
        frame = governor.assess(
            observation=obs,
            remaining_steps=8,
            prior_actions=(),
            prior_outcomes=(),
        )
        packet = build_governor_packet(obs, frame)
        # Should not raise
        assert_no_evaluator_leakage(packet)

    def test_leakage_detection_raises(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet["condition"] = "BLIND"
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)

    def test_nested_leakage_detection(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet["task_summary"] = {"topology_id": "hidden"}
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)

    def test_condition_id_leakage_detection(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet["condition_id"] = "BLIND_NO_GOVERNOR"
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)


class TestPacketHashing:
    def test_packet_sha256_deterministic(self):
        obs = _make_observation()
        p1 = build_base_packet(obs)
        p2 = build_base_packet(obs)
        assert packet_sha256(p1) == packet_sha256(p2)

    def test_packet_json_deterministic(self):
        obs = _make_observation()
        p1 = build_base_packet(obs)
        p2 = build_base_packet(obs)
        assert packet_json(p1) == packet_json(p2)
