"""Tests for I3.5.1 no-leakage — adversarial cases."""
import pytest
from hrm_adaptive_memory.executive.i3_5_1.packet_builder import (
    build_base_packet, build_governor_packet,
    assert_no_evaluator_leakage, FORBIDDEN_KEYS,
)
from hrm_adaptive_memory.executive.i3_5_1.conditions import (
    ConditionID, ExperimentalCondition, ObservationMode,
)
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS


def _make_observation(cognitive_state=None) -> ControllerObservation:
    return ControllerObservation(
        task_id="test_task",
        task_summary="Test task",
        resource_state={"retrieval": 5, "verification": 5, "search": 5, "reasoning": 5, "time_ms": 10000},
        allowed_actions=tuple(V2B_ACTIONS),
        executed_actions=(),
        rejected_actions=(),
        cognitive_state=cognitive_state,
    )


class TestAdversarialLeakage:
    """Each forbidden key, when injected, must be detected."""

    @pytest.mark.parametrize("forbidden_key", sorted(FORBIDDEN_KEYS))
    def test_top_level_leakage_detected(self, forbidden_key):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet[forbidden_key] = "leaked_value"
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)

    def test_nested_in_task_summary_leakage(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet["task_summary"] = {"topology_id": "hidden_topo"}
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)

    def test_nested_in_resource_state_leakage(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet["resource_state"]["difficulty"] = "HARD"
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)

    def test_deeply_nested_leakage(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet["task_summary"] = {"nested": {"deep": {"oracle": "value"}}}
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)

    def test_list_nested_leakage(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        packet["allowed_actions"] = ["ANSWER", {"latent": "leaked"}]
        with pytest.raises(ValueError, match="leak"):
            assert_no_evaluator_leakage(packet)


class TestConditionMetadataNotInPacket:
    """Condition IDs and treatment metadata must never appear in packets."""

    def test_no_condition_id_in_base_packet(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        assert "condition_id" not in packet
        assert "condition" not in packet

    def test_no_governor_enabled_in_base_packet(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        assert "governor_enabled" not in packet

    def test_no_experiment_arm_in_packet(self):
        obs = _make_observation()
        packet = build_base_packet(obs)
        assert "experiment_arm" not in packet
        assert "treatment_group" not in packet
