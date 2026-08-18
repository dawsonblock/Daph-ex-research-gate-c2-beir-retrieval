"""Tests for I3.5.1 governor purity — identical observation → identical frame."""
import pytest
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, DecisionSummary, TemporalStatus, VerificationState,
)


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


class TestGovernorPurity:
    """The governor must consume exactly the visible state — no hidden info."""

    def test_identical_observations_produce_identical_frames(self):
        """Same observation → same governor frame."""
        gov = GeneralGovernor()
        obs = _make_observation()
        f1 = gov.assess(obs, remaining_steps=8, prior_actions=(), prior_outcomes=())
        f2 = gov.assess(obs, remaining_steps=8, prior_actions=(), prior_outcomes=())
        assert f1.governor_top_action == f2.governor_top_action
        assert f1.governor_reason_code == f2.governor_reason_code

    def test_different_task_ids_same_content_same_frame(self):
        """Governor must not depend on task_id (which could leak topology info)."""
        gov = GeneralGovernor()
        obs1 = _make_observation()
        obs1 = ControllerObservation(
            task_id="task_AAAA",
            task_summary=obs1.task_summary,
            resource_state=obs1.resource_state,
            allowed_actions=obs1.allowed_actions,
            executed_actions=obs1.executed_actions,
            rejected_actions=obs1.rejected_actions,
            cognitive_state=obs1.cognitive_state,
        )
        obs2 = ControllerObservation(
            task_id="task_BBBB",
            task_summary=obs1.task_summary,
            resource_state=obs1.resource_state,
            allowed_actions=obs1.allowed_actions,
            executed_actions=obs1.executed_actions,
            rejected_actions=obs1.rejected_actions,
            cognitive_state=obs1.cognitive_state,
        )
        f1 = gov.assess(obs1, remaining_steps=8, prior_actions=(), prior_outcomes=())
        f2 = gov.assess(obs2, remaining_steps=8, prior_actions=(), prior_outcomes=())
        # Governor should produce the same frame for identical visible state
        assert f1.governor_top_action == f2.governor_top_action
        assert f1.governor_reason_code == f2.governor_reason_code

    def test_governor_does_not_access_latent_state(self):
        """Governor must not access latent oracle values or topology labels."""
        gov = GeneralGovernor()
        obs = _make_observation()
        frame = gov.assess(obs, remaining_steps=8, prior_actions=(), prior_outcomes=())
        # The frame should not contain any latent/oracle/topology fields
        from hrm_adaptive_memory.executive.governor.serializer import serialize_frame_dict
        serialized = serialize_frame_dict(frame)
        forbidden = {"oracle", "latent", "topology_id", "topology_hash",
                     "difficulty", "information_class", "expected_terminal",
                     "task_success", "gold", "held_out"}
        def _check(obj):
            if isinstance(obj, dict):
                for k in obj:
                    assert k not in forbidden, f"Governor frame leaks: {k}"
                    _check(obj[k])
            elif isinstance(obj, list):
                for item in obj:
                    _check(item)
        _check(serialized)
