"""Tests for I3.5 runner integration: governor + model packet + prompt."""
import pytest
import json
from hrm_adaptive_memory.executive.i3_5_model_packet import (
    serialize_governor_packet, governor_packet_sha256,
    governor_packet_json, assert_no_governor_leakage,
    PACKET_SCHEMA)
from hrm_adaptive_memory.executive.i3_5_model_prompt import (
    SYSTEM_PROMPT, prompt_sha256, PROMPT_ID)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.executive.actions import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, VerificationSummary, ConflictSummary,
    VerificationState, TemporalStatus)


def _make_observation(cognitive_state=None):
    return ControllerObservation(
        task_id="test_001",
        task_summary="Test task",
        resource_state={
            "retrieval_calls_remaining": 5,
            "verification_calls_remaining": 5,
            "search_calls_remaining": 5,
            "reasoning_tokens_remaining": 512,
            "executive_steps_remaining": 8,
        },
        allowed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY,
                         DecisionAction.SEARCH_MORE, DecisionAction.REASON_MORE,
                         DecisionAction.ANSWER, DecisionAction.DEFER),
        executed_actions=(),
        rejected_actions=(),
        cognitive_state=cognitive_state,
        policy_feedback=(),
    )


def _make_cognitive_state(verification_state=VerificationState.UNVERIFIED):
    return CognitiveStateSnapshot(
        task_id="test_001",
        task_summary="Test task",
        relevant_memories=(),
        verification_states=(VerificationSummary(
            target_id="src_1", state=verification_state,
            evidence_count=1, last_verified=None),),
        provenance_summaries=("src_1",),
        temporal_status=TemporalStatus.CURRENT,
        unresolved_conflicts=(),
        prior_decisions=(),
        prior_outcomes=(),
        resource_state={
            "retrieval_calls_remaining": 5,
            "verification_calls_remaining": 5,
            "search_calls_remaining": 5,
            "reasoning_tokens_remaining": 512,
            "executive_steps_remaining": 8,
        },
        policy_facts=(),
        observation_signals=(),
    )


class TestI35ModelPacket:
    def test_packet_has_governor_field(self):
        """I3.5 packet must contain a governor field."""
        obs = _make_observation()
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=25)
        packet = serialize_governor_packet(obs, frame)
        assert "governor" in packet
        assert "current_bottlenecks" in packet["governor"]
        assert "candidate_actions" in packet["governor"]

    def test_packet_schema_is_i3_5(self):
        """I3.5 packet must have I3.5 schema."""
        obs = _make_observation()
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=25)
        packet = serialize_governor_packet(obs, frame)
        assert packet["schema"] == PACKET_SCHEMA

    def test_packet_sha256_deterministic(self):
        """Same packet produces same SHA-256."""
        obs = _make_observation()
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=25)
        packet = serialize_governor_packet(obs, frame)
        h1 = governor_packet_sha256(packet)
        h2 = governor_packet_sha256(packet)
        assert h1 == h2

    def test_no_governor_leakage(self):
        """Governor packet must not leak evaluator metadata."""
        import re
        obs = _make_observation()
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=25)
        packet = serialize_governor_packet(obs, frame)
        # Should not raise
        assert_no_governor_leakage(packet)

    def test_packet_json_is_valid(self):
        """Packet JSON is valid JSON."""
        obs = _make_observation()
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=25)
        packet = serialize_governor_packet(obs, frame)
        j = governor_packet_json(packet)
        parsed = json.loads(j)
        assert parsed["schema"] == PACKET_SCHEMA


class TestI35Prompt:
    def test_prompt_id_is_i3_5(self):
        """Prompt ID must be I3.5."""
        assert PROMPT_ID == "DAPH_V2B_I3_5_SYSTEM_PROMPT_V1"

    def test_prompt_sha256_deterministic(self):
        """Prompt SHA-256 is deterministic."""
        h1 = prompt_sha256()
        h2 = prompt_sha256()
        assert h1 == h2

    def test_prompt_mentions_governor(self):
        """Prompt must instruct the model to use the governor frame."""
        assert "governor" in SYSTEM_PROMPT.lower()

    def test_prompt_mentions_consequence(self):
        """Prompt must mention consequence-aware reasoning."""
        assert "consequence" in SYSTEM_PROMPT.lower()

    def test_prompt_mentions_repeat_penalty(self):
        """Prompt must mention repeat_penalty."""
        assert "repeat_penalty" in SYSTEM_PROMPT

    def test_prompt_has_seven_actions(self):
        """Prompt must list all seven actions."""
        for action in ("RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE",
                       "ANSWER", "DEFER", "STOP"):
            assert action in SYSTEM_PROMPT
