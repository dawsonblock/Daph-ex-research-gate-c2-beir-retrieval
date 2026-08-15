"""Tests for the General Governor executive layer."""
import pytest
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.governor.state import build_governor_state
from hrm_adaptive_memory.executive.governor.action_semantics import (
    FROZEN_ACTION_SEMANTICS, get_action_semantics, ActionSemantics)
from hrm_adaptive_memory.executive.governor.bottlenecks import (
    detect_bottlenecks, DecisionBottleneck, NONE, LOW, MEDIUM, HIGH, CRITICAL)
from hrm_adaptive_memory.executive.governor.transition_model import predict_outcome
from hrm_adaptive_memory.executive.governor.candidate_features import assess_candidate
from hrm_adaptive_memory.executive.governor.redundancy import compute_redundancy
from hrm_adaptive_memory.executive.governor.value_of_information import estimate_voi
from hrm_adaptive_memory.executive.governor.option_value import estimate_option_value
from hrm_adaptive_memory.executive.governor.identity import compute_governor_identity
from hrm_adaptive_memory.executive.governor.serializer import (
    serialize_frame, serialize_frame_dict, frame_sha256)
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.executive.actions import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, VerificationSummary, ConflictSummary,
    VerificationState, TemporalStatus)


# ─── Fixtures ───

def _make_observation(
    *,
    cognitive_state=None,
    executed_actions=(),
    allowed_actions=None,
    resource_state=None,
):
    if allowed_actions is None:
        allowed_actions = (
            DecisionAction.RETRIEVE, DecisionAction.VERIFY,
            DecisionAction.SEARCH_MORE, DecisionAction.REASON_MORE,
            DecisionAction.ANSWER, DecisionAction.DEFER,
        )
    if resource_state is None:
        resource_state = {"retrieval": 2, "verification": 2, "search": 2, "reasoning": 2}
    return ControllerObservation(
        task_id="test_001",
        task_summary="Test task",
        resource_state=resource_state,
        allowed_actions=allowed_actions,
        executed_actions=executed_actions,
        rejected_actions=(),
        cognitive_state=cognitive_state,
        policy_feedback=(),
    )


def _make_cognitive_state(
    *,
    verification_state=VerificationState.UNVERIFIED,
    temporal_status=TemporalStatus.CURRENT,
    conflicts=(),
    prior_outcomes=(),
    observation_signals=(),
):
    return CognitiveStateSnapshot(
        task_id="test_001",
        task_summary="Test task",
        relevant_memories=(),
        verification_states=(VerificationSummary(
            target_id="src_1", state=verification_state,
            evidence_count=1, last_verified=None),),
        provenance_summaries=("src_1",),
        temporal_status=temporal_status,
        unresolved_conflicts=conflicts,
        prior_decisions=(),
        prior_outcomes=prior_outcomes,
        resource_state={"retrieval": 2, "verification": 2, "search": 2, "reasoning": 2},
        policy_facts=(),
        observation_signals=observation_signals,
    )


# ─── Action Semantics Tests ───

class TestActionSemantics:
    def test_all_actions_have_semantics(self):
        """Every DecisionAction must have frozen semantics."""
        for action in DecisionAction:
            assert action.value in FROZEN_ACTION_SEMANTICS, f"Missing semantics for {action.value}"

    def test_semantics_are_frozen(self):
        """ActionSemantics must be frozen dataclasses."""
        sem = get_action_semantics("VERIFY")
        with pytest.raises(AttributeError):
            sem.can_add_evidence = True

    def test_terminal_actions(self):
        """ANSWER, DEFER, STOP are terminal."""
        for action in ("ANSWER", "DEFER", "STOP"):
            assert get_action_semantics(action).is_terminal
            assert get_action_semantics(action).can_terminate

    def test_non_terminal_actions(self):
        """RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE are not terminal."""
        for action in ("RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"):
            assert not get_action_semantics(action).is_terminal

    def test_external_information_actions(self):
        """RETRIEVE, VERIFY, SEARCH_MORE provide external information."""
        for action in ("RETRIEVE", "VERIFY", "SEARCH_MORE"):
            assert get_action_semantics(action).external_information
        assert not get_action_semantics("REASON_MORE").external_information

    def test_reason_more_is_internal_compute(self):
        """REASON_MORE is internal compute only."""
        sem = get_action_semantics("REASON_MORE")
        assert sem.internal_compute
        assert not sem.external_information
        assert not sem.can_add_evidence

    def test_verify_can_reduce_conflict(self):
        """VERIFY can reduce conflict."""
        assert get_action_semantics("VERIFY").can_reduce_conflict

    def test_search_more_can_add_evidence(self):
        """SEARCH_MORE can add evidence."""
        assert get_action_semantics("SEARCH_MORE").can_add_evidence

    def test_unknown_action_raises(self):
        """Unknown action should raise ValueError."""
        with pytest.raises(ValueError):
            get_action_semantics("UNKNOWN_ACTION")


# ─── Governor State Tests ───

class TestGovernorState:
    def test_blind_state_has_no_cognitive_state(self):
        """Blind condition: cognitive_state is None."""
        obs = _make_observation(cognitive_state=None)
        state = build_governor_state(obs, remaining_steps=25)
        assert not state.has_cognitive_state

    def test_aware_state_has_cognitive_state(self):
        """Aware condition: cognitive_state is present."""
        cs = _make_cognitive_state()
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        assert state.has_cognitive_state

    def test_action_count(self):
        """Action count tracks prior actions."""
        obs = _make_observation(
            executed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=23,
            prior_actions=("RETRIEVE", "VERIFY"))
        assert state.action_count("RETRIEVE") == 1
        assert state.action_count("VERIFY") == 1
        assert state.action_count("SEARCH_MORE") == 0

    def test_repeated_no_gain_detection(self):
        """Repeated same action triggers repeated_no_gain."""
        obs = _make_observation(
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"))
        assert state.repeated_no_gain

    def test_legal_actions_from_observation(self):
        """Legal actions come from observation.allowed_actions."""
        obs = _make_observation(
            allowed_actions=(DecisionAction.ANSWER, DecisionAction.DEFER))
        state = build_governor_state(obs, remaining_steps=25)
        assert set(state.legal_actions) == {"ANSWER", "DEFER"}


# ─── Bottleneck Detection Tests ───

class TestBottleneckDetection:
    def test_no_bottleneck_when_ready(self):
        """SUFFICIENT verification + no conflicts → READY_TO_ANSWER."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            conflicts=(),
        )
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        assert bottlenecks[0].kind == "READY_TO_ANSWER"
        assert bottlenecks[0].severity == NONE

    def test_unverified_evidence_bottleneck(self):
        """UNVERIFIED state → UNVERIFIED_EVIDENCE bottleneck."""
        cs = _make_cognitive_state(verification_state=VerificationState.UNVERIFIED)
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "UNVERIFIED_EVIDENCE" in kinds

    def test_conflict_bottleneck(self):
        """Unresolved conflict → UNRESOLVED_CONFLICT bottleneck."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            conflicts=(ConflictSummary(
                conflict_id="c1", relation="CONTRADICTS",
                source_lineage_count=2, status="RESOLVABLE"),),
        )
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        assert any(b.kind == "UNRESOLVED_CONFLICT" for b in bottlenecks)

    def test_blind_no_evidence_bottleneck(self):
        """Blind condition with no retrieval → NO_EVIDENCE bottleneck."""
        obs = _make_observation(cognitive_state=None)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        assert any(b.kind == "NO_EVIDENCE" for b in bottlenecks)

    def test_bottleneck_severity_ordering(self):
        """Bottlenecks are ordered by severity (most severe first)."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.UNVERIFIED,
            conflicts=(ConflictSummary(
                conflict_id="c1", relation="CONTRADICTS",
                source_lineage_count=2, status="RESOLVABLE"),),
        )
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        severity_order = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, NONE: 4}
        for i in range(len(bottlenecks) - 1):
            assert severity_order[bottlenecks[i].severity] <= severity_order[bottlenecks[i+1].severity]


# ─── Transition Model Tests ───

class TestTransitionModel:
    def test_verify_prediction(self):
        """VERIFY predicts verification_status change."""
        cs = _make_cognitive_state(verification_state=VerificationState.UNVERIFIED)
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        outcome = predict_outcome(state, "VERIFY", bottlenecks)
        assert "verification_status" in outcome.possible_changes
        assert not outcome.terminal
        # VERIFY accesses external sources but does not add new evidence
        assert not get_action_semantics("VERIFY").can_add_evidence

    def test_retrieve_adds_information(self):
        """RETRIEVE adds new external information."""
        cs = _make_cognitive_state(verification_state=VerificationState.MISSING)
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        outcome = predict_outcome(state, "RETRIEVE", bottlenecks)
        assert outcome.adds_new_information
        assert "evidence_count" in outcome.possible_changes

    def test_answer_is_terminal(self):
        """ANSWER is terminal."""
        cs = _make_cognitive_state(verification_state=VerificationState.SUFFICIENT)
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=25)
        bottlenecks = detect_bottlenecks(state)
        outcome = predict_outcome(state, "ANSWER", bottlenecks)
        assert outcome.terminal

    def test_repeated_action_may_repeat_failure(self):
        """Action tried 2+ times → may_repeat_failed_strategy=True."""
        cs = _make_cognitive_state(verification_state=VerificationState.UNVERIFIED)
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY),
        )
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"))
        bottlenecks = detect_bottlenecks(state)
        outcome = predict_outcome(state, "VERIFY", bottlenecks)
        assert outcome.may_repeat_failed_strategy


# ─── Redundancy Tests ───

class TestRedundancy:
    def test_no_redundancy_for_new_action(self):
        """Action never tried → NONE."""
        obs = _make_observation()
        state = build_governor_state(obs, remaining_steps=25)
        assert compute_redundancy(state, "VERIFY") == NONE

    def test_low_redundancy_for_one_attempt(self):
        """Action tried once, not last → LOW."""
        obs = _make_observation(
            executed_actions=(DecisionAction.VERIFY, DecisionAction.RETRIEVE))
        state = build_governor_state(
            obs, remaining_steps=23,
            prior_actions=("VERIFY", "RETRIEVE"))
        assert compute_redundancy(state, "VERIFY") == LOW

    def test_medium_redundancy_for_last_action(self):
        """Action tried once, is last → MEDIUM."""
        obs = _make_observation(
            executed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=23,
            prior_actions=("RETRIEVE", "VERIFY"))
        assert compute_redundancy(state, "VERIFY") == MEDIUM

    def test_high_redundancy_for_repeated_action(self):
        """Action tried 2+ times → HIGH."""
        obs = _make_observation(
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"))
        assert compute_redundancy(state, "VERIFY") == HIGH


# ─── Governor Assessor Tests ───

class TestGovernorAssessor:
    def test_governor_produces_frame(self):
        """Governor.assess() produces a valid frame."""
        governor = GeneralGovernor()
        obs = _make_observation()
        frame = governor.assess(obs, remaining_steps=25)
        assert frame is not None
        assert len(frame.candidates) > 0
        assert frame.governor_top_action is not None

    def test_governor_recommends_search_more_for_conflict(self):
        """When conflict is unresolved and VERIFY was tried, governor recommends SEARCH_MORE."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.UNVERIFIED,
            conflicts=(ConflictSummary(
                conflict_id="c1", relation="CONTRADICTS",
                source_lineage_count=2, status="RESOLVABLE"),),
        )
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY),
            allowed_actions=(DecisionAction.SEARCH_MORE, DecisionAction.REASON_MORE,
                             DecisionAction.ANSWER, DecisionAction.DEFER),
            resource_state={"retrieval": 0, "verification": 0, "search": 2, "reasoning": 2},
        )
        governor = GeneralGovernor()
        frame = governor.assess(
            obs, remaining_steps=22,
            prior_actions=("RETRIEVE", "VERIFY"))
        assert frame.governor_top_action == "SEARCH_MORE"

    def test_governor_penalizes_repeated_verify(self):
        """VERIFY tried twice gets HIGH redundancy and ranks below SEARCH_MORE."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.UNVERIFIED,
            conflicts=(ConflictSummary(
                conflict_id="c1", relation="CONTRADICTS",
                source_lineage_count=2, status="RESOLVABLE"),),
        )
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY, DecisionAction.VERIFY),
            allowed_actions=(DecisionAction.VERIFY, DecisionAction.SEARCH_MORE,
                             DecisionAction.REASON_MORE, DecisionAction.ANSWER, DecisionAction.DEFER),
            resource_state={"retrieval": 0, "verification": 1, "search": 2, "reasoning": 2},
        )
        governor = GeneralGovernor()
        frame = governor.assess(
            obs, remaining_steps=21,
            prior_actions=("RETRIEVE", "VERIFY", "VERIFY"))
        # VERIFY should not be the top action
        assert frame.governor_top_action != "VERIFY"
        # VERIFY should have HIGH repeat penalty
        verify_candidate = next(c for c in frame.candidates if c.action == "VERIFY")
        assert verify_candidate.repeat_penalty == HIGH

    def test_governor_blind_recommends_retrieve_first(self):
        """Blind condition with no prior actions → RETRIEVE is top."""
        obs = _make_observation(cognitive_state=None)
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=25)
        assert frame.governor_top_action == "RETRIEVE"

    def test_governor_ready_to_answer(self):
        """SUFFICIENT + CURRENT + no conflicts → READY_TO_ANSWER."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            conflicts=(),
        )
        obs = _make_observation(cognitive_state=cs)
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=25)
        assert frame.active_bottlenecks[0].kind == "READY_TO_ANSWER"

    def test_frame_serialization(self):
        """Frame can be serialized to dict and JSON."""
        governor = GeneralGovernor()
        obs = _make_observation()
        frame = governor.assess(obs, remaining_steps=25)
        d = serialize_frame_dict(frame)
        assert "current_bottlenecks" in d
        assert "candidate_actions" in d
        assert len(d["candidate_actions"]) > 0

    def test_frame_sha256_deterministic(self):
        """Same frame produces same SHA-256."""
        governor = GeneralGovernor()
        obs = _make_observation()
        frame = governor.assess(obs, remaining_steps=25)
        h1 = frame_sha256(frame)
        h2 = frame_sha256(frame)
        assert h1 == h2

    def test_no_oracle_leakage(self):
        """Governor frame must not contain oracle values, latent state, or topology IDs."""
        import re
        governor = GeneralGovernor()
        obs = _make_observation()
        frame = governor.assess(obs, remaining_steps=25)
        d = serialize_frame_dict(frame)
        serialized = str(d)
        # Check for forbidden fields using word boundaries
        for forbidden in ("oracle", "latent", "topology", "V_L", "V_O",
                          "latent_optimal", "observable_optimal", "topology_id"):
            pattern = r'\b' + re.escape(forbidden) + r'\b'
            assert not re.search(pattern, serialized), f"Governor frame leaks '{forbidden}'"


# ─── Identity Tests ───

class TestGovernorIdentity:
    def test_identity_is_deterministic(self):
        """Same configuration produces same identity hash."""
        id1 = compute_governor_identity()
        id2 = compute_governor_identity()
        assert id1["governor_sha256"] == id2["governor_sha256"]

    def test_identity_has_action_semantics_hash(self):
        """Identity includes action semantics hash."""
        identity = compute_governor_identity()
        assert "action_semantics_sha256" in identity
        assert len(identity["action_semantics_sha256"]) == 64

    def test_identity_has_scoring_weights(self):
        """Identity includes scoring weights."""
        identity = compute_governor_identity()
        assert "scoring_weights" in identity["configuration"]
        weights = identity["configuration"]["scoring_weights"]
        assert set(weights.keys()) == {"progress", "information", "cost", "risk", "redundancy", "options"}
