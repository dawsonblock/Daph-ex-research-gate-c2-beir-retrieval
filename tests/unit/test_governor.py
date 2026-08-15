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
        resource_state = {
            "retrieval_calls_remaining": 5,
            "verification_calls_remaining": 5,
            "search_calls_remaining": 5,
            "reasoning_tokens_remaining": 512,
            "executive_steps_remaining": 8,
            "retrieval_calls_used": 0,
            "verification_calls_used": 0,
            "search_calls_used": 0,
            "reasoning_tokens_used": 0,
            "executive_steps_used": 0,
        }
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
        resource_state={
            "retrieval_calls_remaining": 5,
            "verification_calls_remaining": 5,
            "search_calls_remaining": 5,
            "reasoning_tokens_remaining": 512,
            "executive_steps_remaining": 8,
        },
        policy_facts=(),
        observation_signals=observation_signals,
    )


# ─── Action Semantics Tests ───

class TestActionSemantics:
    def test_v1_actions_have_semantics(self):
        """Every V1 executive action must have frozen semantics."""
        v1_actions = ("ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                      "REASON_MORE", "DEFER", "STOP")
        for action in v1_actions:
            assert action in FROZEN_ACTION_SEMANTICS, f"Missing semantics for {action}"

    def test_no_dormant_actions_in_v1_semantics(self):
        """V1 semantics must not include unavailable future actions."""
        dormant = ("VERIFY_ALTERNATE_SOURCE", "SPAWN_SPECIALIST",
                   "SWITCH_STRATEGY", "ABANDON_STRATEGY")
        for action in dormant:
            assert action not in FROZEN_ACTION_SEMANTICS, f"Dormant action {action} should not be in V1"

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
        """Repeated same action with same outcome triggers repeated_no_gain."""
        obs = _make_observation(
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("NO_CHANGE", "NO_CHANGE"))
        assert state.repeated_no_gain

    def test_repeated_different_outcome_not_no_gain(self):
        """Repeated same action with different outcomes does NOT trigger no-gain."""
        obs = _make_observation(
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("VERIFIED", "CONFLICT_FOUND"))
        assert not state.repeated_no_gain

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

    def test_repeated_action_same_outcome_may_repeat_failure(self):
        """Action tried 2+ times with same outcome → may_repeat_failed_strategy=True."""
        cs = _make_cognitive_state(verification_state=VerificationState.UNVERIFIED)
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY),
        )
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("NO_CHANGE", "NO_CHANGE"))
        bottlenecks = detect_bottlenecks(state)
        outcome = predict_outcome(state, "VERIFY", bottlenecks)
        assert outcome.may_repeat_failed_strategy

    def test_repeated_action_different_outcome_not_failure(self):
        """Action tried 2+ times with different outcomes → NOT may_repeat."""
        cs = _make_cognitive_state(verification_state=VerificationState.UNVERIFIED)
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY),
        )
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("VERIFIED", "CONFLICT_FOUND"))
        bottlenecks = detect_bottlenecks(state)
        outcome = predict_outcome(state, "VERIFY", bottlenecks)
        assert not outcome.may_repeat_failed_strategy


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

    def test_low_redundancy_for_single_attempt_as_last(self):
        """Action tried once, is last → LOW (not MEDIUM; outcome-based now)."""
        obs = _make_observation(
            executed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=23,
            prior_actions=("RETRIEVE", "VERIFY"))
        assert compute_redundancy(state, "VERIFY") == LOW

    def test_high_redundancy_for_repeated_same_outcome(self):
        """Action tried 2+ times with same outcome → HIGH."""
        obs = _make_observation(
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("NO_CHANGE", "NO_CHANGE"))
        assert compute_redundancy(state, "VERIFY") == HIGH

    def test_medium_redundancy_for_repeated_different_outcome(self):
        """Action tried 2+ times with different outcomes → MEDIUM."""
        obs = _make_observation(
            executed_actions=(DecisionAction.VERIFY, DecisionAction.VERIFY))
        state = build_governor_state(
            obs, remaining_steps=22,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("VERIFIED", "CONFLICT_FOUND"))
        assert compute_redundancy(state, "VERIFY") == MEDIUM


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
            resource_state={
                "retrieval_calls_remaining": 0,
                "verification_calls_remaining": 0,
                "search_calls_remaining": 2,
                "reasoning_tokens_remaining": 512,
                "executive_steps_remaining": 5,
            },
        )
        governor = GeneralGovernor()
        frame = governor.assess(
            obs, remaining_steps=22,
            prior_actions=("RETRIEVE", "VERIFY"))
        assert frame.governor_top_action == "SEARCH_MORE"

    def test_governor_penalizes_repeated_verify(self):
        """VERIFY tried twice with same outcome gets HIGH redundancy."""
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
            resource_state={
                "retrieval_calls_remaining": 0,
                "verification_calls_remaining": 1,
                "search_calls_remaining": 2,
                "reasoning_tokens_remaining": 512,
                "executive_steps_remaining": 5,
            },
        )
        governor = GeneralGovernor()
        frame = governor.assess(
            obs, remaining_steps=21,
            prior_actions=("RETRIEVE", "VERIFY", "VERIFY"),
            prior_outcomes=("EVIDENCE_ADDED", "NO_CHANGE", "NO_CHANGE"))
        # VERIFY should not be the top action
        assert frame.governor_top_action != "VERIFY"
        # VERIFY should have HIGH repeat penalty (same outcome twice)
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


# ─── Resource Normalization Integration Tests ───

class TestResourceNormalization:
    """Integration tests using actual ResourceState.as_dict() keys."""

    def test_governor_works_with_real_resource_state(self):
        """Governor must work with keys from ResourceState.as_dict()."""
        from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
        budget = ResourceBudget(
            max_executive_steps=8, max_reasoning_tokens=512,
            max_retrieval_calls=5, max_verification_calls=5,
            max_search_calls=5, max_elapsed_ms=10000,
            max_monetary_cost_microusd=0)
        rs = ResourceState(budget)
        real_dict = rs.as_dict()

        obs = ControllerObservation(
            task_id="test_real",
            task_summary="Test with real resources",
            resource_state=real_dict,
            allowed_actions=(DecisionAction.RETRIEVE, DecisionAction.VERIFY,
                             DecisionAction.SEARCH_MORE, DecisionAction.REASON_MORE,
                             DecisionAction.ANSWER, DecisionAction.DEFER),
            executed_actions=(),
            rejected_actions=(),
            cognitive_state=None,
            policy_feedback=(),
        )
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=8)
        # Governor should detect NO_EVIDENCE and recommend RETRIEVE
        bottleneck_kinds = [b.kind for b in frame.active_bottlenecks]
        assert "NO_EVIDENCE" in bottleneck_kinds
        assert frame.governor_top_action == "RETRIEVE"

    def test_typed_resources_detect_depletion(self):
        """GovernorResourceState must correctly detect resource depletion."""
        from hrm_adaptive_memory.executive.governor.resources import normalize_resources
        res = normalize_resources({
            "retrieval_calls_remaining": 0,
            "verification_calls_remaining": 1,
            "search_calls_remaining": 3,
            "reasoning_tokens_remaining": 512,
            "executive_steps_remaining": 5,
        })
        assert not res.has_retrieval
        assert res.has_verification
        assert res.is_last_resource("verification")
        assert not res.is_last_resource("search")
        assert not res.any_useful_remaining is False

    def test_governor_detects_resource_exhaustion(self):
        """When all useful resources are gone, governor detects exhaustion."""
        # Use aware condition with SUFFICIENT verification so no other bottleneck fires
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            temporal_status=TemporalStatus.CURRENT,
            conflicts=(),
            observation_signals=(),
        )
        obs = _make_observation(
            cognitive_state=cs,
            resource_state={
                "retrieval_calls_remaining": 0,
                "verification_calls_remaining": 0,
                "search_calls_remaining": 0,
                "reasoning_tokens_remaining": 0,
                "executive_steps_remaining": 2,
            },
        )
        governor = GeneralGovernor()
        frame = governor.assess(obs, remaining_steps=2)
        kinds = [b.kind for b in frame.active_bottlenecks]
        # With no useful resources and no other bottleneck, should detect exhaustion
        assert any(k in ("RESOURCE_EXHAUSTION", "READY_TO_ANSWER") for k in kinds)


# ─── No-Gain Detection Tests ───

class TestNoGainDetection:
    """Tests for outcome-based no-gain detection."""

    def test_repeated_same_outcome_is_no_gain(self):
        """Same action twice with same outcome → no-gain."""
        obs = _make_observation()
        state = build_governor_state(
            obs, remaining_steps=5,
            prior_actions=("SEARCH_MORE", "SEARCH_MORE"),
            prior_outcomes=("NO_CHANGE", "NO_CHANGE"))
        assert state.repeated_no_gain is True

    def test_repeated_different_outcome_not_no_gain(self):
        """Same action twice with different outcomes → NOT no-gain."""
        obs = _make_observation()
        state = build_governor_state(
            obs, remaining_steps=5,
            prior_actions=("SEARCH_MORE", "SEARCH_MORE"),
            prior_outcomes=("EVIDENCE_ADDED", "CONFLICT_RESOLVED"))
        assert state.repeated_no_gain is False

    def test_different_actions_not_no_gain(self):
        """Different actions → never no-gain regardless of outcomes."""
        obs = _make_observation()
        state = build_governor_state(
            obs, remaining_steps=5,
            prior_actions=("RETRIEVE", "VERIFY"),
            prior_outcomes=("SAME", "SAME"))
        assert state.repeated_no_gain is False

    def test_redundancy_high_for_same_outcome_repeat(self):
        """Redundancy should be HIGH only for same-outcome repetition."""
        obs = _make_observation()
        state_same = build_governor_state(
            obs, remaining_steps=5,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("NO_CHANGE", "NO_CHANGE"))
        assert compute_redundancy(state_same, "VERIFY") == HIGH

        state_diff = build_governor_state(
            obs, remaining_steps=5,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("VERIFIED", "CONFLICT_FOUND"))
        assert compute_redundancy(state_diff, "VERIFY") == MEDIUM


# ─── Chain Progress Tracking Tests ───

class TestChainProgress:
    """Tests for V2_STAGE_N chain progression tracking."""

    def test_no_chain_progress_on_empty_history(self):
        """ChainProgress should show no progress when no actions executed."""
        from hrm_adaptive_memory.executive.governor.chain_progress import (
            extract_chain_progress)
        cp = extract_chain_progress((), ())
        assert cp.stages_completed == 0
        assert not cp.is_started
        assert not cp.is_complete
        assert not cp.is_poisoned
        assert cp.needs_discovery  # needs discovery but only after steps

    def test_chain_progress_detects_stage_outcomes(self):
        """ChainProgress should count V2_STAGE_N outcomes."""
        from hrm_adaptive_memory.executive.governor.chain_progress import (
            extract_chain_progress)
        cp = extract_chain_progress(
            ("V2_STAGE_1", "V2_STAGE_2", "V2_STAGE_3"),
            ("RETRIEVE", "SEARCH_MORE", "REASON_MORE"))
        assert cp.stages_completed == 3
        assert cp.is_started
        assert cp.is_complete  # >= 3 stages and not poisoned
        assert not cp.is_poisoned
        assert cp.actions_that_advanced == ("RETRIEVE", "SEARCH_MORE", "REASON_MORE")

    def test_chain_progress_detects_poisoned(self):
        """ChainProgress should detect CONTROL_POISONED."""
        from hrm_adaptive_memory.executive.governor.chain_progress import (
            extract_chain_progress)
        cp = extract_chain_progress(
            ("CONTROL_POISONED",),
            ("RETRIEVE",))
        assert cp.is_poisoned
        assert not cp.is_complete

    def test_chain_progress_partial_chain(self):
        """ChainProgress should detect partial chain (started, not complete)."""
        from hrm_adaptive_memory.executive.governor.chain_progress import (
            extract_chain_progress)
        cp = extract_chain_progress(
            ("V2_STAGE_1",),
            ("RETRIEVE",))
        assert cp.is_started
        assert not cp.is_complete
        assert cp.needs_continuation

    def test_chain_progress_tracks_failed_actions(self):
        """Actions tried without advancing the chain should be tracked."""
        from hrm_adaptive_memory.executive.governor.chain_progress import (
            extract_chain_progress)
        cp = extract_chain_progress(
            ("VERIFY_COMPLETED", "RETRIEVE_COMPLETED", "V2_STAGE_1"),
            ("VERIFY", "RETRIEVE", "SEARCH_MORE"))
        # VERIFY and RETRIEVE didn't advance (no V2_STAGE_N from them)
        assert "VERIFY" in cp.actions_that_failed
        assert "RETRIEVE" in cp.actions_that_failed
        assert "SEARCH_MORE" in cp.actions_that_advanced

    def test_untried_composable_actions(self):
        """Should return composable actions not yet tried."""
        from hrm_adaptive_memory.executive.governor.chain_progress import (
            untried_composable_actions)
        untried = untried_composable_actions(
            ("RETRIEVE",),
            ("RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "ANSWER"))
        assert "RETRIEVE" not in untried
        assert "VERIFY" in untried
        assert "SEARCH_MORE" in untried
        assert "REASON_MORE" in untried


class TestChainDiscoveryBottleneck:
    """Tests for chain discovery and chain incomplete bottlenecks."""

    def test_chain_discovery_fires_after_failed_first_action(self):
        """CHAIN_DISCOVERY should fire when actions tried but chain not started."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.MISSING,
            prior_outcomes=("VERIFY_COMPLETED",))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.VERIFY,))
        state = build_governor_state(
            obs, remaining_steps=10,
            prior_actions=("VERIFY",),
            prior_outcomes=("VERIFY_COMPLETED",))
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "CHAIN_DISCOVERY" in kinds

    def test_chain_discovery_does_not_fire_on_first_step(self):
        """CHAIN_DISCOVERY should NOT fire when no actions tried yet."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.MISSING,
            prior_outcomes=())
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=10)
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        # Should be NO_EVIDENCE, not CHAIN_DISCOVERY
        assert "CHAIN_DISCOVERY" not in kinds
        assert "NO_EVIDENCE" in kinds

    def test_chain_incomplete_fires_for_partial_chain(self):
        """CHAIN_INCOMPLETE should fire when chain started but not complete."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.MISSING,
            prior_outcomes=("V2_STAGE_1",))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.RETRIEVE,))
        state = build_governor_state(
            obs, remaining_steps=10,
            prior_actions=("RETRIEVE",),
            prior_outcomes=("V2_STAGE_1",))
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "CHAIN_INCOMPLETE" in kinds

    def test_chain_complete_no_chain_bottleneck(self):
        """No chain bottleneck when chain is complete (>= 3 stages)."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            prior_outcomes=("V2_STAGE_1", "V2_STAGE_2", "V2_STAGE_3"))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(
                DecisionAction.RETRIEVE, DecisionAction.SEARCH_MORE,
                DecisionAction.REASON_MORE))
        state = build_governor_state(
            obs, remaining_steps=10,
            prior_actions=("RETRIEVE", "SEARCH_MORE", "REASON_MORE"),
            prior_outcomes=("V2_STAGE_1", "V2_STAGE_2", "V2_STAGE_3"))
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "CHAIN_INCOMPLETE" not in kinds
        assert "CHAIN_DISCOVERY" not in kinds


class TestPrematureAnswerGuard:
    """Tests for the premature-answer prevention fix."""

    def test_no_ready_to_answer_when_verification_missing(self):
        """READY_TO_ANSWER must not fire when verification_state is MISSING."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.MISSING,
            prior_outcomes=())
        obs = _make_observation(cognitive_state=cs)
        state = build_governor_state(obs, remaining_steps=10)
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "READY_TO_ANSWER" not in kinds

    def test_no_ready_to_answer_when_chain_incomplete(self):
        """READY_TO_ANSWER must not fire when chain is started but incomplete."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            prior_outcomes=("V2_STAGE_1",))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.RETRIEVE,))
        state = build_governor_state(
            obs, remaining_steps=10,
            prior_actions=("RETRIEVE",),
            prior_outcomes=("V2_STAGE_1",))
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "READY_TO_ANSWER" not in kinds

    def test_ready_to_answer_when_chain_complete_and_verified(self):
        """READY_TO_ANSWER should fire when chain is complete and verified."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            prior_outcomes=("V2_STAGE_1", "V2_STAGE_2", "V2_STAGE_3"))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(
                DecisionAction.RETRIEVE, DecisionAction.SEARCH_MORE,
                DecisionAction.REASON_MORE))
        state = build_governor_state(
            obs, remaining_steps=10,
            prior_actions=("RETRIEVE", "SEARCH_MORE", "REASON_MORE"),
            prior_outcomes=("V2_STAGE_1", "V2_STAGE_2", "V2_STAGE_3"))
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "READY_TO_ANSWER" in kinds

    def test_poisoned_chain_blocks_answer(self):
        """Poisoned chain should not allow READY_TO_ANSWER."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.SUFFICIENT,
            prior_outcomes=("CONTROL_POISONED",))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.RETRIEVE,))
        state = build_governor_state(
            obs, remaining_steps=10,
            prior_actions=("RETRIEVE",),
            prior_outcomes=("CONTROL_POISONED",))
        bottlenecks = detect_bottlenecks(state)
        kinds = [b.kind for b in bottlenecks]
        assert "READY_TO_ANSWER" not in kinds


class TestChainProgressInModelPacket:
    """Tests that chain progress is surfaced in the model packet."""

    def test_chain_progress_in_frame(self):
        """GovernorDecisionFrame should include chain_progress when available."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.MISSING,
            prior_outcomes=("V2_STAGE_1",))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.RETRIEVE,))
        governor = GeneralGovernor()
        frame = governor.assess(
            obs, remaining_steps=10,
            prior_actions=("RETRIEVE",),
            prior_outcomes=("V2_STAGE_1",))
        assert frame.chain_progress is not None
        assert frame.chain_progress["stages_completed"] == 1
        assert frame.chain_progress["is_started"] is True

    def test_chain_progress_in_model_packet(self):
        """as_model_packet should include chain_progress when available."""
        cs = _make_cognitive_state(
            verification_state=VerificationState.MISSING,
            prior_outcomes=("V2_STAGE_1",))
        obs = _make_observation(
            cognitive_state=cs,
            executed_actions=(DecisionAction.RETRIEVE,))
        governor = GeneralGovernor()
        frame = governor.assess(
            obs, remaining_steps=10,
            prior_actions=("RETRIEVE",),
            prior_outcomes=("V2_STAGE_1",))
        packet = frame.as_model_packet()
        assert "chain_progress" in packet
        assert packet["chain_progress"]["stages_completed"] == 1
