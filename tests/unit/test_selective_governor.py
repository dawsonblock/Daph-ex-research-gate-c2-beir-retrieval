"""Unit tests for the selective governor intervention gate."""
import pytest
from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot,
    DecisionSummary,
    TemporalStatus,
    VerificationSummary,
    VerificationState,
)
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.executive.selective_governor import (
    BaseInterventionPredictor,
    CalibratedLinearPredictor,
    InterventionDecision,
    InterventionFeatures,
    InterventionPrediction,
    RuleBasedInterventionPredictor,
    SelectiveGovernorGate,
    compute_gate_identity,
    decision_sha256,
    extract_features,
    serialize_decision,
)


def _make_observation(cognitive_state=None) -> ControllerObservation:
    return ControllerObservation(
        task_id="test_task_001",
        task_summary="Analyze evidence for hypothesis verification.",
        resource_state={
            "retrieval": 5,
            "verification": 5,
            "search": 5,
            "reasoning_tokens": 512,
            "time_ms": 10000,
            "executive_steps_remaining": 8,
            "retrieval_calls_remaining": 5,
            "verification_calls_remaining": 5,
            "search_calls_remaining": 5,
            "reasoning_tokens_remaining": 512,
        },
        allowed_actions=tuple(V2B_ACTIONS),
        executed_actions=(),
        rejected_actions=(),
        cognitive_state=cognitive_state,
    )


class TestFeatureExtraction:
    def test_extract_features_no_cognitive_state(self):
        obs = _make_observation(cognitive_state=None)
        feats = extract_features(
            obs, remaining_steps=24, prior_actions=(), prior_outcomes=(),
        )
        assert feats.has_cognitive_state is False
        assert feats.remaining_steps == 24
        assert feats.prior_action_count == 0
        assert feats.last_action is None
        assert feats.verification_state == "NONE"
        assert feats.evidence_count == 0

    def test_extract_features_with_cognitive_state(self):
        v_rec = VerificationSummary(
            target_id="ev_0",
            state=VerificationState.SUFFICIENT,
            evidence_count=2,
            last_verified=100,
        )
        cs = CognitiveStateSnapshot(
            task_id="test_task_001",
            task_summary="Test",
            relevant_memories=("ev_0", "ev_1"),
            verification_states=(v_rec,),
            provenance_summaries=(),
            temporal_status=TemporalStatus.CURRENT,
            unresolved_conflicts=(),
            prior_decisions=(),
            prior_outcomes=(),
            resource_state={},
            policy_facts=(),
            observation_signals={},
        )
        obs = _make_observation(cognitive_state=cs)
        feats = extract_features(
            obs, remaining_steps=20, prior_actions=("RETRIEVE",), prior_outcomes=("OK",),
        )
        assert feats.has_cognitive_state is True
        assert feats.remaining_steps == 20
        assert feats.prior_action_count == 1
        assert feats.last_action == "RETRIEVE"
        assert feats.verification_state == "SUFFICIENT"
        assert feats.verified_count == 1
        assert feats.evidence_count == 2
        assert feats.temporal_status == "CURRENT"

    def test_numeric_vector_length(self):
        obs = _make_observation(cognitive_state=None)
        feats = extract_features(
            obs, remaining_steps=24, prior_actions=(), prior_outcomes=(),
        )
        vec = feats.to_numeric_vector()
        assert isinstance(vec, list)
        assert len(vec) == 28


class TestRuleBasedPredictor:
    def test_step0_sufficient_stop_hazard(self):
        pred = RuleBasedInterventionPredictor()
        feats = InterventionFeatures(
            remaining_steps=24,
            prior_action_count=0,
            last_action=None,
            last_outcome=None,
            repeated_no_gain=False,
            has_cognitive_state=True,
            evidence_count=1,
            verified_count=1,
            verification_state="SUFFICIENT",
            temporal_status="CURRENT",
            conflict_count=0,
            reasoning_depth=0,
            retrieval_budget_remaining=5,
            verification_budget_remaining=5,
            search_budget_remaining=5,
            reasoning_budget_remaining=512,
            chain_started=False,
            chain_completed=False,
            chain_length=0,
            chain_stage=0,
        )
        p = pred.predict(feats)
        assert p.expected_delta_utility <= -100.0
        assert p.harm_probability >= 0.99
        assert "STOP_HAZARD" in p.reason

    def test_step0_missing_verify_hazard(self):
        pred = RuleBasedInterventionPredictor()
        feats = InterventionFeatures(
            remaining_steps=24,
            prior_action_count=0,
            last_action=None,
            last_outcome=None,
            repeated_no_gain=False,
            has_cognitive_state=True,
            evidence_count=1,
            verified_count=0,
            verification_state="MISSING",
            temporal_status="UNKNOWN",
            conflict_count=1,
            reasoning_depth=0,
            retrieval_budget_remaining=5,
            verification_budget_remaining=5,
            search_budget_remaining=5,
            reasoning_budget_remaining=512,
            chain_started=False,
            chain_completed=False,
            chain_length=0,
            chain_stage=0,
        )
        p = pred.predict(feats)
        assert p.expected_delta_utility < 0.0
        assert p.harm_probability >= 0.80
        assert "PREMATURE_INTERVENTION" in p.reason

    def test_post_verify_safe_help_region(self):
        pred = RuleBasedInterventionPredictor()
        feats = InterventionFeatures(
            remaining_steps=22,
            prior_action_count=2,
            last_action="VERIFY",
            last_outcome="UNRESOLVED",
            repeated_no_gain=False,
            has_cognitive_state=True,
            evidence_count=1,
            verified_count=0,
            verification_state="MISSING",
            temporal_status="UNKNOWN",
            conflict_count=0,
            reasoning_depth=0,
            retrieval_budget_remaining=4,
            verification_budget_remaining=4,
            search_budget_remaining=5,
            reasoning_budget_remaining=512,
            chain_started=False,
            chain_completed=False,
            chain_length=0,
            chain_stage=0,
        )
        p = pred.predict(feats)
        assert p.expected_delta_utility > 0.0
        assert p.harm_probability <= 0.05
        assert "SAFE_HELP" in p.reason


class TestSelectiveGovernorGate:
    def test_gate_defaults_to_skip_under_harm(self):
        gate = SelectiveGovernorGate(predictor=RuleBasedInterventionPredictor())
        obs = _make_observation()
        decision = gate.assess(
            obs, remaining_steps=24, prior_actions=(), prior_outcomes=(),
        )
        assert decision.intervene is False
        assert decision.harm_probability >= 0.50
        assert "SKIP" in decision.reason_code

    def test_gate_approves_intervention_when_beneficial(self):
        class MockBeneficialPredictor(BaseInterventionPredictor):
            def predict(self, features):
                return InterventionPrediction(
                    expected_delta_utility=15.0,
                    harm_probability=0.05,
                    help_probability=0.90,
                    confidence=0.95,
                    reason="MOCK_HIGH_VALUE_DISCOVERY",
                )

        gate = SelectiveGovernorGate(
            predictor=MockBeneficialPredictor(),
            delta_u_threshold=5.0,
            max_harm_probability=0.15,
            min_confidence=0.60,
        )
        obs = _make_observation()
        decision = gate.assess(
            obs, remaining_steps=20, prior_actions=("RETRIEVE",), prior_outcomes=("OK",),
        )
        assert decision.intervene is True
        assert decision.expected_delta_utility == 15.0
        assert decision.harm_probability == 0.05
        assert "INTERVENE_APPROVED" in decision.reason_code

    def test_gate_fails_closed_on_error(self):
        class CrashingPredictor(BaseInterventionPredictor):
            def predict(self, features):
                raise RuntimeError("Predictor internal failure")

        gate = SelectiveGovernorGate(predictor=CrashingPredictor())
        obs = _make_observation()
        decision = gate.assess(
            obs, remaining_steps=24, prior_actions=(), prior_outcomes=(),
        )
        assert decision.intervene is False
        assert "SKIP_ERROR_FALLBACK" in decision.reason_code


class TestGateIdentityAndSerialization:
    def test_identity_deterministic(self):
        id1 = compute_gate_identity()
        id2 = compute_gate_identity()
        assert id1["gate_identity_sha256"] == id2["gate_identity_sha256"]
        assert len(id1["gate_identity_sha256"]) == 64

    def test_serializer_roundtrip(self):
        decision = InterventionDecision(
            intervene=False,
            expected_delta_utility=-12.5,
            harm_probability=0.95,
            confidence=0.90,
            reason_code="SKIP_PREDICTION:TEST",
            feature_summary={"remaining_steps": 24},
        )
        d = serialize_decision(decision)
        assert d["decision"] == "SKIP"
        assert d["intervene"] is False
        assert d["expected_delta_utility"] == -12.5
        sha = decision_sha256(decision)
        assert len(sha) == 64
