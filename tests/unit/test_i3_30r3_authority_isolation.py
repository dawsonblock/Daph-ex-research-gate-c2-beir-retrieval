"""Tests for I3.30R3 authority isolation — treatment purity.

These tests verify the core invariant of the authority-isolation experiment:
V3_SHADOW and V3_HARD must execute identical code up to the final override
decision. The ONLY difference is whether the hard override is applied.
"""
import pytest
from daph.authority.isolation import (
    ArmMode,
    AuthorityDecisionV3,
    AuthorityEffect,
    evaluate_v3_authority,
    apply_authority,
    classify_authority_effect,
    state_sha,
    build_normalized_receipt,
)
from daph.authority.policy_v3 import StructuralStateV3


def _make_structural(**overrides) -> StructuralStateV3:
    """Build a StructuralStateV3 with sensible defaults for testing."""
    defaults = dict(
        has_competing_unverified_support=False,
        n_hyp_unverified_support=0,
        n_hyp_unverified_contradiction=0,
        can_verify=False,
        verify_budget_exhausted=True,
        all_evidence_verified=True,
        n_hyp_with_verified_support=1,
        n_hyp_with_verified_contradiction=0,
        n_hyp_with_mixed_verified=0,
        n_viable_hypotheses=1,
        n_eliminated_hypotheses=0,
        has_unique_verified_supported_hypothesis=True,
        has_verified_unresolved_competition=False,
        verified_hyp_action_is_answer=True,
        verified_hyp_action_is_defer=False,
    )
    defaults.update(overrides)
    return StructuralStateV3(**defaults)


def _make_q_values(**overrides) -> dict[str, float]:
    """Build Q values with a clear ANSWER gap for testing."""
    defaults = {"ANSWER": 95.0, "VERIFY": 80.0, "DEFER": 20.0, "REASON_MORE": 70.0}
    defaults.update(overrides)
    return defaults


# ============================================================
# Test 1: evaluate_v3_authority is arm-agnostic
# ============================================================

class TestEvaluateV3AuthorityIsArmAgnostic:
    """The evaluation function must not know which arm it is running."""

    def test_evaluation_does_not_require_arm(self):
        """evaluate_v3_authority must work without an arm parameter."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        assert decision.arm == ""  # arm is filled by caller
        assert decision.certificate_evaluated is True

    def test_evaluation_is_deterministic(self):
        """Same inputs → same decision, every time."""
        struct = _make_structural()
        q = _make_q_values()
        args = dict(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        d1 = evaluate_v3_authority(**args)
        d2 = evaluate_v3_authority(**args)
        assert d1.q_values == d2.q_values
        assert d1.epsilon_set == d2.epsilon_set
        assert d1.certificate_passed == d2.certificate_passed
        assert d1.would_force == d2.would_force
        assert d1.forced_action == d2.forced_action


# ============================================================
# Test 2: V3_SHADOW and V3_HARD produce identical decisions up to force
# ============================================================

class TestTreatmentPurity:
    """V3_SHADOW and V3_HARD must be identical before force application."""

    def test_shadow_does_not_force(self):
        """V3_SHADOW must never apply force, even when would_force is True."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        assert decision.would_force is True
        assert decision.forced_action == "ANSWER"

        llm_action = "VERIFY"
        executed, updated = apply_authority(decision, ArmMode.V3_SHADOW, llm_action)
        assert updated.force_applied is False
        assert executed == llm_action
        assert updated.executed_action == llm_action
        assert updated.action_changed is False

    def test_hard_does_force_when_would_force(self):
        """V3_HARD must apply force when would_force is True."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        assert decision.would_force is True

        llm_action = "VERIFY"
        executed, updated = apply_authority(decision, ArmMode.V3_HARD, llm_action)
        assert updated.force_applied is True
        assert executed == "ANSWER"
        assert updated.executed_action == "ANSWER"
        assert updated.action_changed is True

    def test_hard_does_not_force_when_would_force_false(self):
        """V3_HARD must not force when would_force is False."""
        struct = _make_structural(
            has_unique_verified_supported_hypothesis=False,
            verified_hyp_action_is_answer=False,
        )
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        assert decision.would_force is False

        llm_action = "VERIFY"
        executed, updated = apply_authority(decision, ArmMode.V3_HARD, llm_action)
        assert updated.force_applied is False
        assert executed == llm_action

    def test_shadow_and_hard_share_identical_evaluation(self):
        """The evaluation step must produce identical results for both arms."""
        struct = _make_structural()
        q = _make_q_values()
        legal = ["ANSWER", "VERIFY", "DEFER", "REASON_MORE"]

        # Both arms call the same evaluate_v3_authority
        decision = evaluate_v3_authority(
            q_values=q, legal_actions=legal, structural=struct,
        )

        # The decision is arm-agnostic
        # Only apply_authority differs
        llm_action = "VERIFY"
        _, shadow_result = apply_authority(decision, ArmMode.V3_SHADOW, llm_action)
        _, hard_result = apply_authority(decision, ArmMode.V3_HARD, llm_action)

        # Everything except force_applied and executed_action must match
        assert shadow_result.q_values == hard_result.q_values
        assert shadow_result.q_argmax == hard_result.q_argmax
        assert shadow_result.q_gap == hard_result.q_gap
        assert shadow_result.epsilon_set == hard_result.epsilon_set
        assert shadow_result.certificate_passed == hard_result.certificate_passed
        assert shadow_result.certificate_type == hard_result.certificate_type
        assert shadow_result.would_force == hard_result.would_force
        assert shadow_result.forced_action == hard_result.forced_action
        assert shadow_result.llm_proposed_action == hard_result.llm_proposed_action

        # Only these differ:
        assert shadow_result.force_applied is False
        assert hard_result.force_applied is True
        assert shadow_result.executed_action == "VERIFY"
        assert hard_result.executed_action == "ANSWER"

    def test_neutral_when_llm_already_chose_forced_action(self):
        """If LLM already chose the forced action, force_applied but action_changed=False."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        llm_action = "ANSWER"  # LLM already chose ANSWER
        executed, updated = apply_authority(decision, ArmMode.V3_HARD, llm_action)
        assert updated.force_applied is True
        assert updated.action_changed is False  # no behavioral change
        assert executed == "ANSWER"


# ============================================================
# Test 3: Authority effect classification
# ============================================================

class TestAuthorityEffectClassification:
    """Test the RESCUE/BREAK/BENEFICIAL/HARMFUL/NEUTRAL classification."""

    def test_rescue(self):
        """Forced succeeds, shadow fails → RESCUE."""
        effect = classify_authority_effect(
            forced_success=True, shadow_success=False,
            forced_utility=90.0, shadow_utility=-10.0,
        )
        assert effect == AuthorityEffect.RESCUE

    def test_break(self):
        """Forced fails, shadow succeeds → BREAK."""
        effect = classify_authority_effect(
            forced_success=False, shadow_success=True,
            forced_utility=-10.0, shadow_utility=90.0,
        )
        assert effect == AuthorityEffect.BREAK

    def test_beneficial_nonrescue(self):
        """Both succeed, forced higher utility → BENEFICIAL_NONRESCUE."""
        effect = classify_authority_effect(
            forced_success=True, shadow_success=True,
            forced_utility=95.0, shadow_utility=80.0,
        )
        assert effect == AuthorityEffect.BENEFICIAL_NONRESCUE

    def test_harmful_nonbreak(self):
        """Both succeed, forced lower utility → HARMFUL_NONBREAK."""
        effect = classify_authority_effect(
            forced_success=True, shadow_success=True,
            forced_utility=80.0, shadow_utility=95.0,
        )
        assert effect == AuthorityEffect.HARMFUL_NONBREAK

    def test_neutral_same_utility(self):
        """Both succeed, same utility → NEUTRAL."""
        effect = classify_authority_effect(
            forced_success=True, shadow_success=True,
            forced_utility=90.0, shadow_utility=90.0,
        )
        assert effect == AuthorityEffect.NEUTRAL

    def test_neutral_both_fail(self):
        """Both fail, same utility → NEUTRAL."""
        effect = classify_authority_effect(
            forced_success=False, shadow_success=False,
            forced_utility=-10.0, shadow_utility=-10.0,
        )
        assert effect == AuthorityEffect.NEUTRAL

    def test_neutral_within_tolerance(self):
        """Utility difference within tolerance → NEUTRAL."""
        effect = classify_authority_effect(
            forced_success=True, shadow_success=True,
            forced_utility=90.005, shadow_utility=90.0,
            utility_tolerance=0.01,
        )
        assert effect == AuthorityEffect.NEUTRAL


# ============================================================
# Test 4: Receipt normalization
# ============================================================

class TestNormalizedReceipt:
    """Test that receipts contain all required fields."""

    def test_receipt_has_all_fields(self):
        """Receipt must contain all normalized fields."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        llm_action = "VERIFY"
        _, updated = apply_authority(decision, ArmMode.V3_HARD, llm_action)

        receipt = build_normalized_receipt(
            task_id="test_task",
            arm="v3_hard",
            step=2,
            state_features={"n_verified": 2, "steps_remaining": 5},
            decision=updated,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
        )

        required_fields = [
            "task_id", "arm", "step", "state_sha",
            "legal_actions", "epistemically_admissible_actions",
            "q_values", "q_argmax", "q_second_best", "q_gap",
            "epsilon_set",
            "certificate_evaluated", "certificate_passed",
            "certificate_type", "certificate_components",
            "authority_mode", "would_force", "forced_action",
            "llm_proposed_action", "executed_action",
            "force_applied", "action_changed",
            "structural_state", "resource_state",
        ]
        for field in required_fields:
            assert field in receipt, f"Missing field: {field}"

    def test_receipt_state_sha_is_deterministic(self):
        """Same state → same state_sha."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )

        sf = {"n_verified": 2, "steps_remaining": 5}
        r1 = build_normalized_receipt(
            task_id="t", arm="v3_hard", step=0,
            state_features=sf, decision=decision,
            legal_actions=["ANSWER"],
        )
        r2 = build_normalized_receipt(
            task_id="t", arm="v3_hard", step=0,
            state_features=sf, decision=decision,
            legal_actions=["ANSWER"],
        )
        assert r1["state_sha"] == r2["state_sha"]


# ============================================================
# Test 5: ArmMode enum
# ============================================================

class TestArmMode:
    """Test the ArmMode enum."""

    def test_three_arms(self):
        assert len(ArmMode) == 3

    def test_arm_values(self):
        assert ArmMode.V1.value == "v1"
        assert ArmMode.V3_SHADOW.value == "v3_shadow"
        assert ArmMode.V3_HARD.value == "v3_hard"

    def test_arm_is_string_enum(self):
        assert isinstance(ArmMode.V1, str)
        assert isinstance(ArmMode.V3_SHADOW, str)
        assert isinstance(ArmMode.V3_HARD, str)


# ============================================================
# Test 6: Invariants for force application
# ============================================================

class TestForceInvariants:
    """Invariants that must hold for every force application."""

    def test_force_implies_certificate_passed(self):
        """If force_applied, then certificate_passed must be True."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        _, updated = apply_authority(decision, ArmMode.V3_HARD, "VERIFY")
        if updated.force_applied:
            assert updated.certificate_passed is True

    def test_force_implies_would_force(self):
        """If force_applied, then would_force must be True."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        _, updated = apply_authority(decision, ArmMode.V3_HARD, "VERIFY")
        if updated.force_applied:
            assert updated.would_force is True

    def test_force_implies_forced_action_equals_executed(self):
        """If force_applied, then executed_action == forced_action."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        _, updated = apply_authority(decision, ArmMode.V3_HARD, "VERIFY")
        if updated.force_applied:
            assert updated.executed_action == updated.forced_action

    def test_shadow_never_forces(self):
        """V3_SHADOW must never have force_applied=True."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        _, updated = apply_authority(decision, ArmMode.V3_SHADOW, "VERIFY")
        assert updated.force_applied is False

    def test_shadow_executed_equals_llm(self):
        """V3_SHADOW executed_action must equal llm_proposed_action."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        _, updated = apply_authority(decision, ArmMode.V3_SHADOW, "VERIFY")
        assert updated.executed_action == updated.llm_proposed_action

    def test_forced_action_is_terminal(self):
        """Forced actions must be terminal (ANSWER or DEFER)."""
        struct = _make_structural()
        q = _make_q_values()
        decision = evaluate_v3_authority(
            q_values=q,
            legal_actions=["ANSWER", "VERIFY", "DEFER", "REASON_MORE"],
            structural=struct,
        )
        if decision.forced_action:
            assert decision.forced_action in ("ANSWER", "DEFER")
