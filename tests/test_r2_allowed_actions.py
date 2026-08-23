#!/usr/bin/env python3
"""
Exhaustive unit tests for R2 allowed-action logic.

This module is part of the scientific intervention, so its logic must be
verified exhaustively.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest
from r2_allowed_actions import (
    ACTION_VOCABULARY,
    ALWAYS_LEGAL,
    ActionState,
    AllowedActionDecision,
    R2Arm,
    C0, D, E, DE,
    ALL_ARMS,
    compute_legal_actions,
    compute_epistemically_admissible_actions,
    compute_allowed_actions,
    allowed_actions_sha256,
)
from r2_schema import (
    build_action_schema,
    schema_sha256,
    c0_schema_identity_check,
    verify_schema_invariant,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_state(t2=False, can_retrieve=True, can_search=True, can_verify=True,
               steps_remaining=10):
    return ActionState(
        t2=t2,
        executive_steps_remaining=steps_remaining,
        can_retrieve=can_retrieve,
        can_search=can_search,
        can_verify=can_verify,
    )


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------

class TestLegalActions:
    def test_always_legal_present(self):
        state = make_state(can_retrieve=False, can_search=False, can_verify=False)
        legal = compute_legal_actions(state)
        assert ALWAYS_LEGAL <= legal

    def test_all_budgets_available(self):
        state = make_state(can_retrieve=True, can_search=True, can_verify=True)
        legal = compute_legal_actions(state)
        assert legal == ACTION_VOCABULARY

    def test_no_budgets(self):
        state = make_state(can_retrieve=False, can_search=False, can_verify=False)
        legal = compute_legal_actions(state)
        assert legal == ALWAYS_LEGAL

    def test_only_verify(self):
        state = make_state(can_retrieve=False, can_search=False, can_verify=True)
        legal = compute_legal_actions(state)
        assert legal == ALWAYS_LEGAL | {"VERIFY"}

    def test_only_retrieve(self):
        state = make_state(can_retrieve=True, can_search=False, can_verify=False)
        legal = compute_legal_actions(state)
        assert legal == ALWAYS_LEGAL | {"RETRIEVE"}

    def test_only_search(self):
        state = make_state(can_retrieve=False, can_search=True, can_verify=False)
        legal = compute_legal_actions(state)
        assert legal == ALWAYS_LEGAL | {"SEARCH_MORE"}


# ---------------------------------------------------------------------------
# Epistemically admissible actions
# ---------------------------------------------------------------------------

class TestEpistemicallyAdmissible:
    def test_c0_no_gate(self):
        """C0 never gates VERIFY."""
        state = make_state(t2=True)
        epistemic, gate_applied, reason = compute_epistemically_admissible_actions(state, C0)
        assert epistemic == ACTION_VOCABULARY
        assert gate_applied is False
        assert reason is None

    def test_e_no_gate(self):
        """E does not gate VERIFY (only label change)."""
        state = make_state(t2=True)
        epistemic, gate_applied, reason = compute_epistemically_admissible_actions(state, E)
        assert epistemic == ACTION_VOCABULARY
        assert gate_applied is False
        assert reason is None

    def test_d_gate_at_t2(self):
        """D gates VERIFY at T2."""
        state = make_state(t2=True)
        epistemic, gate_applied, reason = compute_epistemically_admissible_actions(state, D)
        assert "VERIFY" not in epistemic
        assert epistemic == ACTION_VOCABULARY - {"VERIFY"}
        assert gate_applied is True
        assert reason == "ALL_HYPOTHESES_ELIMINATED"

    def test_de_gate_at_t2(self):
        """DE gates VERIFY at T2."""
        state = make_state(t2=True)
        epistemic, gate_applied, reason = compute_epistemically_admissible_actions(state, DE)
        assert "VERIFY" not in epistemic
        assert gate_applied is True
        assert reason == "ALL_HYPOTHESES_ELIMINATED"

    def test_d_no_gate_when_not_t2(self):
        """D does not gate VERIFY when T2 is false."""
        state = make_state(t2=False)
        epistemic, gate_applied, reason = compute_epistemically_admissible_actions(state, D)
        assert epistemic == ACTION_VOCABULARY
        assert gate_applied is False
        assert reason is None

    def test_de_no_gate_when_not_t2(self):
        """DE does not gate VERIFY when T2 is false."""
        state = make_state(t2=False)
        epistemic, gate_applied, reason = compute_epistemically_admissible_actions(state, DE)
        assert epistemic == ACTION_VOCABULARY
        assert gate_applied is False
        assert reason is None

    def test_epistemic_independent_of_legal(self):
        """EpistemicallyAdmissible is the same regardless of budget state."""
        state_full = make_state(t2=False, can_retrieve=True, can_search=True, can_verify=True)
        state_empty = make_state(t2=False, can_retrieve=False, can_search=False, can_verify=False)
        for arm in ALL_ARMS:
            ep1, _, _ = compute_epistemically_admissible_actions(state_full, arm)
            ep2, _, _ = compute_epistemically_admissible_actions(state_empty, arm)
            assert ep1 == ep2, f"Epistemic admissibility changed with budget for {arm.name}"


# ---------------------------------------------------------------------------
# Allowed actions (intersection)
# ---------------------------------------------------------------------------

class TestAllowedActions:
    def test_c0_all_available(self):
        state = make_state(t2=False)
        decision = compute_allowed_actions(state, C0)
        assert decision.allowed == ACTION_VOCABULARY
        assert decision.verify_gate_applied is False

    def test_c0_t2_all_available(self):
        """C0 at T2 still has all actions (no gate)."""
        state = make_state(t2=True)
        decision = compute_allowed_actions(state, C0)
        assert decision.allowed == ACTION_VOCABULARY
        assert decision.verify_gate_applied is False

    def test_d_t2_verify_removed(self):
        """D at T2 removes VERIFY from allowed."""
        state = make_state(t2=True)
        decision = compute_allowed_actions(state, D)
        assert "VERIFY" not in decision.allowed
        assert decision.verify_gate_applied is True
        assert decision.verify_gate_reason == "ALL_HYPOTHESES_ELIMINATED"

    def test_d_not_t2_verify_present(self):
        """D without T2 keeps VERIFY in allowed."""
        state = make_state(t2=False)
        decision = compute_allowed_actions(state, D)
        assert "VERIFY" in decision.allowed
        assert decision.verify_gate_applied is False

    def test_de_t2_verify_removed(self):
        """DE at T2 removes VERIFY from allowed."""
        state = make_state(t2=True)
        decision = compute_allowed_actions(state, DE)
        assert "VERIFY" not in decision.allowed
        assert decision.verify_gate_applied is True

    def test_e_t2_verify_present(self):
        """E at T2 keeps VERIFY (only label change)."""
        state = make_state(t2=True)
        decision = compute_allowed_actions(state, E)
        assert "VERIFY" in decision.allowed
        assert decision.verify_gate_applied is False

    def test_allowed_is_intersection(self):
        """Allowed = Legal ∩ EpistemicallyAdmissible."""
        state = make_state(t2=True, can_verify=True, can_retrieve=False, can_search=False)
        decision = compute_allowed_actions(state, D)
        # Legal = ALWAYS_LEGAL + VERIFY (can_verify=True)
        assert "VERIFY" in decision.legal
        # Epistemic = ACTION_VOCABULARY - VERIFY (T2 gate)
        assert "VERIFY" not in decision.epistemically_admissible
        # Allowed = intersection → no VERIFY
        assert "VERIFY" not in decision.allowed
        assert decision.allowed == decision.legal & decision.epistemically_admissible

    def test_verify_legal_but_not_allowed(self):
        """VERIFY can be legal but not allowed (gated by epistemic admissibility)."""
        state = make_state(t2=True, can_verify=True)
        decision = compute_allowed_actions(state, D)
        assert "VERIFY" in decision.legal
        assert "VERIFY" not in decision.allowed

    def test_non_empty_allowed(self):
        """Allowed set is never empty (ANSWER/DEFER/STOP/REASON_MORE always present)."""
        state = make_state(t2=True, can_retrieve=False, can_search=False, can_verify=False)
        for arm in ALL_ARMS:
            decision = compute_allowed_actions(state, arm)
            assert len(decision.allowed) >= 4, f"Empty-ish allowed for {arm.name}"

    def test_allowed_sha256_deterministic(self):
        """SHA256 of allowed actions is deterministic."""
        state = make_state(t2=True)
        d1 = compute_allowed_actions(state, D)
        d2 = compute_allowed_actions(state, D)
        assert allowed_actions_sha256(d1.allowed) == allowed_actions_sha256(d2.allowed)

    def test_allowed_sha256_differs_when_gated(self):
        """SHA256 differs when VERIFY is gated vs not."""
        state = make_state(t2=True)
        c0_decision = compute_allowed_actions(state, C0)
        d_decision = compute_allowed_actions(state, D)
        assert allowed_actions_sha256(c0_decision.allowed) != allowed_actions_sha256(d_decision.allowed)


# ---------------------------------------------------------------------------
# Exhaustive: all arms × all T2 states × all budget combinations
# ---------------------------------------------------------------------------

class TestExhaustive:
    @pytest.mark.parametrize("arm", ALL_ARMS)
    @pytest.mark.parametrize("t2", [True, False])
    @pytest.mark.parametrize("can_retrieve", [True, False])
    @pytest.mark.parametrize("can_search", [True, False])
    @pytest.mark.parametrize("can_verify", [True, False])
    def test_allowed_non_empty(self, arm, t2, can_retrieve, can_search, can_verify):
        """Allowed set is never empty for any state/arm combination."""
        state = ActionState(
            t2=t2,
            executive_steps_remaining=5,
            can_retrieve=can_retrieve,
            can_search=can_search,
            can_verify=can_verify,
        )
        decision = compute_allowed_actions(state, arm)
        assert len(decision.allowed) >= 4  # ANSWER/DEFER/STOP/REASON_MORE always legal+admissible

    @pytest.mark.parametrize("arm", ALL_ARMS)
    @pytest.mark.parametrize("t2", [True, False])
    @pytest.mark.parametrize("can_retrieve", [True, False])
    @pytest.mark.parametrize("can_search", [True, False])
    @pytest.mark.parametrize("can_verify", [True, False])
    def test_allowed_is_intersection(self, arm, t2, can_retrieve, can_search, can_verify):
        """Allowed = Legal ∩ EpistemicallyAdmissible for all combinations."""
        state = ActionState(
            t2=t2,
            executive_steps_remaining=5,
            can_retrieve=can_retrieve,
            can_search=can_search,
            can_verify=can_verify,
        )
        decision = compute_allowed_actions(state, arm)
        assert decision.allowed == decision.legal & decision.epistemically_admissible

    @pytest.mark.parametrize("arm", ALL_ARMS)
    @pytest.mark.parametrize("t2", [True, False])
    @pytest.mark.parametrize("can_retrieve", [True, False])
    @pytest.mark.parametrize("can_search", [True, False])
    @pytest.mark.parametrize("can_verify", [True, False])
    def test_schema_enum_matches_allowed(self, arm, t2, can_retrieve, can_search, can_verify):
        """Schema enum always matches allowed action set."""
        state = ActionState(
            t2=t2,
            executive_steps_remaining=5,
            can_retrieve=can_retrieve,
            can_search=can_search,
            can_verify=can_verify,
        )
        decision = compute_allowed_actions(state, arm)
        schema = build_action_schema(decision.allowed)
        verify_schema_invariant(schema, decision.allowed)


# ---------------------------------------------------------------------------
# C0 schema identity (Q2)
# ---------------------------------------------------------------------------

class TestC0SchemaIdentity:
    def test_c0_schema_equals_r13(self):
        """Schema_R2(Allowed=ACTION_VOCABULARY) == Schema_R13."""
        passed, r2_sha, r13_sha = c0_schema_identity_check()
        assert passed, f"C0 schema mismatch: {r2_sha} != {r13_sha}"

    def test_c0_schema_sha_stable(self):
        """C0 schema SHA is stable across calls."""
        _, sha1, _ = c0_schema_identity_check()
        _, sha2, _ = c0_schema_identity_check()
        assert sha1 == sha2


# ---------------------------------------------------------------------------
# Arm isolation
# ---------------------------------------------------------------------------

class TestArmIsolation:
    def test_c0_vs_d_differs_only_at_t2(self):
        """C0 and D differ only when T2 is true."""
        state_no_t2 = make_state(t2=False)
        d0 = compute_allowed_actions(state_no_t2, C0)
        d1 = compute_allowed_actions(state_no_t2, D)
        assert d0.allowed == d1.allowed

        state_t2 = make_state(t2=True)
        d0t = compute_allowed_actions(state_t2, C0)
        d1t = compute_allowed_actions(state_t2, D)
        assert d0t.allowed != d1t.allowed
        assert "VERIFY" in d0t.allowed
        assert "VERIFY" not in d1t.allowed

    def test_c0_vs_e_same_allowed(self):
        """C0 and E have the same allowed actions (E only changes label)."""
        for t2 in [True, False]:
            state = make_state(t2=t2)
            d0 = compute_allowed_actions(state, C0)
            d1 = compute_allowed_actions(state, E)
            assert d0.allowed == d1.allowed
            assert d0.verify_gate_applied == d1.verify_gate_applied

    def test_de_equals_d_at_t2(self):
        """DE and D have the same allowed actions (both gate at T2)."""
        state = make_state(t2=True)
        d_dec = compute_allowed_actions(state, D)
        de_dec = compute_allowed_actions(state, DE)
        assert d_dec.allowed == de_dec.allowed
        assert d_dec.verify_gate_applied == de_dec.verify_gate_applied

    def test_de_equals_e_when_not_t2(self):
        """DE and E have the same allowed actions when T2 is false."""
        state = make_state(t2=False)
        e_dec = compute_allowed_actions(state, E)
        de_dec = compute_allowed_actions(state, DE)
        assert e_dec.allowed == de_dec.allowed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
