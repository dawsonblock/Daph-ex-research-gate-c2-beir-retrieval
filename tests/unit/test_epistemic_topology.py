"""Semantic truth-table tests for canonical epistemic topology.

Tests every fundamental combination defined in EPISTEMIC_SEMANTICS_V1.md §3.3.
These tests must pass before any downstream component is wired to the topology.
"""
import pytest
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)

from daph.epistemic import (
    HypothesisState,
    HypothesisTopology,
    TerminalReadiness,
    derive_hypothesis_topology,
    classify_terminal_readiness,
    is_answer_ready,
    is_defer_ready,
    is_continue_required,
)


def make_ev(
    eid: str,
    supports: tuple[str, ...] = (),
    contradicts: tuple[str, ...] = (),
    vstate: VerificationState = VerificationState.UNVERIFIED,
    tstatus: TemporalStatus = TemporalStatus.CURRENT,
    retrieved: bool = True,
) -> dict:
    """Create a minimal evidence dict for testing."""
    return {
        "evidence_id": eid,
        "supports": supports,
        "contradicts": contradicts,
        "verification_state": vstate,
        "temporal_status": tstatus,
        "retrieved": retrieved,
    }


class TestVerificationTruthTable:
    """Test the §3.3 truth table: how verification state × relation affects H."""

    def test_sufficient_support_gives_verified_support(self):
        """SUFFICIENT + supports(H) → H gains verified support."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.SUPPORTED
        assert "E1" in topo.verified_support_by_hypothesis.get("H1", ())

    def test_sufficient_contradict_gives_verified_contradiction(self):
        """SUFFICIENT + contradicts(H) → H gains verified contradiction."""
        ev = [make_ev("E1", contradicts=("H1",), vstate=VerificationState.SUFFICIENT)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.CONTRADICTED
        assert "E1" in topo.verified_contradiction_by_hypothesis.get("H1", ())

    def test_falsified_support_no_effect_on_h(self):
        """FALSIFIED + supports(H) → no effect on H (support claim failed)."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.FALSIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.WEAKENED
        assert "E1" not in topo.verified_support_by_hypothesis.get("H1", ())
        assert "E1" not in topo.verified_contradiction_by_hypothesis.get("H1", ())
        assert "E1" in topo.falsified_support_by_hypothesis.get("H1", ())

    def test_falsified_contradict_no_effect_on_h(self):
        """FALSIFIED + contradicts(H) → no effect on H (contradiction claim failed)."""
        ev = [make_ev("E1", contradicts=("H1",), vstate=VerificationState.FALSIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        # H1 has a falsified contradiction, no verified support or contradiction
        # It should be UNTESTED (falsified_contradiction is NOT evidence of anything)
        assert topo.hypothesis_states["H1"] == HypothesisState.UNTESTED
        assert "E1" not in topo.verified_contradiction_by_hypothesis.get("H1", ())
        assert "E1" not in topo.verified_support_by_hypothesis.get("H1", ())
        assert "E1" in topo.falsified_contradiction_by_hypothesis.get("H1", ())

    def test_unverified_support_gives_unverified_support(self):
        """UNVERIFIED + supports(H) → H gains unverified support (weak)."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.UNVERIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.UNTESTED
        assert "E1" in topo.unverified_support_by_hypothesis.get("H1", ())

    def test_unverified_contradict_gives_unverified_contradiction(self):
        """UNVERIFIED + contradicts(H) → H gains unverified contradiction (weak)."""
        ev = [make_ev("E1", contradicts=("H1",), vstate=VerificationState.UNVERIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.UNTESTED
        assert "E1" in topo.unverified_contradiction_by_hypothesis.get("H1", ())

    def test_stale_evidence_no_effect(self):
        """STALE verification state → no effect."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.STALE)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.UNTESTED

    def test_missing_evidence_no_effect(self):
        """MISSING verification state → no effect."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.MISSING)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.UNTESTED

    def test_temporal_stale_no_effect(self):
        """CURRENT verification but STALE temporal status → no current effect."""
        ev = [make_ev("E1", supports=("H1",),
                      vstate=VerificationState.SUFFICIENT,
                      tstatus=TemporalStatus.STALE)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        # H1 has stale evidence only
        assert topo.hypothesis_states["H1"] == HypothesisState.STALE


class TestHypothesisClassification:
    """Test §4 hypothesis state classification."""

    def test_supported_hypothesis(self):
        """H with SUFFICIENT support, no contradiction → SUPPORTED."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.SUPPORTED

    def test_contradicted_hypothesis(self):
        """H with SUFFICIENT contradiction → CONTRADICTED."""
        ev = [make_ev("E1", contradicts=("H1",), vstate=VerificationState.SUFFICIENT)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.CONTRADICTED

    def test_mixed_verified_evidence_contradicted(self):
        """H with both SUFFICIENT support and SUFFICIENT contradiction → CONTRADICTED (priority)."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H1",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.CONTRADICTED
        assert topo.n_hyp_with_mixed_verified == 1

    def test_weakened_hypothesis(self):
        """H with FALSIFIED support only → WEAKENED."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.FALSIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.WEAKENED

    def test_untested_hypothesis(self):
        """H with no verified evidence → UNTESTED."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.UNVERIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.UNTESTED

    def test_falsified_contradiction_does_not_eliminate(self):
        """H with FALSIFIED contradiction → NOT eliminated (contradiction claim failed)."""
        ev = [make_ev("E1", contradicts=("H1",), vstate=VerificationState.FALSIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] != HypothesisState.CONTRADICTED


class TestAggregateCounts:
    """Test aggregate topology counts."""

    def test_viable_count(self):
        """n_viable_hypotheses counts SUPPORTED only."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H2",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.n_viable_hypotheses == 1
        assert topo.n_eliminated_hypotheses == 1

    def test_competing_verified_support(self):
        """Two hypotheses with SUFFICIENT support → unresolved competition."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.n_hyp_with_verified_support == 2
        assert topo.has_verified_unresolved_competition is True
        assert topo.has_unique_verified_supported is False
        assert topo.unique_supported_hypothesis is None

    def test_unique_verified_support(self):
        """Exactly one hypothesis with SUFFICIENT support → unique."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H2",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.n_hyp_with_verified_support == 1
        assert topo.has_unique_verified_supported is True
        assert topo.has_verified_unresolved_competition is False
        assert topo.unique_supported_hypothesis == "H1"

    def test_no_verified_evidence(self):
        """No verified evidence at all."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.UNVERIFIED),
            make_ev("E2", contradicts=("H2",), vstate=VerificationState.UNVERIFIED),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.n_hyp_with_verified_support == 0
        assert topo.n_hyp_with_verified_contradiction == 0
        assert topo.unique_supported_hypothesis is None


class TestTerminalReadiness:
    """Test §6 terminal readiness classification."""

    def test_answer_ready_unique_supported(self):
        """Unique supported hypothesis → ANSWER_READY."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert is_answer_ready(topo) is True

    def test_not_answer_ready_competing_support(self):
        """Two supported hypotheses → NOT ANSWER_READY (unresolved competition)."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert is_answer_ready(topo) is False

    def test_not_answer_ready_no_support(self):
        """No supported hypotheses → NOT ANSWER_READY."""
        ev = [make_ev("E1", contradicts=("H1",), vstate=VerificationState.SUFFICIENT)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert is_answer_ready(topo) is False

    def test_defer_ready_no_continuation(self):
        """No answer ready, no continuation available → DEFER_READY."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", contradicts=("H1",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        readiness = classify_terminal_readiness(
            topo,
            can_verify=False,
            can_retrieve=False,
            can_search=False,
        )
        assert readiness == TerminalReadiness.DEFER_READY

    def test_continue_required_with_verify_available(self):
        """Not answer ready, verify available with discriminating evidence → CONTINUE_REQUIRED."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
            make_ev("E3", contradicts=("H2",), vstate=VerificationState.UNVERIFIED),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        readiness = classify_terminal_readiness(
            topo,
            can_verify=True,
            can_retrieve=False,
            can_search=False,
            has_unverified_discriminating_evidence=True,
        )
        assert readiness == TerminalReadiness.CONTINUE_REQUIRED

    def test_continue_required_with_retrieve_available(self):
        """Not answer ready, retrieve available with hidden evidence → CONTINUE_REQUIRED."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.UNVERIFIED)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"], hidden_evidence_count=1)
        readiness = classify_terminal_readiness(
            topo,
            can_verify=False,
            can_retrieve=True,
            can_search=False,
            has_hidden_evidence=True,
        )
        assert readiness == TerminalReadiness.CONTINUE_REQUIRED

    def test_defer_ready_resource_exhausted(self):
        """No answer ready, resources exhausted, no continuation → DEFER_READY."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.UNVERIFIED),
            make_ev("E2", supports=("H2",), vstate=VerificationState.UNVERIFIED),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        readiness = classify_terminal_readiness(
            topo,
            can_verify=False,
            can_retrieve=False,
            can_search=False,
        )
        assert readiness == TerminalReadiness.DEFER_READY


class TestObservabilityBoundary:
    """Test that topology derivation does not consume prohibited inputs."""

    def test_no_verify_result_needed(self):
        """Topology derivation works without verify_result field."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT)]
        # No verify_result field present
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.hypothesis_states["H1"] == HypothesisState.SUPPORTED

    def test_hidden_evidence_count_only(self):
        """Topology derivation uses count, not content, of hidden evidence."""
        ev = [make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT)]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"], hidden_evidence_count=5)
        assert topo.hidden_evidence_count == 5
        # Hidden evidence content is never accessed


class TestD5Pattern:
    """Test the specific D5 pattern: competing verified support."""

    def test_d5_competing_support_not_answer_ready(self):
        """D5 pattern: H1 SUFFICIENT, H2 SUFFICIENT → NOT ANSWER_READY."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        assert topo.has_verified_unresolved_competition is True
        assert is_answer_ready(topo) is False
        assert topo.unique_supported_hypothesis is None

    def test_d5_with_unverified_discriminator_continue(self):
        """D5 with an unverified discriminating evidence + verify available → CONTINUE_REQUIRED."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
            make_ev("E3", contradicts=("H2",), vstate=VerificationState.UNVERIFIED),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        readiness = classify_terminal_readiness(
            topo,
            can_verify=True,
            can_retrieve=False,
            can_search=False,
            has_unverified_discriminating_evidence=True,
        )
        assert readiness == TerminalReadiness.CONTINUE_REQUIRED

    def test_d5_without_discriminator_defer(self):
        """D5 without any discriminating continuation → DEFER_READY."""
        ev = [
            make_ev("E1", supports=("H1",), vstate=VerificationState.SUFFICIENT),
            make_ev("E2", supports=("H2",), vstate=VerificationState.SUFFICIENT),
        ]
        topo = derive_hypothesis_topology(ev, ["H1", "H2"])
        readiness = classify_terminal_readiness(
            topo,
            can_verify=False,
            can_retrieve=False,
            can_search=False,
        )
        assert readiness == TerminalReadiness.DEFER_READY
