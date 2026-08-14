"""Tests for scripts/run_c5_integrated_ladder.py (c5_integrated_v1).

The integrated ladder's whole value is attribution: a 2x2 over {fusion} x
{selector} that says whether a Q gain came from R1, from S2, or from their
interaction. That only holds if the crossover really is a crossover, so the
parity checker is the critical piece and is tested against deliberately
confounded rows as well as clean ones.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_c5_integrated", ROOT / "scripts/run_c5_integrated_ladder.py")
lad = importlib.util.module_from_spec(_spec)
sys.modules["_c5_integrated"] = lad
_spec.loader.exec_module(lad)

PROTOCOL = ROOT / "configs/gate_c5_integrated_v1.json"
DRY = ROOT / "evidence/gate_c4/diagnosis/development_c5_integrated_dry.json"


def _row(**overrides):
    """A clean crossover row: pools keyed by fusion, identity shared."""
    base = {}
    for arm in lad.ARM_ORDER:
        fusion, selector = lad.ARM_SPEC[arm]
        base[arm] = {
            "candidate_pool_hash": f"pool_{fusion}",
            "identity_status": "EXACT",
            "fusion": fusion,
            "selector": selector,
        }
    for arm, patch in overrides.items():
        base[arm].update(patch)
    return {"arms": base}


class TestArmMatrixIsAProperCrossover:
    def test_the_2x2_covers_both_factors_at_both_levels(self):
        core = {a: lad.ARM_SPEC[a] for a in ("I0", "I1", "I2", "I3")}
        assert core["I0"] == ("frozen_rrf", "S0")
        assert core["I1"] == ("R1_max_reciprocal", "S0")
        assert core["I2"] == ("frozen_rrf", "S2")
        assert core["I3"] == ("R1_max_reciprocal", "S2")

    def test_ceiling_arms_are_not_promotable_in_the_protocol(self):
        arms = json.loads(PROTOCOL.read_text())["arms"]
        for ceiling in ("I0", "I4", "I5"):
            assert arms[ceiling]["promotable"] is False

    def test_primary_arm_is_i3(self):
        assert "PRIMARY" in json.loads(PROTOCOL.read_text())["arms"]["I3"]["role"]


class TestCrossoverParityChecker:
    def test_clean_row_has_no_violations(self):
        assert lad.check_crossover_parity(_row()) == []

    def test_detects_differing_pools_across_a_selector_only_pair(self):
        """I0 vs I2 must share a pool -- otherwise the 'selector effect' is
        contaminated by a retrieval difference and E_selector is meaningless."""
        violations = lad.check_crossover_parity(
            _row(I2={"candidate_pool_hash": "pool_SOMETHING_ELSE"}))
        assert any("I0 vs I2" in v and "candidate pools differ" in v
                   for v in violations)

    def test_detects_differing_pools_on_the_other_selector_pair(self):
        violations = lad.check_crossover_parity(
            _row(I3={"candidate_pool_hash": "pool_MISMATCH"}))
        assert any("I1 vs I3" in v for v in violations)

    def test_detects_identity_drift_across_a_fusion_only_pair(self):
        violations = lad.check_crossover_parity(
            _row(I1={"identity_status": "RESOLVED"}))
        assert any("I0 vs I1" in v and "identity status differs" in v
                   for v in violations)

    def test_detects_an_arm_whose_reported_selector_drifted_from_spec(self):
        """The pair comparisons are derived from ARM_SPEC, which is only sound
        if each executed arm really is its spec. A row reporting a different
        selector than the ladder declares must be caught, or the derivation
        would be trusting the thing it depends on."""
        violations = lad.check_crossover_parity(_row(I3={"selector": "S0"}))
        assert any("I3" in v and "does not match the ladder spec" in v
                   for v in violations)

    def test_detects_an_arm_whose_reported_fusion_drifted_from_spec(self):
        violations = lad.check_crossover_parity(
            _row(I3={"fusion": "frozen_rrf"}))
        assert any("I3" in v and "reported fusion" in v for v in violations)

    def test_j_ladder_parity_pairs_are_derived_not_hardcoded(self):
        """Renaming the arms must not silently disable the checks."""
        lad.use_ladder("J")
        try:
            clean = {"arms": {a: {"candidate_pool_hash": f"pool_{lad.ARM_SPEC[a][0]}",
                                  "identity_status": "EXACT",
                                  "fusion": lad.ARM_SPEC[a][0],
                                  "selector": lad.ARM_SPEC[a][1]}
                              for a in lad.ARM_ORDER}}
            assert lad.check_crossover_parity(clean) == []
            # J0/J1/J2/J3 all share frozen_rrf, so a pool mismatch must be caught.
            clean["arms"]["J1"]["candidate_pool_hash"] = "pool_DIFFERENT"
            violations = lad.check_crossover_parity(clean)
            assert any("candidate pools differ" in v for v in violations)
        finally:
            lad.use_ladder("I")


class TestConfigHashesArePinned:
    def test_hashes_are_stable_across_calls(self):
        assert lad.config_hashes() == lad.config_hashes()

    def test_both_mechanism_hashes_are_present(self):
        h = lad.config_hashes()
        assert h["retrieval_config_hash"] and h["selector_config_hash"]
        assert h["retrieval_config_hash"] != h["selector_config_hash"]


class TestProtocolFreezesWhatItMustBeforeGpuSpend:
    def test_thresholds_are_frozen(self):
        p = json.loads(PROTOCOL.read_text())
        assert p["primary_criterion"]["threshold"] == 0.15
        assert p["subgroup_safety"]["family_q_regression_tolerance"] == -0.05
        assert p["subgroup_safety"]["entity_regime_q_regression_tolerance"] == -0.05

    def test_reuse_of_the_v2_1_threshold_is_justified_not_silent(self):
        """The instruction was not to inherit +0.15 by default."""
        rationale = json.loads(PROTOCOL.read_text())[
            "primary_criterion"]["threshold_rationale"]
        assert "DELIBERATELY" in rationale and "comparability" in rationale

    def test_both_subgroup_axes_are_monitored(self):
        """v2.1's D5 watched family only, and the real regression landed on
        entity_regime. Both axes must be gated here."""
        axes = json.loads(PROTOCOL.read_text())["subgroup_safety"]["axes"]
        assert set(axes) == {"family", "entity_regime"}

    def test_selector_repair_has_a_hard_retention_floor(self):
        gates = json.loads(PROTOCOL.read_text())["selector_regime_safety_gates"]
        assert "0.60" in gates["hard_requirement"]

    def test_preconditions_are_marked_required_before_gpu(self):
        p = json.loads(PROTOCOL.read_text())
        assert p["determinism_precondition"]["run_before_spending_gpu"] is True
        assert p["dry_pass_precondition"]["run_before_spending_gpu"] is True

    def test_consumed_split_is_excluded(self):
        """The guarantee that matters is the prohibition itself, not the key
        name: the old qualification split informed this mechanism's design and
        so cannot also be its test."""
        excl = json.loads(PROTOCOL.read_text())["scope_exclusions"]
        clause = excl["old_qualification_split_is_consumed"]
        assert "must never be used to evaluate this mechanism" in clause
        assert "diagnostic historical data" in clause

    def test_fresh_confirmation_split_is_required_and_specified(self):
        fresh = json.loads(PROTOCOL.read_text())[
            "fresh_confirmation_split_requirements"]
        assert fresh["must_be_built_before_looking_at_confirmation_results"] is True
        assert fresh["minimum_tasks"] >= 500
        assert "exactly once" in fresh["runs"]


class TestAgainstTheCommittedDryPass:
    """Invariants on the real 120-task dry pass, so a later change cannot
    silently lose either mechanism's effect."""

    def _dry(self):
        return json.loads(DRY.read_text())

    def test_crossover_parity_actually_held(self):
        assert self._dry()["crossover_parity"]["passed"] is True

    def test_r1_retrieval_effect_survived_integration(self):
        """R1's B1 result was +2.5pp candidate CES; it must still be there."""
        arms = self._dry()["arms"]
        assert arms["I1"]["candidate_ces"] > arms["I0"]["candidate_ces"]
        assert arms["I3"]["candidate_ces"] > arms["I2"]["candidate_ces"]

    def test_s2_selector_repair_survived_integration(self):
        """The hard requirement: I3 must retain substantially all of the
        EXACT_bridged repair, not just deliver aggregate CES."""
        arms = self._dry()["arms"]
        i3 = arms["I3"]["answer_retention"]["EXACT_bridged"]
        i0 = arms["I0"]["answer_retention"]["EXACT_bridged"]
        assert i3 >= 0.60, f"repair lost under integration: {i3}"
        assert i3 - i0 > 0.4

    def test_packet_budget_never_exceeded_by_any_arm(self):
        report = self._dry()
        for arm, row in report["arms"].items():
            assert row["max_packet_size"] <= report["packet_budget"], arm

    def test_i3_bridge_retention_meets_the_structural_gate(self):
        """>= I0 - 0.01. R1 alone slightly REDUCES bridge retention, so this
        passing depends on S2's connectivity constraint offsetting it."""
        arms = self._dry()["arms"]
        assert arms["I3"]["bridge_retention"] >= arms["I0"]["bridge_retention"] - 0.01

    def test_identity_retention_is_unharmed_everywhere(self):
        for arm, row in self._dry()["arms"].items():
            assert row["identity_retention"] >= 0.99, arm
