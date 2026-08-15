"""Tests for fail-closed C4 protocol validation.

The validator's job is to reject. Most of these tests corrupt one field of the
real protocol and assert that validation notices, because a validator that only
ever passes is indistinguishable from no validator at all.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.protocol_validation import (
    EXPECTED_METRIC_MODULE,
    ProtocolViolation,
    load_and_validate_protocol,
    validate_c4_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/gate_c4_protocol_v2_1.json"
SUPERSEDED_PROTOCOL_PATH = ROOT / "configs/gate_c4_protocol_v2.json"


@pytest.fixture()
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


class TestActiveProtocol:
    """The shipped protocol must validate against the shipped code."""

    def test_active_protocol_validates(self, protocol):
        checks = validate_c4_protocol(protocol)
        assert checks["protocol_identity"] is True
        assert checks["one_pass_primary"] is True
        assert checks["metric_module"] == EXPECTED_METRIC_MODULE

    def test_sha_matches_sidecar(self):
        _protocol, sha, _checks = load_and_validate_protocol(PROTOCOL_PATH)
        declared = (ROOT / "configs/gate_c4_protocol_v2_1.sha256").read_text().strip()
        assert sha == declared.split()[0]

    def test_resolves_fields_the_notebook_reported_as_na(self, protocol):
        """Identity/selector invariants must be resolved, not 'N/A'.

        The Colab audit printed "Identity Resolution: N/A" because it read
        architecture.identity_resolution, which does not exist. The facts live
        in the arm registry, and the validator checks them there.
        """
        checks = validate_c4_protocol(protocol)
        assert checks["arm_policies"] is True
        assert "C4_4" in checks["arms_present"]
        assert checks["policy_versions"]["selector_policy_version"] == \
            "s2c_deterministic_v1"


class TestMissingFieldsAbort:
    """A missing required field aborts; it never becomes a default."""

    @pytest.mark.parametrize("key", [
        "protocol_id", "protocol_version", "architecture", "arms",
        "iterative_retrieval_status", "metric_definitions",
        "packet_ordering_policy", "policy_versions", "membership_vs_ordering",
        "fail_closed_runner", "frozen_before_any_c4_v2_measurement",
    ])
    def test_missing_top_level_key_raises(self, protocol, key):
        broken = copy.deepcopy(protocol)
        del broken[key]
        with pytest.raises(ProtocolViolation):
            validate_c4_protocol(broken)

    def test_missing_arm_raises(self, protocol):
        broken = copy.deepcopy(protocol)
        del broken["arms"]["C4_4"]
        with pytest.raises(ProtocolViolation, match="missing primary arms"):
            validate_c4_protocol(broken)


class TestSemanticMismatchesAbort:
    """Wrong values abort, even when every key is present."""

    def test_wrong_protocol_id(self, protocol):
        broken = dict(protocol, protocol_id="c4_v1_something")
        with pytest.raises(ProtocolViolation, match="protocol_id"):
            validate_c4_protocol(broken)

    def test_unfrozen_protocol(self, protocol):
        broken = dict(protocol, frozen_before_any_c4_v2_measurement=False)
        with pytest.raises(ProtocolViolation, match="frozen"):
            validate_c4_protocol(broken)

    def test_iterative_retrieval_inside_primary(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["iterative_retrieval_status"]["classification"] = "INSIDE_PRIMARY_C4"
        with pytest.raises(ProtocolViolation, match="OUTSIDE_PRIMARY_C4"):
            validate_c4_protocol(broken)

    def test_ogc_numerator_using_c4_5_is_rejected(self, protocol):
        """The historical bug: OGC numerator using the oracle selector."""
        broken = copy.deepcopy(protocol)
        broken["metric_definitions"]["oracle_gap_capture"]["numerator_uses"] = \
            "C4_5 (oracle selector)"
        with pytest.raises(ProtocolViolation, match="C4_4"):
            validate_c4_protocol(broken)

    def test_non_authoritative_metric_module_rejected(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["metric_definitions"]["quality"]["authoritative_module"] = \
            "scripts.analyze_gate_c4"
        with pytest.raises(ProtocolViolation, match="authoritative_module"):
            validate_c4_protocol(broken)

    def test_quality_truth_table_must_match_code(self, protocol):
        """Overloading quality with accuracy is rejected."""
        broken = copy.deepcopy(protocol)
        broken["metric_definitions"]["quality"]["states"]["complete_and_incorrect"] = 0.0
        with pytest.raises(ProtocolViolation, match="quality state"):
            validate_c4_protocol(broken)

    def test_ordering_policy_id_must_match_code(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["packet_ordering_policy"]["policy_id"] = "c4_packet_ordering_v2"
        with pytest.raises(ProtocolViolation, match="policy_id"):
            validate_c4_protocol(broken)

    def test_chain_order_must_agree_with_role_priority(self, protocol):
        broken = copy.deepcopy(protocol)
        order = broken["packet_ordering_policy"]["chain_order"]
        # Put distractor first: contradicts ROLE_PRIORITY.
        broken["packet_ordering_policy"]["chain_order"] = \
            ["distractor"] + [r for r in order if r != "distractor"]
        with pytest.raises(ProtocolViolation, match="chain_order"):
            validate_c4_protocol(broken)

    def test_unknown_chain_order_role_rejected(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["packet_ordering_policy"]["chain_order"].append("not_a_real_role")
        with pytest.raises(ProtocolViolation, match="no priority"):
            validate_c4_protocol(broken)

    def test_abort_conditions_must_include_test_failure(self, protocol):
        """The fail-open bug this whole sprint fixes must stay declared."""
        broken = copy.deepcopy(protocol)
        broken["fail_closed_runner"]["abort_conditions"] = [
            c for c in broken["fail_closed_runner"]["abort_conditions"]
            if c != "test suite fails"]
        with pytest.raises(ProtocolViolation, match="test suite fails"):
            validate_c4_protocol(broken)


class TestCodeSideInvariants:
    """Validation covers the code, not only the JSON."""

    def test_arm_registry_drift_is_caught(self, protocol, monkeypatch):
        """If C4_4's selector is changed in code, validation fails."""
        from dataclasses import replace
        from hrm_adaptive_memory.c4 import arms as arms_module

        drifted = dict(arms_module.ARMS)
        drifted["C4_4"] = replace(drifted["C4_4"], selector_policy="s0")
        monkeypatch.setattr(arms_module, "ARMS", drifted)
        with pytest.raises(ProtocolViolation, match="C4_4.selector_policy"):
            validate_c4_protocol(protocol)

    def test_degenerate_ordering_arms_are_caught(self, protocol, monkeypatch):
        """If C4_4m also applied deterministic order, the 2x2 collapses."""
        from hrm_adaptive_memory.c4 import packet_stage

        monkeypatch.setattr(packet_stage, "_DETERMINISTIC_ORDER_ARMS",
                            {"C4_4", "C4_3o", "C4_4m"})
        with pytest.raises(ProtocolViolation, match="deterministic-order arms"):
            validate_c4_protocol(protocol)


class TestSupersededProtocolRejected:
    """v2 is retained for lineage but is no longer certifiable."""

    def test_v2_file_still_exists_for_lineage(self):
        assert SUPERSEDED_PROTOCOL_PATH.is_file()

    def test_v2_is_not_a_valid_active_protocol(self):
        with pytest.raises(ProtocolViolation, match="protocol_id"):
            load_and_validate_protocol(SUPERSEDED_PROTOCOL_PATH)

    def test_v2_contained_the_contradiction_this_version_resolves(self):
        """Documents why v2 was superseded, so the reason cannot be lost."""
        v2 = json.loads(SUPERSEDED_PROTOCOL_PATH.read_text())
        # v2 said selector score first...
        assert v2["determinism_requirements"]["tie_break_policy"][0].startswith(
            "1. selector score descending")
        # ...and role tier first, in the same document.
        assert v2["packet_ordering_policy"]["within_role_tier"][0] == \
            "selector score descending"
        assert v2["packet_ordering_policy"]["chain_order"][0] == "identity"

    def test_lineage_binds_the_superseded_file_by_hash(self, protocol):
        import hashlib
        actual = hashlib.sha256(SUPERSEDED_PROTOCOL_PATH.read_bytes()).hexdigest()
        assert protocol["lineage"]["supersedes_sha256"] == actual


class TestSingleOrderingDefinition:
    """v2_1 must define packet order exactly once."""

    def test_canonical_sort_key_is_role_retrieval_id(self, protocol):
        key = protocol["packet_ordering_policy"]["canonical_sort_key"]
        assert len(key) == 3
        assert "role" in key[0].lower()
        assert "retrieval" in key[1].lower()
        assert "record_id" in key[2].lower()

    def test_competing_definition_is_rejected(self, protocol):
        """Re-adding the v2 tie_break_policy must fail validation."""
        broken = copy.deepcopy(protocol)
        broken["determinism_requirements"]["tie_break_policy"] = [
            "1. selector score descending",
            "2. evidence role priority ascending",
        ]
        with pytest.raises(ProtocolViolation, match="contradicts"):
            validate_c4_protocol(broken)

    def test_duplicate_within_role_tier_is_rejected(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["packet_ordering_policy"]["within_role_tier"] = [
            "selector score descending"]
        with pytest.raises(ProtocolViolation, match="one definition"):
            validate_c4_protocol(broken)

    def test_wrong_number_of_sort_rules_rejected(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["packet_ordering_policy"]["canonical_sort_key"] = [
            "1. evidence role priority ascending"]
        with pytest.raises(ProtocolViolation, match="exactly 3 rules"):
            validate_c4_protocol(broken)

    def test_reordered_sort_rules_rejected(self, protocol):
        """Putting score before role must fail: that was the v2 contradiction."""
        broken = copy.deepcopy(protocol)
        key = broken["packet_ordering_policy"]["canonical_sort_key"]
        broken["packet_ordering_policy"]["canonical_sort_key"] = [
            key[1], key[0], key[2]]
        with pytest.raises(ProtocolViolation, match="rule 1 must be role"):
            validate_c4_protocol(broken)

    def test_selector_score_cannot_be_reintroduced_in_the_protocol(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["packet_ordering_policy"]["selector_score_excluded"]["excluded"] = False
        with pytest.raises(ProtocolViolation, match="scores chains, not records"):
            validate_c4_protocol(broken)

    def test_selector_score_cannot_be_reintroduced_in_the_code(self, protocol,
                                                               monkeypatch):
        """If order_packet regained a selector_scores parameter, abort."""
        from hrm_adaptive_memory.c4 import packet_ordering

        def fake_order_packet(selected_ids, *, selector_scores=None,
                              retrieval_scores=None):
            return list(selected_ids)

        monkeypatch.setattr(packet_ordering, "order_packet", fake_order_packet)
        with pytest.raises(ProtocolViolation, match="still accepts selector_scores"):
            validate_c4_protocol(protocol)

    def test_applies_to_arms_must_match_code(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["packet_ordering_policy"]["applies_to_arms"] = ["C4_4"]
        with pytest.raises(ProtocolViolation, match="applies_to_arms"):
            validate_c4_protocol(broken)

    def test_selector_order_must_declare_its_scope(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["determinism_requirements"]["selector_tie_break_policy"]["scope"] = \
            "the packet order"
        with pytest.raises(ProtocolViolation, match="scope must state"):
            validate_c4_protocol(broken)


class TestLineageRequirements:
    def test_mechanism_change_must_be_false(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["lineage"]["mechanism_change"] = True
        with pytest.raises(ProtocolViolation, match="mechanism_change"):
            validate_c4_protocol(broken)

    def test_missing_lineage_rejected(self, protocol):
        broken = copy.deepcopy(protocol)
        del broken["lineage"]
        with pytest.raises(ProtocolViolation, match="lineage"):
            validate_c4_protocol(broken)

    @pytest.mark.parametrize("field", [
        "reason", "conformance_defect_repaired", "affected_prior_results"])
    def test_empty_lineage_field_rejected(self, protocol, field):
        broken = copy.deepcopy(protocol)
        broken["lineage"][field] = ""
        with pytest.raises(ProtocolViolation, match=field):
            validate_c4_protocol(broken)


class TestPromptBindingRequirement:
    def test_prompt_binding_rule_required(self, protocol):
        broken = copy.deepcopy(protocol)
        del broken["pre_hrm_freeze"]["prompt_binding_rule"]
        with pytest.raises(ProtocolViolation, match="prompt_binding_rule"):
            validate_c4_protocol(broken)

    def test_rule_must_reference_the_ordered_packet(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["pre_hrm_freeze"]["prompt_binding_rule"] = \
            "HRM generates from a prompt whose prompt_hash is recorded."
        with pytest.raises(ProtocolViolation, match="ORDERED"):
            validate_c4_protocol(broken)

    def test_binding_abort_condition_required(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["fail_closed_runner"]["abort_conditions"] = [
            c for c in broken["fail_closed_runner"]["abort_conditions"]
            if "prompt binding" not in c]
        with pytest.raises(ProtocolViolation, match="prompt binding"):
            validate_c4_protocol(broken)


class TestDiagnosticArmIsolation:
    """C4_3o/C4_4m explain C4_4; they never join the primary ladder."""

    def test_declared_diagnostic_arms(self, protocol):
        assert protocol["diagnostic_arms"]["arms"] == ["C4_3o", "C4_4m"]

    def test_isolation_rule_names_the_exclusions(self, protocol):
        rule = protocol["diagnostic_arms"]["isolation_rule"]
        assert "promotion threshold" in rule
        assert "primary" in rule

    def test_primary_order_excludes_diagnostic_arms(self):
        from hrm_adaptive_memory.c4.arms import DIAGNOSTIC_ORDER, PRIMARY_ORDER
        assert PRIMARY_ORDER == ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4",
                                 "C4_5", "C4_6"]
        assert DIAGNOSTIC_ORDER == ["C4_3o", "C4_4m"]
        assert not set(PRIMARY_ORDER) & set(DIAGNOSTIC_ORDER)

    def test_diagnostic_arms_are_classified_diagnostic(self):
        from hrm_adaptive_memory.c4.arms import ARMS
        for arm_id in ("C4_3o", "C4_4m"):
            assert ARMS[arm_id].classification == "DIAGNOSTIC"

    def test_promoting_a_diagnostic_arm_is_rejected(self, protocol, monkeypatch):
        from dataclasses import replace
        from hrm_adaptive_memory.c4 import arms as arms_module

        drifted = dict(arms_module.ARMS)
        drifted["C4_4m"] = replace(drifted["C4_4m"], classification="PRIMARY")
        monkeypatch.setattr(arms_module, "ARMS", drifted)
        with pytest.raises(ProtocolViolation, match="must be classified DIAGNOSTIC"):
            validate_c4_protocol(protocol)

    def test_diagnostic_arm_in_primary_order_is_rejected(self, protocol,
                                                         monkeypatch):
        from hrm_adaptive_memory.c4 import arms as arms_module
        monkeypatch.setattr(arms_module, "PRIMARY_ORDER",
                            arms_module.PRIMARY_ORDER + ["C4_4m"])
        with pytest.raises(ProtocolViolation, match="PRIMARY_ORDER"):
            validate_c4_protocol(protocol)

    def test_gap_capture_formulas_reference_only_primary_arms(self, protocol):
        for name in ("oracle_gap_capture", "selector_gap_capture"):
            formula = protocol["metric_definitions"][name]["formula"]
            assert "C4_3o" not in formula
            assert "C4_4m" not in formula


class TestCertificationRequirements:
    def test_required_recomputed_metrics_declared(self, protocol):
        declared = set(protocol["certification"]["recompute_from_raw_receipts"])
        for metric in ("arm quality", "binary accuracy",
                       "complete-set retention (CSR)", "primary delta",
                       "selector_gap_capture", "oracle_gap_capture",
                       "family-grouped CI", "cluster-grouped CI",
                       "arm receipt counts", "task-set equality"):
            assert metric in declared, metric

    def test_dropping_a_recomputed_metric_is_rejected(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["certification"]["recompute_from_raw_receipts"] = ["arm quality"]
        with pytest.raises(ProtocolViolation, match="recompute_from_raw_receipts"):
            validate_c4_protocol(broken)

    def test_valid_run_must_be_a_conjunction(self, protocol):
        broken = copy.deepcopy(protocol)
        broken["certification"]["valid_run_rule"] = \
            "VALID_RUN = protocol hash matches"
        with pytest.raises(ProtocolViolation, match="conjunction"):
            validate_c4_protocol(broken)


class TestLoadAndValidate:
    """Missing files are violations, not 'NOT_FOUND' strings."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ProtocolViolation, match="not found"):
            load_and_validate_protocol(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ProtocolViolation, match="not valid JSON"):
            load_and_validate_protocol(bad)
