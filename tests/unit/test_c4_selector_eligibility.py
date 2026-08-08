"""Tests for scripts/diagnose_c4_selector_eligibility.py (Sprint B2 pre-freeze).

This check exists to stop an underpowered arm from being frozen, so the thing
that must be right is the rule's fire condition and the shape labelling. An
earlier ad-hoc version of this measurement was WRONG because it fed the rule
the ORACLE canonical surface instead of identity.canonical -- the signal a
runtime selector actually has -- and reported 0% instead of 21.4% on bridged
tasks. These tests pin the distinction so that cannot recur silently.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_diagnose_selector_eligibility",
    ROOT / "scripts/diagnose_c4_selector_eligibility.py")
diag = importlib.util.module_from_spec(_spec)
sys.modules["_diagnose_selector_eligibility"] = diag
_spec.loader.exec_module(diag)

PROTOCOL = ROOT / "configs/gate_c4_selector_v1.json"
ARTIFACT = ROOT / "evidence/gate_c4/diagnosis/development_selector_eligibility.json"


def _task(question, *, bridge=None, answer_node="T#value", edges=()):
    return {
        "task_id": "t-1",
        "question": question,
        "_oracle_metadata": {
            "answer_node": answer_node,
            "latent_bridge": bridge,
            "proof_edges": list(edges),
        },
    }


class TestShapeLabelling:
    def test_latent_bridge_means_bridged(self):
        assert diag.task_shape(_task("q", bridge="T#bridge")) == "bridged"

    def test_absent_bridge_means_unbridged(self):
        assert diag.task_shape(_task("q")) == "unbridged"

    def test_missing_oracle_metadata_is_unbridged_not_a_crash(self):
        assert diag.task_shape({"task_id": "t"}) == "unbridged"


class TestTerminalRecordIdentification:
    def test_picks_edges_terminating_at_the_answer_node(self):
        task = _task("q", edges=[
            {"record_id": "T/link", "target": "T#bridge"},
            {"record_id": "T/value", "target": "T#value"},
        ])
        assert diag.terminal_records(task) == ["T/value"]

    def test_multiple_terminal_records_are_all_returned(self):
        task = _task("q", edges=[
            {"record_id": "a", "target": "T#value"},
            {"record_id": "b", "target": "T#value"},
        ])
        assert diag.terminal_records(task) == ["a", "b"]


class TestRuleFireCondition:
    QUESTION = "Which ownership tier applies to Sparrow intake manifold?"

    def test_fires_when_content_has_both_subject_and_relation(self):
        task = _task(self.QUESTION, edges=[{"record_id": "r", "target": "T#value"}])
        texts = {"r": "Sparrow intake manifold has ownership tier 3529."}
        assert diag.rule_fires(task, "Sparrow intake manifold", texts)

    def test_does_not_fire_when_the_record_names_only_the_bridge(self):
        """The structural reason bridged tasks under-fire: the terminal record
        carries the relation and the BRIDGE, never the query subject."""
        task = _task(self.QUESTION, bridge="T#bridge",
                     edges=[{"record_id": "r", "target": "T#value"}])
        texts = {"r": "- updated Finch control module: ownership tier now 3529"}
        assert not diag.rule_fires(task, "Sparrow intake manifold", texts)

    def test_does_not_fire_without_the_relation(self):
        task = _task(self.QUESTION, edges=[{"record_id": "r", "target": "T#value"}])
        texts = {"r": "Sparrow intake manifold was commissioned in March."}
        assert not diag.rule_fires(task, "Sparrow intake manifold", texts)

    def test_does_not_fire_when_identity_supplied_no_canonical(self):
        """UNRESOLVED/AMBIGUOUS identity yields canonical=None; the rule must
        decline rather than fall back to some other anchor."""
        task = _task(self.QUESTION, edges=[{"record_id": "r", "target": "T#value"}])
        texts = {"r": "Sparrow intake manifold has ownership tier 3529."}
        assert not diag.rule_fires(task, None, texts)

    def test_unparseable_question_declines(self):
        task = _task("???", edges=[{"record_id": "r", "target": "T#value"}])
        texts = {"r": "anything at all"}
        assert not diag.rule_fires(task, "Subject", texts)

    def test_uses_the_runtime_canonical_it_is_given(self):
        """Pinning the exact bug that produced the wrong 0% figure: the rule
        must key off the canonical it is PASSED (identity.canonical), so a
        different canonical changes the outcome. If it silently consulted
        oracle metadata instead, this would not hold."""
        task = _task(self.QUESTION, edges=[{"record_id": "r", "target": "T#value"}])
        texts = {"r": "Finch control module: ownership tier now 3529"}
        assert not diag.rule_fires(task, "Sparrow intake manifold", texts)
        assert diag.rule_fires(task, "Finch control module", texts)


class TestNoForbiddenSignalsInTheRule:
    def test_rule_source_does_not_read_record_kind(self):
        """record_kind values here are generator answer-key labels, so the
        eligibility rule must never consult them."""
        import inspect
        source = inspect.getsource(diag.rule_fires)
        assert "record_kind" not in source
        assert "required_evidence_ids" not in source

    def test_protocol_declares_record_kind_forbidden(self):
        audit = json.loads(PROTOCOL.read_text())[
            "LEAKAGE_CORRECTION_APPLIED"]["runtime_signal_audit"]
        assert "record_kind" in audit["forbidden"]
        assert "required_evidence_ids" in audit["forbidden"]


class TestProtocolIsStillUnfrozenOnTheOpenDecision:
    """No arm may run while the eligibility rule is unresolved."""

    def test_open_decision_is_recorded(self):
        protocol = json.loads(PROTOCOL.read_text())
        assert "OPEN_DECISION_BEFORE_ANY_ARM_RUNS" in protocol
        assert protocol["status"].startswith("PREREGISTERED")

    def test_promotion_criteria_are_frozen_before_arms(self):
        protocol = json.loads(PROTOCOL.read_text())
        assert protocol["promotion_criteria_S1"]["frozen_before_any_arm_runs"] is True

    def test_packet_budget_invariant_is_declared(self):
        """A gain must not come from handing HRM more evidence."""
        fixed = json.loads(PROTOCOL.read_text())["fixed_conditions"]
        assert "never increases" in fixed["packet_budget_invariant"]


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="feasibility artifact absent")
class TestAgainstTheCommittedArtifact:
    @pytest.fixture(scope="class")
    def report(self):
        return json.loads(ARTIFACT.read_text())

    def test_defect_is_concentrated_in_bridged_exact_tasks(self, report):
        """The finding that reframed B2. Pinned so a later change cannot erase
        the evidence for why the eligibility rule had to be corrected."""
        assert report["exact_defect_share_bridged"] > 0.8

    def test_exact_bridged_keep_rate_is_far_below_resolved_bridged(self, report):
        rates = report["keep_rate_by_identity_and_shape"]
        assert rates["EXACT_bridged"]["keep_rate"] < 0.3
        assert rates["RESOLVED_bridged"]["keep_rate"] > 0.85

    def test_rule_fires_much_less_on_bridged_than_unbridged(self, report):
        fire = report["original_rule_fire_rate_by_shape"]
        assert fire["unbridged"]["fire_rate"] == 1.0
        assert fire["bridged"]["fire_rate"] < 0.5
