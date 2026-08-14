"""Acceptance tests for the bounded Semantica-donor cognitive-control layer."""
from __future__ import annotations

import json

import pytest

from hrm_adaptive_memory.cognitive_control import (
    CognitiveControlStore, DatalogFact, DatalogReasoner, DatalogRule,
    DecisionAction, PolicyEffect, PolicyGate, PolicyRule,
)

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-06-01T00:00:00+00:00"
T2 = "2026-08-01T00:00:00+00:00"
UTC_T0 = "2026-01-01T00:00:00.000000Z"
UTC_T1 = "2026-06-01T00:00:00.000000Z"
UTC_T2 = "2026-08-01T00:00:00.000000Z"


def _provenance(store, entity="claim-1", operation="prov-1", created_at=T1):
    return store.record_provenance(
        entity_id=entity, entity_type="claim", payload={"value": 6},
        activity_id="verification", agent_id="daph-v2a", agent_type="software_agent",
        source_id="evidence-1", created_at=created_at, operation_id=operation,
        derived_from=("evidence-1",), bundle_id="run-1")


def test_provenance_identity_payload_and_hash_chain_are_bound(tmp_path):
    store = CognitiveControlStore(tmp_path)
    record = _provenance(store)
    assert record.entity_id == "claim-1" and record.payload_hash
    assert CognitiveControlStore(tmp_path).provenance[record.provenance_id] == record

    event = json.loads(store.log_path.read_text())
    event["payload"]["entity_id"] = "relabeled"
    store.log_path.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        CognitiveControlStore(tmp_path)


def test_operation_retries_are_idempotent_but_drift_fails(tmp_path):
    store = CognitiveControlStore(tmp_path)
    first = _provenance(store)
    assert _provenance(store) == first
    with pytest.raises(ValueError, match="different content"):
        _provenance(store, entity="other")


def test_manifest_detects_tail_deletion(tmp_path):
    store = CognitiveControlStore(tmp_path)
    _provenance(store)
    store.log_path.write_text("")
    with pytest.raises(ValueError, match="manifest"):
        CognitiveControlStore(tmp_path)


def test_bitemporal_queries_and_conflict_detection_preserve_disagreement(tmp_path):
    store = CognitiveControlStore(tmp_path)
    p1 = _provenance(store)
    p2 = _provenance(store, operation="prov-2")
    store.record_fact(entity="carbon", relation="atomic_number", value=6,
                      source_evidence_id="ev-a", source_lineage_id="lin-a",
                      provenance_id=p1.provenance_id, valid_from=T0, valid_until=None,
                      recorded_at=T1, operation_id="fact-a")
    store.record_fact(entity="carbon", relation="atomic_number", value=8,
                      source_evidence_id="ev-b", source_lineage_id="lin-b",
                      provenance_id=p2.provenance_id, valid_from=T0, valid_until=None,
                      recorded_at=T2, operation_id="fact-b")
    assert len(store.query_facts(entity="carbon", relation="atomic_number",
                                 valid_at=T2, known_at=T1)) == 1
    conflicts = store.detect_conflicts(valid_at=T2, known_at=T2,
                                       detected_at=T2, operation_prefix="detect")
    assert len(conflicts) == 1
    assert conflicts[0].status == "UNRESOLVED"
    assert conflicts[0].distinct_values == ("6", "8")


def test_invalidation_and_supersession_change_views_without_deletion(tmp_path):
    store = CognitiveControlStore(tmp_path)
    provenance = _provenance(store)
    fact = store.record_fact(entity="x", relation="role", value="ceo",
                             source_evidence_id="ev", source_lineage_id="lin",
                             provenance_id=provenance.provenance_id,
                             valid_from=T0, valid_until=None, recorded_at=T1,
                             operation_id="fact")
    store.invalidate_provenance(provenance.provenance_id, by="reviewer", reason="withdrawn",
                                at=T2, operation_id="invalidate")
    assert provenance.provenance_id in store.provenance
    assert provenance.provenance_id in store.invalidated_provenance
    assert store.query_facts(valid_at=T1, known_at=T1) == (fact,)
    assert store.query_facts(valid_at=T2, known_at=T2) == ()
    superseded = store.supersede_fact(fact.fact_id, at=T2, operation_id="supersede")
    assert superseded.superseded_at == UTC_T2
    assert store.supersede_fact(fact.fact_id, at=T2, operation_id="supersede") == superseded
    assert store.query_facts(valid_at=T2, known_at=T2) == ()


def test_decisions_form_auditable_causal_graph_and_record_outcomes(tmp_path):
    store = CognitiveControlStore(tmp_path)
    first = store.record_decision(
        task_id="task", selected_action=DecisionAction.RETRIEVE,
        alternatives_considered=(DecisionAction.ANSWER,), observations=("missing evidence",),
        evidence_used=(), memory_used=(), policy_id="policy-v1", reason_code="INSUFFICIENT",
        resource_state={"tokens": 1000}, expected_utility=None, uncertainty="HIGH",
        timestamp=T1, operation_id="decision-1")
    second = store.record_decision(
        task_id="task", selected_action=DecisionAction.VERIFY,
        alternatives_considered=(DecisionAction.ANSWER,), observations=("unverified",),
        evidence_used=("ev",), memory_used=("claim",), policy_id="policy-v1",
        reason_code="VERIFY_FIRST", resource_state={"tokens": 800},
        expected_utility=0.2, uncertainty="MEDIUM", timestamp=T2,
        operation_id="decision-2", parent_decision_id=first.decision_id)
    assert store.causal_ancestors(second.decision_id) == (first,)
    completed = store.record_outcome(second.decision_id, outcome={"verified": True}, at=T2,
                                     operation_id="outcome-2")
    assert completed.outcome == {"verified": True}
    assert completed.outcome_at == UTC_T2
    assert store.record_outcome(second.decision_id, outcome={"verified": True}, at=T2,
                                operation_id="outcome-2") == completed
    assert CognitiveControlStore(tmp_path).decisions[second.decision_id].outcome == {"verified": True}


def test_datalog_recursion_and_policy_gate_are_deterministic():
    reasoner = DatalogReasoner()
    reasoner.add_fact("parent", "alice", "bob")
    reasoner.add_fact("parent", "bob", "cara")
    reasoner.add_rule(DatalogRule(DatalogFact("ancestor", ("X", "Y")),
                                  (DatalogFact("parent", ("X", "Y")),)))
    reasoner.add_rule(DatalogRule(DatalogFact("ancestor", ("X", "Z")),
                                  (DatalogFact("parent", ("X", "Y")),
                                   DatalogFact("ancestor", ("Y", "Z")))))
    assert DatalogFact("ancestor", ("alice", "cara")) in reasoner.derive()

    gate = PolicyGate((PolicyRule(
        "high-stakes-unverified",
        DatalogFact("require", ("T", "verify", "high_stakes_unverified")),
        (DatalogFact("high_stakes", ("T",)), DatalogFact("unverified", ("T",)))),))
    result = gate.evaluate("task", DecisionAction.ANSWER, {
        DatalogFact("high_stakes", ("task",)), DatalogFact("unverified", ("task",))})
    assert result.effect is PolicyEffect.REQUIRE
    assert result.required_action is DecisionAction.VERIFY


def test_policy_gate_denies_incompatible_requirements_instead_of_sorting_them():
    gate = PolicyGate((
        PolicyRule(
            "require-verify",
            DatalogFact("require", ("T", "verify", "unverified")),
            (DatalogFact("unverified", ("T",)),)),
        PolicyRule(
            "require-defer",
            DatalogFact("require", ("T", "defer", "human_review")),
            (DatalogFact("human_review", ("T",)),)),
    ))
    facts = {DatalogFact("unverified", ("task",)),
             DatalogFact("human_review", ("task",))}

    # Even if the proposal satisfies one requirement, another incompatible
    # requirement makes execution unsafe until explicit policy resolves it.
    result = gate.evaluate("task", DecisionAction.VERIFY, facts)
    assert result.effect is PolicyEffect.DENY
    assert result.required_action is None
    assert result.reason_codes == (
        "POLICY_CONFLICT",
        "POLICY_CONFLICT_REQUIRE_DEFER",
        "POLICY_CONFLICT_REQUIRE_VERIFY",
    )


def test_policy_gate_coalesces_duplicate_requirements_for_one_action():
    gate = PolicyGate((
        PolicyRule(
            "require-verify-unverified",
            DatalogFact("require", ("T", "verify", "unverified")),
            (DatalogFact("unverified", ("T",)),)),
        PolicyRule(
            "require-verify-high-stakes",
            DatalogFact("require", ("T", "verify", "high_stakes")),
            (DatalogFact("high_stakes", ("T",)),)),
    ))
    result = gate.evaluate("task", DecisionAction.ANSWER, {
        DatalogFact("unverified", ("task",)),
        DatalogFact("high_stakes", ("task",)),
    })
    assert result.effect is PolicyEffect.REQUIRE
    assert result.required_action is DecisionAction.VERIFY
    assert result.reason_codes == ("high_stakes", "unverified")


def test_timestamps_require_timezones_and_are_stored_as_canonical_utc(tmp_path):
    store = CognitiveControlStore(tmp_path)
    provenance = _provenance(store, created_at="2026-06-01T02:30:00+02:30")
    assert provenance.created_at == UTC_T1
    fact = store.record_fact(
        entity="carbon", relation="atomic_number", value=6,
        source_evidence_id="ev", source_lineage_id="lineage",
        provenance_id=provenance.provenance_id,
        valid_from="2026-01-01T03:00:00+03:00", valid_until=None,
        recorded_at="2026-06-01T00:00:00Z", operation_id="fact")
    assert fact.valid_from == UTC_T0
    assert fact.recorded_at == UTC_T1
    with pytest.raises(ValueError, match="timezone"):
        _provenance(store, operation="naive-provenance", created_at="2026-06-01T00:00:00")
    with pytest.raises(ValueError, match="timezone"):
        store.query_facts(valid_at="2026-08-01T00:00:00", known_at=T2)


def test_temporal_lifecycle_events_cannot_predate_their_causes(tmp_path):
    store = CognitiveControlStore(tmp_path)
    provenance = _provenance(store)
    fact = store.record_fact(entity="x", relation="role", value="ceo",
                             source_evidence_id="ev", source_lineage_id="lin",
                             provenance_id=provenance.provenance_id,
                             valid_from=T0, valid_until=None, recorded_at=T1,
                             operation_id="fact")
    with pytest.raises(ValueError, match="later than valid_from"):
        store.record_fact(entity="x", relation="invalid", value="ceo",
                          source_evidence_id="ev", source_lineage_id="lin",
                          provenance_id=provenance.provenance_id,
                          valid_from=T1, valid_until=T1, recorded_at=T1,
                          operation_id="invalid-interval")
    with pytest.raises(ValueError, match="predate provenance"):
        store.invalidate_provenance(provenance.provenance_id, by="reviewer", reason="withdrawn",
                                    at=T0, operation_id="early-invalidation")
    with pytest.raises(ValueError, match="predate recorded_at"):
        store.supersede_fact(fact.fact_id, at=T0, operation_id="early-supersession")

    decision = store.record_decision(
        task_id="task", selected_action=DecisionAction.VERIFY,
        alternatives_considered=(DecisionAction.ANSWER,), observations=(), evidence_used=(),
        memory_used=(), policy_id="policy-v1", reason_code="VERIFY_FIRST",
        resource_state={}, expected_utility=None, uncertainty="HIGH", timestamp=T1,
        operation_id="decision")
    with pytest.raises(ValueError, match="predate decision"):
        store.record_outcome(decision.decision_id, outcome={"verified": False}, at=T0,
                             operation_id="early-outcome")


def test_policy_gate_rejects_malformed_actions_and_rule_shapes_at_construction():
    condition = (DatalogFact("unverified", ("T",)),)
    with pytest.raises(ValueError, match="invalid require action"):
        PolicyGate((PolicyRule(
            "invalid-action", DatalogFact("require", ("T", "destroy", "bad")), condition),))
    rule = PolicyRule(
        "duplicate", DatalogFact("require", ("T", "verify", "unverified")), condition)
    with pytest.raises(ValueError, match="duplicate policy rule_id"):
        PolicyGate((rule, rule))
    with pytest.raises(ValueError, match="unbound head variables"):
        PolicyGate((PolicyRule(
            "unsafe", DatalogFact("require", ("T", "verify", "unverified")),
            (DatalogFact("unverified", ("OTHER",)),)),))


def test_policy_gate_rejects_caller_injected_policy_effects():
    gate = PolicyGate(())
    with pytest.raises(ValueError, match="reserved predicate"):
        gate.evaluate("task", DecisionAction.ANSWER, {
            DatalogFact("require", ("task", "destroy", "injected")),
        })
