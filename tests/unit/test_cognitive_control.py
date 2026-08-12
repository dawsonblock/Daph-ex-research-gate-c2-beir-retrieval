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


def _provenance(store, entity="claim-1", operation="prov-1"):
    return store.record_provenance(
        entity_id=entity, entity_type="claim", payload={"value": 6},
        activity_id="verification", agent_id="daph-v2a", agent_type="software_agent",
        source_id="evidence-1", created_at=T1, operation_id=operation,
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
    assert store.supersede_fact(fact.fact_id, at=T2, operation_id="supersede").superseded_at == T2
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
    completed = store.record_outcome(second.decision_id, outcome={"verified": True},
                                     operation_id="outcome-2")
    assert completed.outcome == {"verified": True}
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
