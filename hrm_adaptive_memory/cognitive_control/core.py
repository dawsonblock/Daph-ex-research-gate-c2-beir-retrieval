"""Hash-chained provenance, bitemporal memory, conflicts, and decisions.

Semantica supplied the donor concepts. DAPH keeps its own append-only event
contract and deliberately provides conflict detection without truth voting.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .datalog import DatalogFact, DatalogReasoner, DatalogRule

SCHEMA_VERSION = "DAPH_COGNITIVE_CONTROL_V1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _json(value).encode()
    return hashlib.sha256(raw).hexdigest()


def _moment(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class ProvenanceRecord:
    provenance_id: str
    entity_id: str
    entity_type: str
    payload_hash: str
    activity_id: str
    agent_id: str
    agent_type: str
    source_id: str
    derived_from: tuple[str, ...]
    previous_version_id: str | None
    bundle_id: str | None
    created_at: str
    operation_id: str


@dataclass(frozen=True)
class TemporalFact:
    fact_id: str
    entity: str
    relation: str
    value: Any
    source_evidence_id: str
    source_lineage_id: str
    provenance_id: str
    valid_from: str
    valid_until: str | None
    recorded_at: str
    superseded_at: str | None = None

    def valid_at(self, when: str) -> bool:
        point = _moment(when)
        return _moment(self.valid_from) <= point and (
            self.valid_until is None or point < _moment(self.valid_until))

    def known_at(self, when: str) -> bool:
        point = _moment(when)
        return _moment(self.recorded_at) <= point and (
            self.superseded_at is None or point < _moment(self.superseded_at))


@dataclass(frozen=True)
class ConflictEvent:
    conflict_id: str
    entity: str
    relation: str
    fact_ids: tuple[str, ...]
    distinct_values: tuple[str, ...]
    source_lineage_ids: tuple[str, ...]
    detected_at: str
    status: str = "UNRESOLVED"


class DecisionAction(str, Enum):
    ANSWER = "ANSWER"
    RETRIEVE = "RETRIEVE"
    VERIFY = "VERIFY"
    VERIFY_ALTERNATE_SOURCE = "VERIFY_ALTERNATE_SOURCE"
    SEARCH_MORE = "SEARCH_MORE"
    REASON_MORE = "REASON_MORE"
    SPAWN_SPECIALIST = "SPAWN_SPECIALIST"
    SWITCH_STRATEGY = "SWITCH_STRATEGY"
    ABANDON_STRATEGY = "ABANDON_STRATEGY"
    DEFER = "DEFER"
    STOP = "STOP"


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    task_id: str
    selected_action: DecisionAction
    alternatives_considered: tuple[DecisionAction, ...]
    observations: tuple[str, ...]
    evidence_used: tuple[str, ...]
    memory_used: tuple[str, ...]
    policy_id: str
    reason_code: str
    resource_state: Mapping[str, Any]
    expected_utility: float | None
    uncertainty: str
    timestamp: str
    parent_decision_id: str | None
    outcome: Mapping[str, Any] | None = None


class PolicyEffect(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE = "REQUIRE"


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    head: DatalogFact
    body: tuple[DatalogFact, ...]


@dataclass(frozen=True)
class PolicyDecision:
    effect: PolicyEffect
    proposed_action: DecisionAction
    required_action: DecisionAction | None
    reason_codes: tuple[str, ...]


class PolicyGate:
    """Deterministic positive-rule gate around an executive proposal."""

    def __init__(self, rules: Iterable[PolicyRule]):
        self.rules = tuple(rules)

    def evaluate(self, task_id: str, proposed: DecisionAction,
                 facts: Iterable[DatalogFact]) -> PolicyDecision:
        engine = DatalogReasoner()
        engine.facts.update(facts)
        engine.add_fact("proposed", task_id, proposed.value.lower())
        for rule in self.rules:
            engine.add_rule(DatalogRule(rule.head, rule.body))
        derived = engine.derive()
        denied = sorted(f.args[2] for f in derived
                        if f.predicate == "deny" and len(f.args) >= 3
                        and f.args[:2] == (task_id, proposed.value.lower()))
        required = sorted(f for f in derived
                          if f.predicate == "require" and len(f.args) >= 3
                          and f.args[0] == task_id)
        if denied:
            return PolicyDecision(PolicyEffect.DENY, proposed, None, tuple(denied))
        required_actions = tuple(sorted({fact.args[1] for fact in required}))
        if len(required_actions) > 1:
            # Policy rules are constraints, not ranked suggestions. Selecting
            # the first Datalog fact would make a Python sort order an
            # accidental safety policy, so incompatible requirements deny the
            # proposed action until a separately frozen resolution rule exists.
            reasons = ("POLICY_CONFLICT",) + tuple(
                f"POLICY_CONFLICT_REQUIRE_{action.upper()}"
                for action in required_actions)
            return PolicyDecision(PolicyEffect.DENY, proposed, None, reasons)
        if required_actions and required_actions[0] != proposed.value.lower():
            action = DecisionAction(required_actions[0].upper())
            return PolicyDecision(PolicyEffect.REQUIRE, proposed, action,
                                  tuple(f.args[2] for f in required))
        return PolicyDecision(PolicyEffect.ALLOW, proposed, None, ())


class CognitiveControlStore:
    """Canonical append-only store with hash-chain and operation idempotency."""

    EVENT_TYPES = {
        "PROVENANCE_RECORDED", "PROVENANCE_INVALIDATED", "TEMPORAL_FACT_RECORDED",
        "TEMPORAL_FACT_SUPERSEDED", "CONFLICT_RECORDED", "DECISION_RECORDED",
        "DECISION_OUTCOME_RECORDED",
    }

    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "cognitive_control_events.jsonl"
        self.manifest_path = self.root / "COGNITIVE_CONTROL_MANIFEST.json"
        self.events: list[dict[str, Any]] = []
        self.by_operation: dict[str, dict[str, Any]] = {}
        self.provenance: dict[str, ProvenanceRecord] = {}
        self.invalidated_provenance: dict[str, str] = {}
        self.facts: dict[str, TemporalFact] = {}
        self.conflicts: dict[str, ConflictEvent] = {}
        self.decisions: dict[str, DecisionRecord] = {}
        self._replay()

    def _replay(self) -> None:
        previous = "GENESIS"
        if not self.log_path.exists():
            return
        for raw in self.log_path.read_text().splitlines():
            event = json.loads(raw)
            if event.get("event_type") not in self.EVENT_TYPES:
                raise ValueError("unknown cognitive-control event type")
            if event.get("previous_event_hash") != previous:
                raise ValueError("cognitive-control hash chain is broken")
            unsigned = {k: v for k, v in event.items()
                        if k not in {"event_hash", "event_id"}}
            if event.get("event_hash") != _sha(unsigned):
                raise ValueError("cognitive-control event hash mismatch")
            if event.get("event_id") != "cce-" + event["event_hash"][:24]:
                raise ValueError("cognitive-control event identity mismatch")
            self.events.append(event); self.by_operation[event["operation_id"]] = event
            self._apply(event); previous = event["event_hash"]
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if (manifest.get("event_count") != len(self.events)
                    or manifest.get("head_event_hash") != previous
                    or manifest.get("event_log_sha256") != _sha(self.log_path.read_bytes())):
                raise ValueError("cognitive-control manifest does not match canonical history")

    def _write_manifest(self) -> None:
        manifest = _json({
            "schema_version": SCHEMA_VERSION,
            "event_count": len(self.events),
            "head_event_hash": self.events[-1]["event_hash"] if self.events else "GENESIS",
            "event_log_sha256": _sha(self.log_path.read_bytes()) if self.log_path.exists() else _sha(b""),
        }) + "\n"
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w") as handle:
            handle.write(manifest); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, self.manifest_path)

    def _apply(self, event: Mapping[str, Any]) -> None:
        kind, payload = event["event_type"], event["payload"]
        if kind == "PROVENANCE_RECORDED":
            raw = dict(payload); raw["derived_from"] = tuple(raw["derived_from"])
            record = ProvenanceRecord(**raw); self.provenance[record.provenance_id] = record
        elif kind == "PROVENANCE_INVALIDATED":
            self.invalidated_provenance[payload["provenance_id"]] = payload["at"]
        elif kind == "TEMPORAL_FACT_RECORDED":
            fact = TemporalFact(**payload); self.facts[fact.fact_id] = fact
        elif kind == "TEMPORAL_FACT_SUPERSEDED":
            old = self.facts[payload["fact_id"]]
            self.facts[old.fact_id] = TemporalFact(**{**asdict(old), "superseded_at": payload["at"]})
        elif kind == "CONFLICT_RECORDED":
            raw = dict(payload)
            for key in ("fact_ids", "distinct_values", "source_lineage_ids"):
                raw[key] = tuple(raw[key])
            conflict = ConflictEvent(**raw); self.conflicts[conflict.conflict_id] = conflict
        elif kind == "DECISION_RECORDED":
            raw = dict(payload); raw["selected_action"] = DecisionAction(raw["selected_action"])
            raw["alternatives_considered"] = tuple(DecisionAction(v) for v in raw["alternatives_considered"])
            for key in ("observations", "evidence_used", "memory_used"):
                raw[key] = tuple(raw[key])
            decision = DecisionRecord(**raw); self.decisions[decision.decision_id] = decision
        elif kind == "DECISION_OUTCOME_RECORDED":
            old = self.decisions[payload["decision_id"]]
            self.decisions[old.decision_id] = DecisionRecord(**{**asdict(old), "outcome": payload["outcome"]})

    def append(self, event_type: str, payload: Mapping[str, Any], operation_id: str) -> dict[str, Any]:
        if event_type not in self.EVENT_TYPES or not operation_id:
            raise ValueError("known event type and operation_id are required")
        base = {"schema_version": SCHEMA_VERSION, "event_type": event_type,
                "operation_id": operation_id, "payload": dict(payload),
                "previous_event_hash": self.events[-1]["event_hash"] if self.events else "GENESIS"}
        digest = _sha(base); event = {**base, "event_hash": digest, "event_id": "cce-" + digest[:24]}
        prior = self.by_operation.get(operation_id)
        if prior is not None:
            if prior["event_type"] != event_type or prior["payload"] != dict(payload):
                raise ValueError("operation_id was already committed with different content")
            return prior
        with self.log_path.open("a") as handle:
            handle.write(_json(event) + "\n"); handle.flush(); os.fsync(handle.fileno())
        self.events.append(event); self.by_operation[operation_id] = event; self._apply(event)
        self._write_manifest()
        return event

    def record_provenance(self, *, entity_id: str, entity_type: str, payload: Any,
                          activity_id: str, agent_id: str, agent_type: str,
                          source_id: str, created_at: str, operation_id: str,
                          derived_from: Iterable[str] = (), previous_version_id: str | None = None,
                          bundle_id: str | None = None) -> ProvenanceRecord:
        body = {"entity_id": entity_id, "entity_type": entity_type, "payload_hash": _sha(payload),
                "activity_id": activity_id, "agent_id": agent_id, "agent_type": agent_type,
                "source_id": source_id, "derived_from": tuple(sorted(derived_from)),
                "previous_version_id": previous_version_id, "bundle_id": bundle_id,
                "created_at": created_at, "operation_id": operation_id}
        body["provenance_id"] = "prv-" + _sha(body)[:24]
        event = self.append("PROVENANCE_RECORDED", body, operation_id)
        return self.provenance[event["payload"]["provenance_id"]]

    def record_fact(self, *, entity: str, relation: str, value: Any,
                    source_evidence_id: str, source_lineage_id: str, provenance_id: str,
                    valid_from: str, valid_until: str | None, recorded_at: str,
                    operation_id: str) -> TemporalFact:
        if provenance_id not in self.provenance or provenance_id in self.invalidated_provenance:
            raise ValueError("fact requires active provenance")
        if valid_until is not None and _moment(valid_until) <= _moment(valid_from):
            raise ValueError("valid_until must be later than valid_from")
        body = {"entity": entity, "relation": relation, "value": value,
                "source_evidence_id": source_evidence_id,
                "source_lineage_id": source_lineage_id, "provenance_id": provenance_id,
                "valid_from": valid_from, "valid_until": valid_until,
                "recorded_at": recorded_at, "superseded_at": None}
        body["fact_id"] = "tf-" + _sha(body)[:24]
        event = self.append("TEMPORAL_FACT_RECORDED", body, operation_id)
        return self.facts[event["payload"]["fact_id"]]

    def invalidate_provenance(self, provenance_id: str, *, by: str, reason: str,
                              at: str, operation_id: str) -> None:
        if provenance_id not in self.provenance:
            raise KeyError(provenance_id)
        self.append("PROVENANCE_INVALIDATED", {
            "provenance_id": provenance_id, "invalidated_by": by,
            "reason": reason, "at": at}, operation_id)

    def supersede_fact(self, fact_id: str, *, at: str, operation_id: str) -> TemporalFact:
        if fact_id not in self.facts:
            raise KeyError(fact_id)
        self.append("TEMPORAL_FACT_SUPERSEDED", {"fact_id": fact_id, "at": at}, operation_id)
        return self.facts[fact_id]

    def query_facts(self, *, entity: str | None = None, relation: str | None = None,
                    valid_at: str, known_at: str) -> tuple[TemporalFact, ...]:
        return tuple(sorted((fact for fact in self.facts.values()
                            if (entity is None or fact.entity == entity)
                            and (relation is None or fact.relation == relation)
                            and fact.valid_at(valid_at) and fact.known_at(known_at)
                            and (fact.provenance_id not in self.invalidated_provenance
                                 or _moment(known_at) < _moment(
                                     self.invalidated_provenance[fact.provenance_id]))),
                           key=lambda fact: fact.fact_id))

    def detect_conflicts(self, *, valid_at: str, known_at: str,
                         detected_at: str, operation_prefix: str) -> tuple[ConflictEvent, ...]:
        groups: dict[tuple[str, str], list[TemporalFact]] = {}
        for fact in self.query_facts(valid_at=valid_at, known_at=known_at):
            groups.setdefault((fact.entity, fact.relation), []).append(fact)
        out = []
        for (entity, relation), facts in sorted(groups.items()):
            values = tuple(sorted({_json(fact.value) for fact in facts}))
            if len(values) < 2:
                continue
            body = {"entity": entity, "relation": relation,
                    "fact_ids": tuple(sorted(f.fact_id for f in facts)),
                    "distinct_values": values,
                    "source_lineage_ids": tuple(sorted({f.source_lineage_id for f in facts})),
                    "detected_at": detected_at, "status": "UNRESOLVED"}
            body["conflict_id"] = "cnf-" + _sha(body)[:24]
            event = self.append("CONFLICT_RECORDED", body,
                                f"{operation_prefix}:{body['conflict_id']}")
            out.append(self.conflicts[event["payload"]["conflict_id"]])
        return tuple(out)

    def record_decision(self, *, task_id: str, selected_action: DecisionAction,
                        alternatives_considered: Iterable[DecisionAction], observations: Iterable[str],
                        evidence_used: Iterable[str], memory_used: Iterable[str], policy_id: str,
                        reason_code: str, resource_state: Mapping[str, Any],
                        expected_utility: float | None, uncertainty: str, timestamp: str,
                        operation_id: str, parent_decision_id: str | None = None) -> DecisionRecord:
        if parent_decision_id is not None and parent_decision_id not in self.decisions:
            raise ValueError("parent decision is absent")
        body = {"task_id": task_id, "selected_action": selected_action.value,
                "alternatives_considered": tuple(v.value for v in alternatives_considered),
                "observations": tuple(observations), "evidence_used": tuple(evidence_used),
                "memory_used": tuple(memory_used), "policy_id": policy_id,
                "reason_code": reason_code, "resource_state": dict(resource_state),
                "expected_utility": expected_utility, "uncertainty": uncertainty,
                "timestamp": timestamp, "parent_decision_id": parent_decision_id, "outcome": None}
        body["decision_id"] = "dec-" + _sha(body)[:24]
        event = self.append("DECISION_RECORDED", body, operation_id)
        return self.decisions[event["payload"]["decision_id"]]

    def causal_ancestors(self, decision_id: str) -> tuple[DecisionRecord, ...]:
        out, seen = [], set()
        current = self.decisions[decision_id]
        while current.parent_decision_id is not None:
            if current.parent_decision_id in seen:
                raise ValueError("decision causal loop detected")
            seen.add(current.parent_decision_id)
            current = self.decisions[current.parent_decision_id]; out.append(current)
        return tuple(out)

    def record_outcome(self, decision_id: str, *, outcome: Mapping[str, Any],
                       operation_id: str) -> DecisionRecord:
        if decision_id not in self.decisions:
            raise KeyError(decision_id)
        self.append("DECISION_OUTCOME_RECORDED", {
            "decision_id": decision_id, "outcome": dict(outcome)}, operation_id)
        return self.decisions[decision_id]
