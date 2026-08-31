"""Hash-chained provenance, bitemporal memory, conflicts, and decisions.

Semantica supplied the donor concepts. DAPH keeps its own append-only event
contract and deliberately provides conflict detection without truth voting.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .datalog import DatalogFact, DatalogReasoner, DatalogRule

SCHEMA_VERSION = "DAPH_COGNITIVE_CONTROL_V2"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _json(value).encode()
    return hashlib.sha256(raw).hexdigest()


def _moment(value: str, *, field_name: str = "timestamp") -> datetime:
    """Parse an aware ISO-8601 timestamp and convert it to UTC.

    Callers may provide any aware ISO-8601 representation at an API boundary.
    The event log itself is stricter: `_canonical_timestamp` emits the sole
    stored representation, UTC RFC3339 with microsecond precision and `Z`.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp string")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_timestamp(value: str, *, field_name: str = "timestamp") -> str:
    return _moment(value, field_name=field_name).isoformat(timespec="microseconds").replace(
        "+00:00", "Z")


def _stored_timestamp(value: str, *, field_name: str) -> str:
    """Reject replayed values that are not in the canonical event representation."""
    canonical = _canonical_timestamp(value, field_name=field_name)
    if value != canonical:
        raise ValueError(f"{field_name} must use canonical UTC RFC3339 format")
    return canonical


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
    outcome_at: str | None = None


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

    _POLICY_HEADS = frozenset({"deny", "require"})
    _RESERVED_INPUT_PREDICATES = frozenset({"proposed", "deny", "require"})
    _ACTIONS_BY_POLICY_NAME = {
        action.value.lower(): action for action in DecisionAction
    }

    def __init__(self, rules: Iterable[PolicyRule]):
        self.rules = tuple(rules)

        seen_rule_ids: set[str] = set()
        predicate_arities: dict[str, int] = {"proposed": 2, "deny": 3, "require": 3}
        for rule in self.rules:
            self._validate_rule(rule, seen_rule_ids, predicate_arities)

    @staticmethod
    def _is_variable(value: str) -> bool:
        return bool(value) and value[0].isupper()

    @classmethod
    def _validate_atom(cls, atom: DatalogFact, predicate_arities: dict[str, int], *,
                       location: str) -> None:
        if not isinstance(atom, DatalogFact):
            raise ValueError(f"{location} must be a DatalogFact")
        if (not isinstance(atom.predicate, str) or not atom.predicate
                or not isinstance(atom.args, tuple) or not atom.args
                or any(not isinstance(arg, str) or not arg for arg in atom.args)):
            raise ValueError(f"{location} must have a predicate and nonempty string arguments")
        existing_arity = predicate_arities.setdefault(atom.predicate, len(atom.args))
        if existing_arity != len(atom.args):
            raise ValueError(
                f"predicate {atom.predicate!r} has inconsistent arity in policy rules")

    @classmethod
    def _validate_rule(cls, rule: PolicyRule, seen_rule_ids: set[str],
                       predicate_arities: dict[str, int]) -> None:
        if not isinstance(rule, PolicyRule):
            raise ValueError("policy rules must be PolicyRule instances")
        if not isinstance(rule.rule_id, str) or not rule.rule_id:
            raise ValueError("policy rule_id is required")
        if rule.rule_id in seen_rule_ids:
            raise ValueError(f"duplicate policy rule_id: {rule.rule_id}")
        seen_rule_ids.add(rule.rule_id)
        if not isinstance(rule.body, tuple) or not rule.body:
            raise ValueError(f"policy rule {rule.rule_id!r} must have a nonempty body")

        cls._validate_atom(rule.head, predicate_arities, location="policy rule head")
        if rule.head.predicate not in cls._POLICY_HEADS or len(rule.head.args) != 3:
            raise ValueError(
                f"policy rule {rule.rule_id!r} head must be deny(task, action, reason) "
                "or require(task, action, reason)")
        action_name = rule.head.args[1]
        if cls._is_variable(action_name) or action_name not in cls._ACTIONS_BY_POLICY_NAME:
            raise ValueError(
                f"policy rule {rule.rule_id!r} has invalid {rule.head.predicate} action: "
                f"{action_name!r}")

        body_variables: set[str] = set()
        for atom in rule.body:
            cls._validate_atom(atom, predicate_arities, location="policy rule body atom")
            body_variables.update(arg for arg in atom.args if cls._is_variable(arg))
        head_variables = {arg for arg in rule.head.args if cls._is_variable(arg)}
        unbound = sorted(head_variables - body_variables)
        if unbound:
            raise ValueError(
                f"policy rule {rule.rule_id!r} has unbound head variables: {', '.join(unbound)}")

    def evaluate(self, task_id: str, proposed: DecisionAction,
                 facts: Iterable[DatalogFact]) -> PolicyDecision:
        engine = DatalogReasoner()
        for fact in facts:
            if not isinstance(fact, DatalogFact):
                raise ValueError("policy input facts must be DatalogFact instances")
            if fact.predicate in self._RESERVED_INPUT_PREDICATES:
                raise ValueError(
                    f"policy input facts cannot assert reserved predicate {fact.predicate!r}")
            engine.add_fact(fact.predicate, *fact.args)
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
            action = self._ACTIONS_BY_POLICY_NAME[required_actions[0]]
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
            if event.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported cognitive-control event schema")
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
            prior = self.by_operation.get(event["operation_id"])
            if prior is not None:
                raise ValueError("canonical history repeats an operation_id")
            self.events.append(event); self.by_operation[event["operation_id"]] = event
            self._apply(event); previous = event["event_hash"]
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if (manifest.get("schema_version") != SCHEMA_VERSION
                    or manifest.get("event_count") != len(self.events)
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
            raw = dict(payload)
            raw["derived_from"] = tuple(raw["derived_from"])
            raw["created_at"] = _stored_timestamp(raw["created_at"], field_name="created_at")
            record = ProvenanceRecord(**raw); self.provenance[record.provenance_id] = record
        elif kind == "PROVENANCE_INVALIDATED":
            provenance_id = payload["provenance_id"]
            if provenance_id not in self.provenance:
                raise ValueError("invalidation references absent provenance")
            at = _stored_timestamp(payload["at"], field_name="invalidated_at")
            if _moment(at) < _moment(self.provenance[provenance_id].created_at):
                raise ValueError("invalidated_at cannot predate provenance creation")
            self.invalidated_provenance[provenance_id] = at
        elif kind == "TEMPORAL_FACT_RECORDED":
            raw = dict(payload)
            if (raw["provenance_id"] not in self.provenance
                    or raw["provenance_id"] in self.invalidated_provenance):
                raise ValueError("fact requires active provenance")
            for field_name in ("valid_from", "recorded_at"):
                raw[field_name] = _stored_timestamp(raw[field_name], field_name=field_name)
            for field_name in ("valid_until", "superseded_at"):
                if raw[field_name] is not None:
                    raw[field_name] = _stored_timestamp(raw[field_name], field_name=field_name)
            if (raw["valid_until"] is not None
                    and _moment(raw["valid_until"]) <= _moment(raw["valid_from"])):
                raise ValueError("valid_until must be later than valid_from")
            if (raw["superseded_at"] is not None
                    and _moment(raw["superseded_at"]) < _moment(raw["recorded_at"])):
                raise ValueError("superseded_at cannot predate recorded_at")
            fact = TemporalFact(**raw); self.facts[fact.fact_id] = fact
        elif kind == "TEMPORAL_FACT_SUPERSEDED":
            old = self.facts[payload["fact_id"]]
            if old.superseded_at is not None:
                raise ValueError("fact is already superseded")
            at = _stored_timestamp(payload["at"], field_name="superseded_at")
            if _moment(at) < _moment(old.recorded_at):
                raise ValueError("superseded_at cannot predate recorded_at")
            self.facts[old.fact_id] = TemporalFact(**{**asdict(old), "superseded_at": at})
        elif kind == "CONFLICT_RECORDED":
            raw = dict(payload)
            for key in ("fact_ids", "distinct_values", "source_lineage_ids"):
                raw[key] = tuple(raw[key])
            raw["detected_at"] = _stored_timestamp(raw["detected_at"], field_name="detected_at")
            conflict = ConflictEvent(**raw); self.conflicts[conflict.conflict_id] = conflict
        elif kind == "DECISION_RECORDED":
            raw = dict(payload); raw["selected_action"] = DecisionAction(raw["selected_action"])
            raw["alternatives_considered"] = tuple(DecisionAction(v) for v in raw["alternatives_considered"])
            for key in ("observations", "evidence_used", "memory_used"):
                raw[key] = tuple(raw[key])
            raw["timestamp"] = _stored_timestamp(raw["timestamp"], field_name="decision timestamp")
            raw.setdefault("outcome_at", None)
            if raw["outcome_at"] is not None:
                raw["outcome_at"] = _stored_timestamp(raw["outcome_at"], field_name="outcome_at")
                if _moment(raw["outcome_at"]) < _moment(raw["timestamp"]):
                    raise ValueError("outcome_at cannot predate decision timestamp")
            parent_id = raw["parent_decision_id"]
            if parent_id is not None:
                if parent_id not in self.decisions:
                    raise ValueError("decision parent is absent")
                if _moment(raw["timestamp"]) < _moment(self.decisions[parent_id].timestamp):
                    raise ValueError("decision timestamp cannot predate parent decision")
            decision = DecisionRecord(**raw); self.decisions[decision.decision_id] = decision
        elif kind == "DECISION_OUTCOME_RECORDED":
            old = self.decisions[payload["decision_id"]]
            if old.outcome is not None:
                raise ValueError("decision outcome is already recorded")
            outcome_at = _stored_timestamp(payload["outcome_at"], field_name="outcome_at")
            if _moment(outcome_at) < _moment(old.timestamp):
                raise ValueError("outcome_at cannot predate decision timestamp")
            self.decisions[old.decision_id] = DecisionRecord(
                **{**asdict(old), "outcome": payload["outcome"], "outcome_at": outcome_at})

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

    def _is_same_committed_operation(self, event_type: str, payload: Mapping[str, Any],
                                     operation_id: str) -> bool:
        prior = self.by_operation.get(operation_id)
        return bool(prior and prior["event_type"] == event_type
                    and prior["payload"] == dict(payload))

    def record_provenance(self, *, entity_id: str, entity_type: str, payload: Any,
                          activity_id: str, agent_id: str, agent_type: str,
                          source_id: str, created_at: str, operation_id: str,
                          derived_from: Iterable[str] = (), previous_version_id: str | None = None,
                          bundle_id: str | None = None) -> ProvenanceRecord:
        created_at = _canonical_timestamp(created_at, field_name="created_at")
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
        valid_from = _canonical_timestamp(valid_from, field_name="valid_from")
        valid_until = (None if valid_until is None else _canonical_timestamp(
            valid_until, field_name="valid_until"))
        recorded_at = _canonical_timestamp(recorded_at, field_name="recorded_at")
        if valid_until is not None and _moment(valid_until) <= _moment(valid_from):
            raise ValueError("valid_until must be later than valid_from")
        body = {"entity": entity, "relation": relation, "value": value,
                "source_evidence_id": source_evidence_id,
                "source_lineage_id": source_lineage_id, "provenance_id": provenance_id,
                "valid_from": valid_from, "valid_until": valid_until,
                "recorded_at": recorded_at, "superseded_at": None}
        body["fact_id"] = "tf-" + _sha(body)[:24]
        if (provenance_id not in self.provenance or provenance_id in self.invalidated_provenance) and not (
                self._is_same_committed_operation("TEMPORAL_FACT_RECORDED", body, operation_id)):
            raise ValueError("fact requires active provenance")
        event = self.append("TEMPORAL_FACT_RECORDED", body, operation_id)
        return self.facts[event["payload"]["fact_id"]]

    def invalidate_provenance(self, provenance_id: str, *, by: str, reason: str,
                              at: str, operation_id: str) -> None:
        if provenance_id not in self.provenance:
            raise KeyError(provenance_id)
        at = _canonical_timestamp(at, field_name="invalidated_at")
        if _moment(at) < _moment(self.provenance[provenance_id].created_at):
            raise ValueError("invalidated_at cannot predate provenance creation")
        self.append("PROVENANCE_INVALIDATED", {
            "provenance_id": provenance_id, "invalidated_by": by,
            "reason": reason, "at": at}, operation_id)

    def supersede_fact(self, fact_id: str, *, at: str, operation_id: str) -> TemporalFact:
        if fact_id not in self.facts:
            raise KeyError(fact_id)
        fact = self.facts[fact_id]
        at = _canonical_timestamp(at, field_name="superseded_at")
        body = {"fact_id": fact_id, "at": at}
        if fact.superseded_at is not None and not self._is_same_committed_operation(
                "TEMPORAL_FACT_SUPERSEDED", body, operation_id):
            raise ValueError("fact is already superseded")
        if _moment(at) < _moment(fact.recorded_at):
            raise ValueError("superseded_at cannot predate recorded_at")
        self.append("TEMPORAL_FACT_SUPERSEDED", body, operation_id)
        return self.facts[fact_id]

    def query_facts(self, *, entity: str | None = None, relation: str | None = None,
                    valid_at: str, known_at: str) -> tuple[TemporalFact, ...]:
        valid_at = _canonical_timestamp(valid_at, field_name="valid_at")
        known_at = _canonical_timestamp(known_at, field_name="known_at")
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
        valid_at = _canonical_timestamp(valid_at, field_name="valid_at")
        known_at = _canonical_timestamp(known_at, field_name="known_at")
        detected_at = _canonical_timestamp(detected_at, field_name="detected_at")
        if _moment(detected_at) < _moment(known_at):
            raise ValueError("detected_at cannot predate known_at")
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
        timestamp = _canonical_timestamp(timestamp, field_name="decision timestamp")
        if (parent_decision_id is not None
                and _moment(timestamp) < _moment(self.decisions[parent_decision_id].timestamp)):
            raise ValueError("decision timestamp cannot predate parent decision")
        body = {"task_id": task_id, "selected_action": selected_action.value,
                "alternatives_considered": tuple(v.value for v in alternatives_considered),
                "observations": tuple(observations), "evidence_used": tuple(evidence_used),
                "memory_used": tuple(memory_used), "policy_id": policy_id,
                "reason_code": reason_code, "resource_state": dict(resource_state),
                "expected_utility": expected_utility, "uncertainty": uncertainty,
                "timestamp": timestamp, "parent_decision_id": parent_decision_id,
                "outcome": None, "outcome_at": None}
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
                       at: str, operation_id: str) -> DecisionRecord:
        if decision_id not in self.decisions:
            raise KeyError(decision_id)
        decision = self.decisions[decision_id]
        at = _canonical_timestamp(at, field_name="outcome_at")
        if _moment(at) < _moment(decision.timestamp):
            raise ValueError("outcome_at cannot predate decision timestamp")
        body = {"decision_id": decision_id, "outcome": dict(outcome), "outcome_at": at}
        if decision.outcome is not None and not self._is_same_committed_operation(
                "DECISION_OUTCOME_RECORDED", body, operation_id):
            raise ValueError("decision outcome is already recorded")
        self.append("DECISION_OUTCOME_RECORDED", body, operation_id)
        return self.decisions[decision_id]
