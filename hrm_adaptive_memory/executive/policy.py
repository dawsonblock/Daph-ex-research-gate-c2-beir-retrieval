"""Versioned, immutable deterministic policy loading for V2B development."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from hrm_adaptive_memory.cognitive_control.core import PolicyGate, PolicyRule
from hrm_adaptive_memory.cognitive_control.datalog import DatalogFact


POLICY_SCHEMAS = frozenset({"DAPH_V2B_I2_POLICY_V1", "DAPH_V2B_I3_POLICY_V1"})


def _fact(raw: object) -> DatalogFact:
    if not isinstance(raw, dict) or not isinstance(raw.get("predicate"), str):
        raise ValueError("policy atoms require a predicate")
    args = raw.get("args")
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        raise ValueError("policy atom args must be strings")
    return DatalogFact(raw["predicate"], tuple(args))


@dataclass(frozen=True)
class FrozenPolicy:
    policy_id: str
    sha256: str
    gate: PolicyGate


def load_frozen_policy(path: str | Path) -> FrozenPolicy:
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") not in POLICY_SCHEMAS or payload.get("status") != "FROZEN_FOR_DEVELOPMENT":
        raise ValueError("V2B policy must be a frozen development policy")
    rules = tuple(PolicyRule(
        rule_id=entry["rule_id"], head=_fact(entry["head"]),
        body=tuple(_fact(atom) for atom in entry["body"]),
    ) for entry in payload.get("rules", ()))
    if not rules or not isinstance(payload.get("policy_id"), str):
        raise ValueError("V2B policy must have an id and nonempty rules")
    return FrozenPolicy(payload["policy_id"], hashlib.sha256(raw).hexdigest(), PolicyGate(rules))
