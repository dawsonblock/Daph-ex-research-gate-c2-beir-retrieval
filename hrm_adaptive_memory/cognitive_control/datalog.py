"""Small deterministic Datalog engine for executive invariants and policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class DatalogFact:
    predicate: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class DatalogRule:
    head: DatalogFact
    body: tuple[DatalogFact, ...]


class DatalogReasoner:
    """Finite, positive Horn-clause evaluation; no negation or side effects."""

    def __init__(self):
        self.facts: set[DatalogFact] = set()
        self.rules: list[DatalogRule] = []

    def add_fact(self, predicate: str, *args: str) -> DatalogFact:
        fact = DatalogFact(predicate, tuple(args))
        if not predicate or not args or any(not arg or arg[0].isupper() for arg in args):
            raise ValueError("ground facts require a predicate and lowercase constants")
        self.facts.add(fact)
        return fact

    def add_rule(self, rule: DatalogRule) -> None:
        if not rule.body:
            raise ValueError("a policy rule must have at least one body atom")
        self.rules.append(rule)

    @staticmethod
    def _variable(value: str) -> bool:
        return bool(value) and value[0].isupper()

    def _matches(self, pattern: DatalogFact, fact: DatalogFact,
                 bindings: Mapping[str, str]) -> dict[str, str] | None:
        if pattern.predicate != fact.predicate or len(pattern.args) != len(fact.args):
            return None
        out = dict(bindings)
        for wanted, actual in zip(pattern.args, fact.args):
            if self._variable(wanted):
                if wanted in out and out[wanted] != actual:
                    return None
                out[wanted] = actual
            elif wanted != actual:
                return None
        return out

    def _bindings(self, body: Iterable[DatalogFact]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = [{}]
        for atom in body:
            next_rows = []
            for row in rows:
                for fact in self.facts:
                    match = self._matches(atom, fact, row)
                    if match is not None:
                        next_rows.append(match)
            rows = next_rows
            if not rows:
                break
        return rows

    def derive(self) -> set[DatalogFact]:
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                for binding in self._bindings(rule.body):
                    args = tuple(binding.get(arg, arg) for arg in rule.head.args)
                    if any(self._variable(arg) for arg in args):
                        raise ValueError("unsafe rule contains an unbound head variable")
                    fact = DatalogFact(rule.head.predicate, args)
                    if fact not in self.facts:
                        self.facts.add(fact)
                        changed = True
        return set(self.facts)

    def query(self, predicate: str, *args: str) -> tuple[DatalogFact, ...]:
        self.derive()
        pattern = DatalogFact(predicate, tuple(args))
        return tuple(sorted(fact for fact in self.facts
                            if self._matches(pattern, fact, {}) is not None))
