"""Typed parameterized actions for DAPH-X.

Unlike V3R2's coarse actions (VERIFY, REASON_MORE), DAPH-X uses
concrete parameterized actions where the target is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Canonical action types."""
    # Terminal actions
    ANSWER = "ANSWER"              # ANSWER(hypothesis_id)
    DEFER = "DEFER"                # DEFER(reason)
    STOP = "STOP"                  # STOP(reason)

    # Information-gathering actions
    VERIFY = "VERIFY"              # VERIFY(evidence_id)
    RETRIEVE = "RETRIEVE"          # RETRIEVE(query, source_scope)
    SEARCH = "SEARCH"              # SEARCH(query, source_scope)
    TEST = "TEST"                  # TEST(test_id)

    # Cognitive operators
    COMPARE = "COMPARE"            # COMPARE(h1, h2)
    CHECK_CONSISTENCY = "CHECK_CONSISTENCY"  # CHECK_CONSISTENCY(target)
    GENERATE_ALTERNATIVE = "GENERATE_ALTERNATIVE"  # GENERATE_ALTERNATIVE(goal)
    DECOMPOSE = "DECOMPOSE"        # DECOMPOSE(goal)
    DERIVE = "DERIVE"              # DERIVE(claim)
    TEST_ASSUMPTION = "TEST_ASSUMPTION"  # TEST_ASSUMPTION(assumption)


class ExpectedObservation(str, Enum):
    """What type of observation an action produces."""
    VERIFICATION_RESULT = "VERIFICATION_RESULT"  # SUFFICIENT/FALSIFIED/INCONCLUSIVE
    NEW_EVIDENCE = "NEW_EVIDENCE"               # retrieved or searched evidence
    TEST_RESULT = "TEST_RESULT"                 # pass/fail/error
    REASONING_OUTPUT = "REASONING_OUTPUT"       # cognitive operator output
    NONE = "NONE"                               # terminal actions


@dataclass(frozen=True)
class Action:
    """A typed parameterized action.

    Every action has:
      - action_type: what kind of action
      - target: the specific object (evidence ID, hypothesis ID, query, etc.)
      - expected_cost: resource cost estimate
      - expected_observation: what type of observation it produces
      - reversible: whether the action can be undone
    """
    action_type: ActionType
    target: str | tuple[str, str] | None = None
    expected_cost: float = 1.0
    expected_observation: ExpectedObservation = ExpectedObservation.NONE
    reversible: bool = True

    def __str__(self) -> str:
        if self.target is None:
            return self.action_type.value
        if isinstance(self.target, tuple):
            return f"{self.action_type.value}({','.join(self.target)})"
        return f"{self.action_type.value}({self.target})"

    def __hash__(self):
        return hash((self.action_type, self.target))


# Convenience constructors

def answer(hypothesis_id: str) -> Action:
    """ANSWER(h) — answer with hypothesis h."""
    return Action(
        action_type=ActionType.ANSWER,
        target=hypothesis_id,
        expected_cost=0.0,
        expected_observation=ExpectedObservation.NONE,
        reversible=False,  # Terminal
    )


def defer(reason: str = "") -> Action:
    """DEFER(r) — defer with reason r."""
    return Action(
        action_type=ActionType.DEFER,
        target=reason or None,
        expected_cost=0.0,
        expected_observation=ExpectedObservation.NONE,
        reversible=False,  # Terminal
    )


def verify(evidence_id: str) -> Action:
    """VERIFY(e) — verify evidence item e."""
    return Action(
        action_type=ActionType.VERIFY,
        target=evidence_id,
        expected_cost=1.0,
        expected_observation=ExpectedObservation.VERIFICATION_RESULT,
        reversible=False,  # Verification is irreversible
    )


def retrieve(query: str, source_scope: str = "all") -> Action:
    """RETRIEVE(q, scope) — retrieve evidence matching query."""
    return Action(
        action_type=ActionType.RETRIEVE,
        target=(query, source_scope),
        expected_cost=1.0,
        expected_observation=ExpectedObservation.NEW_EVIDENCE,
        reversible=True,
    )


def search(query: str, source_scope: str = "all") -> Action:
    """SEARCH(q, scope) — search for new evidence."""
    return Action(
        action_type=ActionType.SEARCH,
        target=(query, source_scope),
        expected_cost=1.0,
        expected_observation=ExpectedObservation.NEW_EVIDENCE,
        reversible=True,
    )


def test(test_id: str) -> Action:
    """TEST(t) — run a test."""
    return Action(
        action_type=ActionType.TEST,
        target=test_id,
        expected_cost=1.0,
        expected_observation=ExpectedObservation.TEST_RESULT,
        reversible=False,
    )


def compare(h1: str, h2: str) -> Action:
    """COMPARE(h1, h2) — compare two hypotheses."""
    return Action(
        action_type=ActionType.COMPARE,
        target=(h1, h2),
        expected_cost=1.0,
        expected_observation=ExpectedObservation.REASONING_OUTPUT,
        reversible=True,
    )


def check_consistency(target: str) -> Action:
    """CHECK_CONSISTENCY(target) — check consistency of a claim or hypothesis."""
    return Action(
        action_type=ActionType.CHECK_CONSISTENCY,
        target=target,
        expected_cost=1.0,
        expected_observation=ExpectedObservation.REASONING_OUTPUT,
        reversible=True,
    )


def stop(reason: str = "") -> Action:
    """STOP(r) — stop with reason r."""
    return Action(
        action_type=ActionType.STOP,
        target=reason or None,
        expected_cost=0.0,
        expected_observation=ExpectedObservation.NONE,
        reversible=False,
    )
