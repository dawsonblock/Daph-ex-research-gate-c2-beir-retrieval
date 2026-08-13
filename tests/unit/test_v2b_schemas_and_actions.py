"""The initial V2B action space and schema registry are explicit and bounded."""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.cognitive_control import (
    DecisionAction, V2B_ACTIONS, schema_identity, validate_v2b_action)


def test_v2b_action_space_is_exactly_the_initial_non_spawning_contract():
    assert tuple(action.value for action in V2B_ACTIONS) == (
        "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP")
    assert validate_v2b_action(DecisionAction.VERIFY) is DecisionAction.VERIFY
    with pytest.raises(ValueError, match="frozen V2B action space"):
        validate_v2b_action(DecisionAction.VERIFY_ALTERNATE_SOURCE)


def test_v2b_schema_identity_covers_decision_outcome_temporal_conflict_and_provenance():
    identity = schema_identity()
    assert identity["sha256"]
    assert set(identity["definitions"]) == {
        "provenance", "temporal_fact", "conflict", "decision", "outcome", "policy",
        "cognitive_state_snapshot", "resource_state", "executive_action"}
