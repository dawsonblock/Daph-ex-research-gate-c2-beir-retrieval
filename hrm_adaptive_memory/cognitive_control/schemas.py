"""Versioned cognitive-control schema descriptors used by V2B identity binding."""
from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA_REGISTRY_VERSION = "DAPH_COGNITIVE_SCHEMA_REGISTRY_V3"

SCHEMAS: dict[str, dict[str, Any]] = {
    "provenance": {
        "schema": "DAPH_PROVENANCE_RECORD_V1",
        "required": ["provenance_id", "entity_id", "entity_type", "payload_hash", "activity_id",
                     "agent_id", "agent_type", "source_id", "derived_from", "created_at", "operation_id"],
    },
    "temporal_fact": {
        "schema": "DAPH_TEMPORAL_FACT_V2",
        "required": ["fact_id", "entity", "relation", "value", "source_evidence_id",
                     "source_lineage_id", "provenance_id", "valid_from", "recorded_at"],
        "invariants": ["timezone_aware_utc", "valid_until_gt_valid_from",
                       "superseded_at_gte_recorded_at"],
    },
    "conflict": {
        "schema": "DAPH_CONFLICT_EVENT_V1",
        "required": ["conflict_id", "entity", "relation", "fact_ids", "distinct_values",
                     "source_lineage_ids", "detected_at", "status"],
        "status_values": ["UNRESOLVED"],
    },
    "decision": {
        "schema": "DAPH_DECISION_RECORD_V2",
        "required": ["decision_id", "task_id", "selected_action", "policy_id", "reason_code",
                     "resource_state", "timestamp", "parent_decision_id"],
        "invariants": ["timezone_aware_utc", "parent_timestamp_lte_child_timestamp"],
    },
    "outcome": {
        "schema": "DAPH_DECISION_OUTCOME_V2",
        "required": ["decision_id", "outcome", "outcome_at"],
        "invariants": ["timezone_aware_utc", "outcome_at_gte_decision_timestamp"],
    },
    "policy": {
        "schema": "DAPH_COGNITIVE_POLICY_V2",
        "rule_heads": ["deny(task,action,reason)", "require(task,action,reason)"],
        "invariants": ["unique_rule_ids", "range_restricted_variables", "known_actions",
                       "conflicting_requirements_deny"],
    },
    "cognitive_state_snapshot": {
        "schema": "DAPH_COGNITIVE_STATE_SNAPSHOT_V1",
        "bounded_categories": 16,
        "required": ["task_id", "relevant_memories", "verification_states", "temporal_status",
                     "unresolved_conflicts", "prior_decisions", "prior_outcomes", "resource_state",
                     "policy_facts", "observation_signals"],
    },
    "resource_state": {
        "schema": "DAPH_V2B_RESOURCE_STATE_V1",
        "required": ["executive_steps", "reasoning_tokens", "retrieval_calls", "verification_calls",
                     "search_calls", "elapsed_ms", "monetary_cost_microusd"],
        "invariants": ["hard_budget_limits", "action_costs_accounted"],
    },
    "executive_action": {
        "schema": "DAPH_V2B_EXECUTIVE_ACTIONS_V1",
        "allowed": ["ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"],
        "forbidden": ["SPAWN_SPECIALIST", "SWITCH_STRATEGY", "ABANDON_STRATEGY", "VERIFY_ALTERNATE_SOURCE"],
    },
    "metareasoning_action_trace": {
        "schema": "DAPH_V2B_I3_ACTION_TRACE_V1",
        "required": ["proposed_action", "policy_resolved_action", "execution_status",
                     "executed_action", "pre_state_hash", "post_state_hash", "state_delta"],
        "invariants": ["deny_records_rejection_without_execution", "state_delta_per_executed_action",
                       "latent_terminal_labels_not_controller_input"],
    },
}


def schema_identity() -> dict[str, Any]:
    payload = {"schema": SCHEMA_REGISTRY_VERSION, "definitions": SCHEMAS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}
