"""C4 receipt schema — task-level receipts with runtime/evaluator separation.

Runtime payload contains NO oracle keys. Evaluator annotation contains
oracle metadata for scoring only. The separation is physically enforced.
"""
from __future__ import annotations

from typing import Any, Mapping
from dataclasses import asdict

from .contracts import (
    PreHRMResult, TaskReceipt, QueryResult, RetrievalResult,
    IdentityResolution, SelectionResult, PacketResult, HRMResult)


# Oracle keys that must NEVER appear in runtime_payload
ORACLE_KEYS = frozenset({
    "_oracle_metadata", "proof_edges", "required_evidence_ids",
    "oracle_evidence_ids", "answer_node", "family", "entity_regime",
    "latent_subject", "latent_bridge", "answer",
})


def assert_runtime_clean(payload: Mapping[str, Any]) -> None:
    """Assert that a runtime payload contains no oracle keys."""
    def _check(obj, path="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ORACLE_KEYS:
                    raise AssertionError(
                        f"Runtime payload contains oracle key {k!r} at {path}")
                _check(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _check(v, f"{path}[{i}]")

    _check(payload)


def build_pre_hrm_receipt(result: PreHRMResult) -> TaskReceipt:
    """Build a pre-HRM receipt with runtime/evaluator separation."""

    runtime_payload = {
        "task_id": result.task_id,
        "arm_id": result.arm_id,
        "split": result.split,
        "query": {
            "original_question": result.query.original_question,
            "rendered_query": result.query.rendered_query,
            "query_hash": result.query.query_hash,
            "query_policy": result.query.query_policy,
            "query_policy_version": result.query.query_policy_version,
        },
        "retrieval": {
            "candidate_ids": list(result.retrieval.candidate_ids),
            "candidate_budget": result.retrieval.candidate_budget,
            "retrieval_policy": result.retrieval.retrieval_policy,
            "bm25_backend": result.retrieval.bm25_backend,
            "bge_model_id": result.retrieval.bge_model_id,
            "bge_revision": result.retrieval.bge_revision,
            "rrf_k": result.retrieval.rrf_k,
            "bm25_ranked": list(result.retrieval.bm25_ranked),
            "bge_ranked": list(result.retrieval.bge_ranked),
            "fusion_ranked": list(result.retrieval.fusion_ranked),
        },
        "identity": {
            "status": result.identity.status,
            "surface": result.identity.surface,
            "canonical": result.identity.canonical,
            "evidence_ids": list(result.identity.evidence_ids),
            "candidate_mappings": list(result.identity.candidate_mappings),
            "resolution_needed": result.identity.resolution_needed,
            "resolution_attempted": result.identity.resolution_attempted,
            "resolution_changed_state": result.identity.resolution_changed_state,
        },
        "selection": {
            "selector": result.selection.selector,
            "selected_ids": list(result.selection.selected_ids),
            "selector_policy": result.selection.selector_policy,
            "identity_status": result.selection.identity_status,
        },
        "packet": {
            "packet_ids": list(result.packet.packet_ids),
            "packet_token_count": result.packet.packet_token_count,
            "packet_hash": result.packet.packet_hash,
            "packet_budget": result.packet.packet_budget,
        },
        "information_state_before": dict(result.information_state_before),
        "information_state_after": dict(result.information_state_after),
    }

    # Validate no oracle leakage
    assert_runtime_clean(runtime_payload)

    return TaskReceipt(
        task_id=result.task_id,
        arm_id=result.arm_id,
        split=result.split,
        runtime_payload=runtime_payload,
        evaluator_annotation={},  # filled after HRM + verification
    )


def build_full_receipt(pre_hrm: TaskReceipt, hrm: HRMResult,
                       evaluator: Mapping[str, Any]) -> TaskReceipt:
    """Build a full receipt after HRM generation and evaluator scoring."""

    runtime_payload = dict(pre_hrm.runtime_payload)
    runtime_payload["hrm"] = {
        "output": hrm.output,
        "prompt_hash": hrm.prompt_hash,
        "prompt_tokens": hrm.prompt_tokens,
        "completion_tokens": hrm.completion_tokens,
        "model_id": hrm.model_id,
        "model_revision": hrm.model_revision,
        "latency_seconds": hrm.latency_seconds,
    }

    assert_runtime_clean(runtime_payload)

    return TaskReceipt(
        task_id=pre_hrm.task_id,
        arm_id=pre_hrm.arm_id,
        split=pre_hrm.split,
        runtime_payload=runtime_payload,
        evaluator_annotation=evaluator,
    )
