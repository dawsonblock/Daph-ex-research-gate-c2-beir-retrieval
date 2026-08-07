"""C4 packet ordering — separates membership selection from packet ordering.

Phase 3-4 of the C4 determinism repair.

The S2c selector determines WHICH records survive (membership).
This module determines IN WHAT ORDER they are presented to HRM.

These are distinct decisions and are represented separately in receipts:
    selected_set_ids:      lexically sorted, for identity checks
    ordered_selected_ids:  actual HRM order, for prompt construction

The frozen packet-ordering policy (C4 protocol v2):

    identity → direct/link → intermediate/bridge → current/value/fact
    → support → superseded → accepted → rejected → dead-end → distractor

Within each role tier:
    selector score descending → retrieval score descending → record_id ascending

This is NOT claimed to be optimal. It is explicit, versioned, testable,
and deterministic. Later, ordering can become a separate ablation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from ..retrieval_bench.selectors.chain import ROLE_PRIORITY, _quantize, _role_priority


# Packet ordering policy version — frozen under C4 protocol v2
ORDERING_POLICY_ID = "c4_packet_ordering_v1"
ORDERING_POLICY_VERSION = "1.0.0"


def order_packet(
    selected_ids: Sequence[str],
    *,
    selector_scores: dict[str, float] | None = None,
    retrieval_scores: dict[str, float] | None = None,
) -> list[str]:
    """Apply the frozen packet-ordering policy to a set of selected IDs.

    Args:
        selected_ids: The membership-selected record IDs (any order).
        selector_scores: Optional mapping from record_id to S2c score.
        retrieval_scores: Optional mapping from record_id to retrieval score.

    Returns:
        The same IDs in the deterministic packet order.

    The sort key is a total order:
        1. role priority ascending (identity < direct < ... < distractor)
        2. selector score descending (quantized)
        3. retrieval score descending (quantized)
        4. record_id lexical ascending

    No two different records compare equal.
    """
    selector_scores = selector_scores or {}
    retrieval_scores = retrieval_scores or {}

    def sort_key(rid: str) -> tuple:
        return (
            _role_priority(rid),
            -_quantize(selector_scores.get(rid, 0.0)),
            -_quantize(retrieval_scores.get(rid, 0.0)),
            rid,
        )

    return sorted(selected_ids, key=sort_key)


def canonical_membership_hash(selected_ids: Sequence[str]) -> str:
    """Hash of the membership set (lexically sorted, order-independent).

    Two packets with the same members but different order produce the
    same membership hash.
    """
    canonical = json.dumps(sorted(selected_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def canonical_order_hash(ordered_ids: Sequence[str]) -> str:
    """Hash of the ordered packet (order-sensitive).

    Two packets with the same members but different order produce
    different order hashes.
    """
    canonical = json.dumps(list(ordered_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def packet_receipt(
    task_id: str,
    selected_ids: Sequence[str],
    ordered_ids: Sequence[str],
    *,
    query_hash: str = "",
    canonical_subject: str = "",
    candidate_pool_hash: str = "",
    selector_policy_id: str = "",
    selector_scores: dict[str, float] | None = None,
    retrieval_scores: dict[str, float] | None = None,
) -> dict:
    """Build a full packet receipt with all hashes.

    This gives clear boundaries for debugging:
        retrieval changed?    → candidate_pool_hash differs
        membership changed?   → membership_hash differs
        ordering changed?     → order_hash differs
        prompt changed?       → prompt_hash differs
    """
    membership_hash = canonical_membership_hash(selected_ids)
    order_hash = canonical_order_hash(ordered_ids)

    return {
        "task_id": task_id,
        "query_hash": query_hash,
        "canonical_subject": canonical_subject,
        "candidate_pool_hash": candidate_pool_hash,
        "selector_policy_id": selector_policy_id,
        "ordering_policy_id": ORDERING_POLICY_ID,
        "ordering_policy_version": ORDERING_POLICY_VERSION,
        "selected_set_ids": sorted(selected_ids),
        "ordered_selected_ids": list(ordered_ids),
        "membership_hash": membership_hash,
        "order_hash": order_hash,
        "selector_scores": {k: _quantize(v) for k, v in (selector_scores or {}).items()},
        "retrieval_scores": {k: _quantize(v) for k, v in (retrieval_scores or {}).items()},
    }
