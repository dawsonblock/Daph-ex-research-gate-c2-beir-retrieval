"""C4 packet stage — compose the bounded evidence packet.

Uses the existing compose_evidence_prompt. The packet budget is frozen at
C4_PRIMARY_PACKET_BUDGET = 6 across all arms. No budget sweeps during C4.

For arms with deterministic packet ordering (C4_4, C4_3o), the frozen
c4_packet_ordering_v1 policy is applied. For other arms (C4_0-C4_3, C4_4m),
pool order from the selector is used as-is.

The prompt returned here is the ONLY prompt HRM may consume. Callers must not
recompose a prompt from ``SelectionResult.selected_ids``: that bypasses the
ordering policy and breaks the pre-HRM freeze guarantee (protocol v2
``pre_hrm_freeze.rule``).  ``PacketResult.prompt_hash`` binds the packet to
that exact prompt text so the bypass is detectable in receipts.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

from ..evidence.packing import compose_evidence_prompt
from .contracts import C4Arm, RetrievalResult, SelectionResult, PacketResult
from .packet_ordering import (
    ORDERING_POLICY_ID, canonical_candidate_pool_hash, canonical_membership_hash,
    canonical_order_hash, canonical_prompt_hash, order_packet)

# Arms that use the deterministic packet ordering policy
_DETERMINISTIC_ORDER_ARMS = {"C4_4", "C4_3o"}

# Ordering policy id recorded for arms that deliberately keep pool order
_POOL_ORDER_POLICY_ID = "pool_order"


def run_packet_stage(arm: C4Arm, question: str,
                     selection: SelectionResult,
                     texts: Mapping[str, str],
                     retrieval: RetrievalResult | None = None,
                     ) -> tuple[str, PacketResult]:
    """Compose the evidence packet from selected IDs.

    Returns (prompt, PacketResult).

    ``retrieval`` supplies the candidate pool (for candidate_pool_hash) and the
    fusion scores used by rule 2 of the canonical sort key. It is optional only
    so that unit tests can exercise ordering in isolation; the production runner
    always passes it, and ``score_sources`` records what was actually available
    so an inert ordering term can never be mistaken for an active one.
    """
    retrieval_scores: dict[str, float] = {}
    candidate_ids: tuple[str, ...] = ()
    score_sources = {"retrieval_scores": False}

    if retrieval is not None:
        candidate_ids = tuple(retrieval.candidate_ids)
        retrieval_scores = {eid: score for eid, score in retrieval.fusion_ranked}
        score_sources["retrieval_scores"] = bool(retrieval_scores)

    # Apply deterministic ordering for arms that use it
    if arm.arm_id in _DETERMINISTIC_ORDER_ARMS:
        ordered_ids = order_packet(
            selection.selected_ids, retrieval_scores=retrieval_scores)
        ordering_policy_id = ORDERING_POLICY_ID
        ordering_applied = True
    else:
        ordered_ids = list(selection.selected_ids)
        ordering_policy_id = _POOL_ORDER_POLICY_ID
        ordering_applied = False

    contents = [texts.get(eid, "") for eid in ordered_ids if eid in texts]
    prompt = compose_evidence_prompt(question, contents)

    # Token count: simple whitespace split for now (HRM tokenizer may differ)
    token_count = sum(len(c.split()) for c in contents)
    packet_hash = hashlib.sha256(
        "|".join(ordered_ids).encode()).hexdigest()

    return prompt, PacketResult(
        packet_ids=tuple(ordered_ids),
        packet_contents=tuple(contents),
        packet_token_count=token_count,
        packet_hash=packet_hash,
        packet_budget=arm.packet_budget,
        candidate_pool_hash=canonical_candidate_pool_hash(candidate_ids),
        membership_hash=canonical_membership_hash(selection.selected_ids),
        order_hash=canonical_order_hash(ordered_ids),
        prompt_hash=canonical_prompt_hash(prompt),
        ordering_policy_id=ordering_policy_id,
        ordering_applied=ordering_applied,
        score_sources=score_sources,
    )
