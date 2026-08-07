"""C4 selection stage — S0, S2c, or oracle, with deterministic fallback.

The frozen fallback rule for C4-4:
    if identity_status in {EXACT, RESOLVED}:
        selector = S2c
    else:
        selector = S0

No adaptive router. No entity-regime labels at runtime.

C4-5 (oracle selector) is constrained to G_i ∩ C_i (gold ∩ candidate pool),
NOT the full gold set. Only C4-6 may inject required evidence directly.
"""
from __future__ import annotations

from typing import Sequence, Mapping, Any

from ..retrieval_bench.selectors import s0_raw, s2_connectivity, s5_oracle
from ..retrieval_bench.selectors.chain import s2c_chain_plus_relation
from .contracts import C4Arm, IdentityResolution, RetrievalResult, SelectionResult


def run_selection_stage(arm: C4Arm,
                        question: str,
                        retrieval: RetrievalResult,
                        identity: IdentityResolution,
                        texts: Mapping[str, str],
                        required_evidence_ids: Sequence[str],
                        canonical_question: str | None = None) -> SelectionResult:
    """Select evidence from the candidate pool.

    For "s0": pool order baseline.
    For "s2c_with_s0_fallback": S2c if identity resolved, else S0.
    For "s2c_with_srel_fallback": S2c if identity resolved, else s_rel_only (diagnostic).
    For "oracle": oracle selector constrained to gold ∩ candidate pool.
    For "oracle_evidence": directly inject required evidence (C4-6 only).
    """
    budget = arm.packet_budget
    candidates = [{"document_id": eid} for eid in retrieval.candidate_ids]

    if arm.selector_policy == "oracle_evidence":
        # C4-6: directly use required evidence, truncated to budget
        selected = list(required_evidence_ids[:budget])
        return SelectionResult(
            selector="oracle_evidence",
            selected_ids=tuple(selected),
            selector_policy=arm.selector_policy,
            identity_status=identity.status,
        )

    if arm.selector_policy == "oracle":
        # C4-5: oracle selector constrained to gold ∩ candidate pool
        candidate_set = set(retrieval.candidate_ids)
        gold_in_pool = [eid for eid in required_evidence_ids if eid in candidate_set]
        selected = s5_oracle(candidates, budget=budget, required=gold_in_pool)
        return SelectionResult(
            selector="oracle",
            selected_ids=tuple(selected),
            selector_policy=arm.selector_policy,
            identity_status=identity.status,
        )

    # Determine which selector to use based on identity status
    use_s2c = False
    use_srel = False

    if arm.selector_policy == "s2c_with_s0_fallback":
        use_s2c = identity.status in ("EXACT", "RESOLVED")
    elif arm.selector_policy == "s2c_with_srel_fallback":
        use_s2c = identity.status in ("EXACT", "RESOLVED")
        use_srel = not use_s2c
    elif arm.selector_policy == "s0":
        use_s2c = False
    else:
        raise ValueError(f"Unknown selector_policy: {arm.selector_policy}")

    # Use the canonical question (with resolved entity) if available
    query_for_selection = canonical_question or question

    if use_s2c and identity.canonical:
        # Reformulate question with canonical entity for S2c
        resolved_question = question
        if identity.surface and identity.canonical:
            resolved_question = question.replace(identity.surface, identity.canonical)
        selected = s2c_chain_plus_relation(
            candidates, budget=budget, question=resolved_question, texts=texts)
        return SelectionResult(
            selector="s2c",
            selected_ids=tuple(selected),
            selector_policy=arm.selector_policy,
            identity_status=identity.status,
        )
    elif use_srel:
        selected = s2_connectivity(
            candidates, budget=budget, question=question, texts=texts)
        return SelectionResult(
            selector="srel",
            selected_ids=tuple(selected),
            selector_policy=arm.selector_policy,
            identity_status=identity.status,
        )
    else:
        selected = s0_raw(candidates, budget=budget)
        return SelectionResult(
            selector="s0",
            selected_ids=tuple(selected),
            selector_policy=arm.selector_policy,
            identity_status=identity.status,
        )
