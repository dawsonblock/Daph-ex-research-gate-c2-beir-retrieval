"""Candidate fusion policies for the c4_retrieval_fusion_v1 experiment.

ADDITIVE AND OPT-IN. The frozen C4 v2.1 path in retrieval_stage.py is not
modified and does not import this module -- `frozen_rrf` here delegates to
retrieval_stage._rrf precisely so the experiment's baseline cannot drift away
from the certified behavior it is supposed to reproduce.

Why these policies exist (see configs/gate_c4_retrieval_fusion_v1.json):

RRF scores a record as the SUM over constituent lists of 1/(k + rank), and
contributes nothing where a record is absent from a list. A record both
retrievers rank therefore earns roughly double what a record only one
retriever ranks can earn. Algebraically a single-list record at rank s is
outranked by any both-list record at rank b whenever b < 2s + k, so with the
frozen k=10 a record BM25 ranks 37th -- inside the candidate budget of 50 --
is displaced by anything both retrievers rank better than 84th.

That penalty lands on exactly the signal that is most reliable on this corpus:
Gate B established BM25 dominates the tested dense representation here. Sprint
A measured the cost as 24% of all missing required evidence on both splits,
worth +5.8pp CES on development and +10.8pp on qualification at UNCHANGED
budget.

Every policy is deterministic: scored policies order by (-score, evidence_id)
so equal scores resolve identically across processes and platforms, and the
interleave policy is positionally deterministic. No policy samples, and none
depends on dict or set iteration order.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .retrieval_stage import _rrf

# Signature shared by every policy: (ranked lists, k_rrf, limit) -> [(id, score)]
FusionPolicy = Callable[[Sequence[Sequence[str]], int, int], list[tuple[str, float]]]


def frozen_rrf(rankings: Sequence[Sequence[str]], k: int,
               limit: int) -> list[tuple[str, float]]:
    """R0 baseline: the frozen C4 v2.1 fusion, by delegation.

    Deliberately not reimplemented. A local copy of the baseline would be free
    to drift from the certified path, and then every measured delta would be
    against something that is not actually what v2.1 does.
    """
    return _rrf([list(r) for r in rankings], k, limit)


def max_reciprocal(rankings: Sequence[Sequence[str]], k: int,
                   limit: int) -> list[tuple[str, float]]:
    """R1: like RRF but aggregate by MAX instead of SUM.

    A record's score becomes its single best constituent rank, which removes
    the consensus bonus exactly rather than compensating for it: a record no
    constituent ranks highly can no longer outrank one that a constituent does.
    Keeps k and the rank-reciprocal shape, so this is a change of aggregation,
    not a new scoring family. Parameter-free beyond the frozen k.
    """
    best: dict[str, float] = {}
    for ranking in rankings:
        for position, evidence_id in enumerate(ranking, 1):
            score = 1.0 / (k + position)
            if score > best.get(evidence_id, 0.0):
                best[evidence_id] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit]


def reserved_slot_interleave(rankings: Sequence[Sequence[str]], k: int,
                             limit: int) -> list[tuple[str, float]]:
    """R2: deterministic round-robin across constituent lists.

    bm25[0], bge[0], bm25[1], bge[1], ... deduplicating on first occurrence
    until the budget fills.

    NOT INDEPENDENT OF :func:`max_reciprocal`. Round-robin emits a record at the
    position equal to its min-rank across lists, and max aggregation scores a
    record as 1/(k + min-rank), so both policies produce the SAME min-rank
    ordering and differ only in tie-breaking (by evidence_id here vs by list
    position there). Verified over 300 randomized two-list trials at budget 50:
    identical membership in 264, differing in 36, entirely from tie-breaks at
    the budget boundary. Retained as a tie-break sensitivity check only --
    agreement between R1 and R2 is circular and must never be reported as
    independent corroboration.

    Guarantee depth: at most ``len(rankings) * r`` records can have min-rank
    <= r, so survival is guaranteed only while that is within the budget, i.e.
    to depth ``limit // len(rankings)`` -- only 25 for the frozen budget of 50
    over two lists. Sprint A measured the median displaced record at rank 37-41,
    beyond that depth, which is why this family of rules recovers only part of
    the displacement loss.

    ``k`` is unused for ordering and is accepted only to keep the policy
    signature uniform; the returned scores are descending rank-order surrogates
    so callers can treat every policy's output identically.
    """
    del k  # not used: ordering here is positional, not scored
    if not rankings:
        return []

    selected: list[str] = []
    seen: set[str] = set()
    depth = max((len(r) for r in rankings), default=0)
    for position in range(depth):
        for ranking in rankings:
            if position >= len(ranking):
                continue
            evidence_id = ranking[position]
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            selected.append(evidence_id)
            if len(selected) >= limit:
                # Surrogate scores: strictly descending, so downstream
                # consumers see the same shape as a scored policy.
                return [(eid, 1.0 / (index + 1))
                        for index, eid in enumerate(selected)]
    return [(eid, 1.0 / (index + 1)) for index, eid in enumerate(selected)]


def oracle_fusion(rankings: Sequence[Sequence[str]], k: int, limit: int, *,
                  required: Iterable[str]) -> list[tuple[str, float]]:
    """R3 ceiling: bound what ANY reordering of these lists can reach at k.

    Places every required record the constituent lists contain anywhere, then
    fills the remaining budget in frozen RRF order. Uses oracle labels, so it
    is a headroom measurement and is NEVER promotable -- the same status
    C4_5/C4_6 have in the main ladder.

    If this sits far below 100%, the residual loss is beyond the constituents'
    own depth and no fusion rule can recover it; that is the budget question
    (Sprint B3), not a fusion question.
    """
    available: set[str] = {eid for ranking in rankings for eid in ranking}
    protected = [eid for eid in dict.fromkeys(required) if eid in available]

    out: list[tuple[str, float]] = [(eid, 1.0) for eid in protected[:limit]]
    if len(out) >= limit:
        return out

    chosen = {eid for eid, _ in out}
    for eid, score in frozen_rrf(rankings, k, len(available)):
        if eid in chosen:
            continue
        out.append((eid, score))
        chosen.add(eid)
        if len(out) >= limit:
            break
    return out


POLICIES: dict[str, FusionPolicy] = {
    "R0_frozen_rrf": frozen_rrf,
    "R1_max_reciprocal": max_reciprocal,
    "R2_reserved_slot_interleave": reserved_slot_interleave,
}

ORACLE_POLICY = "R3_oracle_fusion"
