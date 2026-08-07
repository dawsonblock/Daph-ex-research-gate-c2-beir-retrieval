"""C4 retrieval stage — BM25 and BM25+BGE fusion with frozen configuration.

Reuses the existing CanonicalRetrievalBackend. Does not introduce a new dense
model or retune RRF. The BGE revision and fusion config are frozen from C2.

Backends are cached per evidence corpus to avoid reloading the BGE model on
every task. The cache key is the corpus digest, so a different evidence set
produces a different backend.
"""
from __future__ import annotations

import asyncio
from typing import Sequence

from ..backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from ..contracts import IndexRecord
from ..retrieval.embedding import BGE_SMALL, EmbeddingSpec
from .contracts import C4Arm, RetrievalResult, C4_CANDIDATE_BUDGET, C4_RRF_K


def _rrf(rankings: list[list[str]], k: int, lim: int) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, eid in enumerate(ranking, 1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + i)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:lim]


# --- Backend cache ----------------------------------------------------------
# Keyed by (mode, frozenset of evidence_ids) so the BGE model loads once per
# evidence corpus, not once per task. The cache is module-level and lives for
# the duration of a run.

_backend_cache: dict[tuple, CanonicalRetrievalBackend] = {}


def get_cached_backend(mode: CanonicalRetrievalMode,
                       records: Sequence[IndexRecord]) -> CanonicalRetrievalBackend:
    """Get or create a cached retrieval backend for the given evidence set."""
    key = (mode, frozenset(r.evidence_id for r in records))
    if key not in _backend_cache:
        _backend_cache[key] = CanonicalRetrievalBackend(mode, records)
    return _backend_cache[key]


def clear_backend_cache() -> None:
    """Clear the backend cache (use between splits or runs)."""
    _backend_cache.clear()


def run_retrieval_stage(query: str, arm: C4Arm,
                        records: Sequence[IndexRecord]) -> RetrievalResult:
    """Run retrieval and return ranked candidates.

    For "bm25_only" (C4-0, C4-1): BM25 only.
    For "bm25_bge_fusion" (C4-2+): BM25 + BGE with RRF fusion.
    """
    budget = C4_CANDIDATE_BUDGET

    # BM25 is always run (cached)
    bm25_backend = get_cached_backend(CanonicalRetrievalMode.BM25, records)
    bm25_result = asyncio.run(bm25_backend.search(query, k=budget))
    bm25_ranked = tuple((e.evidence_id, e.lexical_score or 0.0) for e in bm25_result.evidence)

    if arm.retrieval_policy == "bm25_only":
        candidate_ids = tuple(eid for eid, _ in bm25_ranked)
        bge_ranked: tuple[tuple[str, float], ...] = ()
        fusion_ranked = bm25_ranked
    elif arm.retrieval_policy == "bm25_bge_fusion":
        bge_backend = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
        bge_result = asyncio.run(bge_backend.search(query, k=budget))
        bge_ranked = tuple((e.evidence_id, e.dense_score or 0.0) for e in bge_result.evidence)
        fused = _rrf(
            [[eid for eid, _ in bm25_ranked], [eid for eid, _ in bge_ranked]],
            C4_RRF_K, budget)
        fusion_ranked = tuple(fused)
        candidate_ids = tuple(eid for eid, _ in fusion_ranked)
    else:
        raise ValueError(f"Unknown retrieval_policy: {arm.retrieval_policy}")

    return RetrievalResult(
        bm25_ranked=bm25_ranked,
        bge_ranked=bge_ranked,
        fusion_ranked=fusion_ranked,
        candidate_ids=candidate_ids,
        candidate_budget=budget,
        retrieval_policy=arm.retrieval_policy,
        bm25_backend="okapibm25",
        bge_model_id=BGE_SMALL.get("model_id", ""),
        bge_revision=BGE_SMALL.get("revision", ""),
        rrf_k=C4_RRF_K,
    )
