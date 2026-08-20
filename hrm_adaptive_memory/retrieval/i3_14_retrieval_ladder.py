"""I3.14a: Retrieval ladder for retrieval sufficiency repair.

Implements Q0-Q4 retrievers on the frozen I3.13 corpus:
  Q0_BM25:     Frozen BM25 baseline
  Q1_DENSE:    BGE-small-en-v1.5 dense retrieval
  Q2_HYBRID:   BM25 + dense with reciprocal rank fusion
  Q3_RERANKED: Hybrid + cross-encoder reranking
  Q4_ORACLE:   Oracle retrieval ceiling

All downstream components remain frozen.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from hrm_adaptive_memory.memory.chunking import Chunk
from hrm_adaptive_memory.retrieval.lexical import BM25Retriever, tokenize


class Retriever(Protocol):
    retriever_id: str

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        ...


# ---------------------------------------------------------------------------
# Q0: BM25 (frozen baseline)
# ---------------------------------------------------------------------------

class Q0BM25Retriever:
    """Frozen BM25 retriever — identical to I3.13 R1_REAL."""

    retriever_id = "Q0_BM25"

    def __init__(self, chunks: Sequence[Chunk]):
        self.bm25 = BM25Retriever(chunks, k1=1.5, b=0.75)
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        return self.bm25.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Q1: Dense retrieval (BGE-small-en-v1.5)
# ---------------------------------------------------------------------------

class Q1DenseRetriever:
    """Dense retrieval with pinned BGE-small-en-v1.5 embeddings."""

    retriever_id = "Q1_DENSE_BGE_SMALL"

    def __init__(self, chunks: Sequence[Chunk]):
        from hrm_adaptive_memory.retrieval.embedding import (
            EmbeddingSpec, PinnedTransformerEmbedder, BGE_SMALL,
        )
        self.spec = EmbeddingSpec(**BGE_SMALL)
        self.embedder = PinnedTransformerEmbedder(self.spec)
        self.chunks = list(chunks)
        # Pre-compute document embeddings
        texts = [c.content for c in self.chunks]
        self.doc_embeddings = np.array(
            self.embedder.embed_documents(texts), dtype=np.float32
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        # Cosine similarity (embeddings are already normalized)
        scores = self.doc_embeddings @ q_vec
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Q2: Hybrid BM25 + Dense with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class Q2HybridRetriever:
    """Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion."""

    retriever_id = "Q2_HYBRID_RRF"

    def __init__(self, chunks: Sequence[Chunk], rrf_k: int = 60):
        self.bm25 = Q0BM25Retriever(chunks)
        self.dense = Q1DenseRetriever(chunks)
        self.chunks = list(chunks)
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        # Get larger candidate pools from each retriever
        pool_k = max(top_k * 4, 20)
        bm25_results = self.bm25.search(query, top_k=pool_k)
        dense_results = self.dense.search(query, top_k=pool_k)

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(bm25_results):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[cid] = chunk

        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(chunk_map[cid], score) for cid, score in ranked]


# ---------------------------------------------------------------------------
# Q3: Hybrid + Cross-encoder Reranking
# ---------------------------------------------------------------------------

class Q3RerankedRetriever:
    """Hybrid candidate generation + cross-encoder reranking."""

    retriever_id = "Q3_RERANKED_GTE"

    def __init__(self, chunks: Sequence[Chunk], candidate_k: int = 20):
        self.hybrid = Q2HybridRetriever(chunks)
        self.chunks = list(chunks)
        self.candidate_k = candidate_k
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            model_id = "Alibaba-NLP/gte-reranker-modernbert-base"
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_id, trust_remote_code=True
            ).eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)
            self._reranker = True
        return self._reranker

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        import torch

        # Get hybrid candidates
        candidates = self.hybrid.search(query, top_k=self.candidate_k)
        if not candidates:
            return []

        self._get_reranker()

        # Rerank with cross-encoder
        pairs = [(query, chunk.content) for chunk, _ in candidates]
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            )
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            scores = self._model(**inputs).logits.squeeze(-1).float()

        scored = list(zip(candidates, scores.tolist()))
        scored.sort(key=lambda x: -x[1])
        return [(chunk, score) for (chunk, _), score in scored[:top_k]]


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def build_retriever(
    condition: str, chunks: Sequence[Chunk]
) -> Retriever | None:
    """Build a retriever for the given condition.

    Returns None for Q4_ORACLE (handled separately).
    """
    if condition == "Q0_BM25":
        return Q0BM25Retriever(chunks)
    elif condition == "Q1_DENSE":
        return Q1DenseRetriever(chunks)
    elif condition == "Q2_HYBRID":
        return Q2HybridRetriever(chunks)
    elif condition == "Q3_RERANKED":
        return Q3RerankedRetriever(chunks)
    elif condition == "Q4_ORACLE":
        return None  # Oracle handled by caller
    else:
        raise ValueError(f"Unknown retrieval condition: {condition}")


RETRIEVAL_CONDITIONS = ["Q0_BM25", "Q1_DENSE", "Q2_HYBRID", "Q3_RERANKED", "Q4_ORACLE"]


def retriever_digest(condition: str) -> str:
    """Return a short digest identifying the retriever configuration."""
    digests = {
        "Q0_BM25": "bm25_k1.5_b0.75",
        "Q1_DENSE": "bge-small-en-v1.5_5c38ec7c",
        "Q2_HYBRID": "bm25+bge_rrf_k60",
        "Q3_RERANKED": "bm25+bge_rrf+gte-reranker-modernbert",
        "Q4_ORACLE": "oracle",
    }
    return digests.get(condition, "unknown")
