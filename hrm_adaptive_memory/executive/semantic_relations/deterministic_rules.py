"""Deterministic relation extractor for I3.12 S1.

This extractor uses keyword and pattern matching to infer
SUPPORT / CONTRADICT / NEUTRAL relations from proposition text.

It is deliberately simple. The goal is to test whether MDSG/R1
survives when relations are derived rather than given, not to
build the best possible NLI system.

Supported semantic patterns:
  - literal entailment (keyword overlap + positive language)
  - explicit contradiction (negation, "not", "denies", "refutes")
  - temporal mismatch ("stale", "outdated", "old" vs "current", "recent")
  - temporal match ("current", "recent", "updated")
  - no overlap (neutral)
  - keyword contradiction (positive vs negative sentiment words)
"""
from __future__ import annotations

import re
import time
from typing import Any

from hrm_adaptive_memory.executive.semantic_relations.extractor import (
    SemanticRelationExtractor,
    ExtractionInput,
    ExtractionResult,
)
from hrm_adaptive_memory.executive.semantic_relations.schema import (
    SemanticRelation,
    RelationType,
    ExtractorReasonCode,
)
from hrm_adaptive_memory.executive.semantic_relations.identity import (
    compute_extractor_identity,
    ExtractorIdentity,
)


# --- Normalization ---

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", text.lower().strip())


# --- Keyword sets ---

POSITIVE_KEYWORDS = frozenset({
    "confirms", "confirmed", "supports", "validates", "establishes",
    "demonstrates", "shows", "proves", "verifies", "verified",
    "active", "operational", "current", "recent", "updated",
    "sufficient", "available", "present", "exists", "true",
    "yes", "correct", "accurate", "reliable", "established",
    "documented", "reported", "stated", "indicates", "suggests",
})

NEGATIVE_KEYWORDS = frozenset({
    "not", "no", "denies", "refutes", "contradicts", "disputes",
    "false", "incorrect", "wrong", "unverified", "unconfirmed",
    "missing", "absent", "lacking", "insufficient", "stale",
    "outdated", "old", "expired", "removed", "deleted", "discontinued",
    "inactive", "offline", "unavailable", "none", "never",
    "fails", "failed", "broken", "invalid",
})

CONTRADICTION_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"\bnever\b",
    r"\bdenies\b",
    r"\brefutes\b",
    r"\bcontradicts\b",
    r"\bdisputes\b",
    r"\bhowever\b.*\bnot\b",
    r"\bbut\b.*\bnot\b",
]

TEMPORAL_STALE_KEYWORDS = frozenset({
    "stale", "outdated", "old", "expired", "archived",
    "historical", "former", "previous", "past",
})

TEMPORAL_CURRENT_KEYWORDS = frozenset({
    "current", "recent", "updated", "latest", "active",
    "present", "now", "today", "ongoing",
})


def _has_negation(text: str) -> bool:
    lower = text.lower()
    for pattern in CONTRADICTION_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _keyword_overlap(ev_text: str, hyp_text: str) -> set[str]:
    """Return overlapping content words (excluding stop words)."""
    ev_words = set(ev_text.split())
    hyp_words = set(hyp_text.split())
    return ev_words & hyp_words


def _has_positive_keywords(text: str) -> set[str]:
    words = set(text.split())
    return words & POSITIVE_KEYWORDS


def _has_negative_keywords(text: str) -> set[str]:
    words = set(text.split())
    return words & NEGATIVE_KEYWORDS


def _has_temporal_stale(text: str) -> set[str]:
    words = set(text.split())
    return words & TEMPORAL_STALE_KEYWORDS


def _has_temporal_current(text: str) -> set[str]:
    words = set(text.split())
    return words & TEMPORAL_CURRENT_KEYWORDS


# --- Extractor ---

class DeterministicRelationExtractor(SemanticRelationExtractor):
    """Deterministic keyword/pattern-based relation extractor.

    Rules (in priority order):
      1. If evidence has negation + keyword overlap with hypothesis -> CONTRADICT
      2. If evidence has temporal-stale and hypothesis wants current -> CONTRADICT
      3. If evidence has temporal-current and hypothesis wants current -> SUPPORT
      4. If evidence has positive keywords + keyword overlap -> SUPPORT
      5. If evidence has negative keywords + keyword overlap -> CONTRADICT
      6. If no keyword overlap -> NEUTRAL
      7. Default -> NEUTRAL
    """

    def __init__(self) -> None:
        identity = compute_extractor_identity(
            extractor_class="DeterministicRelationExtractor",
            extractor_version="1.0.0",
            relation_schema_version="1",
            normalization_rules=(
                "lowercase",
                "collapse_whitespace",
                "strip",
            ),
            thresholds={
                "keyword_overlap_min": 1,
                "negation_weight": 1.0,
                "temporal_weight": 1.0,
            },
            prompt_template=None,
            model_name=None,
            model_version=None,
            hypothesis_serializer="default_text",
            evidence_serializer="default_text",
        )
        super().__init__(identity)

    def _extract_one(self, inp: ExtractionInput) -> ExtractionResult:
        t0 = time.time()
        ev_norm = normalize_text(inp.evidence_proposition)
        hyp_norm = normalize_text(inp.hypothesis_proposition)

        overlap = _keyword_overlap(ev_norm, hyp_norm)
        has_neg = _has_negation(inp.evidence_proposition)
        pos_kw = _has_positive_keywords(ev_norm)
        neg_kw = _has_negative_keywords(ev_norm)
        temp_stale = _has_temporal_stale(ev_norm)
        temp_current = _has_temporal_current(ev_norm)

        # Hypothesis temporal orientation
        hyp_wants_current = bool(_has_temporal_current(hyp_norm))
        hyp_wants_stale = bool(_has_temporal_stale(hyp_norm))

        relation: RelationType
        reason: ExtractorReasonCode

        # Rule 1: Negation + overlap -> CONTRADICT
        if has_neg and overlap:
            relation = RelationType.CONTRADICT
            reason = ExtractorReasonCode.NEGATION
        # Rule 2: Temporal stale vs hypothesis wanting current -> CONTRADICT
        elif temp_stale and hyp_wants_current and not temp_current:
            relation = RelationType.CONTRADICT
            reason = ExtractorReasonCode.TEMPORAL_MISMATCH
        # Rule 3: Temporal current + hypothesis wants current -> SUPPORT
        elif temp_current and hyp_wants_current:
            relation = RelationType.SUPPORT
            reason = ExtractorReasonCode.TEMPORAL_MATCH
        # Rule 4: Positive keywords + overlap -> SUPPORT
        elif pos_kw and overlap:
            relation = RelationType.SUPPORT
            reason = ExtractorReasonCode.KEYWORD_MATCH
        # Rule 5: Negative keywords + overlap -> CONTRADICT
        elif neg_kw and overlap:
            relation = RelationType.CONTRADICT
            reason = ExtractorReasonCode.KEYWORD_CONTRADICTION
        # Rule 6: No overlap -> NEUTRAL
        elif not overlap:
            relation = RelationType.NEUTRAL
            reason = ExtractorReasonCode.NO_OVERLAP
        # Rule 7: Default
        else:
            relation = RelationType.NEUTRAL
            reason = ExtractorReasonCode.DEFAULT_NEUTRAL

        latency_ms = int((time.time() - t0) * 1000)

        rel = SemanticRelation(
            evidence_id=inp.evidence_id,
            hypothesis_id=inp.hypothesis_id,
            relation=relation,
            confidence=None,  # deterministic
            reason_code=reason,
            evidence_sha256=inp.evidence_sha256,
            hypothesis_sha256=inp.hypothesis_sha256,
        )

        return ExtractionResult(
            relation=rel,
            input=inp,
            latency_ms=latency_ms,
        )
