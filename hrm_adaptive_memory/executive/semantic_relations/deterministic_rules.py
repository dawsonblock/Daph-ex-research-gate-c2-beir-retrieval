"""Deterministic relation extractor for I3.12 S1.

This extractor uses keyword and pattern matching to infer
SUPPORT / CONTRADICT / NEUTRAL relations from proposition text.

It is deliberately simple. The goal is to test whether MDSG/R1
survives when relations are derived rather than given, not to
build the best possible NLI system.

Supported semantic patterns:
  - explicit support/contradiction verbs ("confirms", "refutes")
  - temporal alignment (current vs stale)
  - negation detection
  - status keyword alignment (operational vs expired)
  - neutral/no-overlap detection

The extractor understands that hypotheses have semantic orientations
(e.g., H1 wants "current and confirmed", H2 wants "stale or unconfirmed")
and matches evidence keywords directionally against those orientations.
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

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "and", "or", "but", "because", "so", "that", "this", "these",
    "those", "it", "its", "has", "have", "had", "in", "on", "at",
    "to", "from", "of", "for", "with", "by", "as", "not", "no",
    "should", "system", "about", "into", "than", "then", "also",
    "only", "may", "might", "can", "could", "would", "will",
    "there", "here", "which", "who", "what", "when", "where",
    "how", "why", "whether", "if", "then", "else",
})

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _strip_punct(word: str) -> str:
    """Strip leading/trailing punctuation from a word."""
    return word.strip(".,;:!?\"'()[]{}")


def _words(text: str) -> set[str]:
    """Return set of normalized, punctuation-stripped words."""
    norm = normalize_text(text)
    return {_strip_punct(w) for w in norm.split() if _strip_punct(w)}


def content_words(text: str) -> set[str]:
    """Return content words (excluding stop words)."""
    return {w for w in _words(text) if w not in STOP_WORDS and len(w) > 1}


# --- Semantic orientation keywords ---

# Entailment verbs (indicate the evidence supports whatever status it describes)
ENTAILMENT_VERBS = frozenset({
    "confirms", "confirmed", "validates", "establishes",
    "demonstrates", "shows", "proves", "verifies", "verified",
    "documents", "reported", "indicates", "states",
})

# Status keywords indicating "current/active/operational" claim
CURRENT_STATUS_KEYWORDS = frozenset({
    "current", "recent", "updated", "latest", "active",
    "operational", "sufficient", "available", "present",
    "true", "correct", "accurate", "reliable",
})

# Status keywords indicating "stale/inactive/expired" claim
STALE_STATUS_KEYWORDS = frozenset({
    "stale", "outdated", "old", "expired", "archived",
    "historical", "former", "previous", "past",
    "unverified", "unconfirmed", "missing", "absent",
    "lacking", "insufficient", "inactive", "offline",
    "unavailable", "none", "never", "fails", "failed",
    "broken", "invalid", "removed", "deleted", "discontinued",
})

# Explicit contradiction verbs
CONTRADICTION_VERBS = frozenset({
    "refutes", "denies", "contradicts", "disputes",
})

# Negation context patterns (suppress status keywords)
NEGATION_CONTEXT_PATTERNS = [
    r"\bwithout\b",
    r"\bnot\b.*\b(current|confirmed|operational|active|stale|outdated)\b",
    r"\bsilent\b",
    r"\btangential\b",
    r"\bunrelated\b",
    r"\bpassing\b",
    r"\bdifferent topic\b",
]

# Simple negation patterns (excluding \bno\b which matches "no longer" false positives)
NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bdenies\b",
    r"\brefutes\b",
    r"\bcontradicts\b",
    r"\bdisputes\b",
]

# Hypothesis orientation markers
HYP_CURRENT_MARKERS = frozenset({
    "current", "confirmed", "answer", "operational",
})

HYP_STALE_MARKERS = frozenset({
    "stale", "unconfirmed",
})

HYP_AMBIGUOUS_MARKERS = frozenset({
    "ambiguous",
})


def _has_negation(text: str) -> bool:
    lower = text.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _has_negation_context(text: str) -> bool:
    """Check if status keywords are in a negation/suppression context."""
    lower = text.lower()
    for pattern in NEGATION_CONTEXT_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _has_contradiction_verb(text: str) -> bool:
    return bool(_words(text) & CONTRADICTION_VERBS)


def _has_entailment_verb(text: str) -> bool:
    return bool(_words(text) & ENTAILMENT_VERBS)


def _has_current_status(text: str) -> set[str]:
    return _words(text) & CURRENT_STATUS_KEYWORDS


def _has_stale_status(text: str) -> set[str]:
    return _words(text) & STALE_STATUS_KEYWORDS


def _hyp_orientation(hyp_text: str) -> str:
    """Determine hypothesis orientation: 'current', 'stale', 'ambiguous', or 'unknown'."""
    words = _words(hyp_text)
    current_score = len(words & HYP_CURRENT_MARKERS)
    stale_score = len(words & HYP_STALE_MARKERS)
    ambiguous_score = len(words & HYP_AMBIGUOUS_MARKERS)
    if ambiguous_score > 0 and current_score == 0 and stale_score == 0:
        return "ambiguous"
    if current_score > stale_score:
        return "current"
    elif stale_score > current_score:
        return "stale"
    return "unknown"


def _subject_overlap(ev_text: str, hyp_text: str) -> set[str]:
    """Return overlapping content words between evidence and hypothesis."""
    ev_words = content_words(ev_text)
    hyp_words = content_words(hyp_text)
    return ev_words & hyp_words


# --- Extractor ---

class DeterministicRelationExtractor(SemanticRelationExtractor):
    """Deterministic keyword/pattern-based relation extractor v2.0.0.

    Rules (in priority order):
      1. Explicit contradiction verb + hypothesis reference -> CONTRADICT
      2. Temporal/status alignment (requires status keywords in evidence):
         - Evidence has "current" status + hypothesis wants "current" -> SUPPORT
         - Evidence has "current" status + hypothesis wants "stale" -> CONTRADICT
         - Evidence has "stale" status + hypothesis wants "stale" -> SUPPORT
         - Evidence has "stale" status + hypothesis wants "current" -> CONTRADICT
      3. Negation + status keywords opposing hypothesis -> CONTRADICT
      4. No clear status signal -> NEUTRAL (even with subject overlap)
      5. Default -> NEUTRAL
    """

    def __init__(self) -> None:
        identity = compute_extractor_identity(
            extractor_class="DeterministicRelationExtractor",
            extractor_version="2.5.0",
            relation_schema_version="1",
            normalization_rules=(
                "lowercase",
                "collapse_whitespace",
                "strip",
                "stopword_removal_for_content_words",
            ),
            thresholds={
                "subject_overlap_min": 1,
                "temporal_weight": 1.5,
                "contradiction_verb_weight": 2.0,
                "require_status_for_support": True,
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

        overlap = _subject_overlap(ev_norm, hyp_norm)
        has_neg = _has_negation(inp.evidence_proposition)
        has_neg_ctx = _has_negation_context(inp.evidence_proposition)
        has_contrad_verb = _has_contradiction_verb(ev_norm)
        has_entail_verb = _has_entailment_verb(ev_norm)
        ev_current = _has_current_status(ev_norm)
        ev_stale = _has_stale_status(ev_norm)
        hyp_orient = _hyp_orientation(hyp_norm)

        # Check if evidence explicitly references this hypothesis
        hyp_id = inp.hypothesis_id.lower()
        ev_mentions_hyp = hyp_id in ev_norm or f"hypothesis {hyp_id[-1]}" in ev_norm

        # Check if evidence explicitly references a DIFFERENT hypothesis
        ev_mentions_other_hyp = False
        for other_id in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            if other_id != hyp_id and (other_id in ev_norm or f"hypothesis {other_id[-1]}" in ev_norm):
                ev_mentions_other_hyp = True
                break

        # Suppress status keywords if in negation context
        if has_neg_ctx:
            ev_current = set()
            ev_stale = set()

        # Determine evidence claim direction
        # If evidence has entailment verb + current status -> claims "current"
        # If evidence has entailment verb + stale status -> claims "stale"
        # If evidence has contradiction verb + current status -> claims "not current" = "stale"
        # If evidence has contradiction verb + stale status -> claims "not stale" = "current"
        claims_current = False
        claims_stale = False

        if has_entail_verb and not has_contrad_verb:
            if ev_current and not ev_stale:
                claims_current = True
            elif ev_stale and not ev_current:
                claims_stale = True
            elif ev_current and ev_stale:
                # Both present - use the one closer to the entailment verb
                # Simplified: prefer stale if "stale" appears, since it's more specific
                claims_stale = True
        elif has_contrad_verb:
            if ev_current and not ev_stale:
                claims_stale = True  # contradicting current -> claims stale
            elif ev_stale and not ev_current:
                claims_current = True  # contradicting stale -> claims current
        elif ev_current and not ev_stale and not has_neg:
            claims_current = True
        elif ev_stale and not ev_current and not has_neg:
            claims_stale = True

        relation: RelationType
        reason: ExtractorReasonCode

        # Rule 1: Explicit contradiction verb + hypothesis reference -> CONTRADICT
        if has_contrad_verb and ev_mentions_hyp:
            relation = RelationType.CONTRADICT
            reason = ExtractorReasonCode.EXPLICIT_CONTRADICTION
        # Rule 2: Status alignment (skip for ambiguous hypotheses)
        elif claims_current and hyp_orient != "ambiguous":
            if hyp_orient == "current":
                relation = RelationType.SUPPORT
                reason = ExtractorReasonCode.TEMPORAL_MATCH
            elif hyp_orient == "stale":
                relation = RelationType.CONTRADICT
                reason = ExtractorReasonCode.TEMPORAL_MISMATCH
            else:
                relation = RelationType.NEUTRAL
                reason = ExtractorReasonCode.AMBIGUOUS
        elif claims_stale and hyp_orient != "ambiguous":
            if hyp_orient == "stale":
                relation = RelationType.SUPPORT
                reason = ExtractorReasonCode.TEMPORAL_MATCH
            elif hyp_orient == "current":
                relation = RelationType.CONTRADICT
                reason = ExtractorReasonCode.TEMPORAL_MISMATCH
            else:
                relation = RelationType.NEUTRAL
                reason = ExtractorReasonCode.AMBIGUOUS
        # Rule 3: Negation + overlap (but no status signal, no other-hyp reference) -> CONTRADICT
        elif has_neg and overlap and not claims_current and not claims_stale and hyp_orient != "ambiguous" and not ev_mentions_other_hyp:
            relation = RelationType.CONTRADICT
            reason = ExtractorReasonCode.NEGATION
        # Rule 4: No clear status signal or ambiguous hypothesis -> NEUTRAL
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
