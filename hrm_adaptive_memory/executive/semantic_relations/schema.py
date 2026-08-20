"""Core schema for semantic relations in I3.12.

A SemanticRelation is a single evidence x hypothesis relation:
  SUPPORT, CONTRADICT, NEUTRAL, or UNKNOWN.

A RelationGraph is the complete set of relations for one task state.

These are distinct from the oracle supports/contradicts fields in
EvidenceItem. The oracle fields are gold-standard evaluator-side data.
SemanticRelation values are inferred by the extractor and are
controller-visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class RelationType(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class ExtractorReasonCode(str, Enum):
    """Why the extractor produced this relation."""
    LITERAL_ENTAILMENT = "LITERAL_ENTAILMENT"
    EXPLICIT_CONTRADICTION = "EXPLICIT_CONTRADICTION"
    NEGATION = "NEGATION"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"
    TEMPORAL_MATCH = "TEMPORAL_MATCH"
    NO_OVERLAP = "NO_OVERLAP"
    AMBIGUOUS = "AMBIGUOUS"
    KEYWORD_MATCH = "KEYWORD_MATCH"
    KEYWORD_CONTRADICTION = "KEYWORD_CONTRADICTION"
    DEFAULT_NEUTRAL = "DEFAULT_NEUTRAL"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"


@dataclass(frozen=True)
class SemanticRelation:
    """A single inferred evidence x hypothesis relation.

    Attributes:
        evidence_id: the evidence item this relation is about
        hypothesis_id: the hypothesis this relation relates to
        relation: SUPPORT, CONTRADICT, NEUTRAL, or UNKNOWN
        confidence: optional confidence in [0, 1]; None for deterministic
        reason_code: why the extractor produced this relation
        evidence_sha256: hash of the evidence proposition text
        hypothesis_sha256: hash of the hypothesis proposition text
    """
    evidence_id: str
    hypothesis_id: str
    relation: RelationType
    confidence: float | None
    reason_code: ExtractorReasonCode
    evidence_sha256: str
    hypothesis_sha256: str

    def as_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "hypothesis_id": self.hypothesis_id,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "reason_code": self.reason_code.value,
            "evidence_sha256": self.evidence_sha256,
            "hypothesis_sha256": self.hypothesis_sha256,
        }


@dataclass(frozen=True)
class RelationGraph:
    """Complete set of inferred relations for one task state.

    This is the controller-visible epistemic graph that replaces
    the oracle supports/contradicts fields in S1 (raw proposition)
    condition.
    """
    task_id: str
    relations: tuple[SemanticRelation, ...]
    extractor_identity_sha256: str

    def relations_for_evidence(self, evidence_id: str) -> tuple[SemanticRelation, ...]:
        """All relations involving a given evidence item."""
        return tuple(r for r in self.relations if r.evidence_id == evidence_id)

    def supports_for_hypothesis(self, hypothesis_id: str) -> tuple[SemanticRelation, ...]:
        return tuple(
            r for r in self.relations
            if r.hypothesis_id == hypothesis_id and r.relation is RelationType.SUPPORT
        )

    def contradicts_for_hypothesis(self, hypothesis_id: str) -> tuple[SemanticRelation, ...]:
        return tuple(
            r for r in self.relations
            if r.hypothesis_id == hypothesis_id and r.relation is RelationType.CONTRADICT
        )

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "relations": [r.as_dict() for r in self.relations],
            "extractor_identity_sha256": self.extractor_identity_sha256,
        }

    @property
    def relation_graph_sha256(self) -> str:
        """Stable hash of the relation graph contents."""
        import hashlib
        import json
        payload = json.dumps(
            [{"e": r.evidence_id, "h": r.hypothesis_id, "r": r.relation.value}
             for r in sorted(self.relations, key=lambda x: (x.evidence_id, x.hypothesis_id))],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def text_sha256(text: str) -> str:
    """SHA-256 of a text string."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
