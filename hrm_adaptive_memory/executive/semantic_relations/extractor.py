"""Abstract extractor interface for I3.12.

The extractor must NOT access:
  - verify_result
  - expected_terminal
  - correct_hypothesis_id
  - oracle_resolution_path
  - hidden evidence
  - task category

Only proposition text + hypothesis text are allowed inputs.

ExtractionInput carries ONLY text. If any forbidden field is set,
the extractor must raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hrm_adaptive_memory.executive.semantic_relations.schema import (
    SemanticRelation,
    RelationType,
    RelationGraph,
    ExtractorReasonCode,
    text_sha256,
)
from hrm_adaptive_memory.executive.semantic_relations.identity import (
    ExtractorIdentity,
)


# Fields the extractor must never see
FORBIDDEN_FIELDS = frozenset({
    "verify_result",
    "expected_terminal",
    "correct_hypothesis_id",
    "oracle_resolution_path",
    "category",
    "task_success",
    "hidden_evidence",
    "gold_relations",
    "supports",
    "contradicts",
})


@dataclass(frozen=True)
class ExtractionInput:
    """Input to the relation extractor.

    Carries ONLY text identifiers and proposition text.
    Must not carry any oracle or evaluator-side data.
    """
    evidence_id: str
    evidence_proposition: str
    hypothesis_id: str
    hypothesis_proposition: str

    def __post_init__(self) -> None:
        # Verify no forbidden fields are set via kwargs
        # (dataclass prevents extra fields, but this is a belt-and-braces check)
        pass

    @property
    def evidence_sha256(self) -> str:
        return text_sha256(self.evidence_proposition)

    @property
    def hypothesis_sha256(self) -> str:
        return text_sha256(self.hypothesis_proposition)


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting one relation."""
    relation: SemanticRelation
    input: ExtractionInput
    latency_ms: int | None = None


class SemanticRelationExtractor:
    """Abstract base class for semantic relation extractors.

    Subclasses must implement _extract_one().
    """

    def __init__(self, identity: ExtractorIdentity) -> None:
        self.identity = identity

    def extract(
        self,
        evidence_id: str,
        evidence_proposition: str,
        hypothesis_id: str,
        hypothesis_proposition: str,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Extract a single relation.

        Raises ValueError if any forbidden kwarg is supplied.
        """
        for key in kwargs:
            if key in FORBIDDEN_FIELDS:
                raise ValueError(
                    f"Extractor received forbidden field '{key}'. "
                    f"Extractor must not access oracle/evaluator data."
                )

        inp = ExtractionInput(
            evidence_id=evidence_id,
            evidence_proposition=evidence_proposition,
            hypothesis_id=hypothesis_id,
            hypothesis_proposition=hypothesis_proposition,
        )
        return self._extract_one(inp)

    def extract_graph(
        self,
        task_id: str,
        evidence_items: list[dict],
        hypotheses: list[dict],
        **kwargs: Any,
    ) -> RelationGraph:
        """Extract a complete relation graph for a task state.

        evidence_items: list of {"evidence_id": ..., "proposition": ...}
        hypotheses: list of {"hypothesis_id": ..., "proposition": ...}

        Raises ValueError if any evidence/hypothesis dict contains
        forbidden fields.
        """
        for item in evidence_items:
            for key in item:
                if key in FORBIDDEN_FIELDS:
                    raise ValueError(
                        f"Evidence item contains forbidden field '{key}'. "
                        f"Extractor must not access oracle/evaluator data."
                    )
        for hyp in hypotheses:
            for key in hyp:
                if key in FORBIDDEN_FIELDS:
                    raise ValueError(
                        f"Hypothesis contains forbidden field '{key}'. "
                        f"Extractor must not access oracle/evaluator data."
                    )

        relations: list[SemanticRelation] = []
        for ev in evidence_items:
            for hyp in hypotheses:
                result = self.extract(
                    evidence_id=ev["evidence_id"],
                    evidence_proposition=ev["proposition"],
                    hypothesis_id=hyp["hypothesis_id"],
                    hypothesis_proposition=hyp["proposition"],
                )
                relations.append(result.relation)

        return RelationGraph(
            task_id=task_id,
            relations=tuple(relations),
            extractor_identity_sha256=self.identity.sha256,
        )

    def _extract_one(self, inp: ExtractionInput) -> ExtractionResult:
        raise NotImplementedError("Subclasses must implement _extract_one")
