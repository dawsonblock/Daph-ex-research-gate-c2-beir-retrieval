"""Semantic relation extraction for I3.12.

This package provides the upstream semantic-relation layer that
transforms raw proposition text into controller-visible epistemic
relations (SUPPORT / CONTRADICT / NEUTRAL).

The extractor must NOT access:
  - verify_result
  - expected_terminal
  - correct_hypothesis_id
  - oracle_resolution_path
  - hidden evidence
  - task category

Only proposition text + hypothesis text are allowed inputs.
"""
from hrm_adaptive_memory.executive.semantic_relations.schema import (
    SemanticRelation,
    RelationType,
    RelationGraph,
    ExtractorReasonCode,
)
from hrm_adaptive_memory.executive.semantic_relations.extractor import (
    SemanticRelationExtractor,
    ExtractionInput,
    ExtractionResult,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.semantic_relations.identity import (
    ExtractorIdentity,
    compute_extractor_identity,
)
from hrm_adaptive_memory.executive.semantic_relations.metrics import (
    RelationMetrics,
    compute_relation_metrics,
    ConfusionMatrix,
)
from hrm_adaptive_memory.executive.semantic_relations.receipts import (
    RelationExtractionReceipt,
)

__all__ = [
    "SemanticRelation",
    "RelationType",
    "RelationGraph",
    "ExtractorReasonCode",
    "SemanticRelationExtractor",
    "ExtractionInput",
    "ExtractionResult",
    "DeterministicRelationExtractor",
    "ExtractorIdentity",
    "compute_extractor_identity",
    "RelationMetrics",
    "compute_relation_metrics",
    "ConfusionMatrix",
    "RelationExtractionReceipt",
]
