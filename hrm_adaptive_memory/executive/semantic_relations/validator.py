"""Validator for semantic relation extraction.

Ensures that:
  - No forbidden fields are present in extractor inputs
  - Relation graphs are well-formed
  - Behavioral equivalence holds (same text -> same relations)
"""
from __future__ import annotations

from hrm_adaptive_memory.executive.semantic_relations.extractor import FORBIDDEN_FIELDS
from hrm_adaptive_memory.executive.semantic_relations.schema import (
    RelationGraph,
    RelationType,
)


def validate_extraction_input(data: dict) -> list[str]:
    """Check that an extraction input dict contains no forbidden fields.

    Returns a list of violations (empty if valid).
    """
    violations = []
    for key in data:
        if key in FORBIDDEN_FIELDS:
            violations.append(f"forbidden field '{key}' present in extraction input")
    return violations


def validate_relation_graph(graph: RelationGraph) -> list[str]:
    """Check that a relation graph is well-formed.

    Returns a list of violations (empty if valid).
    """
    violations = []
    seen = set()
    for rel in graph.relations:
        # Check for duplicates
        key = (rel.evidence_id, rel.hypothesis_id)
        if key in seen:
            violations.append(f"duplicate relation for {key}")
        seen.add(key)

        # Check relation type is valid
        if not isinstance(rel.relation, RelationType):
            violations.append(
                f"invalid relation type for {key}: {rel.relation}"
            )

        # Check hashes are present
        if not rel.evidence_sha256:
            violations.append(f"missing evidence_sha256 for {key}")
        if not rel.hypothesis_sha256:
            violations.append(f"missing hypothesis_sha256 for {key}")

    return violations


def check_behavioral_equivalence(
    extractor,
    evidence_text: str,
    hypothesis_text: str,
    n_runs: int = 3,
) -> bool:
    """Verify that the extractor produces the same output for the same input.

    For deterministic extractors, this should always pass.
    For LLM extractors, this may fail and should be reported.
    """
    results = []
    for _ in range(n_runs):
        result = extractor.extract(
            evidence_id="E1",
            evidence_proposition=evidence_text,
            hypothesis_id="H1",
            hypothesis_proposition=hypothesis_text,
        )
        results.append(result.relation.relation)

    return len(set(results)) == 1
