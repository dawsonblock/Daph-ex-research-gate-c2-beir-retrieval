"""V2B typed verifier, deliberately separate from the frozen V2A verifier."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping

from hrm_adaptive_memory.memory_write.claim_store import ClaimRecord
from hrm_adaptive_memory.memory_write.verification import VerificationResult

from .comparators import (ComparisonOutcome, RelationSchema,
                          default_comparator_registry)
from .core import (ExternalEvidenceRecord, SourceType, VerificationDecision)


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class RelationSchemaRegistry:
    def __init__(self, schemas: Iterable[RelationSchema]):
        entries = tuple(schemas)
        self._by_relation = {schema.relation: schema for schema in entries}
        if len(entries) != len(self._by_relation):
            raise ValueError("relation schemas must be unique")

    def resolve(self, relation: str) -> RelationSchema | None:
        return self._by_relation.get(relation)

    def identity(self) -> Mapping[str, object]:
        values = [
            {"relation": item.relation, "value_type": item.value_type.value,
             "comparator": item.comparator.value, "canonical_unit": item.canonical_unit,
             "absolute_tolerance": item.absolute_tolerance,
             "relative_tolerance": item.relative_tolerance,
             "allowed_values": list(item.allowed_values)}
            for item in sorted(self._by_relation.values(), key=lambda value: value.relation)
        ]
        return {"schema": "DAPH_RELATION_SCHEMA_REGISTRY_V1", "schemas": values,
                "sha256": _sha256(values)}


@dataclass(frozen=True)
class TypedFieldVerifier:
    """Fail-closed relation-schema verifier for post-V2A experiments."""

    schemas: RelationSchemaRegistry
    method: str = "authoritative_typed_field"
    method_version: str = "1.0.0-v2b"

    def verify(self, claim: ClaimRecord, evidence: ExternalEvidenceRecord) -> VerificationDecision:
        base: dict[str, object] = {
            "entity": claim.canonical_entity, "field": claim.canonical_relation,
            "claim_value": claim.value, "evidence_id": evidence.evidence_id,
        }
        if evidence.source_type not in {
            SourceType.AUTHORITATIVE_STRUCTURED_DATA, SourceType.OFFICIAL_PRIMARY_SOURCE,
        }:
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  "SOURCE_CLASS_NOT_QUALIFIED")
        schema = self.schemas.resolve(claim.canonical_relation)
        if schema is None:
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  "UNSUPPORTED_RELATION_SCHEMA")
        fields = dict(evidence.extracted_fields)
        if "entity" not in fields or str(fields["entity"]).strip().casefold() != claim.canonical_entity.strip().casefold():
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  "ENTITY_MISMATCH_OR_AMBIGUOUS")
        if claim.canonical_relation not in fields:
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  "MISSING_COMPARISON_FIELD")
        base["schema"] = schema.relation
        base["evidence_value"] = fields[claim.canonical_relation]
        comparison = default_comparator_registry().compare(
            schema, claim.value, fields[claim.canonical_relation])
        base["typed_comparison"] = {
            "outcome": comparison.outcome.value,
            "reason_code": comparison.reason_code,
            "claim_value": comparison.claim_value,
            "evidence_value": comparison.evidence_value,
        }
        if comparison.outcome is ComparisonOutcome.INCONCLUSIVE:
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  comparison.reason_code)
        if comparison.outcome is ComparisonOutcome.MATCH:
            return self._decision(VerificationResult.SUPPORTED, claim, evidence, base,
                                  "TYPED_FIELD_MATCH")
        result = (VerificationResult.FALSIFIED
                  if evidence.source_type is SourceType.AUTHORITATIVE_STRUCTURED_DATA
                  else VerificationResult.CONTRADICTED)
        return self._decision(result, claim, evidence, base, "TYPED_FIELD_MISMATCH")

    def _decision(self, result: VerificationResult, claim: ClaimRecord,
                  evidence: ExternalEvidenceRecord, fields: Mapping[str, object],
                  reason_code: str) -> VerificationDecision:
        receipt = _sha256({"result": result.value, "method": self.method,
                          "version": self.method_version, "claim": claim.record_id,
                          "evidence": evidence.evidence_id, "fields": dict(fields),
                          "reason_code": reason_code})
        return VerificationDecision(result, self.method, self.method_version,
                                    claim.record_id, (evidence.evidence_id,), dict(fields),
                                    reason_code, receipt)
