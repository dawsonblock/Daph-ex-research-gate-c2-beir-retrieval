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
from .authority_registry import AuthorityNotRegistered, AuthorityRegistry
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
    authority_registry: AuthorityRegistry
    method: str = "authoritative_typed_field"
    method_version: str = "2.0.0-v2b"

    def _attestation_error(self, claim: ClaimRecord, evidence: ExternalEvidenceRecord) -> str | None:
        """Validate provenance produced by the registered authority path.

        A source-class enum is caller-controlled metadata. Typed evidence can
        only influence truth-bearing results when its recorded acquisition
        matches this frozen registry/extractor contract.
        """
        try:
            self.authority_registry.require_truth_bearing()
        except ValueError:
            return "AUTHORITY_REGISTRY_NOT_FROZEN"
        attestation = dict(evidence.authority_attestation)
        required = {
            "schema", "authority_id", "registry_sha256", "registry_status", "relation",
            "extractor_id", "extractor_module", "extractor_symbol", "extractor_sha256",
            "schema_id", "source_uri", "final_uri", "peer_ip", "raw_content_sha256",
            "acquisition_method", "acquisition_version",
        }
        if not required.issubset(attestation):
            return "AUTHORITY_ATTESTATION_MISSING"
        if (attestation["schema"] != "DAPH_AUTHORITY_ATTESTATION_V1"
                or attestation["relation"] != claim.canonical_relation
                or attestation["registry_sha256"] != self.authority_registry.identity()["sha256"]
                or attestation["registry_status"] != self.authority_registry.status.value
                or attestation["source_uri"] != evidence.source_uri
                or attestation["final_uri"] != evidence.canonical_source_uri
                or attestation["raw_content_sha256"] != evidence.raw_content_hash
                or evidence.acquisition_method != attestation["acquisition_method"]
                or evidence.acquisition_version != attestation["acquisition_version"]
                or attestation["acquisition_method"] != "registered_authority"
                or attestation["acquisition_version"] != "2.0.0-v2b"):
            return "AUTHORITY_ATTESTATION_MISMATCH"
        try:
            definition = self.authority_registry.resolve(
                authority_id=str(attestation["authority_id"]),
                relation=claim.canonical_relation, source_uri=evidence.source_uri)
            final_definition = self.authority_registry.resolve(
                authority_id=str(attestation["authority_id"]),
                relation=claim.canonical_relation, source_uri=evidence.canonical_source_uri)
        except AuthorityNotRegistered:
            return "AUTHORITY_ATTESTATION_URI_NOT_REGISTERED"
        if definition != final_definition or any((
            attestation["extractor_id"] != definition.extractor_id,
            attestation["extractor_module"] != definition.extractor_module,
            attestation["extractor_symbol"] != definition.extractor_symbol,
            attestation["extractor_sha256"] != definition.extractor_sha256,
            attestation["schema_id"] != definition.schema_id,
        )):
            return "AUTHORITY_ATTESTATION_EXTRACTOR_MISMATCH"
        try:
            import ipaddress
            if not ipaddress.ip_address(str(attestation["peer_ip"])).is_global:
                return "AUTHORITY_ATTESTATION_PEER_NOT_PUBLIC"
        except ValueError:
            return "AUTHORITY_ATTESTATION_PEER_INVALID"
        return None

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
        attestation_error = self._attestation_error(claim, evidence)
        if attestation_error is not None:
            return self._decision(VerificationResult.INCONCLUSIVE, claim, evidence, base,
                                  attestation_error)
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
