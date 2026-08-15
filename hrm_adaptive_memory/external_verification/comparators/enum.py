from __future__ import annotations

from .base import ComparisonOutcome, ComparisonResult, RelationSchema
from ._helpers import canonical_string


class EnumComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        try:
            allowed = {canonical_string(item) for item in schema.allowed_values}
            claim, evidence = canonical_string(claim_value), canonical_string(evidence_value)
            if claim not in allowed or evidence not in allowed:
                return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, claim, evidence,
                                        "ENUM_VALUE_NOT_ALLOWED")
            matched = claim == evidence
            return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                    claim, evidence, "ENUM_MATCH" if matched else "ENUM_MISMATCH")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None, "ENUM_PARSE_ERROR")
