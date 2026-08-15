from __future__ import annotations

from .base import ComparisonOutcome, ComparisonResult, RelationSchema
from ._helpers import canonical_string


class CanonicalStringComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        try:
            claim, evidence = canonical_string(claim_value), canonical_string(evidence_value)
            matched = claim == evidence
            return ComparisonResult(ComparisonOutcome.MATCH if matched else ComparisonOutcome.MISMATCH,
                                    claim, evidence,
                                    "CANONICAL_STRING_MATCH" if matched else "CANONICAL_STRING_MISMATCH")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None,
                                    "CANONICAL_STRING_PARSE_ERROR")
