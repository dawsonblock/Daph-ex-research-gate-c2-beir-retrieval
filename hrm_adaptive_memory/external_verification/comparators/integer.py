from __future__ import annotations

from decimal import Decimal

from .base import ComparisonOutcome, ComparisonResult, RelationSchema
from ._helpers import decimal_value


class IntegerComparator:
    def compare(self, schema: RelationSchema, claim_value: object,
                evidence_value: object) -> ComparisonResult:
        try:
            claim, evidence = decimal_value(claim_value), decimal_value(evidence_value)
            if claim != claim.to_integral_value() or evidence != evidence.to_integral_value():
                raise ValueError("integer required")
            return ComparisonResult(ComparisonOutcome.MATCH if claim == evidence else ComparisonOutcome.MISMATCH,
                                    int(claim), int(evidence),
                                    "INTEGER_EXACT_MATCH" if claim == evidence else "INTEGER_EXACT_MISMATCH")
        except ValueError:
            return ComparisonResult(ComparisonOutcome.INCONCLUSIVE, None, None, "INTEGER_PARSE_ERROR")
